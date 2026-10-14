"""Small geometry helpers for building collision envelopes from part meshes (no mesh library needed at run time)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_stl_triangles(path, scale=1.0):
    """Triangles (N, 3, 3) of a binary or ASCII STL file."""
    data = Path(path).read_bytes()
    n = struct.unpack('<I', data[80:84])[0] if len(data) >= 84 else -1
    if n >= 0 and len(data) == 84 + 50 * n:                    # binary
        rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
        T = rec['v'].astype(float)
    else:
        V = np.array([[float(x) for x in line.split()[1:4]] for line in data.decode('ascii', 'ignore').splitlines()
                      if line.strip().startswith('vertex')], float)
        T = V.reshape(-1, 3, 3)
    return T * scale


def read_stl(path, scale=1.0, spacing=None):
    """Points of an STL surface: its unique vertices, or (with ``spacing``) a deterministic sample of every triangle with
    about that distance between the points (barycentric lattice), so that large flat faces are represented as well."""
    T = read_stl_triangles(path, scale)
    if spacing is None:
        return np.unique(np.round(T.reshape(-1, 3), 9), axis=0)
    out = [T.reshape(-1, 3)]
    edge = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
    for k in np.unique(np.maximum(1, np.ceil(edge / spacing).astype(int))):
        if k < 2:
            continue
        sel = T[np.maximum(1, np.ceil(edge / spacing).astype(int)) == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')
        keep = i + j <= k
        a, b = i[keep] / k, j[keep] / k
        w = np.c_[1. - a - b, a, b]                              # (points, 3)
        out.append(np.einsum('pc,tcx->tpx', w, sel).reshape(-1, 3))
    return np.unique(np.round(np.vstack(out), 6), axis=0)


def sphere_chain(V, axis, count, shrink=1.0, min_radius=0.05):
    """Chain of bounding spheres along coordinate ``axis`` of the vertices V: the vertices are cut into ``count`` slices of
    equal length; every slice gets the sphere around the centre of its bounding box that contains its vertices
    (radius times ``shrink``).  Returns (centres (count, 3), radii (count,)); empty slices are skipped."""
    V = np.asarray(V, float)
    lo, hi = V[:, axis].min(), V[:, axis].max()
    edges = np.linspace(lo, hi, count + 1)
    C, R = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        P = V[(V[:, axis] >= a - 1e-9) & (V[:, axis] <= b + 1e-9)]
        if len(P) == 0:
            continue
        c = 0.5 * (P.min(0) + P.max(0))
        c[axis] = 0.5 * (a + b)
        other = [k for k in range(3) if k != axis]
        # the slice is covered when the sphere contains the bounding rectangle of the slice at both of its ends
        half = 0.5 * (P[:, other].max(0) - P[:, other].min(0))
        r = float(np.sqrt(half @ half + (0.5 * (b - a)) ** 2))
        C.append(c)
        R.append(max(min_radius, shrink * r))
    return np.asarray(C), np.asarray(R)


def bounding_box(V):
    """(centre, half sizes) of the axis-aligned bounding box of the vertices."""
    V = np.asarray(V, float)
    lo, hi = V.min(0), V.max(0)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def centre_line(V, axis, count):
    """Centres of the bounding rectangles of ``count`` slices along ``axis`` and their half extents (count, 3)."""
    V = np.asarray(V, float)
    lo, hi = V[:, axis].min(), V[:, axis].max()
    edges = np.linspace(lo, hi, count + 1)
    C, H = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        P = V[(V[:, axis] >= a - 1e-9) & (V[:, axis] <= b + 1e-9)]
        if len(P) == 0:
            continue
        c, h = bounding_box(P)
        c[axis], h[axis] = 0.5 * (a + b), 0.5 * (b - a)
        C.append(c)
        H.append(h)
    return np.asarray(C), np.asarray(H)
