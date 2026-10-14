"""Signed-distance primitives for the obstacle side of a clearance pair, with unit gradients and Hessians.

Robot-side collision geometry is a set of spheres attached to graph bodies (a sphere set bounds links, capsules and
convex shells); the obstacle side is one of the primitives below, expressed in its owner frame (the inertial
frame or a graph body).  All functions are vectorised over K query points (K, 3) given in the owner frame and
return  d (K,), n (K, 3), H (K, 3, 3)  with  d = signed distance to the primitive surface,  n = grad d,
H = Hessian of d.  The distance to a convex set is continuously differentiable outside the set; its Hessian is
piecewise continuous (it jumps across the face/edge/vertex regions of a box and across the end caps of a
segment), which is what a second-order barrier condition needs (bounded, piecewise-continuous h_ddot).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def sdf_plane(x, normal, offset):
    """Half-space  normal . x >= offset  is the allowed side (any non-zero normal: the signed distance is
    (normal . x - offset) / |normal|)."""
    n = np.asarray(normal, float)
    length = float(np.linalg.norm(n))
    n = n / length
    d = x @ n - float(offset) / length
    return d, np.broadcast_to(n, x.shape).copy(), np.zeros(x.shape + (3,))


def _unit_or_fallback(diff, dist, fallback):
    """Unit direction diff / dist; where the query point lies on the primitive's centre or axis (dist = 0, no unique
    normal) a fixed unit vector is returned and the curvature is set to zero by the caller."""
    degenerate = dist < _EPS
    safe = np.where(degenerate, 1., dist)
    n = diff / safe[:, None]
    if degenerate.any():
        n[degenerate] = fallback
    return n, safe, degenerate


def sdf_sphere(x, center, radius):
    diff = x - np.asarray(center, float)
    dist = np.linalg.norm(diff, axis=1)
    n, safe, degenerate = _unit_or_fallback(diff, dist, np.array([0., 0., 1.]))
    H = (np.eye(3)[None] - n[:, :, None] * n[:, None, :]) / safe[:, None, None]
    H[degenerate] = 0.
    return dist - float(radius), n, H


def sdf_segment(x, a, b, radius):
    """Capsule obstacle: distance to the segment a-b minus ``radius`` (a pipe, a rail, a bar)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if not L > 0.:
        return sdf_sphere(x, a, radius)
    e = (b - a) / L
    t_raw = (x - a) @ e
    t = np.clip(t_raw, 0., L)
    diff = x - (a + t[:, None] * e)
    dist = np.linalg.norm(diff, axis=1)
    side = np.cross(e, [1., 0., 0.]) if abs(e[0]) < 0.9 else np.cross(e, [0., 1., 0.])
    n, safe, degenerate = _unit_or_fallback(diff, dist, side / np.linalg.norm(side))
    P = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
    interior = (t_raw > 0.) & (t_raw < L)
    P = P - interior[:, None, None] * (e[:, None] * e[None, :])[None]
    H = P / safe[:, None, None]
    H[degenerate] = 0.
    return dist - float(radius), n, H


def sdf_box(x, half):
    """Axis-aligned box centred at the owner-frame origin of the primitive (the frame transform is applied by the caller)."""
    half = np.asarray(half, float)
    q = np.abs(x) - half
    sgn = np.where(x >= 0., 1., -1.)
    qp = np.maximum(q, 0.)
    out_dist = np.linalg.norm(qp, axis=1)
    outside = out_dist > 0.
    K = len(x)
    d = np.empty(K)
    n = np.zeros((K, 3))
    H = np.zeros((K, 3, 3))
    if outside.any():
        do = out_dist[outside]
        no = sgn[outside] * qp[outside] / do[:, None]
        act = (q[outside] > 0.).astype(float)
        Ho = np.zeros((int(outside.sum()), 3, 3))
        idx = np.arange(3)
        Ho[:, idx, idx] = act
        Ho = (Ho - no[:, :, None] * no[:, None, :]) / do[:, None, None]
        d[outside], n[outside], H[outside] = do, no, Ho
    inside = ~outside
    if inside.any():
        qi = q[inside]
        k = np.argmax(qi, axis=1)
        rows = np.arange(len(k))
        d[inside] = qi[rows, k]
        ni = np.zeros((len(k), 3))
        ni[rows, k] = sgn[inside][rows, k]
        n[inside] = ni
    return d, n, H


def rpy_matrix(rpy):
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class Obstacle:
    """One obstacle primitive in its owner frame; ``evaluate`` takes points in that owner frame."""

    def __init__(self, record):
        self.id = record['id']
        self.frame = record.get('frame', 'world')
        self.type = record['type']
        self.record = dict(record)
        if self.type == 'plane':                           # allowed side  normal . x >= offset, for any non-zero normal
            self.normal = np.asarray(record['normal'], float)
            length = float(np.linalg.norm(self.normal))
            self.normal = self.normal / length
            self.offset = float(record['offset']) / length
        elif self.type == 'box':
            self.center = np.asarray(record['center'], float)
            self.half = np.asarray(record['half'], float)
            self.R = rpy_matrix(record.get('rpy', (0., 0., 0.)))
        elif self.type == 'capsule':
            self.a, self.b, self.radius = np.asarray(record['a'], float), np.asarray(record['b'], float), float(record['radius'])
        elif self.type == 'sphere':
            self.center, self.radius = np.asarray(record['center'], float), float(record['radius'])
        else:
            raise ValueError(f'{self.id}: unsupported obstacle type {self.type}')

    def evaluate(self, x):
        if self.type == 'plane':
            return sdf_plane(x, self.normal, self.offset)
        if self.type == 'box':
            d, n, H = sdf_box((x - self.center) @ self.R, self.half)
            return d, n @ self.R.T, self.R[None] @ H @ self.R.T[None]
        if self.type == 'capsule':
            return sdf_segment(x, self.a, self.b, self.radius)
        return sdf_sphere(x, self.center, self.radius)
