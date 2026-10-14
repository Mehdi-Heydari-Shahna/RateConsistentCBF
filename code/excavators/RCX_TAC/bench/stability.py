"""Static support of a machine on its base: centre of mass of the compiled graph at a configuration and the tipping
moment margins about the edges of the support polygon.

Design aid for layouts and references (where may a loaded bucket be carried?) and a diagnostic of the reports.  It
uses the records of the graph only (masses, centres of mass) with the generated frames of the body plan; no plant.

Convention: everything in the graph root frame with z up; the support polygon is a convex polygon in the plane z = 0
given counter-clockwise.  The margin of an edge is the restoring moment about that edge divided by the weight, i.e. the
distance of the centre of pressure from the edge (positive inside); the base starts to tip about an edge when its
margin reaches zero.
"""
from __future__ import annotations

import numpy as np

from .reference import closed_configuration

G = 9.81


class Stability:
    def __init__(self, physical, xc):
        self.xc, self.ci, self.plan = xc, xc.ci, xc.plan
        bodies = {b['id']: b for b in physical['bodies']}
        self.frames = list(self.plan.frame_ids)
        self.mass = np.array([float(bodies[f]['mass_kg']) if f in bodies else 0. for f in self.frames])
        self.com = np.array([np.asarray(bodies[f]['com_m'], float) if f in bodies else np.zeros(3) for f in self.frames])
        self.total = float(self.mass.sum())

    def body_coms(self, u):
        ci = self.ci
        q = closed_configuration(ci, self.xc.compose(np.asarray(u, float)))
        kin = self.plan.evaluate(q[ci.tree_rows], np.zeros(ci.nt), dynamics=False)
        return kin.p + np.einsum('nij,nj->ni', kin.R, self.com)

    def centre_of_mass(self, u):
        c = self.body_coms(u)
        return self.total, (self.mass[:, None] * c).sum(axis=0) / self.total

    def margins(self, u, polygon, loads=()):
        """Edge margins (m) of the centre of pressure for configuration u.  ``loads``: iterable of (force (3,), point (3,))
        external forces on the machine in the root frame (e.g. payload weight at the bucket, soil reaction).
        Returns dict(mass, com, cop, vertical_N, margins (one per edge), min_margin)."""
        mass, com = self.centre_of_mass(u)
        forces = [(np.array([0., 0., -mass * G]), com)] + [(np.asarray(f, float), np.asarray(p, float)) for f, p in loads]
        Fz = sum(f[2] for f, _ in forces)                       # negative: pressing on the ground
        # moment of all forces about the origin; the ground reaction (0, 0, N), N = -Fz, at the centre of pressure c balances
        # it:  c x (0, 0, N) + M = 0,  i.e.  c = (M_y, -M_x) / N
        M = sum(np.cross(p, f) for f, p in forces)
        N = -Fz
        cop = np.array([M[1] / N, -M[0] / N]) if N > 1e-9 else np.array([np.nan, np.nan])
        poly = np.asarray(polygon, float)
        out = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            e = (b - a) / np.linalg.norm(b - a)
            n = np.array([-e[1], e[0]])                          # inward normal of a counter-clockwise polygon
            out.append(float(n @ (cop - a)))
        return dict(mass=mass, com=com, cop=cop, vertical_N=float(N), margins=out, min_margin=float(min(out)))


def rectangle(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
