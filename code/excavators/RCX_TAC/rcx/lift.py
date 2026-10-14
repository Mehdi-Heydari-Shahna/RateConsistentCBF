"""Extended lift of the compiled interface: tangent map E, its time derivative as a matrix, and task maps.

RoboCompiler (Sec. III-B, Listing 4) provides  v = E u  and the curvature vector  c = Edot u.  The VDC evaluator
needs the derivative of the *required* velocity,  d/dt (E u_r) = E u_r_dot + Edot u_r,  i.e. Edot applied to a
vector other than u.  With the selected rows S and the passive block  R_p = S C_p  of the closure Jacobian C,

    S C E = 0   =>   Edot_p = - R_p^{-1} S (Cdot E),      Edot_a = 0,

where Cdot is the directional derivative of the analytic closure Jacobian along the actual velocity  v = E u,
evaluated by the same centred difference as the paper's curvature term.  ``Edot @ u`` reproduces that term.

Held coordinates.  A mechanism that stands on the ground has independent coordinates that the ground holds (the
anchors of its contacts, rcx.spec section ``support``).  With ``free`` (positions of the commanded independent
coordinates in the compiled order) the lift is returned for those columns: ``E``, ``ET``, ``Edot``, ``EdotT`` map the
commanded speeds, and the columns of the held coordinates are kept as ``EX``, ``EXT`` (their generalized forces are the
contact forces).  The velocity ``v`` and the curvature vector ``cT`` are those of the complete motion (the held speeds
are part of ``u``; they are zero while the contacts hold).
"""
from __future__ import annotations

import numpy as np


class Lift:
    __slots__ = ('q', 'u', 'E', 'ET', 'Edot', 'EdotT', 'v', 'vT', 'cT', 'rows', 'info', 'closure_J', 'EX', 'EXT')


def extended_lift(ci, q, u, h0=1e-5, free=None, held=None):
    """Tangent lift, its derivative matrix and the tree rows at a closed augmented configuration ``q``."""
    u = np.asarray(u, float)
    E, info = ci.pacdm.mapping(q)
    if E is None:
        raise ValueError(f'Tangent map rejected: {info}')
    v = E @ u
    J = ci.graph.residual(q)[1]
    eps = h0 / max(1., float(np.max(np.abs(v))) if v.size else 1.)
    Cdot = (ci.graph.residual(q + eps * v)[1] - ci.graph.residual(q - eps * v)[1]) / (2. * eps)
    rows = np.asarray(info['rows'], int)
    Edot = np.zeros_like(E)
    Edot[ci.passive] = -np.linalg.solve(J[np.ix_(rows, ci.passive)], (Cdot @ E)[rows])
    out = Lift()
    out.q, out.u, out.E, out.Edot, out.v, out.rows, out.info, out.closure_J = q, u, E, Edot, v, rows, info, J
    out.ET, out.EdotT = E[ci.tree_rows], Edot[ci.tree_rows]
    out.vT = v[ci.tree_rows]
    out.cT = out.EdotT @ u
    out.EX = out.EXT = None
    if (free is None) != (held is None):
        raise ValueError('free and held coordinates are given together')
    if free is not None:
        out.EX, out.EXT = out.E[:, held], out.ET[:, held]
        out.E, out.Edot, out.ET, out.EdotT = out.E[:, free], out.Edot[:, free], out.ET[:, free], out.EdotT[:, free]
        out.u = u[free]
    return out


class TaskMap:
    """Control objective in physical task coordinates  x = selected augmented coordinates.

    x_dot = Jx u  with  Jx = E[task rows];  Jx_dot = Edot[task rows].  For the partition whose independent
    coordinates are the task coordinates themselves Jx = I and Jx_dot = 0; for any other valid partition the
    same physical required motion is obtained through Jx^{-1} (representation covariance)."""

    def __init__(self, ci, task_ids, held=0):
        self.ids = list(task_ids)
        self.rows = np.array([ci.coordinate_ids.index(k) for k in self.ids], int)
        if len(self.ids) != ci.nu - held:
            raise ValueError('The number of task coordinates must equal the mobility of the controlled mechanism'
                             + (' with its held coordinates fixed' if held else ''))

    def evaluate(self, lift):
        Jx, Jxd = lift.E[self.rows], lift.Edot[self.rows]
        return lift.q[self.rows], Jx, Jxd
