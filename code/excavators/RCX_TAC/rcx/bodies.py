"""Generated body-level kinematics of the compiled tree on a moving base, and the VDC body modules.

Conventions (same as the ported tree oracle ``core.tree_dynamics`` and ``core.reduced.floating_rows``):
generalized velocity  nu = [v_base (3, world), omega_base (3, world), v_T (tree speeds)];  body Jacobians are
linear-first, world-aligned and taken at the body origin, ``V_B = J_B nu``; ``D_B`` is the time derivative of
``J_B`` along the actual motion, so that the origin acceleration is ``J_B nu_dot + D_B nu``.

Everything is generated from the compiled structure (edges of the spanning tree with their attachment
frames); no mechanism-specific expression is written here.  The implementation is independent of the ported
oracle and is checked against it and against Pinocchio in ``tests/test_rcx.py``.

VDC body module (world-aligned, at the centre of mass), for a required generalized velocity ``nu_r`` and its
derivative ``nu_r_dot``:

    v_r = Jc nu_r,                    w_r = Jw nu_r                         required twist
    a_r = Jc nu_r_dot + Dc nu_r,      al_r = Jw nu_r_dot + Dw nu_r          required acceleration
    F_r = m (a_r - g) + k m (v_r - v)                                       required net force
    T_r = I al_r + w x (I w_r) + k I (w_r - w)                              required net moment
    nu_B = 1/2 m |v_r - v|^2 + 1/2 (w_r - w)^T I (w_r - w)                  storage (accompanying function)

The factorization ``w x (I w_r)`` is the one for which  (V_r - V)^T (F*_r - F*) = d(nu_B)/dt + 2 k nu_B.
"""
from __future__ import annotations

import math

import numpy as np

_I3 = np.eye(3)


def skew(v):
    x, y, z = v
    return np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))


def skew_stack(v):
    """(B, 3) -> (B, 3, 3) cross-product matrices."""
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1], out[:, 0, 2] = -v[:, 2], v[:, 1]
    out[:, 1, 0], out[:, 1, 2] = v[:, 2], -v[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -v[:, 1], v[:, 0]
    return out


class Kin:
    """Result of one body-plan evaluation (plain attribute container)."""
    __slots__ = ('nu', 'R', 'p', 'J', 'D', 'v', 'w', 'Jc', 'Dc', 'Jw', 'Dw', 'Iw', 'vc', 'wb', 'rc', 'M', 'h', 'C', 'grav',
                 'kinetic')


class BodyPlan:
    """Evaluation plan of every frame of the compiled spanning tree on a moving root frame."""

    def __init__(self, comp, gravity):
        s = comp.structure
        cmg = comp.cmg
        self.nt = int(s['tree_coordinates'])
        self.nf = 6 + self.nt
        self.gravity = np.asarray(gravity, float)
        self.root = s['root']
        edges = sorted(s['edges'], key=lambda e: e['depth'])
        self.frame_ids = [self.root] + [e['child'] for e in edges]
        self.index = {name: i for i, name in enumerate(self.frame_ids)}
        self.steps = []
        self.support = {self.root: []}                       # tree coordinates that move each frame
        self.parent = {self.root: None}
        for e in edges:
            k = e['coord'] if e['kind'] != 'fixed' else None
            if k is not None and k >= self.nt:
                raise ValueError('tree edge with a non-tree coordinate index')
            self.steps.append((self.index[e['parent']], self.index[e['child']], e['kind'], np.array(e['enter_R']),
                               np.array(e['enter_p']), np.array(e['leave_R']), np.array(e['leave_p']),
                               np.array(e['axis'], float), np.array(e['K']), np.array(e['K2']), k))
            self.support[e['child']] = self.support[e['parent']] + ([k] if k is not None else [])
            self.parent[e['child']] = (e['parent'], e['id'])
        bodies = [b for b in cmg['bodies'] if b['kind'] == 'rigid_body']
        self.rigid_ids = [b['id'] for b in bodies]
        self.rigid_frame = np.array([self.index[b['id']] for b in bodies], int)
        self.rigid_index = {b['id']: i for i, b in enumerate(bodies)}
        self.mass = np.array([float(b['mass_kg']) for b in bodies])
        self.com = np.array([b['com_m'] for b in bodies], float)
        self.inertia = np.array([b['inertia_com_kg_m2'] for b in bodies], float)
        self.F = len(self.frame_ids)

    # ------------------------------------------------------------------ kinematics
    def evaluate(self, qt, vt, base_p=None, base_R=None, base_v=None, base_w=None, dynamics=True):
        """Frames, Jacobians and Jacobian derivatives of every frame; with ``dynamics`` also M, h, C and gravity."""
        nf, F = self.nf, self.F
        base_p = np.zeros(3) if base_p is None else np.asarray(base_p, float)
        base_R = _I3 if base_R is None else np.asarray(base_R, float)
        base_v = np.zeros(3) if base_v is None else np.asarray(base_v, float)
        base_w = np.zeros(3) if base_w is None else np.asarray(base_w, float)
        nu = np.empty(nf)
        nu[:3], nu[3:6], nu[6:] = base_v, base_w, vt
        R = np.empty((F, 3, 3))
        p = np.empty((F, 3))
        J = np.zeros((F, 6, nf))
        D = np.zeros((F, 6, nf))
        v = np.empty((F, 3))
        w = np.empty((F, 3))
        R[0], p[0], v[0], w[0] = base_R, base_p, base_v, base_w
        J[0, :3, :3] = _I3
        J[0, 3:, 3:6] = _I3
        for par, ch, kind, eR, ep, lR, lp, axis, K, K2, k in self.steps:
            Rp, pp = R[par], p[par]
            Rpre = Rp @ eR
            ppre = pp + Rp @ ep
            if kind == 'revolute':
                a = qt[k]
                Rj = Rpre @ (_I3 + math.sin(a) * K + (1. - math.cos(a)) * K2)
                pj = ppre
                aw = Rpre @ axis
            elif kind == 'prismatic':
                aw = Rpre @ axis
                Rj = Rpre
                pj = ppre + aw * qt[k]
            else:
                Rj, pj = Rpre, ppre
            Rc = Rj @ lR
            pc = pj + Rj @ lp
            sv = pc - pp
            Ss = skew(sv)
            Jp, Dp = J[par], D[par]
            Jc = J[ch]
            Jc[:3] = Jp[:3] - Ss @ Jp[3:]
            Jc[3:] = Jp[3:]
            if k is not None:
                col = 6 + k
                if kind == 'revolute':
                    lv = pc - ppre
                    Jc[0, col] += aw[1] * lv[2] - aw[2] * lv[1]
                    Jc[1, col] += aw[2] * lv[0] - aw[0] * lv[2]
                    Jc[2, col] += aw[0] * lv[1] - aw[1] * lv[0]
                    Jc[3:, col] += aw
                else:
                    Jc[:3, col] += aw
            vc = Jc[:3] @ nu
            wc = Jc[3:] @ nu
            vp, wp = v[par], w[par]
            Dc = D[ch]
            Dc[:3] = Dp[:3] - skew(vc - vp) @ Jp[3:] - Ss @ Dp[3:]
            Dc[3:] = Dp[3:]
            if k is not None:
                awd = np.array((wp[1] * aw[2] - wp[2] * aw[1], wp[2] * aw[0] - wp[0] * aw[2], wp[0] * aw[1] - wp[1] * aw[0]))
                if kind == 'revolute':
                    ro = ppre - pp
                    vo = vp + np.array((wp[1] * ro[2] - wp[2] * ro[1], wp[2] * ro[0] - wp[0] * ro[2], wp[0] * ro[1] - wp[1] * ro[0]))
                    ld = vc - vo
                    Dc[0, col] += awd[1] * lv[2] - awd[2] * lv[1] + aw[1] * ld[2] - aw[2] * ld[1]
                    Dc[1, col] += awd[2] * lv[0] - awd[0] * lv[2] + aw[2] * ld[0] - aw[0] * ld[2]
                    Dc[2, col] += awd[0] * lv[1] - awd[1] * lv[0] + aw[0] * ld[1] - aw[1] * ld[0]
                    Dc[3:, col] += awd
                else:
                    Dc[:3, col] += awd
            R[ch], p[ch], v[ch], w[ch] = Rc, pc, vc, wc
        out = Kin()
        out.nu, out.R, out.p, out.J, out.D, out.v, out.w = nu, R, p, J, D, v, w
        if not dynamics:
            return out
        idx = self.rigid_frame
        Rb = R[idx]
        rc = np.einsum('bij,bj->bi', Rb, self.com)
        Jl, Ja, Dl, Da = J[idx, :3], J[idx, 3:], D[idx, :3], D[idx, 3:]
        wb = w[idx]
        Sr = skew_stack(rc)
        Jcm = Jl - Sr @ Ja
        Dcm = Dl - skew_stack(np.cross(wb, rc)) @ Ja - Sr @ Da
        Iw = Rb @ self.inertia @ Rb.transpose(0, 2, 1)
        m = self.mass
        IJa = Iw @ Ja
        M = np.einsum('b,bin,bim->nm', m, Jcm, Jcm) + np.einsum('bin,bim->nm', Ja, IJa)
        Cmat = np.einsum('b,bin,bim->nm', m, Jcm, Dcm) + np.einsum('bin,bim->nm', Ja, Iw @ Da + skew_stack(wb) @ IJa)
        grav = -np.einsum('b,bin,i->n', m, Jcm, self.gravity)
        out.Jc, out.Dc, out.Jw, out.Dw, out.Iw, out.wb, out.rc = Jcm, Dcm, Ja, Da, Iw, wb, rc
        out.vc = Jcm @ nu
        out.M = 0.5 * (M + M.T)
        out.C = Cmat
        out.grav = grav
        out.h = Cmat @ nu + grav
        out.kinetic = 0.5 * float(nu @ out.M @ nu)
        return out

    # ------------------------------------------------------------------ points
    def point(self, kin, frame, local):
        """World position, Jacobian (3 x nf) and Jacobian derivative of a point fixed in ``frame``."""
        i = self.index[frame] if isinstance(frame, str) else frame
        r = kin.R[i] @ np.asarray(local, float)
        S = skew(r)
        Jp = kin.J[i, :3] - S @ kin.J[i, 3:]
        Dp = kin.D[i, :3] - skew(np.cross(kin.w[i], r)) @ kin.J[i, 3:] - S @ kin.D[i, 3:]
        return kin.p[i] + r, Jp, Dp

    def points(self, kin, frame, locals_):
        """Vectorised ``point`` for K points of one frame: (K, 3), (K, 3, nf), (K, 3, nf)."""
        i = self.index[frame] if isinstance(frame, str) else frame
        r = np.asarray(locals_, float) @ kin.R[i].T
        S = skew_stack(r)
        Ja, Da = kin.J[i, 3:], kin.D[i, 3:]
        Jp = kin.J[i, :3][None] - S @ Ja
        Dp = kin.D[i, :3][None] - skew_stack(np.cross(kin.w[i], r)) @ Ja - S @ Da
        return kin.p[i] + r, Jp, Dp

    # ------------------------------------------------------------------ VDC body modules
    def vdc(self, kin, nu_r, nud_r, gain):
        """Required net wrenches of every rigid body and their virtual-work assembly.

        ``gain``: velocity-feedback rate k (1/s), scalar or one value per rigid body (K_B = k_B M_B).
        Returns the generalized required force (nf), the per-body storage and the per-body required wrenches."""
        k = np.broadcast_to(np.asarray(gain, float), self.mass.shape)
        m = self.mass
        vr = kin.Jc @ nu_r
        wr = kin.Jw @ nu_r
        ar = kin.Jc @ nud_r + kin.Dc @ nu_r
        alr = kin.Jw @ nud_r + kin.Dw @ nu_r
        ev = vr - kin.vc
        ew = wr - kin.wb
        Iew = np.einsum('bij,bj->bi', kin.Iw, ew)
        Fr = m[:, None] * (ar - self.gravity[None, :]) + (k * m)[:, None] * ev
        Tr = np.einsum('bij,bj->bi', kin.Iw, alr) + np.cross(kin.wb, np.einsum('bij,bj->bi', kin.Iw, wr)) + k[:, None] * Iew
        tau = np.einsum('bin,bi->n', kin.Jc, Fr) + np.einsum('bin,bi->n', kin.Jw, Tr)
        storage = 0.5 * m * np.einsum('bi,bi->b', ev, ev) + 0.5 * np.einsum('bi,bi->b', ew, Iew)
        return dict(tau=tau, storage=storage, dissipation=2. * k * storage, F=Fr, T=Tr, v_r=vr, w_r=wr, a_r=ar, al_r=alr,
                    e_v=ev, e_w=ew)

    def gain_matrix(self, kin, gain):
        """Velocity-feedback matrix of the body modules,  sum_B k_B (m_B Jc_B^T Jc_B + Jw_B^T I_B Jw_B)  (= k M for one k):
        with it the assembly of ``vdc`` is the affine map  M nu_r_dot + C nu_r + grav + K (nu_r - nu)."""
        k = np.broadcast_to(np.asarray(gain, float), self.mass.shape)
        return (np.einsum('b,bin,bim->nm', k * self.mass, kin.Jc, kin.Jc)
                + np.einsum('b,bin,bim->nm', k, kin.Jw, kin.Iw @ kin.Jw))

    def describe(self):
        """Module records for the generated-interface manifest."""
        out = []
        for i, name in enumerate(self.rigid_ids):
            par = self.parent.get(name)
            out.append(dict(body=name, parent=None if par is None else par[0], joint=None if par is None else par[1],
                            support_coordinates=list(self.support.get(name, [])), mass_kg=float(self.mass[i])))
        return out
