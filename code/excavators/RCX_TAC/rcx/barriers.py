"""Generated barrier evaluators: safety functions h(q) with first- and second-order derivative data.

Two barrier families are generated from the specification:

  clearance   h = sdf_O(x) - r - d_min      sphere (centre fixed in a graph body) against an obstacle primitive in
                                             its owner frame O (inertial frame or a graph body, so the same code gives
                                             environment keep-outs, e-fence planes and self-collision)
  coordinate  h = q_k - (lower + margin),    one-sided limits of a joint coordinate of the physical graph (cylinder
              h = (upper - margin) - q_k     stroke end-stops, joint limits).  The joint may be one that the compiler kept
                                             in the spanning tree (q_k is then a tree coordinate) or a revolute joint
                                             that it cut: there q_k is the rotation about the joint axis that takes the
                                             attachment frame on the base body to the one on the follower body,
                                             q_dot = a . (w_F - w_B),  G = a^T (Jw_F - Jw_B),  beta = a^T (Dw_F - Dw_B) nu
                                             (a: world axis; the angle is taken on the branch within pi of the seed
                                             value of the cut coordinate, where the limits must lie as well)

  support     h = n_e . (c - a_e) - margin   static support of a mechanism on declared ground contacts (rcx.spec section
                                             ``support``): the centre of mass c of all bodies of the graph stays inside the
                                             polygon of the contact points, one row per edge e (a_e: a vertex of the edge,
                                             n_e: horizontal inward normal).  G = n_e^T J_c with the mass-weighted mean J_c
                                             of the centre-of-mass Jacobians of the body modules, beta = n_e^T (D_c nu);
                                             the contact points are taken at rest (they are held coordinates).

For every barrier the evaluator returns, in the generalized velocity nu = [v_base, omega_base, v_T] of
``rcx.bodies``:

    h,   G (row, nf)  with  h_dot = G nu,   beta  with  h_ddot = G nu_dot + beta.

For a clearance pair with relative point velocity v_rel = (J_P - J_O@P) nu, obstacle angular velocity w_O, world
normal n and world Hessian H of the signed distance:

    G    = n^T (J_P - J_O@P)
    beta = n^T (D_P - D_O@P) nu + v_rel^T H v_rel + 2 (w_O x n) . v_rel

(the last term is the Coriolis contribution of a rotating owner frame).  These expressions are checked against
finite differences in the tests; no mechanism-specific expression appears.
"""
from __future__ import annotations

import numpy as np

from .bodies import skew_stack
from .geometry import Obstacle


class Rows:
    """Stacked barrier data (R rows)."""
    __slots__ = ('ids', 'cls', 'h', 'G', 'beta', 'alpha1', 'alpha2', 'pair', 'sphere', 'on')

    def __len__(self):
        return len(self.h)


def hull_order(P):
    """Indices of the points P (N, 2) on their convex hull, counter-clockwise (Andrew's monotone chain)."""
    idx = sorted(range(len(P)), key=lambda i: (P[i][0], P[i][1]))

    def cross(o, a, b):
        return (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - (P[a][1] - P[o][1]) * (P[b][0] - P[o][0])
    lower, upper = [], []
    for i in idx:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0.:
            lower.pop()
        lower.append(i)
    for i in reversed(idx):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0.:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def joint_angle(RA, RB, axis, center=0.):
    """Rotation angle about ``axis`` (unit, in the frame RA) that takes the frame RA to the frame RB, on the branch
    (center - pi, center + pi]."""
    Rd = RA.T @ RB
    v = 0.5 * np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])     # sin(angle) * axis
    angle = float(np.arctan2(axis @ v, 0.5 * (np.trace(Rd) - 1.)))
    return center + (angle - center + np.pi) % (2. * np.pi) - np.pi if abs(angle - center) > np.pi else angle


class BarrierSet:
    def __init__(self, plan, spec, tree_ids, cut_joints=None, support=None):
        """``cut_joints``: records (id -> base_body, follower_body, T_BJ, T_FJ, axis, type, seed) of the joints that the
        compiler cut (seed: value of the cut coordinate at the seed configuration), so that a coordinate limit may be
        declared on any moving joint of the physical graph.  ``support``: record of the static support limit with the
        frames of the contact points and their seed positions (rcx.compiler.Support.centroid_plan)."""
        self.plan = plan
        geo = spec.get('geometry', {})
        safety = spec.get('safety', {})
        self.obstacles = {o['id']: Obstacle(o) for o in geo.get('obstacles', [])}
        for o in self.obstacles.values():
            if o.frame != 'world' and o.frame not in plan.index:
                raise ValueError(f'obstacle {o.id}: unknown owner frame {o.frame}')
        spheres = geo.get('spheres', [])
        if len({s['id'] for s in spheres}) != len(spheres):
            raise ValueError('sphere ids must be unique')
        for s in spheres:
            if s['body'] not in plan.index:
                raise ValueError(f"sphere {s['id']}: unknown body {s['body']}")
        # all spheres of one body share one point evaluation
        self.body_spheres = {}
        where = {}
        for s in spheres:
            f = plan.index[s['body']]
            lst = self.body_spheres.setdefault(f, [])
            where[s['id']] = (f, len(lst))
            lst.append(s)
        self.body_centers = {f: np.array([s['center'] for s in lst], float) for f, lst in self.body_spheres.items()}
        # clearance plans: one vectorised block per (pair, body)
        self.blocks = []
        self.ids, self.cls, alpha1, alpha2 = [], [], [], []
        self.row_pair, self.row_sphere = [], []
        for pair in safety.get('clearance', []):
            obs = self.obstacles[pair['obstacle']]
            members = [s for s in spheres if s.get('group') in pair['groups']]
            if not members:
                raise ValueError(f"clearance {pair['id']}: no sphere in groups {pair['groups']}")
            if obs.frame != 'world' and any(s['body'] == obs.frame for s in members):
                raise ValueError(f"clearance {pair['id']}: sphere and obstacle on the same body")
            for body in sorted({s['body'] for s in members}, key=lambda b: plan.index[b]):
                ms = [s for s in members if s['body'] == body]
                start = len(self.ids)
                for s in ms:
                    self.ids.append(f"{pair['id']}:{s['id']}")
                    self.cls.append(pair.get('class', 'clearance'))
                    alpha1.append(float(pair['alpha'][0]))
                    alpha2.append(float(pair['alpha'][1]))
                    self.row_pair.append(pair['id'])
                    self.row_sphere.append(s['id'])
                self.blocks.append(dict(pair=pair['id'], body=plan.index[body], obstacle=obs,
                                        owner=None if obs.frame == 'world' else plan.index[obs.frame],
                                        sel=np.array([where[s['id']][1] for s in ms], int),
                                        offset=np.array([float(s['radius']) + float(pair.get('d_min', 0.)) for s in ms]),
                                        activation=float(pair.get('activation_distance', np.inf)),
                                        rows=slice(start, start + len(ms))))
        self.n_clearance = len(self.ids)
        # coordinate limits
        self.coord = []
        cut_joints = cut_joints or {}
        for lim in safety.get('coordinate_limits', []):
            jid = lim['joint']
            if jid in tree_ids:
                kind, data = 'tree', tree_ids.index(jid)
            elif jid in cut_joints:
                j = cut_joints[jid]
                if j['type'] != 'revolute':
                    raise ValueError(f"limit {lim['id']}: joint {jid} was cut by the compiler and is not revolute; limits on cut "
                                     'joints are generated for revolute joints only')
                seed = float(j.get('seed', 0.))
                if any(lim.get(key) is not None and not abs(float(lim[key]) - seed) < np.pi for key in ('lower', 'upper')):
                    raise ValueError(f"limit {lim['id']}: limits of the cut joint {jid} must lie within pi of its seed value {seed:.3f} rad "
                                     '(the angle of a cut joint is unique on that branch only)')
                axis = np.asarray(j['axis'], float)
                kind, data = 'cut', dict(base=plan.index[j['base_body']], follower=plan.index[j['follower_body']],
                                         RA=np.asarray(j['T_BJ'], float)[:3, :3], RB=np.asarray(j['T_FJ'], float)[:3, :3],
                                         axis=axis / np.linalg.norm(axis), joint=jid, seed=seed)
            else:
                raise ValueError(f"limit {lim['id']}: joint {jid} is not a coordinate of the compiled graph")
            margin = float(lim.get('margin', 0.))
            for side, sign, key in (('lower', 1., 'lower'), ('upper', -1., 'upper')):
                if lim.get(key) is None:
                    continue
                ref = sign * (float(lim[key]) + sign * margin)
                self.coord.append((kind, data, sign, ref))
                self.ids.append(f"{lim['id']}:{side}")
                self.row_pair.append(lim['id'])
                self.row_sphere.append(side)
                self.cls.append(lim.get('class', 'limit'))
                alpha1.append(float(lim['alpha'][0]))
                alpha2.append(float(lim['alpha'][1]))
        # static support: one row per edge of the polygon of the contact points (order: convex hull at the seed)
        self.support = None
        if support is not None:
            frames = [plan.index[f] for f in support['frames']]
            P0 = np.asarray(support['seed_points'], float)[:, :2]
            order = hull_order(P0)
            if len(order) < 3:
                raise ValueError('support: the contact points do not span a polygon')
            edges = [(frames[order[k]], frames[order[(k + 1) % len(order)]]) for k in range(len(order))]
            names = [f"{support['contacts'][order[k]]}-{support['contacts'][order[(k + 1) % len(order)]]}" for k in range(len(order))]
            self.support = dict(edges=edges, margin=float(support.get('margin', 0.)), start=len(self.ids), id=support['id'])
            for name in names:
                self.ids.append(f"{support['id']}:{name}")
                self.row_pair.append(support['id'])
                self.row_sphere.append(name)
                self.cls.append(support.get('class', 'support'))
                alpha1.append(float(support['alpha'][0]))
                alpha2.append(float(support['alpha'][1]))
            self.mass_total = float(plan.mass.sum())
        self.alpha1, self.alpha2 = np.array(alpha1), np.array(alpha2)
        self.R = len(self.ids)
        self.cls = np.array(self.cls)
        self.classes = sorted(set(self.cls.tolist()))
        self.class_rows = {c: np.flatnonzero(self.cls == c) for c in self.classes}
        self.pair_ids = sorted(set(self.row_pair), key=self.row_pair.index)
        rp = np.array(self.row_pair)
        self.pair_rows = {pid: np.flatnonzero(rp == pid) for pid in self.pair_ids}

    # ------------------------------------------------------------------ positions only (monitoring)
    def _support_edges(self, p):
        """Inward unit normals (E, 2) and first vertices (E, 2) of the support polygon for frame positions p."""
        a = np.array([p[i][:2] for i, _ in self.support['edges']])
        b = np.array([p[j][:2] for _, j in self.support['edges']])
        e = b - a
        n = np.c_[-e[:, 1], e[:, 0]] / np.linalg.norm(e, axis=1, keepdims=True)
        return n, a

    def support_values(self, p, com):
        """h of the static-support rows for frame positions p and a given centre of mass (e.g. that of a plant with an
        unmodelled payload); None without a support limit."""
        if self.support is None:
            return None
        n, a = self._support_edges(p)
        return np.einsum('ei,ei->e', n, np.asarray(com, float)[None, :2] - a) - self.support['margin']

    def centre_of_mass(self, R, p):
        plan = self.plan
        f = plan.rigid_frame
        return (plan.mass[:, None] * (p[f] + np.einsum('bij,bj->bi', R[f], plan.com))).sum(axis=0) / self.mass_total

    def values(self, R, p, qt, com=None):
        """h of every barrier from frame poses R (F, 3, 3), p (F, 3) and tree coordinates (no derivatives).  ``com``: centre
        of mass to use in the support rows (a plant's own value; default: that of the body records at the given poses)."""
        h = np.empty(self.R)
        if self.support is not None:
            n, a = self._support_edges(p)
            c = self.centre_of_mass(R, p) if com is None else np.asarray(com, float)
            k = self.support['start']
            h[k:k + len(n)] = np.einsum('ei,ei->e', n, c[None, :2] - a) - self.support['margin']
        pos = {f: p[f] + c @ R[f].T for f, c in self.body_centers.items()}
        for b in self.blocks:
            P = pos[b['body']][b['sel']]
            if b['owner'] is not None:
                P = (P - p[b['owner']]) @ R[b['owner']]
            h[b['rows']] = b['obstacle'].evaluate(P)[0] - b['offset']
        for j, (kind, data, sign, ref) in enumerate(self.coord):
            if kind == 'tree':
                h[self.n_clearance + j] = sign * qt[data] - ref
            else:
                h[self.n_clearance + j] = sign * joint_angle(R[data['base']] @ data['RA'], R[data['follower']] @ data['RB'], data['axis'],
                                                             data['seed']) - ref
        return h

    # ------------------------------------------------------------------ full rows
    def evaluate(self, kin, qt, broad_phase=True):
        """h of every barrier and derivative data of the rows of every block within its activation distance."""
        plan = self.plan
        nf = plan.nf
        nu = kin.nu
        out = Rows()
        out.ids, out.cls, out.alpha1, out.alpha2 = self.ids, self.cls, self.alpha1, self.alpha2
        out.pair, out.sphere = self.row_pair, self.row_sphere
        h = np.empty(self.R)
        G = np.zeros((self.R, nf))
        beta = np.zeros(self.R)
        on = np.zeros(self.R, bool)
        pos = {f: kin.p[f] + c @ kin.R[f].T for f, c in self.body_centers.items()}
        pts = {}
        for b in self.blocks:
            f, sel, obs, rows = b['body'], b['sel'], b['obstacle'], b['rows']
            P = pos[f][sel]
            o = b['owner']
            if o is None:
                d, n, H = obs.evaluate(P)
            else:
                Ro, po, wo = kin.R[o], kin.p[o], kin.w[o]
                r = P - po
                d, g, Hs = obs.evaluate(r @ Ro)
            h[rows] = d - b['offset']
            if broad_phase and float(np.min(h[rows])) > b['activation']:
                continue
            if f not in pts:
                pts[f] = plan.points(kin, f, self.body_centers[f])
            JP, DP = pts[f][1][sel], pts[f][2][sel]
            if o is None:
                vrel = JP @ nu
                bias = np.einsum('ki,ki->k', n, DP @ nu) + np.einsum('ki,kij,kj->k', vrel, H, vrel)
                Jrel = JP
            else:
                n = g @ Ro.T
                H = Ro[None] @ Hs @ Ro.T[None]
                S = skew_stack(r)
                Ja, Da = kin.J[o, 3:], kin.D[o, 3:]
                JO = kin.J[o, :3][None] - S @ Ja
                DO = kin.D[o, :3][None] - skew_stack(np.cross(wo, r)) @ Ja - S @ Da
                Jrel = JP - JO
                vrel = Jrel @ nu
                bias = (np.einsum('ki,ki->k', n, (DP - DO) @ nu) + np.einsum('ki,kij,kj->k', vrel, H, vrel)
                        + 2. * np.einsum('ki,ki->k', np.cross(wo, n), vrel))
            G[rows] = np.einsum('ki,kin->kn', n, Jrel)
            beta[rows] = bias
            on[rows] = True
        for j, (kind, data, sign, ref) in enumerate(self.coord):
            i = self.n_clearance + j
            if kind == 'tree':
                h[i] = sign * qt[data] - ref
                G[i, 6 + data] = sign
            else:
                A, B = data['base'], data['follower']
                RA = kin.R[A] @ data['RA']
                aw = RA @ data['axis']
                h[i] = sign * joint_angle(RA, kin.R[B] @ data['RB'], data['axis'], data['seed']) - ref
                G[i] = sign * (aw @ (kin.J[B, 3:] - kin.J[A, 3:]))
                beta[i] = sign * float(aw @ ((kin.D[B, 3:] - kin.D[A, 3:]) @ nu))
            on[i] = True
        if self.support is not None:
            n, a = self._support_edges(kin.p)
            k = self.support['start']
            rows = slice(k, k + len(n))
            c = self.centre_of_mass(kin.R, kin.p)
            h[rows] = np.einsum('ei,ei->e', n, c[None, :2] - a) - self.support['margin']
            if hasattr(kin, 'Jc'):                              # derivative data need the body modules (dynamics evaluation)
                w = self.plan.mass / self.mass_total
                Jc = np.einsum('b,bin->in', w, kin.Jc[:, :2])
                Dc = np.einsum('b,bin->in', w, kin.Dc[:, :2])
                G[rows] = n @ Jc
                beta[rows] = n @ (Dc @ nu)
            else:                                               # values only: no derivative data of the support rows
                G[rows] = np.nan
                beta[rows] = np.nan
            on[rows] = True
        out.h, out.G, out.beta, out.on = h, G, beta, on
        return out

    def describe(self):
        recs = []
        for b in self.blocks:
            body = self.plan.frame_ids[b['body']]
            owner = 'world' if b['owner'] is None else self.plan.frame_ids[b['owner']]
            sa = set(self.plan.support.get(body, []))
            sb = set(self.plan.support.get(owner, [])) if owner != 'world' else set()
            recs.append(dict(pair=b['pair'], body=body, obstacle=b['obstacle'].id, owner=owner, type=b['obstacle'].type,
                             spheres=int(len(b['sel'])), support_coordinates=sorted(sa ^ sb),
                             depends_on_base=bool(owner == 'world'),
                             activation_distance_m=None if not np.isfinite(b['activation']) else b['activation']))
        for j, (kind, data, sign, ref) in enumerate(self.coord):
            rec = dict(pair=self.row_pair[self.n_clearance + j], side='lower' if sign > 0 else 'upper', bound=float(sign * ref),
                       depends_on_base=False)
            if kind == 'tree':
                rec.update(coordinate=int(data), support_coordinates=[int(data)])
            else:
                a, b = self.plan.frame_ids[data['base']], self.plan.frame_ids[data['follower']]
                rec.update(cut_joint=data['joint'], bodies=[a, b],
                           support_coordinates=sorted(set(self.plan.support.get(a, [])) ^ set(self.plan.support.get(b, []))))
            recs.append(rec)
        if self.support is not None:
            for (i, j), name in zip(self.support['edges'], self.row_sphere[self.support['start']:]):
                recs.append(dict(pair=self.support['id'], edge=name, frames=[self.plan.frame_ids[i], self.plan.frame_ids[j]],
                                 margin_m=self.support['margin'], depends_on_base=False))
        return recs
