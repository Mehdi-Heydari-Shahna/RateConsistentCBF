"""Extended RoboCompiler: the same physical mechanism graph generates the control and the safety interfaces.

    xc = compile_extended(physical, spec, partition='joint_space')

``physical`` is the unchanged RoboCompiler input (bodies, joints, attachment frames, inertias, actuator ports);
``spec`` holds control objectives, collision geometry, safety limits and actuator capabilities (rcx.spec).
The result bundles

  xc.ci         compiled closed-chain interface (tree, cuts, PACDM assembly, tangent lift)        [RoboCompiler]
  xc.plan       generated body plan: frames, Jacobians, VDC body modules                          [new]
  xc.task       task map of the control objective                                                 [new]
  xc.barriers   generated barrier evaluators (clearance pairs, coordinate limits)                 [new]
  xc.channels   generated actuator capability models and the port/channel maps                    [new]
  xc.support    generated support interface of a mechanism that stands on the ground: held          [new]
                coordinates, contact-force map, effort distribution (None without the section ``support``)
  xc.manifest   JSON description of everything that was generated (modules, supports, hashes)

Changing the mechanism or the actuator arrangement and calling ``compile_extended`` again regenerates all of
them; no mechanism-specific expression is written by hand anywhere in this package.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from core.reduced import CompiledInterface

from . import spec as spec_mod
from .actuators import build_channels
from .barriers import BarrierSet
from .bodies import BodyPlan
from .lift import TaskMap, extended_lift


class ExtendedCompilation:
    def __init__(self, physical, spec, partition='joint_space'):
        spec_mod.validate(spec, physical)
        self.physical = deepcopy(physical)
        self.spec = deepcopy(spec)
        self.partition = partition
        self.ci = CompiledInterface(self.physical, partition)
        ci = self.ci
        self.plan = BodyPlan(ci.comp, ci.gravity)
        # held coordinates (anchors of the ground contacts): independent coordinates that are not commanded
        held_ids = spec_mod.held_ids(spec)
        missing = [k for k in held_ids if k not in ci.independent_ids]
        if missing:
            raise spec_mod.SpecError(f'support: the anchor joints {missing} are not independent coordinates of the partition {partition} '
                                     '(the partition request of the physical records must name them)')
        self.held_ids = list(held_ids)
        self.held = np.array([ci.independent_ids.index(k) for k in held_ids], int)
        self.free = np.array([i for i in range(ci.nu) if i not in set(self.held.tolist())], int)
        self.free_ids = [ci.independent_ids[i] for i in self.free]
        self.seed_independent = np.asarray(ci.seed, float)[ci.active].copy()       # ci.seed is the state of the continuation later
        self.task = TaskMap(ci, spec['control']['task_ids'], held=len(self.held))
        cut_joints = {j['id']: dict(j, seed=float(ci.seed[ci.coordinate_ids.index(j['id'])]))
                      for j in ci.comp.cmg['joints'] if j['id'] in ci.cut_ids}
        self.support = Support(self, spec['support']) if 'support' in spec else None
        self.barriers = BarrierSet(self.plan, self._with_activation(spec), ci.tree_ids, cut_joints,
                                   support=None if self.support is None else self.support.centroid_plan())
        groups = physical.get('actuator_groups', {})
        self.channels = build_channels(spec['actuators'], ci.port_ids, ci.group_ids,
                                       {g: [p for p in groups.get(g, []) if p in ci.port_ids] for g in ci.group_ids})
        self.S_port = ci.E.copy()                        # tree x ports selector
        self.G = ci.G.copy()                             # ports x channels
        self.Ts = float(spec['actuators']['sample_time_s'])
        self.d, self.m, self.nports = len(self.free), len(self.channels), len(ci.port_ids)
        self.d_all = ci.nu
        k = spec['control'].get('k', 12.)
        if isinstance(k, dict):
            self.k_body = np.array([float(k.get(b, k.get('default', 12.))) for b in self.plan.rigid_ids])
        else:
            self.k_body = np.full(len(self.plan.rigid_ids), float(k))
        lam = spec['control'].get('lambda', 12.)
        self.Lambda = np.broadcast_to(np.asarray(lam, float), (self.d,)).copy()
        self.port_speed = np.zeros(self.nports)          # scale of the port-speed metric (smaller of the two directions)
        self.port_speed_lo = np.zeros(self.nports)       # retraction / negative-direction limit (negative)
        self.port_speed_hi = np.zeros(self.nports)       # extension / positive-direction limit
        self.e_force = np.zeros(self.m)
        self.scale = np.zeros(self.m)
        self.tau_act = np.zeros(self.m)
        self.theta = np.ones(self.m)                     # fraction of a command step delivered after one sample
        self.theta_bar = np.ones(self.m)                 # mean fraction over the sample
        self.theta2 = np.full(self.m, 0.5)               # fraction of the double time integral (unit Ts^2)
        for j, ch in enumerate(self.channels):
            self.port_speed[ch.port_index] = ch.speed
            self.port_speed_lo[ch.port_index] = -ch.speed_ret
            self.port_speed_hi[ch.port_index] = ch.speed_ext
            self.e_force[j] = ch.force_error
            self.scale[j] = ch.scale
            self.tau_act[j] = ch.response_time
            self.theta[j], self.theta_bar[j], self.theta2[j] = ch.response(self.Ts)
        if self.m < self.d:
            raise ValueError(f'{self.m} actuator channels for mobility {self.d}: the generated effort maps need at least one channel '
                             'per commanded independent coordinate; deficient actuation is not supported')
        if self.m > self.d and self.support is None:
            raise ValueError(f'{self.m} actuator channels for mobility {self.d}: redundant actuation is generated for mechanisms with '
                             'declared ground contacts only (specification section support)')
        self.redundant = self.m > self.d
        self.eta_nominal = float(spec['control'].get('nominal_effort_fraction', 1.0))
        self.speed_headroom = float(spec['control'].get('speed_headroom_gain', 0.))
        self.corr_tau = float(spec['control'].get('correction_filter_s', 0.03))
        unc = spec.get('uncertainty', {})
        self.tool_body = unc.get('tool_body')
        self.tool_point = np.asarray(unc.get('tool_point', [0., 0., 0.]), float)
        self.Wf = float(unc.get('wrench_force_N', 0.))
        self.Wm = float(unc.get('wrench_moment_Nm', 0.))
        self.rho = float(unc.get('effectiveness', 0.))
        self.eps_load = float(unc.get('load_relative', 0.))
        self.jerk = float(unc.get('jerk_m_s3', 0.))
        self._check_seed()
        self.manifest = self._manifest()

    # ------------------------------------------------------------------ consistency of the generated maps at the seed
    def _check_seed(self):
        """The declared task coordinates must be independent (their rates determine the independent speeds), and the
        generated rows of limits on cut joints must agree with the tangent lift (value and rate of the cut coordinate)."""
        ci = self.ci
        q0 = np.asarray(ci.seed, float)
        lift = self.lift(q0, np.zeros(self.d_all))
        Jx = lift.E[self.task.rows]
        sv = np.linalg.svd(Jx, compute_uv=False)
        if sv[-1] <= 1e-9 * sv[0]:
            raise spec_mod.SpecError(f'control.task_ids {self.task.ids} are not independent coordinates of the mechanism at its seed '
                                     f'configuration (singular values of the task map {sv.tolist()})')
        if any(kind == 'cut' for kind, *_ in self.barriers.coord):
            kin = self.plan.evaluate(q0[ci.tree_rows], lift.vT, dynamics=False)
            rows = self.barriers.evaluate(kin, q0[ci.tree_rows], broad_phase=False)
            for j, (kind, data, sign, ref) in enumerate(self.barriers.coord):
                if kind != 'cut':
                    continue
                i = self.barriers.n_clearance + j
                k = ci.coordinate_ids.index(data['joint'])
                if abs(rows.h[i] - (sign * q0[k] - ref)) > 1e-7 or np.max(np.abs(rows.G[i, 6:] @ lift.ET - sign * lift.E[k])) > 1e-7:
                    raise RuntimeError(f"generated limit row of the cut joint {data['joint']} disagrees with the compiled lift")

    # ------------------------------------------------------------------ generated activation distances
    @staticmethod
    def _with_activation(spec):
        """Broad phase.  For |h_dot| <= v_max, |h_ddot| <= a_max (declared motion bounds) a row with
            h > D = (a_max + m_max + (alpha1 + alpha2) v_max) / (alpha1 alpha2)
        has a row function above m_max, the declared bound of the state-dependent margins, so it cannot be active.  A row
        is evaluated once per sample, and h decreases by at most Ts v_max between two samples: rows are therefore taken
        into the problem at  h <= D + Ts v_max,  which leaves a row function of at least m_max at the first sample at
        which a row is seen.  Rows that nevertheless enter below their margin are counted at run time (late rows)."""
        s = deepcopy(spec)
        mb = s.get('safety', {}).get('motion_bounds', {})
        vmax, amax = float(mb.get('point_speed_m_s', np.inf)), float(mb.get('point_acceleration_m_s2', np.inf))
        mmax = float(mb.get('row_margin_m_s2', 0.))
        Ts = float(s['actuators']['sample_time_s'])
        for c in s.get('safety', {}).get('clearance', []):
            if 'activation_distance' not in c and np.isfinite(vmax) and np.isfinite(amax):
                a1, a2 = float(c['alpha'][0]), float(c['alpha'][1])
                c['activation_distance'] = (amax + mmax + (a1 + a2) * vmax) / (a1 * a2) + Ts * vmax
        return s

    # ------------------------------------------------------------------ lift of the commanded coordinates
    def compose(self, u, held=None):
        """All independent coordinates of the compiled partition from the commanded ones: held coordinates at their seed
        values (or as given).  For a mechanism without held coordinates this is u itself."""
        if not len(self.held):
            return np.asarray(u, float)
        out = self.seed_independent.copy()
        out[self.free] = u
        if held is not None:
            out[self.held] = held
        return out

    def lift(self, q, u_all):
        """Extended lift at the closed configuration q for the speeds of all independent coordinates; with held coordinates
        its matrices are those of the commanded coordinates (rcx.lift)."""
        if len(self.held):
            return extended_lift(self.ci, q, u_all, free=self.free, held=self.held)
        return extended_lift(self.ci, q, u_all)

    # ------------------------------------------------------------------ reduced maps at a state
    def reduce(self, lift, kin, a_base):
        """Reduced model of the controlled mechanism on its moving base:  M_r u_dot + h_r = C phi + tau_ext."""
        ET, EdT = lift.ET, lift.EdotT
        Mqq, Mqb = kin.M[6:, 6:], kin.M[6:, :6]
        MqqE = Mqq @ ET
        Mr = ET.T @ MqqE
        Mr = 0.5 * (Mr + Mr.T)
        hr = ET.T @ (kin.h[6:] + Mqq @ lift.cT + Mqb @ a_base)
        L = self.S_port.T @ ET                           # port speeds = L u
        C = L.T @ self.G                                 # reduced effort = C phi
        Cqq = kin.C[6:, 6:]
        Mr_dot = EdT.T @ MqqE + ET.T @ ((Cqq + Cqq.T) @ ET) + MqqE.T @ EdT
        return Mr, hr, L, C, Mr_dot

    def nominal_map(self, C, held=None):
        """Matrix D with  phi = D tau  for the nominal channel efforts of a required reduced effort tau (C phi = tau).
        One channel per commanded coordinate: D = C^-1.  Redundant actuation (mechanism on ground contacts): the efforts
        of smallest weighted norm (rcx.compiler.Support.distribution; the tangential contact forces enter through its
        second return value)."""
        if not self.redundant:
            return np.linalg.inv(C), None
        return self.support.distribution(C, held)

    def bounds(self, port_velocity):
        lo, hi = np.empty(self.m), np.empty(self.m)
        for j, ch in enumerate(self.channels):
            lo[j], hi[j] = ch.bounds(port_velocity[ch.port_index])
        return lo, hi

    # ------------------------------------------------------------------ manifest
    def _manifest(self):
        ci = self.ci
        plan = ci.comp.plan
        return dict(
            schema='rcx.generated-interface/1.0', partition=self.partition,
            physical_sha256=spec_mod.digest(self.physical), spec_sha256=spec_mod.digest(self.spec),
            graph=dict(bodies=plan['physical_bodies'], joints=plan['physical_joints'], tree_coordinates=plan['tree_coordinates'],
                       cuts=plan['cut_joint_ids'], mobility=plan['mobility'], independent_ids=plan['independent_ids'],
                       held_ids=list(self.held_ids), commanded_mobility=int(self.d),
                       modules=plan['module_sizes'], closure_rank=plan['closure_rank']),
            vdc=dict(body_modules=self.plan.describe(), task_ids=self.task.ids, Lambda=self.Lambda.tolist(),
                     k_body=dict(zip(self.plan.rigid_ids, self.k_body.tolist())),
                     required_twist='V_r,B = J_B [nu_base; E_T u_r]  (image of the tangent lift)',
                     required_acceleration='J_B [a_base; E_T u_r_dot + Edot_T u_r] + D_B nu_r'),
            support=None if self.support is None else self.support.describe(),
            barriers=dict(count=self.barriers.R, classes={c: int(len(r)) for c, r in self.barriers.class_rows.items()},
                          pairs={p: int(len(r)) for p, r in self.barriers.pair_rows.items()}, plans=self.barriers.describe()),
            actuators=dict(ports=ci.port_ids, channels=[ch.describe() for ch in self.channels], G=self.G.tolist(),
                           sample_time_s=self.Ts),
            uncertainty=dict(tool_body=self.tool_body, wrench_force_N=self.Wf, wrench_moment_Nm=self.Wm, load_relative=self.eps_load,
                             effectiveness=self.rho, response_time_s=self.tau_act.tolist(), theta=self.theta.tolist(),
                             theta_bar=self.theta_bar.tolist(), theta2=self.theta2.tolist(),
                             jerk_m_s3=self.jerk, force_error=self.e_force.tolist()))


class Support:
    """Generated support interface of a mechanism that stands on declared ground contacts.

    Every contact has an anchor: three prismatic joints of the graph from the root frame (x, y, z) whose coordinates are
    the position of the contact point.  They are independent coordinates of the compiled partition that the ground
    holds.  With the commanded speeds u (d) and the held coordinates at rest, the rows of the reduced dynamics that
    belong to the held coordinates give the contact forces lam (generalized forces of the anchors = force of the ground
    on the mechanism at the contact point, root-frame components), as an affine function of the delivered efforts f:

        M_ff u_dot + h_f = C_f f + tau_ext,f                       (commanded rows: the reduced model of rcx.compiler.reduce)
        M_xf u_dot + h_x = C_x f + tau_ext,x + lam                 (held rows)
        lam = (M_xf M_ff^-1 C_f - C_x) f + h_x - tau_ext,x + M_xf M_ff^-1 (tau_ext,f - h_f)

    No contact Jacobian is written by hand: C_x, M_xf and h_x come from the columns of the tangent lift that belong to
    the anchors.

    Distribution of the nominal efforts (redundant actuation: more channels than commanded coordinates).  Among the
    efforts with C_f phi = tau the nominal ones minimise

        sum_j (phi_j / scale_j)^2 + w sum_contacts |lam_t / lam_scale|^2

    with the tangential contact forces lam_t of the nominal model (lam = lam_0 - C_x phi): joint efforts that carry the
    weight through friction are avoided.  The minimiser is affine, phi = D tau + D_lam lam_0, and is the inverse of C_f
    for a mechanism with one channel per commanded coordinate and no contact.
    """

    def __init__(self, xc, record):
        self.record = record
        self.contacts = []
        ci = xc.ci
        joints = {j['id']: j for j in xc.physical['joints']}
        for k, c in enumerate(record['contacts']):
            self.contacts.append(dict(id=c['id'], anchor=list(c['anchor']), rows=np.arange(3 * k, 3 * k + 3), friction=float(c['friction']),
                                      normal_min=float(c.get('normal_force_min_N', 0.)), frame=joints[c['anchor'][2]]['body_b']))
        self.n = len(self.contacts)
        self.ids = [c['id'] for c in self.contacts]
        self.tangential_weight = float(record.get('tangential_weight', 1.))
        self.reserve = float(record.get('force_reserve_N', 0.))
        self.centroid = record.get('centroid')
        self.xc = xc
        self.weight = float(np.linalg.norm(ci.gravity) * sum(float(b['mass_kg']) for b in xc.physical['bodies']))
        t = np.zeros((2 * self.n, 3 * self.n))                   # selector of the tangential components (root x, y)
        for k in range(self.n):
            t[2 * k, 3 * k], t[2 * k + 1, 3 * k + 1] = 1., 1.
        self.St = t

    def centroid_plan(self):
        """Record of the static support limit for the barrier generator: frames of the contact points and their positions
        at the seed (the anchor coordinates are those positions)."""
        if self.centroid is None:
            return None
        ci = self.xc.ci
        seed = [[float(ci.seed[ci.coordinate_ids.index(a)]) for a in c['anchor']] for c in self.contacts]
        return dict(self.centroid, frames=[c['frame'] for c in self.contacts], contacts=[c['id'] for c in self.contacts], seed_points=seed)

    def held_maps(self, lift, kin, a_base, S_port, G):
        """M_xf, h_x, C_x of the held rows (see the class text)."""
        ET, EXT = lift.ET, lift.EXT
        Mqq, Mqb = kin.M[6:, 6:], kin.M[6:, :6]
        Mxf = EXT.T @ (Mqq @ ET)
        hx = EXT.T @ (kin.h[6:] + Mqq @ lift.cT + Mqb @ a_base)
        Cx = (S_port.T @ EXT).T @ G
        return Mxf, hx, Cx

    def force_map(self, Mr, hr, C, Mxf, hx, Cx, tau_ext, tau_ext_x):
        """lam = A f + b  for the delivered channel efforts f."""
        X = np.linalg.solve(Mr, np.c_[C, tau_ext - hr])
        A = Mxf @ X[:, :-1] - Cx
        b = hx - tau_ext_x + Mxf @ X[:, -1]
        return A, b

    def distribution(self, C, Cx):
        """(D, D_lam) with the nominal efforts  phi = D tau + D_lam lam_0  (lam_0: contact forces of the nominal model at zero
        efforts).  KKT system of the equality-constrained least-squares problem of the class text."""
        xc = self.xc
        d, m = C.shape
        W = np.diag(1. / xc.scale ** 2)
        if Cx is None or self.tangential_weight == 0.:
            T = np.zeros((0, m))
        else:
            T = np.sqrt(self.tangential_weight) / self.weight * (self.St @ Cx) * float(np.sqrt(np.mean(xc.scale ** 2)))
        # efforts are weighted by their scales, tangential forces by the weight of the machine: both terms are of order one
        s2 = float(np.mean(xc.scale ** 2))
        H = W * s2 + T.T @ T
        K = np.zeros((m + d, m + d))
        K[:m, :m], K[:m, m:], K[m:, :m] = H, C.T, C
        rhs = np.zeros((m + d, d + (0 if Cx is None else Cx.shape[0])))
        rhs[m:, :d] = np.eye(d)
        if Cx is not None and len(T):
            rhs[:m, d:] = T.T @ (np.sqrt(self.tangential_weight) / self.weight * float(np.sqrt(s2)) * self.St)
        sol = np.linalg.solve(K, rhs)
        return sol[:m, :d], (sol[:m, d:] if Cx is not None else None)

    def describe(self):
        return dict(contacts=[dict(id=c['id'], anchor=c['anchor'], friction=c['friction'], normal_force_min_N=c['normal_min'], frame=c['frame'])
                              for c in self.contacts], tangential_weight=self.tangential_weight, force_reserve_N=self.reserve,
                    weight_N=self.weight, centroid=self.centroid,
                    contact_force='lam = (M_xf M_ff^-1 C_f - C_x) f + h_x - tau_ext,x + M_xf M_ff^-1 (tau_ext,f - h_f)')


def compile_extended(physical, spec, partition='joint_space'):
    return ExtendedCompilation(physical, spec, partition)
