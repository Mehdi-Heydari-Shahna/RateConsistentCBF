"""Runtime of the generated interfaces: measured state -> geometric VDC evaluator, barrier and storage evaluators
-> coordinated actuator-force filter -> effort commands for the actuator force controllers.

One call of ``SafeVDC.step`` (every sample time Ts):

  1. shadow assembly (PACDM prediction/correction) of the measured independent coordinates, tangent lift E, Edot
  2. generated body plan on the measured base motion: frames, Jacobians, M, C, reduced model  M_r u_dot + h_r = C f + tau_ext
  3. momentum observer of the external load on the measured port efforts (fast estimate for the rows, slow part for compensation)
  4. generated barrier rows  h_i, J_i, gamma_i  and their second-order form in the delivered channel efforts
  5. robust margins and the equivalent standoff of every row
  6. required motion  u_r = shape(Jx^{-1}(x_d_dot + Lambda (x_d - x)))  (first-order safe, inside the port-speed and the
     nominal-effort capability; the generated VDC evaluator is affine in u_r, which gives the effort rows)
  7. VDC body modules: required twists B_B u_r, required net wrenches, virtual-work assembly  tau_r, nominal efforts phi_r
  8. storage evaluator: s = u_r - u, nu = sum_B nu_B = 1/2 s^T M_r s, powers of the storage balance
  9. coordinated actuator-force filter on the efforts delivered at the next sample  ->  phi
 10. assumption monitors: force-loop model error, one-step row prediction residual, late row activation

Mechanism on declared ground contacts (specification section ``support``; rcx.compiler.Support).  ``u`` holds all
independent coordinates of the compiled partition: the commanded ones and the anchors of the contacts, which are
measured in position and taken at rest.  Steps 2-9 work on the commanded coordinates (d of them) with m >= d channels:
the nominal efforts are the distribution of the required reduced effort that keeps the tangential contact forces
small, and the effort filter carries, next to the barrier rows, the contact rows

    lam_n,k >= N_min,k + margin,        mu_k / sqrt(2) lam_n,k -+ lam_t,k >= margin      (two tangential directions)

on the contact forces  lam = A f + b  of the efforts delivered at the end of the sample (margin: force-loop model
error through |A| and the declared reserve).  The filter metric is completed on the null space of C (efforts that
change contact forces only) so that it is positive definite.
"""
from __future__ import annotations

import time

import numpy as np

from .filter import (FilterOptions, barrier_rows, contact_rows, delivered_rows, filter_forces, filter_uncoordinated, load_error_bound,
                     row_margins, shape_required_velocity)
from .lift import extended_lift
from .observer import MomentumObserver


class SafeVDC:
    def __init__(self, xc, options=None, observer_gain=100., load_compensation_rate=2., load_compensation=True,
                 load_compensation_limit=None, base_acceleration_filter_s=0.05, bound_speed_filter_s=0.05):
        self.xc = xc
        self.opt = options or FilterOptions()
        # the discrete observer update is stable for gain * Ts < 2; gain * Ts <= 1 avoids overshoot
        self.observer = MomentumObserver(xc.d, min(float(observer_gain), 1. / xc.Ts), slow_rate=load_compensation_rate)
        self.load_compensation = bool(load_compensation)
        self.load_limit = None if load_compensation_limit is None else np.asarray(load_compensation_limit, float)
        self.acc_tau = float(base_acceleration_filter_s)
        self.corr_tau = float(xc.corr_tau)
        self.bound_tau = float(bound_speed_filter_s)
        self.G_pinv = np.linalg.pinv(xc.G)
        self.uniform_k = bool(np.ptp(xc.k_body) == 0.)
        index = {p: i for i, p in enumerate(xc.barriers.pair_ids)}
        self._row_pair_index = np.array([index[p] for p in xc.barriers.row_pair], int)
        self.reset()

    def reset(self):
        self.observer.reset()
        self._acc = None
        self._t_prev = None
        self._dx_prev = None                              # task-rate correction of the previous sample
        self._u_r_prev = None
        self._ur_state = None                             # required velocity as a controller state (consistent first layer)
        self._uD = None                                   # conditioning state (conditioned required velocity)
        self._port_v = None
        self._rate_x = np.zeros(self.xc.d)                # its filtered derivative
        self._phi_prev = None
        self._f_prev = None
        self._pred = None
        self._on_prev = None
        self._active_prev = False
        self.last = None
        m = self.xc.m
        self.stats = dict(evaluations=0, wall_s=0., wall_max_s=0., qp_infeasible=0, velocity_qp_relaxed=0, velocity_qp_failed=0,
                          intervention_steps=0, bound_steps=0, numerically_flagged=0, no_authority_steps=0, shadow_closure_max=0.,
                          storage_identity_max=0., vdc_affine_max=0.,
                          force_model_checks=0, force_model_violations=0, force_model_checks_active=0,
                          force_model_violations_active=0, force_model_error_max=np.zeros(m),
                          row_residual_checks=0, row_residual_violations=0, row_residual_max=0., row_psi2_min_after_active=np.inf,
                          late_activation_rows=0, initial_condition_rows=0, contact_row_steps=0, contact_slack_steps=0)

    # ------------------------------------------------------------------ one control sample
    def step(self, t, u, ud, base, f_meas, ref):
        started = time.perf_counter()
        xc, opt = self.xc, self.opt
        ci = xc.ci
        st = self.stats
        u, ud = np.asarray(u, float), np.asarray(ud, float)
        dt = None if self._t_prev is None else t - self._t_prev
        # 1. assembly and lift
        q = ci.assemble(u)
        if len(xc.held):                                      # anchors of the ground contacts: measured position, at rest
            ud_all = ud.copy()
            ud_all[xc.held] = 0.
            lift = xc.lift(q, ud_all)
            u, ud = u[xc.free], ud_all[xc.free]
        else:
            lift = extended_lift(ci, q, ud)
        qt = q[ci.tree_rows]
        # 2. body plan on the measured base motion
        a_meas = np.asarray(base['acceleration'], float)
        if self.acc_tau > 0.:
            if self._acc is None:
                self._acc = a_meas.copy()
            elif dt is not None and dt > 0.:
                self._acc += -np.expm1(-dt / self.acc_tau) * (a_meas - self._acc)
            a_base = self._acc.copy()
        else:
            a_base = a_meas
        vb = np.asarray(base['velocity'], float)
        kin = xc.plan.evaluate(qt, lift.vT, base['position'], base['rotation'], vb[:3], vb[3:])
        Mr, hr, L, C, Mr_dot = xc.reduce(lift, kin, a_base)
        ET, EdT = lift.ET, lift.EdotT
        # 3. external-load observer on the measured port efforts, in the momentum conjugate to the task coordinates
        f_meas = np.asarray(f_meas, float)
        x, Jx, Jxd = xc.task.evaluate(lift)
        JxT = Jx.T
        p_x = np.linalg.solve(JxT, Mr @ ud)
        beta_x = np.linalg.solve(JxT, L.T @ f_meas - hr + Mr_dot @ ud - Jxd.T @ p_x)
        tau_ext = JxT @ self.observer.update_momentum(t, p_x, beta_x)
        slow = self.observer.slow.copy() if self.load_compensation else np.zeros(xc.d)
        if self.load_limit is not None:
            slow = np.clip(slow, -self.load_limit, self.load_limit)
        comp = JxT @ slow
        phi_meas = self.G_pinv @ f_meas
        # measured port efforts that are not a channel effort (ports of one channel carrying different efforts): a known
        # generalized force next to C phi_meas, zero for a channel whose ports are loaded equally
        tau_split = L.T @ f_meas - C @ phi_meas
        # 4. barrier rows and their second-order form in the delivered channel efforts
        rows = xc.barriers.evaluate(kin, qt)
        on = np.flatnonzero(rows.on)
        Gon = rows.G[on]
        J = Gon[:, 6:] @ ET
        hb = Gon[:, :6] @ kin.nu[:6]
        hdot = hb + J @ ud
        gamma = Gon[:, 6:] @ lift.cT + Gon[:, :6] @ a_base + rows.beta[on]
        h, a1, a2 = rows.h[on], rows.alpha1[on], rows.alpha2[on]
        port_v = L @ ud
        # effort capability at a low-pass filtered port speed: the capability of a flow-limited actuator moves with its
        # speed, and feeding the raw speed back through the bounds closes a fast loop with the required-motion problem
        if self.bound_tau > 0.:
            if self._port_v is None:
                self._port_v = port_v.copy()
            elif dt is not None and dt > 0.:
                self._port_v += -np.expm1(-dt / self.bound_tau) * (port_v - self._port_v)
            port_vf = self._port_v.copy()
        else:
            port_vf = port_v
        lo, hi = xc.bounds(port_vf)
        a, drift, b, MiJ = barrier_rows(Mr, hr, C, J, hdot, gamma, h, a1, a2, tau_ext + tau_split)
        if opt.actuator_model:
            theta, theta_bar, theta2 = xc.theta, xc.theta_bar, xc.theta2
        else:
            theta, theta_bar, theta2 = np.ones(xc.m), np.ones(xc.m), np.full(xc.m, 0.5)
        c, r0, psi2, need, hdd = delivered_rows(a, drift, b, hdot, a1, a2, phi_meas, theta, theta_bar, theta2,
                                                xc.Ts if opt.sampled_prediction else 0.)
        # 5. robust margins; their state-dependent part is the standoff  sigma_i = m_i^state / (alpha1_i alpha2_i)
        if xc.tool_body is not None:
            _, Jt, _ = xc.plan.point(kin, xc.tool_body, xc.tool_point)
            TF = ET.T @ Jt[:, 6:].T
            TM = ET.T @ kin.J[xc.plan.index[xc.tool_body], 3:, 6:].T
        else:
            TF = TM = np.zeros((xc.d, 3))
        sup = xc.support
        if sup is not None:
            # held rows of the reduced dynamics; the load estimate is attributed to a wrench on the tool body, which gives
            # its part on the held rows (no tool body: the load is taken to act on the commanded coordinates only)
            Mxf, hx, Cx = sup.held_maps(lift, kin, a_base, xc.S_port, xc.G)
            EXT = lift.EXT
            tau_x = (xc.S_port.T @ EXT).T @ f_meas - Cx @ phi_meas
            if xc.tool_body is not None:
                wrench = np.linalg.lstsq(np.c_[TF, TM], tau_ext, rcond=None)[0]
                tau_x = tau_x + np.c_[EXT.T @ Jt[:, 6:].T, EXT.T @ kin.J[xc.plan.index[xc.tool_body], 3:, 6:].T] @ wrench
            A_lam, b_lam = sup.force_map(Mr, hr, C, Mxf, hx, Cx, tau_ext + tau_split, tau_x)
            lam_meas = A_lam @ phi_meas + b_lam            # contact forces of the model at the measured efforts
        load_bound, load_norm, load_floor = load_error_bound(Mr, tau_ext, TF, TM, xc.Wf, xc.Wm, xc.eps_load)
        margin, mcomp = row_margins(opt, a, MiJ, J, need, load_bound, xc.e_force, xc.Ts, xc.jerk, xc.rho, a1, a2)
        sigma = mcomp['standoff'] / (a1 * a2)
        # 6. required motion: nominal task law, then the shaping problem with barrier and capability rows
        x_des, xd_des, xdd_des = (np.asarray(v, float) for v in ref)
        e = x_des - x
        xdot = Jx @ ud
        u_r0 = np.linalg.solve(Jx, xd_des + xc.Lambda * e)
        xr0_ddot = xdd_des + xc.Lambda * (xd_des - xdot)       # nominal required task acceleration
        #    generated VDC evaluator as an affine map of the required motion:  tau_rr = M_r u_r_dot + N u_r + c0
        Mqq, Mqb, Cqq, Cqb = kin.M[6:, 6:], kin.M[6:, :6], kin.C[6:, 6:], kin.C[6:, :6]
        Kr = xc.k_body[0] * Mr if self.uniform_k else ET.T @ xc.plan.gain_matrix(kin, xc.k_body)[6:, 6:] @ ET
        N = ET.T @ (Mqq @ EdT + Cqq @ ET) + Kr
        c0 = ET.T @ (Mqb @ a_base + Cqb @ kin.nu[:6] + kin.grav[6:]) - Kr @ ud
        #    required task acceleration = nominal one + filtered derivative of the task-rate correction  Jx (u_r - u_r0)
        #    (formed in the declared task coordinates, hence independent of the coordinate partition):
        #    u_r_dot = Jx^-1 (x_r0_ddot + rate_x - Jx_dot u_r) = G1 u_r + g0
        if dt is None or dt <= 0. or self._dx_prev is None:
            beta_c, g1 = 0., 0.
            g0 = np.linalg.solve(Jx, xr0_ddot)
        else:
            beta_c = -np.expm1(-dt / self.corr_tau) if self.corr_tau > 0. else 1.      # no filter: backward difference
            g1 = beta_c / dt
            g0 = np.linalg.solve(Jx, xr0_ddot + (1. - beta_c) * self._rate_x - g1 * (Jx @ u_r0 + self._dx_prev))
        G1 = g1 * np.eye(xc.d) - np.linalg.solve(Jx, Jxd)
        if xc.redundant:
            # nominal efforts phi_r = D_eff (tau_rr - comp) + phi_0: distribution with small tangential contact forces; the
            # contact forces of the nominal model at zero efforts are lam_0 = h_x - tau_x + M_xf M_r^-1 (tau_rr - comp + tau_ext - h_r)
            D, D_lam = xc.nominal_map(C, Cx)
            MxfMi = np.linalg.solve(Mr, Mxf.T).T
            D_eff = D + D_lam @ MxfMi
            phi_0 = D_lam @ (hx - tau_x + MxfMi @ (tau_ext + tau_split - hr))
            P = D_eff @ (Mr @ G1 + N)
            p0 = D_eff @ (Mr @ g0 + c0 - comp) + phi_0
        else:
            P = np.linalg.solve(C, Mr @ G1 + N)
            p0 = np.linalg.solve(C, Mr @ g0 + c0 - comp)
        eta = xc.eta_nominal                                # the nominal effort uses this fraction of the capability:
        lo_n = lo + (1. - eta) * np.abs(lo)                 # both bounds move inwards (also in overrun, where they have the
        hi_n = hi - (1. - eta) * np.abs(hi)                 # same sign), and an interval that closes collapses to its centre
        closed = lo_n > hi_n
        lo_n = np.where(closed, 0.5 * (lo_n + hi_n), lo_n)
        hi_n = np.where(closed, lo_n, hi_n)
        A_cap, b_cap, n_speed = [], [], 0
        if opt.speed_rows:
            # port-speed capability with headroom gain g: the required port speed may exceed the capability only in
            # proportion to the distance of the *actual* port speed from it (g = 0: hard cap on the required speed), so a
            # stalled actuator still receives the velocity-error effort while a moving one is not driven past its capability
            g = xc.speed_headroom
            A_cap += [L, -L]
            b_cap += [xc.port_speed_hi + g * (xc.port_speed_hi - port_v), -(xc.port_speed_lo - g * (port_v - xc.port_speed_lo))]
            n_speed = 2 * xc.nports
        if opt.effort_rows:
            Ps = P / xc.scale[:, None]
            A_cap += [Ps, -Ps]
            b_cap += [(hi_n - p0) / xc.scale, (p0 - lo_n) / xc.scale]
        if opt.velocity_metric == 'kinetic':
            Wv = Mr
        elif opt.velocity_metric == 'task':                # equal deviation of the task-coordinate rates
            Wv = Jx.T @ Jx
        else:
            Wv = L.T @ (L / (xc.port_speed ** 2)[:, None])
            Wv = 0.5 * (Wv + Wv.T)
            # regularisation by the kinetic metric, scaled by an invariant of the pair (W_v, M_r): partition-covariant
            Wv = Wv + 1e-9 * float(np.max(np.linalg.eigvals(np.linalg.solve(Mr, Wv)).real)) * Mr
        if opt.velocity_shaping:
            Jv, rv = J, -a1 * (h - sigma) - hb
        else:
            Jv, rv = np.zeros((0, xc.d)), np.zeros(0)
        no_rows = dict(feasible=True, relaxed=False, active=np.zeros(0, bool), mu=np.zeros(0), speed_active=False,
                       effort_active=False, slack=0., level=0, no_authority=np.zeros(0, bool), numerically_ok=True)
        if opt.first_layer == 'acceleration':
            # Consistent first layer (paper).  The required velocity is a controller state; the required acceleration a_r is
            # the projection of the nominal one onto the rows  zs_i' + beta_i zs_i >= 0  (zs_i: slack of the first-order row at
            # the required velocity).  zs_i' = J_i a_r + (Jdot_i u_r + base terms) + alpha1_i hdot_i; the bilinear term is taken
            # at the measured speed (gamma_i), which is exact along s = 0 and differs from it by Jdot_i s otherwise.  The
            # capability rows act on the next required velocity (port speeds) and on the nominal efforts, which are affine
            # in a_r.  The feedforward is exact by construction: u_r_dot = a_r.
            if self._ur_state is None:
                self._ur_state = ud.copy()                        # start with s = 0
            u_r = self._ur_state.copy()
            un_dot = np.linalg.solve(Jx, xr0_ddot - Jxd @ u_r0)   # time derivative of the nominal required velocity u_r0
            a_n = un_dot + opt.kappa_r * (u_r0 - u_r)
            zs_rows = hb + J @ u_r + a1 * (h - sigma)
            beta_i = ((np.minimum(opt.beta_bar, a2) if getattr(opt, 'beta_cap', True) else np.full_like(a2, opt.beta_bar))
                      if opt.beta_bar is not None else a2.copy())
            if opt.velocity_shaping:
                Ja, ra = J, -(gamma + a1 * hdot) - beta_i * zs_rows
            else:
                Ja, ra = np.zeros((0, xc.d)), np.zeros(0)
            A_a, b_a, n_speed_a = [], [], 0
            if opt.speed_rows:
                g = xc.speed_headroom
                hi_s = xc.port_speed_hi + g * (xc.port_speed_hi - port_v)
                lo_s = xc.port_speed_lo - g * (port_v - xc.port_speed_lo)
                A_a += [xc.Ts * L, -xc.Ts * L]
                b_a += [hi_s - L @ u_r, L @ u_r - lo_s]
                n_speed_a = 2 * xc.nports
            if xc.redundant:
                Pa = D_eff @ Mr
                p0a = D_eff @ (N @ u_r + c0 - comp) + phi_0
            else:
                Pa = np.linalg.solve(C, Mr)
                p0a = np.linalg.solve(C, N @ u_r + c0 - comp)
            if opt.effort_rows:
                Pas = Pa / xc.scale[:, None]
                A_a += [Pas, -Pas]
                b_a += [(hi_n - p0a) / xc.scale, (p0a - lo_n) / xc.scale]
            if A_a or len(ra):
                a_r, vinfo = shape_required_velocity(a_n, Wv, Ja, ra, np.vstack(A_a) if A_a else None,
                                                     np.concatenate(b_a) if b_a else None, n_speed_a)
            else:
                a_r, vinfo = a_n.copy(), dict(no_rows)
            if not opt.velocity_shaping:
                vinfo['active'], vinfo['mu'], vinfo['no_authority'] = np.zeros(len(h), bool), np.zeros(len(h)), np.zeros(len(h), bool)
            u_r_dot = a_r
        else:
            if A_cap or len(rv):
                u_r, vinfo = shape_required_velocity(u_r0, Wv, Jv, rv, np.vstack(A_cap) if A_cap else None,
                                                     np.concatenate(b_cap) if b_cap else None, n_speed)
            else:
                u_r, vinfo = u_r0.copy(), dict(no_rows)
            if not opt.velocity_shaping:
                vinfo['active'], vinfo['mu'], vinfo['no_authority'] = np.zeros(len(h), bool), np.zeros(len(h)), np.zeros(len(h), bool)
            dx = Jx @ (u_r - u_r0)
            if g1 > 0.:
                self._rate_x = (1. - beta_c) * self._rate_x + g1 * (dx - self._dx_prev)
            self._dx_prev = dx.copy()
            u_r_dot = G1 @ u_r + g0
            if opt.conditioning:
                # conditioned required velocity (paper): u_r = U + u_D with the output U of the first layer; the required
                # acceleration contains -kappa_c u_D but not the change M^-1 C (phi - phi_r) that the filter commands, which
                # enters the derivative of u_D (update after the filter)
                if self._uD is None:
                    self._uD = np.zeros(xc.d)
                u_r = u_r + self._uD
                u_r_dot = u_r_dot - opt.kappa_c * self._uD
        # 7. VDC body modules and virtual-work assembly
        nu_r = kin.nu.copy()
        nu_r[6:] = ET @ u_r
        nud_r = np.r_[a_base, ET @ u_r_dot + EdT @ u_r]
        mod = xc.plan.vdc(kin, nu_r, nud_r, xc.k_body)
        tau_rr = ET.T @ mod['tau'][6:]
        phi_r = D_eff @ (tau_rr - comp) + phi_0 if xc.redundant else np.linalg.solve(C, tau_rr - comp)
        #    affine form of the VDC evaluator in (u_r_dot, u_r): tau_rr = M_r u_r_dot + N u_r + c0 (for the released first layer
        #    this is P u_r + p0 with u_r_dot = G1 u_r + g0)
        tau_aff = Mr @ u_r_dot + N @ u_r + c0
        phi_aff = D_eff @ (tau_aff - comp) + phi_0 if xc.redundant else np.linalg.solve(C, tau_aff - comp)
        st['vdc_affine_max'] = max(st['vdc_affine_max'], float(np.max(np.abs(phi_r - phi_aff) / xc.scale)))
        # 8. storage evaluator
        s = u_r - ud
        storage = float(mod['storage'].sum())
        storage_check = 0.5 * float(s @ Mr @ s)
        dissipation = float(mod['dissipation'].sum())
        #    power of the required-acceleration error (the feedforward uses a filtered derivative of the shaping correction;
        #    the storage balance holds for the true derivative of u_r, estimated here by its backward difference; at the
        #    first sample the required velocity has not changed yet)
        if self._u_r_prev is not None and dt is not None and dt > 0.:
            P_acc = float(s @ Mr @ ((u_r - self._u_r_prev) / dt - u_r_dot))
        else:
            P_acc = -float(s @ Mr @ u_r_dot)
        self._u_r_prev = u_r.copy()
        # 9. coordinated actuator-force filter on the delivered efforts
        rhs = r0 + margin
        R = len(h)
        # contact rows of a mechanism on ground contacts: on the efforts delivered at the end of the sample
        if sup is not None and opt.contact_rows:
            cc, rc, contact_margin = contact_rows(sup, A_lam, b_lam, phi_meas, theta, xc.e_force if opt.margin_force else np.zeros(xc.m))
        else:
            cc, rc, contact_margin = np.zeros((0, xc.m)), np.zeros(0), np.zeros(0)
        Rc = len(rc)
        phi_box = np.clip(phi_r, lo, hi) if opt.enforce_bounds else phi_r.copy()       # nominal efforts inside the effort bounds
        c_qp, mu_qp, nb = c, np.zeros(R), R                    # rows and multipliers of the effort QP; its first nb rows are barrier rows
        phi_n = phi_r                                          # efforts about which the effort QP is formed
        keys = ('active', 'mu', 'slack', 'no_authority')
        if opt.force_filter and opt.coordinated:
            phi, rec = filter_forces(opt, phi_r, Mr, C, np.vstack([c, cc]) if Rc else c, np.r_[rhs, rc] if Rc else rhs, lo, hi, xc.scale, theta)
            crec = {k: rec[k][R:] for k in keys}
            c_qp, mu_qp = (np.vstack([c, cc]) if Rc else c), rec['mu']
            for k in keys:
                rec[k] = rec[k][:R]
        else:
            if not opt.force_filter:                            # no barrier rows in the effort problem
                phi = phi_box.copy()
                rec = dict(feasible=True, relaxed=False, slack=np.zeros(R), mu=np.zeros(R), active=np.zeros(R, bool),
                           bound_active=np.r_[phi_r > hi, phi_r < lo], bound_mu=np.zeros(2 * xc.m), kkt=None, iterations=0, numerically_ok=True,
                           no_authority=np.zeros(R, bool), H=None)
            else:                                               # ablation: every channel corrects on its own for the barrier rows
                phi = filter_uncoordinated(phi_r, c, rhs, lo, hi, xc.scale, opt.enforce_bounds)
                needs = (rhs - c @ phi_box) > 0.                # rows that the nominal efforts inside the bounds do not meet
                reach = (np.abs(c) * xc.scale[None, :]).max(axis=1) > 0. if R else np.zeros(0, bool)
                rec = dict(feasible=True, relaxed=False, slack=np.maximum(0., rhs - c @ phi), mu=np.zeros(R),
                           active=needs & reach, bound_active=np.r_[phi >= hi, phi <= lo], bound_mu=np.zeros(2 * xc.m),
                           kkt=None, iterations=0, numerically_ok=True, no_authority=needs & ~reach, H=None)
                phi_n = phi
            crec = dict(active=np.zeros(0, bool), mu=np.zeros(0), slack=np.zeros(0), no_authority=np.zeros(0, bool))
            if Rc:
                # A mechanism on ground contacts keeps its contact rows: they belong to the support interface (the model of the
                # controller holds only while the contacts hold), not to the safety limits.  One QP on the contact rows alone,
                # about the command so far (the nominal efforts, or the efforts after the per-channel correction).
                phi, rec_c = filter_forces(opt, phi_n, Mr, C, cc, rc, lo, hi, xc.scale, theta)
                crec = {k: rec_c[k] for k in keys}
                rec.update({k: rec_c[k] for k in ('feasible', 'relaxed', 'bound_active', 'bound_mu', 'kkt', 'iterations', 'numerically_ok', 'H')})
                if opt.force_filter:
                    rec['slack'] = np.maximum(0., rhs - c @ phi)        # barrier rows at the command that the contact rows changed
                c_qp, mu_qp, nb = cc, rec_c['mu'], 0
        dphi = phi - phi_r
        psi2_pred = c @ (phi - phi_meas) - need                 # one-step prediction of the row function for the applied command
        # effort-mismatch powers into the storage:  nu_dot = -dissipation + P_mis - s^T (tau_ext - compensation)
        P_cmd = -float(s @ (C @ dphi))                          # command deviation (safety filter and support interface)
        P_mis = -float(s @ (C @ (phi_meas - phi_r) + tau_split))   # delivered efforts against the nominal ones (filter + force loops)
        # split of P_cmd:  barrier rows + contact rows + effort bounds
        P_box = -float(s @ (C @ (phi_box - phi_r)))             # clipping of the nominal efforts to the bounds
        per_channel = phi_n is not phi_r                        # a per-channel correction was made before the QP
        P_pre = -float(s @ (C @ (phi_n - phi_box))) if per_channel else 0.      # its row part
        if rec['H'] is not None:
            w = np.linalg.solve(rec['H'], C.T @ s)
            P_rows = -float(mu_qp[:nb] @ (c_qp[:nb] @ w)) + P_pre       # multipliers of the barrier rows (and the per-channel part)
            P_contact = -float(mu_qp[nb:] @ (c_qp[nb:] @ w))    # multipliers of the contact rows (mechanism on ground contacts)
            P_bounds = float((rec['bound_mu'][:xc.m] - rec['bound_mu'][xc.m:]) / xc.scale @ w) + (P_box if per_channel else 0.)
        else:
            # no multipliers (no QP, or its fallback): clipping to the bounds first, the rest is due to the rows; it is booked
            # on the contact rows if they are the only rows of the QP, on the barrier rows otherwise
            P_bounds = P_box
            if nb == 0 and Rc:
                P_rows = P_pre
                P_contact = P_cmd - P_box - P_pre
            else:
                P_rows, P_contact = P_cmd - P_box, 0.
        hdot_r = hb + J @ u_r
        zs_now = hdot_r + a1 * (h - sigma)                      # slack of the first-order rows at the required velocity
        # states of the first layer for the next sample
        if opt.first_layer == 'acceleration':
            self._ur_state = u_r + xc.Ts * u_r_dot
        elif opt.conditioning:
            kc = float(opt.kappa_c)
            self._uD = np.exp(-kc * xc.Ts) * self._uD + (-np.expm1(-kc * xc.Ts) / kc) * np.linalg.solve(Mr, C @ dphi)
        # 10. assumption monitors
        if self._phi_prev is not None and self._f_prev is not None:
            err = np.abs(phi_meas - (self._f_prev + xc.theta * (self._phi_prev - self._f_prev)))
            bad = bool(np.any(err > xc.e_force + 1e-9))
            st['force_model_checks'] += 1
            st['force_model_violations'] += int(bad)
            st['force_model_error_max'] = np.maximum(st['force_model_error_max'], err)
            if self._active_prev:
                st['force_model_checks_active'] += 1
                st['force_model_violations_active'] += int(bad)
        else:
            err = np.zeros(xc.m)
        resid_max, psi2_after = 0., np.inf
        if self._pred is not None:
            on_p, pred_p, allow_p, act_p = self._pred
            pos = {int(k): i for i, k in enumerate(on)}
            sel = [(i, pos[int(k)]) for i, k in enumerate(on_p) if act_p[i] and int(k) in pos]
            if sel:
                ip, ic = np.array(sel).T
                resid = pred_p[ip] - psi2[ic]
                resid_max = float(np.max(resid))
                psi2_after = float(np.min(psi2[ic]))
                st['row_residual_checks'] += len(ip)
                st['row_residual_violations'] += int(np.sum(resid > allow_p[ip] + 1e-9))
                st['row_residual_max'] = max(st['row_residual_max'], resid_max)
                st['row_psi2_min_after_active'] = min(st['row_psi2_min_after_active'], psi2_after)
        on_mask = rows.on.copy()
        # a row must enter the problem with its row function above the state-dependent margin (the induction of the safety
        # statement needs only standoff - allowance; this test is the stricter one)
        short = psi2 < mcomp['standoff'] - 1e-9
        if self._on_prev is None:
            st['initial_condition_rows'] += int(np.sum(short))
        else:
            st['late_activation_rows'] += int(np.sum(short & ~self._on_prev[on]))
        self._on_prev = on_mask
        self._pred = (on.copy(), psi2_pred.copy(), mcomp['allowance'].copy(), np.asarray(rec['active'], bool).copy())
        self._active_prev = bool(np.any(rec['active']) or np.any(crec['active']))      # the command was changed by a row
        self._phi_prev, self._f_prev = phi.copy(), phi_meas.copy()
        self._t_prev = t
        intervening = bool(np.linalg.norm(dphi / xc.scale) > 1e-9)
        st['evaluations'] += 1
        st['qp_infeasible'] += int(not rec['feasible'])
        st['velocity_qp_relaxed'] += int(vinfo['relaxed'])
        st['velocity_qp_failed'] += int(not vinfo['feasible'])
        st['intervention_steps'] += int(bool(np.any(rec['active'])))
        st['bound_steps'] += int(bool(np.any(rec['bound_active'])))
        st['numerically_flagged'] += int(not rec['numerically_ok'] or not vinfo['numerically_ok'])
        st['contact_row_steps'] += int(bool(np.any(crec['active'])))
        st['contact_slack_steps'] += int(bool(np.any(crec['slack'] > 1e-6 * (sup.weight if sup is not None else 1.))))
        st['no_authority_steps'] += int(bool(np.any(rec['no_authority'])) or bool(np.any(vinfo['no_authority'])) or bool(np.any(crec['no_authority'])))
        st['shadow_closure_max'] = max(st['shadow_closure_max'], float(ci.mapping['residual_inf']))
        st['storage_identity_max'] = max(st['storage_identity_max'], abs(storage - storage_check) / max(1., storage_check))
        elapsed = time.perf_counter() - started
        st['wall_s'] += elapsed
        st['wall_max_s'] = max(st['wall_max_s'], elapsed)
        self.last = dict(
            t=t, q=q, u=u, ud=ud, x=x, e=e, u_r0=u_r0, u_r=u_r, u_r_dot=u_r_dot, s=s, phi_r=phi_r, phi=phi, dphi=dphi, lo=lo, hi=hi,
            phi_meas=phi_meas, tau_ext=tau_ext, load_comp=comp, storage=storage, dissipation=dissipation, P_cmd=P_cmd, P_mis=P_mis,
            P_rows=P_rows, P_contact=P_contact, P_bounds=P_bounds, P_acc=P_acc, rows=rows, on=on, h=h, hdot=hdot, hdot_r=hdot_r,
            psi1=hdot + a1 * h, psi2=psi2,
            psi2_pred=psi2_pred, hdd=hdd, margin=margin, margin_components=mcomp, sigma=sigma, a=a, b=b, c=c, rhs=rhs,
            active=rec['active'], mu=rec['mu'], slack=rec['slack'], feasible=rec['feasible'], bound_active=rec['bound_active'],
            velocity_rows_active=vinfo['active'], velocity_relaxed=vinfo['relaxed'], velocity_level=vinfo['level'],
            speed_active=vinfo['speed_active'], effort_active=vinfo['effort_active'], intervening=intervening, Mr=Mr, Mr_dot=Mr_dot, C=C, L=L,
            hr=hr,
            kin=kin, lift=lift, a_base=a_base, tau_rr=tau_rr, kkt=rec['kkt'], qp_iterations=rec['iterations'], wall_s=elapsed,
            f_meas=f_meas, load_bound=load_bound, load_norm=load_norm, load_floor=load_floor, no_authority=rec['no_authority'],
            velocity_no_authority=vinfo['no_authority'], tau_split=tau_split, velocity_slack=vinfo['slack'],
            TF=TF, TM=TM, force_model_error=err, row_residual_max=resid_max, psi2_after_active=psi2_after,
            contact_active=crec['active'], contact_slack=crec['slack'], contact_margin=contact_margin, contact_c=cc, contact_rhs=rc,
            contact_force=None if sup is None else lam_meas.reshape(-1, 3),
            contact_force_command=None if sup is None else (A_lam @ (phi_meas + theta * (phi - phi_meas)) + b_lam).reshape(-1, 3),
            contact_force_nominal=None if sup is None else (A_lam @ phi_r + b_lam).reshape(-1, 3),
            zs=zs_now, a_r=np.asarray(u_r_dot, float).copy(), uD=(self._uD.copy() if self._uD is not None else np.zeros(xc.d)),
            first_layer=opt.first_layer)
        self.last['pair_active'] = self._pair_mask(on[rec['active']])
        self.last['pair_velocity_active'] = self._pair_mask(on[vinfo['active']])
        return phi

    def _pair_mask(self, row_idx):
        mask = np.zeros(len(self.xc.barriers.pair_ids), bool)
        mask[self._row_pair_index[row_idx]] = True
        return mask

    # ------------------------------------------------------------------ summaries
    def active_ids(self):
        if self.last is None:
            return []
        ids = self.last['rows'].ids
        return [ids[i] for i in self.last['on'][self.last['active']]]

    def class_minima(self):
        """Smallest h of every barrier class at the last sample (all rows, also outside the broad phase)."""
        rows = self.last['rows']
        return {c: float(np.min(rows.h[idx])) for c, idx in self.xc.barriers.class_rows.items()}

    def diagnostics(self):
        s = dict(self.stats)
        s.update(mean_wall_s=s['wall_s'] / max(1, s['evaluations']), options=self.opt.as_dict(),
                 assembly=dict(self.xc.ci.stats))
        return s
