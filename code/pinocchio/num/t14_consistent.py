"""Tracking of a reference that runs into limits again and again: the first-order first layer of the note against a
required motion that respects the second-order rows (six-joint arm of toy.py, ideal torque sources).

Reference (periodic)   q_d(t) = Q_A + (1 - cos(2 pi t / T_p)) / 2 * (Q_B - Q_A):  Q_B lies below the floor and beyond the
                       elbow limit, so the reference leaves the safe set and returns in every period.

Part 1  First layer of the note (projection of the nominal velocity on the first-order rows), exact feedforward, both
        layers.  At every onset of a hold (time t2 at which a kinematic row becomes held) the lower bound
            |g_i(t2)|_M |s(t2)|_M >= -w_i(t2) = psi1s_i(t2) >= exp(-alpha2_i (t2 - ta)) psi1s_i(ta)     (ta <= t2)
        is evaluated with ta = the time tb at which the effort row began to bind, together with the value
        v / (e alpha2) of the idealized onset for a slack that closes at the rate v.  The velocity error does not converge.

Part 2  Required velocity as a state of the controller,  u_r' = a_r,
            a_r = argmin 1/2 |a - a_n|^2   s.t.   J_i a >= -(Jd_i u_r + alpha1_i J_i u + beta zs_i),      zs_i = J_i u_r + alpha1_i hs_i,
            a_n = u_n' + kappa (u_n - u_r),
        i.e. the rows  zs_i' + beta zs_i >= 0  (derivatives along the actual motion), with beta <= min alpha2.  Then the
        slack defect vanishes and e_a = 0, and the main inequality at the rate alpha_A of the binding set gives
        S' <= -2 (mu - rho) S,  mu = min(k, alpha_min)  (the rate beta of the first layer does not enter); for k below
        the smallest second rate alpha_min the quadratic bound gives  S' <= -2 (k - rho^2 / (4 (alpha_min - k))) S  as well:
        (a) start with u_r = u: s = 0 for all times and the effort layer never binds;
        (b) start with a velocity error: exponential decay;
        (c) bounded mismatch e_m with a standoff: the main inequality with its mismatch term, ultimate bound.
        The inequality of the theorem is evaluated as it is stated, at every step with a binding effort row:
            S' <= -k |s_perp|^2 - min(k, alpha_A) |s_par|^2 + |s_par| d_A[eta + c] - <s, e_m>_M
        (eta: sign defect at the rate alpha_A; c: curvature term).  Its residual is (S' - right-hand side) / (2 S), in units
        of a decay rate.  The slack defect at the rate alpha_A, which the theorem shows to be zero, is evaluated on the
        binding rows and printed separately; it is not put into the bracket.  For every residual the number of steps behind
        it is printed, and the size of |s| at these steps: where |s| is at the level of the rounding errors the residual
        tests nothing.
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np      # noqa: E402

import paths            # noqa: E402
from ldp import Infeasible, min_norm, project      # noqa: E402
from toy import Arm, Controller, advance_with_events, simulate      # noqa: E402
from run_toy import Q_A, Q_B, barriers      # noqa: E402

OUT = {}
T_P = 3.0


def periodic_reference(t):
    w = 2. * np.pi / T_P
    c, sn = np.cos(w * t), np.sin(w * t)
    return Q_A + 0.5 * (1. - c) * (Q_B - Q_A), 0.5 * w * sn * (Q_B - Q_A), 0.5 * w * w * c * (Q_B - Q_A)


# ---------------------------------------------------------------------------------------------------------------- part 1
def part1(periods=4):
    arm = Arm()
    res = []
    print('Part 1: first-order first layer, exact feedforward, periodic reference (period 3 s)')
    for k in (12., 30.):
        bar = barriers(arm)
        ctl = Controller(arm, bar, periodic_reference, lam=12., k=k)
        L = simulate(ctl, Q_A, np.zeros(6), periods * T_P, dt=1e-3)
        t, sn = L['t'], np.sqrt(2 * L['S'])
        act, act_k = L['act'].astype(bool), L['act_k'].astype(bool)
        per = [float(sn[(t >= j * T_P) & (t < (j + 1) * T_P)].max()) for j in range(periods)]
        l2 = [float(np.sum(sn[(t >= j * T_P) & (t < (j + 1) * T_P)] ** 2) * 1e-3) for j in range(periods)]
        onsets = []
        for row in range(bar.R):
            starts = np.flatnonzero(act_k[1:, row] & ~act_k[:-1, row]) + 1          # first held step
            for i2 in starts:
                # the effort row binds on the steps before the hold: go back to its first binding step
                ib = i2 - 1
                if not act[ib, row]:
                    continue
                while ib > 0 and act[ib - 1, row]:
                    ib -= 1
                o2 = ctl(t[i2], L['q'][i2], L['u'][i2], log=True)[2]
                ob = ctl(t[ib], L['q'][ib], L['u'][ib], log=True)[2]
                g2 = float(np.sqrt(o2['J'][row] @ np.linalg.solve(o2['M'], o2['J'][row])))
                lhs = g2 * float(np.sqrt(2 * o2['S']))
                psi2_, psib = float(o2['psi1'][row]), float(ob['psi1'][row])
                low = float(np.exp(-bar.a2[row] * (t[i2] - t[ib])) * psib)
                v = -float(ob['zs_dot'][row])
                onsets.append(dict(row=bar.names[row], t_b=float(t[ib]), t_2=float(t[i2]), g_s=lhs, minus_w=-float(o2['w'][row]), psi1_t2=psi2_,
                                   lower=low, psi1_tb=psib, zs_tb=float(ob['zs'][row]), s_tb=float(np.sqrt(2 * ob['S'])), closing_rate=v,
                                   leading_order=v / (np.e * bar.a2[row]), binding_time=float(t[i2] - t[ib]), inv_alpha2=1. / float(bar.a2[row])))
        last = t >= (periods - 1) * T_P
        dev = np.array([np.sqrt(max((ur_ - un_) @ M_ @ (ur_ - un_), 0.)) for ur_, un_, M_ in
                        ((o_['u_r'], o_['u_n'], o_['M']) for o_ in (ctl(t[i], L['q'][i], L['u'][i], log=True)[2] for i in np.flatnonzero(last)))])
        one = dict(k=k, sup_per_period=per, l2_per_period=l2, onsets=onsets, h_min=float(L['hmin'].min()), psi1_min=float(L['psi1min'].min()),
                   events=int(L['events'][0]), effort_binding_steps=int(act.any(axis=1).sum()), first_layer_binding_steps=int(act_k.any(axis=1).sum()),
                   steps=int(len(t)), e_max_last_period=float(L['e'][last].max()), e_mean_last_period=float(L['e'][last].mean()),
                   dev_mean_last_period=float(dev.mean()), dev_max_last_period=float(dev.max()))
        res.append(one)
        print(f"   k = {k:g}: sup |s|_M per period {np.round(per, 4).tolist()} sqrt(J);  integral of |s|_M^2 per period {np.round(l2, 5).tolist()};  min hs {one['h_min']:+.1e}")
        print(f"        steps with a binding effort row {one['effort_binding_steps']}, with a binding row of the first layer {one['first_layer_binding_steps']} of {one['steps']};  "
              f"last period: max |q_d - q| = {one['e_max_last_period']:.4f}, mean {one['e_mean_last_period']:.4f} rad;  |u_r - u_n|_M mean {one['dev_mean_last_period']:.4f}, max {one['dev_max_last_period']:.4f}")
        for o in onsets:
            print(f"        {o['row']:6s} held from t2 = {o['t_2']:.3f} s (effort row binding since tb = {o['t_b']:.3f} s, {o['binding_time']:.4f} s; 1/alpha2 = {o['inv_alpha2']:.4f}):  "
                  f"|g||s|(t2) = {o['g_s']:.4f} >= psi1s(t2) = {o['psi1_t2']:.4f} >= e^(-alpha2 (t2-tb)) psi1s(tb) = {o['lower']:.4f};  |s(tb)| = {o['s_tb']:.1e};  "
                  f"closing rate v = {o['closing_rate']:.3f}, v/(e alpha2) = {o['leading_order']:.4f}")
        np.savez_compressed(paths.NUM / f't14_first_order_k{k:g}.npz', t=t, snorm=sn, hmin=L['hmin'])
    OUT['first_order'] = res


# ---------------------------------------------------------------------------------------------------------------- part 2
class Consistent:
    """Loop with the required velocity as a controller state (see the header)."""

    def __init__(self, arm, bar, ref, lam=12., k=12., beta=None, kappa=20., e_m=None):
        self.arm, self.bar, self.ref, self.lam, self.k, self.kappa, self.e_m = arm, bar, ref, lam, k, kappa, e_m
        self.beta = float(np.min(bar.a2)) if beta is None else beta
        self.infeasible = 0

    def __call__(self, t, q, u, ur, log=False, active_sets=False):
        """Plant acceleration (with the mismatch) and required acceleration; with ``active_sets`` the binding rows of the
        first layer and of the effort layer (the signature of the event location); with ``log`` the logged quantities."""
        arm, bar, k, beta = self.arm, self.bar, self.k, self.beta
        M, N, _ = arm.dyn(q, u)
        h, J = bar.hJ(q)
        Jd = bar.Jdot(q, u)
        qd, qd_dot, qd_ddot = self.ref(t)
        u_n = qd_dot + self.lam * (qd - q)
        un_dot = qd_ddot + self.lam * (qd_dot - u)
        a_n = un_dot + self.kappa * (u_n - ur)
        zs = J @ ur + bar.a1 * h
        rk = -(Jd @ ur + bar.a1 * (J @ u) + beta * zs)                       # rows  J a >= rk   <=>   zs' + beta zs >= 0
        try:
            a_r, nu, _ = project(np.eye(arm.n), a_n, J, rk)
        except Infeasible:
            if not active_sets:
                self.infeasible += 1
            a_r, nu = a_n.copy(), np.zeros(bar.R)
        s = ur - u
        Mi = np.linalg.inv(M)
        a_nom = a_r + Mi @ (N @ s) + k * s
        rhs = -(Jd @ u + (bar.a1 + bar.a2) * (J @ u) + bar.a1 * bar.a2 * h)
        a, lam, _ = project(M, a_nom, J, rhs)
        act = lam > 1e-10 * (1. + lam.max())
        act_kin = nu > 1e-10 * (1. + nu.max())
        if active_sets:
            return act_kin, act
        em = self.e_m(t, q, u) if self.e_m is not None else np.zeros(arm.n)
        if not log:
            return a + em, a_r
        zs_dot = J @ a_r + Jd @ ur + bar.a1 * (J @ u)
        c = Jd @ s - J @ (Mi @ (N @ s))
        psi1 = J @ u + bar.a1 * h
        S = 0.5 * float(s @ M @ s)
        Pf = -float(s @ M @ (a - a_nom))
        S_dot = -2. * k * S + Pf - float(s @ M @ em)
        out = dict(S=S, snorm=np.sqrt(2 * S), Pf=Pf, S_dot=S_dot, act=act, act_kin=act_kin, zs=zs, zs_dot=zs_dot, psi1=psi1, h=h,
                   cons=zs_dot + beta * zs, e=float(np.max(np.abs(qd - q))), em_norm=float(np.sqrt(em @ M @ em)), bound=np.nan, rho=0., gap=np.nan,
                   alpha_A=np.nan, spar2=0., zeta=0.,
                   c=c, J=J, M=M, s=s, w=J @ s, corr=float(np.sqrt(max((a - a_nom) @ M @ (a - a_nom), 0.))),
                   dev=float(np.sqrt(max((ur - u_n) @ M @ (ur - u_n), 0.))))
        if act.any():
            JA = J[act]
            Lam = JA @ Mi @ JA.T
            s_par = Mi @ JA.T @ np.linalg.solve(Lam, JA @ s)
            p2 = float(s_par @ M @ s_par)
            # the inequality of the theorem on consistent required motions as it is stated: rate min(k, alpha_A), alpha_A the
            # smallest second rate of the binding set, and in the bracket the sign defect at the rate alpha_A and the curvature
            # term.  The slack defect at the rate alpha_A is zero for a consistent required motion; it is evaluated on the
            # binding rows and reported separately (out['zeta']), not put into the bracket.
            a_A = float(np.min(bar.a2[act]))
            mu = min(k, a_A)
            eta = (bar.a2 - a_A) * np.maximum(0., -psi1)
            zeta = np.maximum(0., -(zs_dot + a_A * zs))
            dd, _, _ = min_norm(M, JA, (eta + c)[act])
            dA = float(np.sqrt(max(dd @ M @ dd, 0.)))
            out['bound'] = -k * (2 * S - p2) - mu * p2 + np.sqrt(p2) * dA - float(s @ M @ em)
            out['gap'] = (S_dot - out['bound']) / (2. * S) if S > 0. else np.nan       # scale-free: in units of the decay rate
            out['alpha_A'] = a_A
            out['spar2'] = p2
            out['zeta'] = float(zeta[act].max())
            Gam = Jd[act] - JA @ Mi @ N
            out['rho'] = float(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(Lam), Gam) @ np.linalg.inv(np.linalg.cholesky(M)).T, 2))
        return a + em, a_r, out


def simulate_consistent(ctl, q0, u0, ur0, T, dt=1e-3, log_every=1, events=True):
    """RK4 on the grid dt with the event location of toy.simulate (toy.advance_with_events): a change of the binding rows
    of the first layer or of the effort layer inside a step is located by bisection and the step is split there, as for
    the loop with the first-order first layer.  (The right-hand side of this loop is continuous; the location removes
    the loss of order of RK4 at its kinks.)  ``events=False`` gives plain RK4 steps."""
    q, u, ur = np.array(q0, float), np.array(u0, float), np.array(ur0, float)
    n = int(round(T / dt))
    keys = ('S', 'snorm', 'Pf', 'S_dot', 'e', 'bound', 'rho', 'gap', 'em_norm', 'corr', 'dev', 'zeta', 'alpha_A')
    L = {kk: [] for kk in keys}
    for kk in ('t', 'hmin', 'psi1min', 'zsmin', 'consmin', 'nact', 'nact_kin'):
        L[kk] = []

    def f(t, q, u, ur):
        a, a_r = ctl(t, q, u, ur)
        return u, a, a_r

    def signature(t, q, u, ur):
        ak, ae = ctl(t, q, u, ur, active_sets=True)
        return ak.tobytes() + ae.tobytes()

    counts = dict(events=0)
    for i in range(n + 1):
        t = i * dt
        if i % log_every == 0:
            o = ctl(t, q, u, ur, log=True)[2]
            for kk in keys:
                L[kk].append(o[kk])
            L['t'].append(t); L['hmin'].append(float(o['h'].min())); L['psi1min'].append(float(o['psi1'].min()))
            L['zsmin'].append(float(o['zs'].min())); L['consmin'].append(float(o['cons'].min()))
            L['nact'].append(int(o['act'].sum())); L['nact_kin'].append(int(o['act_kin'].sum()))
        if i == n:
            break
        q, u, ur = advance_with_events(f, signature, t, (q, u, ur), dt, counts, events)
    out = {kk: np.array(v) for kk, v in L.items()}
    out['events'] = np.array([counts['events']])
    return out


def part2(periods=4):
    arm = Arm()
    res = {}
    print('Part 2: required velocity as a controller state with the rows zs\' + beta zs >= 0 (beta = min alpha2 = 16), same reference')
    # (a) start with u_r = u = 0 (in the safe set)
    for k in (12., 30.):
        bar = barriers(arm)
        ctl = Consistent(arm, bar, periodic_reference, k=k)
        L = simulate_consistent(ctl, Q_A, np.zeros(6), np.zeros(6), periods * T_P)
        t = L['t']
        per = [float(L['snorm'][(t >= j * T_P) & (t < (j + 1) * T_P)].max()) for j in range(periods)]
        one = dict(k=k, beta=ctl.beta, sup_per_period=per, effort_binding_steps=int((L['nact'] > 0).sum()), first_layer_binding_steps=int((L['nact_kin'] > 0).sum()),
                   h_min=float(L['hmin'].min()), psi1_min=float(L['psi1min'].min()), zs_min=float(L['zsmin'].min()), cons_min=float(L['consmin'].min()),
                   infeasible=ctl.infeasible, e_max_last_period=float(L['e'][t >= (periods - 1) * T_P].max()),
                   e_mean_last_period=float(L['e'][t >= (periods - 1) * T_P].mean()), dev_mean_last_period=float(L['dev'][t >= (periods - 1) * T_P].mean()),
                   dev_max_last_period=float(L['dev'][t >= (periods - 1) * T_P].max()), largest_correction=float(L['corr'].max()),
                   events=int(L['events'][0]))
        res[f'a_k{k:g}'] = one
        print(f"   (a) k = {k:g}: sup |s|_M per period {['%.1e' % x for x in per]};  steps with a binding effort row {one['effort_binding_steps']}, with a binding row of the first layer "
              f"{one['first_layer_binding_steps']} of {len(t)};  min hs {one['h_min']:+.2e}, min psi1s {one['psi1_min']:+.2e}, min zs {one['zs_min']:+.2e}, min (zs' + beta zs) {one['cons_min']:+.1e};  "
              f"infeasible steps {one['infeasible']};  located active-set changes {one['events']}")
        print(f"        largest correction of the effort layer |a - a_nom|_M = {one['largest_correction']:.1e};  last period: max |q_d - q| = {one['e_max_last_period']:.4f}, "
              f"mean {one['e_mean_last_period']:.4f} rad;  |u_r - u_n|_M mean {one['dev_mean_last_period']:.4f}, max {one['dev_max_last_period']:.4f}")
        np.savez_compressed(paths.NUM / f't14_consistent_k{k:g}.npz', t=t, snorm=L['snorm'], hmin=L['hmin'])
    # (b) start with a velocity error: u(0) = 0, u_r(0) admissible (zs >= 0) and different from u
    rng = np.random.default_rng(3)
    for k, beta in ((12., 16.), (30., 16.), (12., 6.)):
        bar = barriers(arm)
        ctl = Consistent(arm, bar, periodic_reference, k=k, beta=beta)
        h0, J0 = bar.hJ(Q_A)
        ur0 = rng.standard_normal(6) * 0.8
        ur0 = project(np.eye(6), ur0, J0, -bar.a1 * h0 + 0.02)[0]                 # zs(0) >= 0.02
        L = simulate_consistent(ctl, Q_A, np.zeros(6), ur0, 2 * T_P)
        t, sn = L['t'], L['snorm']
        a_min = float(np.min(bar.a2))
        mu = min(k, a_min)
        # rate of the theorem:  max(mu - rho, k - rho^2 / (4 (alpha_min - k)))  if k < alpha_min,  mu - rho otherwise,  mu = min(k, alpha_min)
        om = mu - L['rho']
        if k < a_min:
            om = np.maximum(om, k - L['rho'] ** 2 / (4. * (a_min - k)))
        m = np.concatenate([[0.], np.cumsum(0.5 * (om[1:] + om[:-1]) * np.diff(t))])
        env = np.exp(-m) * sn[0]
        sel = (sn > 1e-9 * sn[0]) & (t > 0.02)
        rate = float(-np.polyfit(t[sel], np.log(sn[sel]), 1)[0]) if sel.sum() > 20 else float('nan')
        bind = np.isfinite(L['gap'])                             # steps with a binding effort row (S > 0 at all of them)
        cnt = bind & (sn > 1e-9 * sn[0])                         # ... at which |s| is above the threshold of the residual
        gap = L['gap'][cnt]
        rel = sn / sn[0]
        nan = float('nan')
        one = dict(k=k, beta=beta, mu=mu, s0=float(sn[0]), s_end=float(sn[-1]), fitted_rate=rate, envelope_excess=float(np.max((sn - env)[sel]) / sn[0]),
                   rate_min=float(om.min()), rate_max=float(om.max()), envelope_tightness=float(np.min((env / np.maximum(sn, 1e-300))[sel])),
                   rho_max=float(L['rho'].max()), effort_binding_steps=int((L['nact'] > 0).sum()), main_gap_max=float(gap.max()) if len(gap) else nan,
                   main_gap_steps=int(cnt.sum()), s_rel_max_binding=float(rel[bind].max()) if bind.any() else nan,
                   s_rel_min_counted=float(rel[cnt].min()) if cnt.any() else nan, zeta_max_binding=float(L['zeta'][bind].max()) if bind.any() else nan,
                   h_min=float(L['hmin'].min()), psi1_min=float(L['psi1min'].min()), zs_min=float(L['zsmin'].min()), infeasible=ctl.infeasible,
                   events=int(L['events'][0]))
        assert int(bind.sum()) == one['effort_binding_steps']
        res[f'b_k{k:g}_beta{beta:g}'] = one
        print(f"   (b) k = {k:g}, beta = {beta:g}: |s(0)|_M = {one['s0']:.3f} -> {one['s_end']:.1e};  fitted decay rate {rate:.3f} 1/s (mu = {mu:g}, largest rho {one['rho_max']:.3f}, rate of the theorem between {one['rate_min']:.4f} and {one['rate_max']:.4f});  "
              f"largest (|s| - envelope)/|s(0)| = {one['envelope_excess']:+.1e};  min hs {one['h_min']:+.2e}, min zs {one['zs_min']:+.2e};  located active-set changes {one['events']}")
        if bind.any():
            print(f"        steps with a binding effort row {one['effort_binding_steps']}, the largest |s|/|s(0)| at such a step is {one['s_rel_max_binding']:.1e};  {one['main_gap_steps']} of them with |s| > 1e-9 |s(0)|: "
                  f"largest (S' - bound)/(2S) there = {one['main_gap_max']:+.2e} 1/s;  largest evaluated slack defect at the rate alpha_A on a binding row {one['zeta_max_binding']:.1e}")
        else:
            print('        no step with a binding effort row')
    # (c) bounded mismatch with a standoff that covers it (as in the mismatch run of run_toy.py)
    em_bar, Jn = 0.3, 2.2

    def e_m(t, q, u):
        return em_bar * np.array([np.sin(7 * t), np.cos(5 * t), np.sin(3 * t + 1.), np.cos(11 * t), np.sin(13 * t), np.cos(2 * t)]) / np.sqrt(6.)
    bar0 = barriers(arm)
    sigma = Jn * em_bar / (bar0.a1 * bar0.a2)
    for k in (12.,):
        bar = barriers(arm, sigma=sigma)
        ctl = Consistent(arm, bar, periodic_reference, k=k, e_m=e_m)
        L = simulate_consistent(ctl, Q_A, np.zeros(6), np.zeros(6), periods * T_P)
        t, sn = L['t'], L['snorm']
        mu = min(k, float(np.min(bar.a2)))
        per = [float(sn[(t >= j * T_P) & (t < (j + 1) * T_P)].max()) for j in range(periods)]
        bind = np.isfinite(L['gap'])
        gap = L['gap'][bind]
        aA = L['alpha_A'][bind]
        one = dict(k=k, sup_per_period=per, em_norm_max=float(L['em_norm'].max()), em_over_mu=float(L['em_norm'].max() / mu), main_gap_max=float(gap.max()) if len(gap) else float('nan'),
                   effort_binding_steps=int((L['nact'] > 0).sum()), main_gap_steps=int(bind.sum()), s_min_binding=float(sn[bind].min()) if bind.any() else float('nan'),
                   zeta_max_binding=float(L['zeta'][bind].max()) if bind.any() else float('nan'),
                   alpha_A_counts={f'{x:g}': int((aA == x).sum()) for x in np.unique(aA)},
                   h_unshifted_min=float(L['hmin'].min() + sigma.min()), hs_min=float(L['hmin'].min()),
                   psi1_min=float(L['psi1min'].min()), infeasible=ctl.infeasible, events=int(L['events'][0]))
        assert one['main_gap_steps'] == one['effort_binding_steps']
        res[f'c_k{k:g}'] = one
        print(f"   (c) k = {k:g}, mismatch |e_m|_2 <= {em_bar}: sup |s|_M per period {np.round(per, 4).tolist()};  largest |e_m|_M / mu = {one['em_over_mu']:.4f};  "
              f"smallest shifted barrier value {one['hs_min']:+.2e}, min psi1s {one['psi1_min']:+.2e};  located active-set changes {one['events']}")
        print(f"        steps with a binding effort row {one['effort_binding_steps']} (alpha_A: " + ', '.join(f'{a} at {c}' for a, c in one['alpha_A_counts'].items())
              + f"), the smallest |s|_M at such a step is {one['s_min_binding']:.1e};  largest (S' - bound)/(2S) at these steps = {one['main_gap_max']:+.2e} 1/s;  "
              f"largest evaluated slack defect at the rate alpha_A on a binding row {one['zeta_max_binding']:.1e}")
        np.savez_compressed(paths.NUM / f't14_consistent_mismatch_k{k:g}.npz', t=t, snorm=sn, hmin=L['hmin'])
    OUT['consistent'] = res


def part3(periods=4):
    """Rate of the first layer above the second rates of the rows: the required motion is then not consistent."""
    arm = Arm()
    res = {}
    print('Part 3: the same first layer with beta = 40 > alpha2 (16 and 20), start with u_r = u')
    for k in (12., 30.):
        bar = barriers(arm)
        ctl = Consistent(arm, bar, periodic_reference, k=k, beta=40.)
        L = simulate_consistent(ctl, Q_A, np.zeros(6), np.zeros(6), periods * T_P)
        t = L['t']
        per = [float(L['snorm'][(t >= j * T_P) & (t < (j + 1) * T_P)].max()) for j in range(periods)]
        one = dict(k=k, beta=40., sup_per_period=per, effort_binding_steps=int((L['nact'] > 0).sum()), first_layer_binding_steps=int((L['nact_kin'] > 0).sum()),
                   h_min=float(L['hmin'].min()), zs_min=float(L['zsmin'].min()), cons_min=float(L['consmin'].min()), largest_correction=float(L['corr'].max()),
                   infeasible=ctl.infeasible, events=int(L['events'][0]))
        res[f'k{k:g}'] = one
        print(f"   k = {k:g}: sup |s|_M per period {np.round(per, 4).tolist()};  steps with a binding effort row {one['effort_binding_steps']};  "
              f"largest correction of the effort layer {one['largest_correction']:.2f};  min hs {one['h_min']:+.2e}, min zs {one['zs_min']:+.1e};  located active-set changes {one['events']}")
    OUT['beta_above_alpha2'] = res


if __name__ == '__main__':
    parts = sys.argv[1] if len(sys.argv) > 1 else '123'
    try:
        OUT.update(json.load(open(paths.NUM / 't14_results.json')))
    except Exception:
        pass
    if '1' in parts:
        part1()
    if '2' in parts:
        part2()
    if '3' in parts:
        part3()
    json.dump(OUT, open(paths.NUM / 't14_results.json', 'w'), indent=1, default=float)
