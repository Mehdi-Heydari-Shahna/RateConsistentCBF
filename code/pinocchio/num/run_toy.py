"""Six-joint arm of Section VII-A: start pose Q_A, reference target Q_B and the five barriers (``barriers``), used by
t14_consistent.py and checks/arm_onset_corollary.py.  Run as a script, the file also performs further experiments with
the continuous-time two-layer loop of toy.py that the paper does not report (results in toy_results.json and toy_<run>.npz).

    python run_toy.py [experiment ...]        (no argument: all)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from toy import Arm, Barriers, Controller, constant_reference, ramp_reference, simulate

NUM = Path(__file__).resolve().parent          # the result files are written next to the scripts
OUT = {}
Q_A = np.array([0., 0.5, 0., 1.0, 0., 0.3])            # end effector at (1.477, 0, 0.948)
Q_B = np.array([0.4, 0.9, 0., 1.4, 0.2, 0.3])          # end effector at (1.529, 0.017, -0.041): below the floor, past the elbow limit


def barriers(arm, floor=(6., 16.), limit=(8., 20.), sigma=0., elbow_hi=1.25):
    return Barriers(arm, limits=[(3, None, elbow_hi, *limit), (1, -0.2, 1.5, *limit)],
                    planes=[((0., 0., 1.), 0.3, *floor), ((-1., 0., 0.), -1.7, *floor)], sigma=sigma)


def summary(name, ctl, L, betas, hold_from=None, quiet=False):
    """Checks of one run; returns a dict of numbers."""
    t, S, Pf = L['t'], L['S'], L['Pf']
    dt = t[1] - t[0]
    res = dict(name=name, k=ctl.k, lam=ctl.lam, steps=len(t), h_min=float(L['hmin'].min()), psi1_min=float(L['psi1min'].min()),
               psi2_min=float(L['psi2min'].min()), active_steps=int((L['nact'] > 0).sum()), kin_active_steps=int((L['nact_k'] > 0).sum()),
               max_rows_active=int(L['nact'].max()), S0=float(S[0]), S_end=float(S[-1]), Pf_max=float(Pf.max()), Pf_min=float(Pf.min()),
               subset_fraction=float(np.mean(L['subset'][L['nact'] > 0])) if (L['nact'] > 0).any() else 1.)
    # energy balance  S(t) - S(0) = int (-dissipation + P_ea + P_em + P_f): the integral is a state of the simulation
    res['balance_err'] = float(np.max(np.abs(S - S[0] - L['balance'])) / max(1e-12, np.max(S)))
    res['S_max'] = float(S.max())
    res['events'] = int(L['events'][0])
    res['t_S_max'] = float(t[S.argmax()])
    for nm in betas:
        b = L[f'beta:{nm}']
        mu = np.minimum(ctl.k, b)
        gap = Pf - L[f'bound:{nm}']
        res[f'viol:{nm}'] = float(np.max(gap / np.maximum(1., np.abs(Pf))))
        res[f'star_err:{nm}'] = float(L[f'star_err:{nm}'].max())
        res[f'dA_max:{nm}'] = float(L[f'dA:{nm}'].max())
        res[f'split_viol:{nm}'] = float(np.max((Pf - L[f'bound_split:{nm}']) / np.maximum(1., np.abs(Pf))))
        # comparison envelope  y' = -mu y + dA + |e_a| + |e_m|,  y(0) = sqrt(2 S(0))
        y = np.zeros_like(S)
        y[0] = np.sqrt(2 * S[0])
        inp = L[f'dA:{nm}'] + L['ea_norm'] + L['em_norm']
        for i in range(len(t) - 1):                       # exact step of the linear comparison system with the input held (upper sum)
            m_ = mu[i]
            y[i + 1] = np.exp(-m_ * dt) * y[i] + (1 - np.exp(-m_ * dt)) / m_ * max(inp[i], inp[i + 1])
        res[f'envelope_viol:{nm}'] = float(np.max(np.sqrt(2 * S) - y))
        res[f'envelope_end:{nm}'] = float(y[-1])
        L[f'env:{nm}'] = y
    if hold_from is not None:
        # decay rate of |s|_M after the last change of an active set (fit of log |s| while |s| > 1e-8)
        chg = np.flatnonzero(np.any(L['act'][1:] != L['act'][:-1], axis=1) | np.any(L['act_k'][1:] != L['act_k'][:-1], axis=1))
        t0 = (t[chg[-1] + 1] if len(chg) else 0.) + 0.05
        sn = np.sqrt(2 * S)
        sel = (t >= max(t0, hold_from)) & (sn > 1e-8)
        if sel.sum() > 20:
            res['hold_rate_fit'] = float(-np.polyfit(t[sel], np.log(sn[sel]), 1)[0])
            res['hold_fit_window'] = [float(t[sel][0]), float(t[sel][-1])]
    if not quiet:
        print(f"[{name}] steps {res['steps']}  active (effort/kinematic) {res['active_steps']}/{res['kin_active_steps']}  rows<= {res['max_rows_active']}"
              f"  h_min {res['h_min']:.2e}  psi1_min {res['psi1_min']:.2e}  psi2_min {res['psi2_min']:.2e}")
        print(f"    S(0) {res['S0']:.3e} -> S(T) {res['S_end']:.3e}   P_f in [{res['Pf_min']:.3e}, {res['Pf_max']:.3e}]   max S {res['S_max']:.3e} at t = {res['t_S_max']:.3f}   balance error {res['balance_err']:.1e}"
              f"   effort-active rows held by the kinematic layer: {100 * res['subset_fraction']:.1f} %")
        for nm in betas:
            print(f"    beta={nm:10s} max(P_f - bound)/scale {res[f'viol:{nm}']:+.2e}  (split {res[f'split_viol:{nm}']:+.2e})  row identity err {res[f'star_err:{nm}']:.1e}"
                  f"  defect max {res[f'dA_max:{nm}']:.3e}  envelope: max(|s| - y) {res[f'envelope_viol:{nm}']:+.2e}")
        if 'hold_rate_fit' in res:
            print(f"    fitted decay rate of |s|_M on t in {np.round(res['hold_fit_window'], 2).tolist()}: {res['hold_rate_fit']:.3f} 1/s")
    return res


def beta_set(ctl, bar):
    """Comparison rates: k, the smallest alpha2 of the active rows (hold), and a fixed small one."""
    def amin(o):
        return float(np.min(bar.a2[o['act']])) if o['act'].any() else ctl.k
    return {'k': lambda o: ctl.k, 'a2min': amin, 'half': lambda o: 0.5 * min(ctl.k, amin(o))}


def run(name, ctl, q0, u0, T, hold_from=None, dt=1e-3, e_m=None, keep=True, betas=None):
    bar = ctl.bar
    betas = betas or beta_set(ctl, bar)
    start = time.time()
    L = simulate(ctl, q0, u0, T, dt=dt, e_m=e_m, betas=betas)
    res = summary(name, ctl, L, betas, hold_from)
    res['wall_s'] = time.time() - start
    OUT[name] = res
    if keep:
        np.savez_compressed(NUM / f'toy_{name}.npz', **{k: v for k, v in L.items()})
    return L, res


def exp_checks():
    """Derivative of the required velocity (analytic against finite differences) for both velocity metrics."""
    arm = Arm()
    rng = np.random.default_rng(3)
    worst = 0.
    for W in ('identity', 'kinetic'):
        bar = barriers(arm)
        ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.0), W=W)
        n_act = 0
        for _ in range(300):
            q = Q_A + rng.uniform(0., 1., 1) * (Q_B - Q_A) + rng.uniform(-0.15, 0.15, 6)
            u = rng.uniform(-1., 1., 6)
            t = rng.uniform(0.1, 0.9)
            M, N, g = arm.dyn(q, u)
            h, J = bar.hJ(q)
            if np.min(h) < 0.:
                continue
            Jd = bar.Jdot(q, u)
            _, _, u_r, ur_dot, act = ctl.required(t, q, u, M, N, h, J, Jd)
            eps = 1e-6
            fd = (ctl.U_r(t + eps, q + eps * u) - ctl.U_r(t - eps, q - eps * u)) / (2 * eps)
            if act.any():
                n_act += 1
            worst = max(worst, float(np.max(np.abs(fd - ur_dot)) / (1. + np.max(np.abs(ur_dot)))))
        print(f'[checks] W={W}: analytic derivative of U_r against central differences, worst relative error {worst:.2e} ({n_act} states with active rows)')
    OUT['checks'] = dict(ur_dot_err=worst)


def exp_hold():
    """E1: the reference leaves the safe set (floor and elbow limit); hold.  (a) k < alpha2, (b) k > alpha2."""
    arm = Arm()
    for tag, k, floor, limit in (('a', 12., (6., 16.), (8., 20.)), ('b', 30., (6., 16.), (8., 20.)), ('c', 12., (6., 4.), (8., 4.))):
        bar = barriers(arm, floor, limit)
        ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=k)
        run(f'hold_{tag}', ctl, Q_A, np.zeros(6), 4.0, hold_from=0.0)


def exp_hold_initial_error():
    """E1d: hold reached with a large initial velocity error (the effort layer acts before the kinematic rows hold)."""
    arm = Arm()
    rng = np.random.default_rng(5)
    for tag, k in (('k12', 12.), ('k30', 30.)):
        bar = barriers(arm)
        ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=k)
        u0 = safe_u0(bar, Q_A, np.array([0.3, 0.8, -0.2, 1.0, 0.4, -0.3]))      # towards the floor and the elbow limit, inside the safe set
        run(f'hold_u0_{tag}', ctl, Q_A, u0, 4.0, hold_from=0.0)


def safe_u0(bar, q0, direction, psi1_left=0.05):
    """Largest multiple of ``direction`` with every first-order function psi1s_i = J_i u + alpha1_i hs_i >= psi1_left."""
    h, J = bar.hJ(q0)
    rate = J @ direction
    lim = np.where(rate < 0., (bar.a1 * h - psi1_left) / np.maximum(-rate, 1e-300), np.inf)
    return float(np.min(lim)) * direction


def exp_recovery():
    """E2: constant reference well inside the safe set; the initial velocity points at the floor and the elbow limit and
    is the largest one of its direction with psi1 >= 0.05 (inside the safe set): the effort layer acts for a while."""
    arm = Arm()
    q_end = Q_A + 0.30 * (Q_B - Q_A)
    direction = np.array([0.5, 1.6, 0., 2.0, 0.5, 0.])
    for tag, k in (('k12', 12.), ('k30', 30.)):
        bar = barriers(arm)
        u0 = safe_u0(bar, Q_A, direction)
        ctl = Controller(arm, bar, constant_reference(q_end), lam=12., k=k)
        h_end = bar.hJ(q_end)[0]
        betas = beta_set(ctl, bar)
        betas['free'] = lambda o, bar=bar, k=k: float(np.min((bar.a1 + bar.a2 - 12.)[o['act']])) if o['act'].any() else k
        L, res = run(f'recovery_{tag}', ctl, Q_A, u0, 3.0, hold_from=0.0, betas=betas)
        res['h_reference_end'] = h_end.tolist()
        res['e_end'] = float(L['e'][-1])
        res['u0'] = u0.tolist()
        last = np.flatnonzero(L['nact'] > 0)
        res['t_filter_last_active'] = float(L['t'][last[-1]]) if len(last) else 0.
        print(f"    u0 = {np.round(u0, 3)}; barrier values at the reference {np.round(h_end, 3)}; effort layer last active at t = {res['t_filter_last_active']:.3f};"
              f" tracking error at the end {res['e_end']:.2e}")


def exp_free():
    """E3: free regime (set-point well inside the safe set, start at rest or with a velocity towards the limits) for three
    sets of rates: matched (lam = alpha1, k = alpha2), the values of the releases, and a slow barrier."""
    arm = Arm()
    q_end = Q_A + 0.30 * (Q_B - Q_A)
    direction = np.array([0.5, 1.6, 0., 2.0, 0.5, 0.])
    for tag, lam, k, floor, limit in (('matched', 6., 16., (6., 16.), (6., 16.)), ('release', 12., 12., (6., 16.), (8., 20.)),
                                      ('slow_barrier', 12., 12., (2., 4.), (2., 4.))):
        for start in ('rest', 'fast'):
            bar = barriers(arm, floor, limit)
            u0 = np.zeros(6) if start == 'rest' else safe_u0(bar, Q_A, direction)
            ctl = Controller(arm, bar, constant_reference(q_end), lam=lam, k=k)
            betas = beta_set(ctl, bar)
            betas['free'] = lambda o, bar=bar, lam=lam, k=k: float(np.min((bar.a1 + bar.a2 - lam)[o['act']])) if o['act'].any() else k
            L, res = run(f'free_{tag}_{start}', ctl, Q_A, u0, 3.0, betas=betas, hold_from=0.0)
            res['e_end'] = float(L['e'][-1])
            res['slack_defect_max:free'] = float(L['dA_slack:free'].max())
            res['curv_defect_max:free'] = float(L['dA_curv:free'].max())
            print(f"    beta = alpha1 + alpha2 - lam: largest slack defect {res['slack_defect_max:free']:.3e}, largest curvature defect {res['curv_defect_max:free']:.3e};"
                  f" tracking error at the end {res['e_end']:.2e}")


def exp_free_identity():
    """Free-regime form of the row identity on random states with an inactive kinematic layer:
    Pi_i^(beta_i) = Psi*_i - c_i - Jdot_i(q, s) u_n,  beta_i = alpha1_i + alpha2_i - lam,  Psi*_i = Z_i + alpha2_i zs_i  along the
    required flow (the second-order row function of the nominal required motion)."""
    arm = Arm()
    rng = np.random.default_rng(11)
    bar = barriers(arm)
    lam, k = 12., 12.
    ctl = Controller(arm, bar, ramp_reference(Q_A, Q_A + 0.3 * (Q_B - Q_A), 1.0), lam=lam, k=k)
    worst, n = 0., 0
    while n < 400:
        q = Q_A + rng.uniform(-0.1, 0.1, 6)
        u = rng.uniform(-1.5, 1.5, 6)
        t = rng.uniform(0.05, 0.95)
        _, _, o = ctl(t, q, u, np.zeros(6), log=True)
        if o['act_k'].any():
            continue
        n += 1
        qd, qd_dot, qd_ddot = ctl.ref(t)
        u_n = o['u_n']
        J, s = o['J'], o['s']
        Jd_n = bar.Jdot(q, u_n)                              # along the required flow
        Z = J @ (qd_ddot + lam * (qd_dot - u_n)) + Jd_n @ u_n + bar.a1 * (J @ u_n)
        psi_star = Z + bar.a2 * o['zs']
        Jd_s = bar.Jdot(q, s)
        beta_i = bar.a1 + bar.a2 - lam
        Pi = J @ (o['a_nom'][None, :] + (beta_i[:, None] - k) * s[None, :]).T
        Pi = np.einsum('ij,ij->i', J, o['a_nom'][None, :] + (beta_i[:, None] - k) * s[None, :]) - o['rhs']
        rhs = psi_star - o['c'] - Jd_s @ u_n
        worst = max(worst, float(np.max(np.abs(Pi - rhs) / (1. + np.abs(Pi)))))
    print(f'[free identity] {n} random states with an inactive kinematic layer: largest relative error {worst:.2e}')
    OUT['free_identity'] = dict(states=n, err=worst)


def exp_kinematic_only():
    """E5: kinematic layer alone (no effort rows) against both layers, from the same initial states inside the safe set.
    With k < alpha1 the kinematic layer alone leaves the safe set; S(t) = S(0) exp(-2kt) holds for it exactly."""
    arm = Arm()
    direction = np.array([0.5, 1.6, 0., 2.0, 0.5, 0.])
    for tag, k in (('k12', 12.), ('k4', 4.)):
        for layers in ('kinematic', 'both'):
            bar = barriers(arm)
            u0 = safe_u0(bar, Q_A, direction)
            ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=k, effort_layer=(layers == 'both'))
            L, res = run(f'layers_{layers}_{tag}', ctl, Q_A, u0, 4.0, hold_from=0.0, betas=beta_set(ctl, bar))
            res['S_rate_err'] = float(np.max(np.abs(L['S'] - L['S'][0] * np.exp(-2 * k * L['t']))) / L['S'][0])
            # sufficient condition of the kinematic layer alone:  hs_i(0) >= gbar_i sqrt(2 S(0)) / (k - alpha1_i),  k > alpha1_i
            gn = np.array([np.sqrt(np.einsum('ri,ir->r', bar.hJ(q)[1], np.linalg.solve(arm.dyn(q, np.zeros(6))[0], bar.hJ(q)[1].T))) for q in L['q'][::25]]).max(axis=0)
            need = np.where(k > bar.a1, gn * np.sqrt(2 * L['S'][0]) / np.maximum(k - bar.a1, 1e-12), np.inf)
            res['kinematic_only_condition'] = dict(h0=L['h'][0].tolist(), needed=[float(x) for x in need])
            print(f"    {layers:9s} k={k:4.1f}: h_min {res['h_min']:+.3e}  psi1_min {res['psi1_min']:+.3e}   |S(t) - S(0) exp(-2kt)| / S(0) = {res['S_rate_err']:.1e}"
                  f"   h(0) {np.round(L['h'][0], 3)}  needed by the sufficient condition {np.round(need, 3)}")


def exp_nonuniform():
    """E6: non-uniform body gains: K with eigenvalues of M^-1 K between k_min and k_max.  Storage may grow during a hold."""
    arm = Arm()
    bar = barriers(arm, floor=(6., 1.), limit=(8., 1.))

    def kmat(M, kmin=0.1, kmax=10.):
        w, V = np.linalg.eigh(M)
        # K = M^(1/2) D M^(1/2), D diagonal in a fixed basis: eigenvalues of M^-1 K are those of D
        Mh = (V * np.sqrt(w)) @ V.T
        D = np.diag(np.geomspace(kmin, kmax, M.shape[0]))
        return Mh @ D @ Mh
    for tag, km in (('nonuniform', kmat), ('uniform', None)):
        ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=2., k=1., k_matrix=km)
        L, res = run(f'gain_{tag}', ctl, Q_A, np.zeros(6), 6.0, betas={'k': lambda o: 1.})
        # growth of the storage with zero defect: largest S_dot over samples at which the effort-active rows are held
        dt = L['t'][1] - L['t'][0]
        Sd = np.gradient(L['S'], dt)
        held = (L['nact'] > 0) & L['subset']
        res['Sdot_max_on_hold'] = float(np.max(Sd[held])) if held.any() else 0.
        res['Sdot_rel_max_on_hold'] = float(np.max(Sd[held] / np.maximum(L['S'][held], 1e-12))) if held.any() else 0.
        print(f"    largest dS/dt on held samples {res['Sdot_max_on_hold']:+.3e}  (relative to S: {res['Sdot_rel_max_on_hold']:+.3f} 1/s)")


def exp_metric():
    """E7: Euclidean metric of the efforts in the effort layer (k > alpha2): the storage can grow during a hold."""
    arm = Arm()
    for tag, metric in (('kinetic', 'kinetic'), ('euclidean', 'euclidean')):
        bar = barriers(arm, floor=(6., 2.), limit=(8., 2.))
        ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=30., metric=metric)
        L, res = run(f'metric_{tag}', ctl, Q_A, np.zeros(6), 4.0, betas={'k': lambda o: 30.})
        dt = L['t'][1] - L['t'][0]
        Sd = np.gradient(L['S'], dt)
        held = (L['nact'] > 0) & L['subset']
        res['Sdot_rel_max_on_hold'] = float(np.max(Sd[held] / np.maximum(L['S'][held], 1e-12))) if held.any() else 0.
        res['Sdot_max_on_hold'] = float(np.max(Sd[held])) if held.any() else 0.
        print(f"    largest dS/dt on held samples {res['Sdot_max_on_hold']:+.3e}  (relative to S: {res['Sdot_rel_max_on_hold']:+.3f} 1/s)")


def exp_feedforward():
    """E8: the feedforward of the release (nominal acceleration + filtered derivative of the correction): e_a is an input."""
    arm = Arm()
    bar = barriers(arm)
    ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=12., feedforward='filtered', tau_c=0.03)
    L, res = run('feedforward_filtered', ctl, Q_A, np.zeros(6), 4.0, hold_from=0.0)
    res['ea_max'] = float(L['ea_norm'].max())
    print(f"    largest |e_a|_M {res['ea_max']:.3e}")


def exp_mismatch():
    """E9: bounded acceleration mismatch e_m (unknown to the controller) with matching row margins (standoff)."""
    arm = Arm()
    em_bar = 0.3

    def e_m(t, q, u):
        return em_bar * np.array([np.sin(7 * t), np.cos(5 * t), np.sin(3 * t + 1.), np.cos(11 * t), np.sin(13 * t), np.cos(2 * t)]) / np.sqrt(6.)
    bar0 = barriers(arm)
    # margin m_i >= sup |J_i e_m|: |J_i e_m| <= |J_i|_2 |e_m|_2 <= |J_i|_2 em_bar
    Jn = 2.2                                                 # bound on |J_i|_2 for these rows on the workspace used (checked below)
    a1, a2 = bar0.a1, bar0.a2
    sigma = Jn * em_bar / (a1 * a2)
    bar = barriers(arm, sigma=sigma)
    ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=12.)
    L, res = run('mismatch', ctl, Q_A, np.zeros(6), 4.0, e_m=e_m)
    # unshifted barrier values
    res['h_unshifted_min'] = float(np.min(L['h'] + sigma[None, :]))
    res['Jnorm_max'] = float(max(np.linalg.norm(bar.hJ(q)[1], axis=1).max() for q in L['q'][::50]))
    print(f"    mismatch |e_m|_2 <= {em_bar}: smallest unshifted h {res['h_unshifted_min']:.3e}, smallest shifted psi1 {res['psi1_min']:.3e}, max |J_i|_2 {res['Jnorm_max']:.2f} (assumed {Jn})")


EXPERIMENTS = dict(checks=exp_checks, hold=exp_hold, hold_u0=exp_hold_initial_error, recovery=exp_recovery, free=exp_free, free_identity=exp_free_identity,
                   kinonly=exp_kinematic_only, nonuniform=exp_nonuniform, metric=exp_metric, feedforward=exp_feedforward, mismatch=exp_mismatch)

if __name__ == '__main__':
    names = sys.argv[1:] or list(EXPERIMENTS)
    for nm in names:
        EXPERIMENTS[nm]()
    try:
        old = json.load(open(NUM / 'toy_results.json'))
    except Exception:
        old = {}
    old.update(OUT)
    json.dump(old, open(NUM / 'toy_results.json', 'w'), indent=1)
