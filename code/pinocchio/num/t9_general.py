"""The general forms: row identity and main inequality without a safe start, with a mismatch, with a standoff that
varies in time, with an additional row margin, and with two estimates of the load.

Part 1 (random instances; inertia and Coriolis matrices of the 6R arm at random states, everything else random)
   plant            M u' + N u + g = C f + tau_e,                 C (d x m), m >= d, kinetic metric H = C^T M^-1 C + R
   VDC law          C phi_r = M a_r + N u_r + g - tau_c + k M s     (compensation with the estimate tau_c)
   rows             J_i a_hat(phi') + Jd_i u + (a1_i + a2_i) J_i u + a1_i a2_i hs_i >= varpi_i,
                    a_hat(phi') = M^-1 (C phi' + tau_w - N u - g)   (rows with the estimate tau_w)
   filter           phi = H-nearest point to phi_r that satisfies the rows;   delivered efforts f = phi + df
   Checked with  e_l = M^-1 (tau_w - tau_c),  e_a = ur_dot - a_r - e_l,  e_m = M^-1 (C df + tau_e - tau_w):
   (B)  S' computed from the plant  =  -k |s|^2 + <s, e_a - e_m>_M + P_f,          P_f = -s^T C (phi - phi_r)
   (R)  Pi_i^(beta) = zs_i' + beta zs_i + (a2_i - beta) psi1s_i - c_i - J_i e_a + a1_i sigma_i' - varpi_i     (all rows)
   (P)  P_f = sum_i lam_i (psi1s_i - zs_i)                                         (multipliers lam_i >= 0 of the rows)
   (M)  S' <= -k |s_perp|^2 - mu |s_par|^2 + |s_par| d_A[zeta + eta + c + varpi - a1 sigma'] + |s| |e_a| - <s, e_m>_M
        for the carrier A = support of lam, beta <= min_A a2_i, mu = min(k, beta),
        zeta_i = [zs_i' + beta zs_i]^-,  eta_i = (a2_i - beta) [psi1s_i]^-.
   The required velocity is arbitrary here (not the output of a kinematic layer) and the state is not in the safe set
   in general: neither is needed for (M).

Part 2 (6R arm, simulation from a state outside the safe set, exact model)
   [psi1s_i(t)]^- <= e^{-a2_i t} [psi1s_i(0)]^-,  inequality (M) at every logged sample, convergence.
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np      # noqa: E402

import paths            # noqa: E402
from ldp import Infeasible, min_norm, project      # noqa: E402

rng = np.random.default_rng(20261008)
OUT = {}


def part1(n_inst=20000):
    from toy import Arm
    from run_toy import Q_A
    arm = Arm()
    d = arm.n
    worst = dict(B=0., R=0., P=0., M=-np.inf, feas=0.)
    cnt = dict(inst=0, binding=0, unsafe=0, evals=0, tight=0, rows_binding=0, infeasible=0)
    while cnt['inst'] < n_inst:
        q = Q_A + rng.uniform(-0.6, 0.6, d)
        u = rng.uniform(-1.5, 1.5, d)
        M, N, g = arm.dyn(q, u)
        Mi = np.linalg.inv(M)
        m = d + int(rng.integers(0, 3))
        C = rng.standard_normal((d, m))
        if np.linalg.cond(C @ C.T) > 1e3:
            continue
        H = C.T @ Mi @ C
        if m > d:
            Z = np.linalg.svd(C)[2][d:].T
            Q = np.linalg.qr(rng.standard_normal((m - d, m - d)))[0]
            H = H + Z @ (Q * rng.uniform(0.05, 2., m - d)) @ Q.T @ Z.T * np.trace(H) / d
        Csh = np.linalg.solve(H, C.T @ Mi)                                # C^# = H^-1 C^T M^-1
        R = int(rng.integers(1, 5))
        J = rng.standard_normal((R, d))
        Jd = rng.standard_normal((R, d)) * rng.uniform(0., 2.)
        a1, a2 = rng.uniform(1., 10., R), rng.uniform(2., 25., R)
        k = rng.uniform(4., 30.)
        hs = rng.uniform(-0.05, 0.4, R)
        sig_dot = rng.uniform(-0.3, 0.3, R) * (rng.random() < 0.7)
        varpi = rng.uniform(0., 0.5, R) * (rng.random() < 0.5)
        u_r = u + rng.standard_normal(d) * rng.uniform(0.05, 1.)
        ur_dot = rng.standard_normal(d) * 3.
        a_r = ur_dot - rng.standard_normal(d) * rng.uniform(0., 1.) * (rng.random() < 0.7)
        tau_e = rng.standard_normal(d) * 20.
        tau_w = tau_e + rng.standard_normal(d) * 5. * (rng.random() < 0.7)
        tau_c = tau_w + rng.standard_normal(d) * 5. * (rng.random() < 0.7)
        df = rng.standard_normal(m) * 3. * (rng.random() < 0.7)
        s = u_r - u
        # controller
        phi_r = Csh @ (M @ a_r + N @ u_r + g - tau_c + k * (M @ s))
        if m > d:
            phi_r = phi_r + Z @ rng.standard_normal(m - d)                # any solution of the VDC equation
        G = J @ Mi @ C
        r = -(J @ Mi @ (tau_w - N @ u - g) + Jd @ u + (a1 + a2) * (J @ u) + a1 * a2 * hs) + varpi
        try:
            phi, lam, info = project(H, phi_r, G, r)
        except Infeasible:
            cnt['infeasible'] += 1
            continue
        cnt['inst'] += 1
        worst['feas'] = max(worst['feas'], info.get('feas', 0.), info.get('stat', 0.))
        Delta = phi - phi_r
        f = phi + df
        # plant
        u_dot = Mi @ (C @ f + tau_e - N @ u - g)
        s_dot = ur_dot - u_dot
        S_dot = float(s @ M @ s_dot + s @ N @ s)                          # M' = N + N^T
        e_l = Mi @ (tau_w - tau_c)
        e_a = ur_dot - a_r - e_l
        e_m = Mi @ (C @ df + tau_e - tau_w)
        Pf = -float(s @ C @ Delta)
        n2 = float(s @ M @ s)
        bal = -k * n2 + float(s @ M @ (e_a - e_m)) + Pf
        worst['B'] = max(worst['B'], abs(S_dot - bal) / (1. + abs(S_dot)))
        # row quantities
        w = J @ s
        zs = J @ u_r + a1 * hs
        zs_dot = J @ ur_dot + Jd @ u_r + a1 * (J @ u) - a1 * sig_dot
        psi1 = J @ u + a1 * hs
        c = Jd @ s - J @ Mi @ (N @ s)
        cnt['unsafe'] += int(np.any(psi1 < 0.) or np.any(hs < 0.))
        A = np.flatnonzero(lam > 1e-12 * max(1., lam.max()))
        cnt['binding'] += int(len(A) > 0)
        cnt['rows_binding'] += len(A)
        worst['P'] = max(worst['P'], abs(Pf - float(lam @ (psi1 - zs))) / (1. + abs(Pf)))
        s_hat = Csh @ (M @ s)
        for beta in ((np.min(a2[A]) if len(A) else k), 0.5 * (np.min(a2[A]) if len(A) else k), -2.):
            # (R) at the comparison efforts of rate beta
            phi_b = phi_r - (k - beta) * s_hat
            Pi = G @ phi_b - r
            star = zs_dot + beta * zs + (a2 - beta) * psi1 - c - J @ e_a + a1 * sig_dot - varpi
            worst['R'] = max(worst['R'], float(np.max(np.abs(Pi - star) / (1. + np.abs(Pi)))))
            # (M)
            mu = min(k, beta)
            if len(A):
                JA = J[A]
                Lam = JA @ Mi @ JA.T
                s_par = Mi @ JA.T @ np.linalg.solve(Lam, w[A])
                p2 = float(s_par @ M @ s_par)
                zeta = np.maximum(0., -(zs_dot + beta * zs))
                eta = (a2 - beta) * np.maximum(0., -psi1)
                arg = (zeta + eta + c + varpi - a1 * sig_dot)[A]
                dd, _, _ = min_norm(M, JA, arg)
                dA = float(np.sqrt(max(dd @ M @ dd, 0.)))
            else:
                p2, dA = 0., 0.
            bound = -k * (n2 - p2) - mu * p2 + np.sqrt(p2) * dA + np.sqrt(n2 * float(e_a @ M @ e_a)) - float(s @ M @ e_m)
            gap = (S_dot - bound) / (1. + abs(S_dot))
            worst['M'] = max(worst['M'], gap)
            cnt['tight'] += int(len(A) > 0 and abs(S_dot - bound) < 1e-6 * (1. + abs(S_dot)))
            cnt['evals'] += 1
    print(f"Part 1: {cnt['inst']} instances ({cnt['infeasible']} infeasible ones skipped); a row binding in {cnt['binding']} ({cnt['rows_binding']} binding rows); "
          f"state outside the safe set in {cnt['unsafe']}")
    print(f"   (B) balance from the plant against the formula, largest relative difference {worst['B']:.1e}")
    print(f"   (R) row identity with standoff rate, margin and two load estimates, largest relative difference {worst['R']:.1e}")
    print(f"   (P) P_f = sum lam_i (psi1s_i - zs_i), largest relative difference {worst['P']:.1e}")
    print(f"   (M) largest (S' - bound) / (1 + |S'|) over {cnt['evals']} evaluations: {worst['M']:+.1e}   (equality to 1e-6 in {cnt['tight']} evaluations; "
          f"residual of the conditions of the projections {worst['feas']:.1e})")
    OUT['part1'] = dict(counts=cnt, worst=worst)


def part2():
    from toy import Arm, Controller, least_norm, ramp_reference, simulate
    from run_toy import Q_A, Q_B, barriers
    arm = Arm()
    res = []
    print('Part 2: 6R arm; the hold of run_toy.py (reference leaves the safe set) is disturbed at t = 1.5 s by a velocity jump that makes the')
    print('        first-order functions of the held rows negative (state outside the safe set); exact model afterwards')
    for tag, k, delta in (('k12', 12., 0.3), ('k30', 30., 0.3), ('k12_large', 12., 1.0)):
        bar = barriers(arm)
        ctl = Controller(arm, bar, ramp_reference(Q_A, Q_B, 1.2), lam=12., k=k)
        L0 = simulate(ctl, Q_A, np.zeros(6), 1.5, dt=1e-3)
        q1, u1 = L0['q'][-1], L0['u'][-1]
        _, _, o = ctl(1.5, q1, u1, log=True)
        B = o['act_k']
        JB = o['J'][B]
        Mi = np.linalg.inv(o['M'])
        du = Mi @ JB.T @ np.linalg.solve(JB @ Mi @ JB.T, -delta - o['psi1'][B])
        u1 = u1 + du
        # continue from the disturbed state (time shifted: the reference is constant after 1.2 s)
        ctl2 = Controller(arm, bar, lambda t: ctl.ref(t + 1.5), lam=12., k=k)
        L = simulate(ctl2, q1, u1, 2.5, dt=1e-3)
        t, psi1, h = L['t'], L['psi1'], L['h']
        neg0 = np.maximum(0., -psi1[0])
        env = np.exp(-bar.a2[None, :] * t[:, None]) * neg0[None, :]
        sign_viol = float(np.max(np.maximum(0., -psi1) - env))
        worst = {'a2min': -np.inf, 'half': -np.inf, 'zero': -np.inf}
        nosign = -np.inf
        n_bind, eta_used, eta_max = 0, 0, 0.
        for i in range(0, len(t), 2):
            if L['S'][i] < 1e-12 * L['S'][0]:                      # below this the comparison is one of rounding errors
                break
            _, _, o = ctl2(t[i], L['q'][i], L['u'][i], log=True)
            A = o['act']
            if not A.any():
                continue
            n_bind += 1
            M, sv, J = o['M'], o['s'], o['J']
            Mi = np.linalg.inv(M)
            JA = J[A]
            Lam = JA @ Mi @ JA.T
            s_par = Mi @ JA.T @ np.linalg.solve(Lam, JA @ sv)
            p2, n2 = float(s_par @ M @ s_par), float(sv @ M @ sv)
            S_dot = -k * n2 + o['Pf']
            a2min = float(np.min(bar.a2[A]))
            eta_used += int(np.any(o['psi1'][A] < 0.))
            for name, beta in (('a2min', a2min), ('half', 0.5 * a2min), ('zero', 0.)):
                mu = min(k, beta)
                zeta = np.maximum(0., -(o['zs_dot'] + beta * o['zs']))
                eta = (bar.a2 - beta) * np.maximum(0., -o['psi1'])
                dA = least_norm(M, JA, (zeta + eta + o['c'])[A])
                bound = -k * (n2 - p2) - mu * p2 + np.sqrt(p2) * dA
                worst[name] = max(worst[name], (S_dot - bound) / max(abs(S_dot), abs(bound), 1e-300))
                if name == 'half':
                    eta_max = max(eta_max, float(np.max(eta[A])))
                    dA0 = least_norm(M, JA, (zeta + o['c'])[A])           # the bound without the sign term (not claimed outside the safe set)
                    b0 = -k * (n2 - p2) - mu * p2 + np.sqrt(p2) * dA0
                    nosign = max(nosign, (S_dot - b0) / max(abs(S_dot), abs(b0), 1e-300))
        last_neg = np.flatnonzero(psi1.min(axis=1) < -1e-9)
        one = dict(run=tag, k=k, delta=delta, rows_disturbed=[bar.names[j] for j in np.flatnonzero(B)], psi1_0_min=float(psi1[0].min()), h_min=float(h.min()),
                   sign_bound_viol=sign_viol, main_viol=worst, without_sign_term=float(nosign), samples_binding=n_bind,
                   samples_binding_negative_psi1=eta_used, eta_max=eta_max, S0=float(L['S'][0]), S_max=float(L['S'].max()), S_end=float(L['S'][-1]),
                   t_unsafe_end=float(t[last_neg[-1]]) if len(last_neg) else 0.)
        rows_B = np.flatnonzero(B)
        unit = ['m' if bar.names[j] == 'plane' else 'rad' for j in rows_B]            # a plane row is a distance, a limit row an angle
        one['rows_units'] = unit
        one['h_min_rows'] = [float(h[:, j].min()) for j in rows_B]
        one['h_min_row'] = bar.names[int(np.unravel_index(np.argmin(h), h.shape)[1])]
        res.append(one)
        print(f"   {tag}: rows {one['rows_disturbed']} set to psi1s = -{delta:g} ({', '.join(un + '/s' for un in unit)}); smallest shifted barrier values afterwards: "
              + ', '.join(f"{bar.names[j]} {v:+.2e} {un}" for j, v, un in zip(rows_B, one['h_min_rows'], unit))
              + f" (smallest over all rows {one['h_min']:+.2e}, row {one['h_min_row']}); "
              f"[psi1s]^- minus its bound e^(-alpha2 t)[psi1s(0)]^-: at most {sign_viol:+.1e}; psi1s < -1e-9 until t = {one['t_unsafe_end']:.3f} s")
        print(f"        (M) at {n_bind} samples with a binding row ({eta_used} with a negative psi1s on a binding row): largest (S' - bound)/max(|S'|, |bound|) for beta = min alpha2: "
              f"{worst['a2min']:+.1e}, half of it: {worst['half']:+.1e}, beta = 0: {worst['zero']:+.1e};  without the term eta (beta = half): {nosign:+.1e};  "
              f"S: {one['S0']:.3e} -> {one['S_end']:.1e}")
    OUT['part2'] = res


def part3(n_inst=20000):
    """Is the sign term eta needed when the required velocity satisfies the kinematic rows (zs >= 0)?  Instances with a
    binding row whose first-order function is negative; exact model, exact feedforward.  For one carrying row the term
    is not needed (then w_i > 0 and P_f = -lam_i w_i <= 0); with several rows it is."""
    from toy import Arm
    from run_toy import Q_A
    arm = Arm()
    d = arm.n
    gen = np.random.default_rng(5)
    n, viol, viol_one, n_one, worst0, worst1, example = 0, 0, 0, 0, -np.inf, -np.inf, None
    while n < n_inst:
        q = Q_A + gen.uniform(-0.6, 0.6, d)
        u = gen.uniform(-1.5, 1.5, d)
        M, N, _ = arm.dyn(q, u)
        Mi = np.linalg.inv(M)
        R = int(gen.integers(1, 5))
        J = gen.standard_normal((R, d))
        Jd = gen.standard_normal((R, d)) * gen.uniform(0., 1.)
        a1, a2 = gen.uniform(1., 10., R), gen.uniform(2., 25., R)
        k = gen.uniform(4., 30.)
        hs = gen.uniform(-0.05, 0.3, R)
        try:
            u_r, _, _ = project(np.eye(d), u + gen.standard_normal(d) * gen.uniform(0.05, 1.), J, -a1 * hs)      # zs >= 0
        except Infeasible:
            continue
        ur_dot = gen.standard_normal(d) * 3.
        s = u_r - u
        a_nom = ur_dot + Mi @ (N @ s) + k * s
        rhs = -(Jd @ u + (a1 + a2) * (J @ u) + a1 * a2 * hs)
        try:
            a, lam, _ = project(M, a_nom, J, rhs)
        except Infeasible:
            continue
        A = np.flatnonzero(lam > 1e-12 * max(1., lam.max()))
        psi1 = J @ u + a1 * hs
        if len(A) == 0 or not np.any(psi1[A] < 0.):
            continue
        n += 1
        n_one += int(len(A) == 1)
        zs = J @ u_r + a1 * hs
        zs_dot = J @ ur_dot + Jd @ u_r + a1 * (J @ u)
        c = Jd @ s - J @ Mi @ (N @ s)
        w = J @ s
        n2 = float(s @ M @ s)
        S_dot = -k * n2 - float(s @ M @ (a - a_nom))
        JA = J[A]
        s_par = Mi @ JA.T @ np.linalg.solve(JA @ Mi @ JA.T, w[A])
        p2 = float(s_par @ M @ s_par)
        for beta in (np.min(a2[A]), 0.5 * np.min(a2[A]), 0.):
            mu = min(k, beta)
            zeta = np.maximum(0., -(zs_dot + beta * zs))
            eta = (a2 - beta) * np.maximum(0., -psi1)
            gaps = []
            for arg in ((zeta + c)[A], (zeta + eta + c)[A]):
                dd, _, _ = min_norm(M, JA, arg)
                b = -k * (n2 - p2) - mu * p2 + np.sqrt(p2 * max(dd @ M @ dd, 0.))
                gaps.append((S_dot - b) / max(abs(S_dot), abs(b), 1e-300))
            worst0, worst1 = max(worst0, gaps[0]), max(worst1, gaps[1])
            if gaps[0] > 1e-9:
                viol += 1
                viol_one += int(len(A) == 1)
                if example is None:
                    example = dict(rows=int(len(A)), psi1s=psi1[A].tolist(), zs=zs[A].tolist(), alpha2=a2[A].tolist(), beta=float(beta), k=float(k), gap=float(gaps[0]))
    print(f"Part 3: {n} instances with zs >= 0 and a binding row with psi1s < 0 ({n_one} with one binding row), three rates each:")
    print(f"   without the term eta the inequality (M) fails in {viol} evaluations ({viol_one} of them with one binding row), largest relative excess {worst0:+.2e};"
          f" with the term: largest (S' - bound)/max(|S'|,|bound|) = {worst1:+.1e}")
    if example:
        print(f"   first failure: {example}")
    OUT['part3'] = dict(instances=n, one_row=n_one, failures_without_eta=viol, failures_one_row=viol_one, worst_without=worst0, worst_with=worst1, example=example)


def part4():
    """The example of the note for the sign defect: d = 2, M = C = H = I, k = 12, two held rows without curvature,
    J_1 = (1, 0), J_2 = (-0.8, 0.6), alpha2 = (4, 20), s = (-2, -1), exact feedforward."""
    k = 12.
    J = np.array([[1., 0.], [-0.8, 0.6]])
    a2 = np.array([4., 20.])
    s = np.array([-2., -1.])
    w = J @ s
    psi1 = -w                                                 # held rows
    Pi_k = (a2 - k) * psi1                                    # row values of the VDC law
    d, lam, _ = min_norm(np.eye(2), J, -Pi_k)                 # correction of the acceleration
    Pf = -float(s @ d)
    beta = float(a2.min())
    eta = (a2 - beta) * np.maximum(0., -psi1)
    d_eta, _, _ = min_norm(np.eye(2), J, eta)
    n2 = float(s @ s)
    print(f"Part 4: w = {w.tolist()}, psi1s = {psi1.tolist()}, row values of the VDC law {Pi_k.tolist()}, multipliers {np.round(lam, 2).tolist()}, correction {np.round(d, 2).tolist()}, "
          f"P_f = {Pf:.2f};  (k - beta)|s|^2 = {(k - beta) * n2:.1f};  eta = {eta.tolist()}, d_A[eta] = {np.linalg.norm(d_eta):.2f}, bound with the sign defect "
          f"{(k - beta) * n2 + np.sqrt(n2) * np.linalg.norm(d_eta):.1f}")
    OUT['part4'] = dict(Pf=Pf, bound_without=(k - beta) * n2, bound_with=(k - beta) * n2 + float(np.sqrt(n2) * np.linalg.norm(d_eta)))


if __name__ == '__main__':
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    parts = sys.argv[2] if len(sys.argv) > 2 else '1234'
    try:
        OUT.update(json.load(open(paths.NUM / 't9_results.json')))
    except Exception:
        pass
    if '1' in parts:
        part1(n_inst)
    if '2' in parts:
        part2()
    if '3' in parts:
        part3(n_inst)
    if '4' in parts:
        part4()
    json.dump(OUT, open(paths.NUM / 't9_results.json', 'w'), indent=1, default=float)
