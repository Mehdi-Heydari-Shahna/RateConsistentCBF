"""Algebraic check of the comparison theorem for the kinetic-metric effort filter (random instances).

Setting at one instant.  M (d x d, s.p.d.), C (d x m, full row rank), filter metric H = C^T M^-1 C + R, R >= 0 on null(C).
Nominal efforts phi_r, admissible set K = {phi : rows, bounds}, filter Delta* = argmin |Delta|_H s.t. phi_r + Delta in K.
VDC with uniform rate k:  S_dot = -2 k S + <s, e_a - e_m>_M + P_f,   P_f = - s^T C Delta*.

Claims checked for every beta >= 0 (mu = min(k, beta)):
  (T)   P_f <= (k - mu) |s|_M^2 + |s|_M dist_H(phi^(beta), K),     phi^(beta) = phi_r - (k - beta) C^# M s,  C^# = H^-1 C^T M^-1
  (F)   rows only:  P_f <= (k - mu) |s_par|_M^2 + |s_par|_M d_A,   s_par = M-orthogonal projection of s on span{M^-1 J_i^T, i active},
        d_A = min{|d|_M : J_i d >= -Pi_i, i active},  Pi_i = J_i a^(beta) - r_i
  (E)   rows only, independent active gradients:  P_f = (k - beta) |s_par|^2 + w_A^T Lam_A^-1 Pi_A,   w = J s
"""
import json
import sys
import numpy as np

import paths                                  # noqa: E402  (locations of the release trees)

sys.path.insert(0, paths.RCX_HEAP)
sys.path.insert(0, str(paths.NUM))
from toy import ProjectionError, project      # noqa: E402


class _Res:
    pass


def solve_qp(H, g, A, b):
    """min 1/2 x^T H x + g^T x  s.t.  A x <= b,  by the exact projection of toy.project (KKT conditions to round-off)."""
    out = _Res()
    x0 = -np.linalg.solve(H, g)
    try:
        out.x, out.mu = project(H, x0, -np.asarray(A, float), -np.asarray(b, float), tol=1e-11)
        out.feasible = True
    except ProjectionError:
        out.x, out.mu, out.feasible = None, None, False
    return out

rng = np.random.default_rng(20261007)


def spd(n, cond=30.):
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    ev = np.exp(rng.uniform(0., np.log(cond), n))
    return (Q * ev) @ Q.T


def instance(d, m, R, bounds):
    M = spd(d)
    C = rng.standard_normal((d, m))
    Z = np.linalg.svd(C)[2][d:].T                              # null space of C (m x (m-d))
    H = C.T @ np.linalg.solve(M, C) + (rng.uniform(0.01, 2.) * (Z @ Z.T) if m > d else 0.)
    H = 0.5 * (H + H.T)
    s = rng.standard_normal(d) * rng.choice([0.1, 1., 5.])
    k = rng.uniform(0.5, 20.)
    J = rng.standard_normal((R, d))
    a_nom = rng.standard_normal(d) * 3.
    r = J @ a_nom + rng.standard_normal(R) * rng.choice([0.3, 3.])         # rows J a >= r; some violated by a_nom
    lo = hi = None
    if bounds:
        w = np.abs(rng.standard_normal(m)) * rng.choice([1., 4.]) + 0.3
        centre = rng.standard_normal(m) * 0.5
        lo, hi = centre - w, centre + w                           # bounds on Delta (phi_r + Delta in the box)
    return dict(M=M, C=C, H=H, s=s, k=k, J=J, a_nom=a_nom, r=r, lo=lo, hi=hi, d=d, m=m)


def constraints(p):
    """A Delta <= b for the rows (in Delta) and the bounds."""
    MiC = np.linalg.solve(p['M'], p['C'])
    A = [-(p['J'] @ MiC)]
    b = [-(p['r'] - p['J'] @ p['a_nom'])]
    if p['lo'] is not None:
        A += [np.eye(p['m']), -np.eye(p['m'])]
        b += [p['hi'], -p['lo']]
    return np.vstack(A), np.concatenate(b)


def check(p, betas):
    M, C, H, s, k = p['M'], p['C'], p['H'], p['s'], p['k']
    A, b = constraints(p)
    res = solve_qp(H, np.zeros(p['m']), A, b)
    if not res.feasible:
        return None
    D = res.x
    Pf = -float(s @ C @ D)
    sM = float(np.sqrt(s @ M @ s))
    Csharp = np.linalg.solve(H, C.T @ np.linalg.inv(M))
    out = dict(Pf=Pf, worst_T=-np.inf, worst_F=-np.inf, worst_E=0., active=int(np.sum(res.mu > 0.)), worst_F_well=-np.inf, worst_E_well=0., cond=1.,
               gram=False, bounds=p['lo'] is not None)
    R = len(p['r'])
    for beta in betas:
        mu = min(k, beta)
        Db = (beta - k) * (Csharp @ (M @ s))
        # distance of the beta-candidate from the admissible set in the filter metric
        rd = solve_qp(H, np.zeros(p['m']), A, b - A @ Db)
        if not rd.feasible:
            continue
        dist = float(np.sqrt(max(rd.x @ H @ rd.x, 0.)))
        out['worst_T'] = max(out['worst_T'], Pf - ((k - mu) * sM ** 2 + sM * dist))
        if p['lo'] is None:
            act = np.flatnonzero(res.mu[:R] > 0.)
            if len(act):
                Ja = p['J'][act]
                G = np.linalg.solve(M, Ja.T)                       # gradients (columns)
                Lam = Ja @ G
                # M-orthogonal projector on span of the active gradients
                U, sv, _ = np.linalg.svd(np.linalg.cholesky(M).T @ G, full_matrices=False)
                U = U[:, sv > 1e-9 * sv[0]]
                Lc = np.linalg.cholesky(M)
                s_hat = Lc.T @ s
                spar2 = float(np.sum((U.T @ s_hat) ** 2))
                a_beta = p['a_nom'] + (beta - k) * s
                Pi = Ja @ a_beta - p['r'][act]
                rq = solve_qp(M, np.zeros(p['d']), -Ja, Pi)        # min |d|_M  s.t.  J d >= -Pi
                dA = float(np.sqrt(max(rq.x @ M @ rq.x, 0.)))
                out['worst_F'] = max(out['worst_F'], Pf - ((k - mu) * spar2 + np.sqrt(spar2) * dA))
                # condition number of the Gram matrix of the active gradients after normalising them to unit length
                dn = np.sqrt(np.diag(Lam))
                cnd = np.linalg.cond(Lam / np.outer(dn, dn))
                out['cond'] = max(out['cond'], cnd)
                out['gram'] = True
                if cnd < 1e4:
                    out['worst_F_well'] = max(out['worst_F_well'], Pf - ((k - mu) * spar2 + np.sqrt(spar2) * dA))
                if cnd < 1e8:
                    w = Ja @ s
                    exact = (k - beta) * spar2 + float(w @ np.linalg.solve(Lam, Pi))
                    out['worst_E'] = max(out['worst_E'], abs(Pf - exact) / max(1., abs(Pf)))
                    if cnd < 1e4:
                        out['worst_E_well'] = max(out['worst_E_well'], abs(Pf - exact) / max(1., abs(Pf)))
            else:
                out['worst_F'] = max(out['worst_F'], Pf)           # no active row: P_f = 0
    return out


def main():
    n, n_feas = 0, 0
    worst = dict(T=-np.inf, F=-np.inf, E=0., Fw=-np.inf, Ew=0., cond=1., nwell=0)
    cnt = dict(nobounds=0, nobounds_active=0, bounds=0)
    tight = dict(T=-np.inf, F=-np.inf)
    acts = np.zeros(8, int)
    for trial in range(40000):
        d = int(rng.integers(1, 7))
        m = d + int(rng.integers(0, 4))
        R = int(rng.integers(1, 7))
        bounds = bool(rng.random() < 0.4)
        p = instance(d, m, R, bounds)
        betas = np.r_[rng.uniform(0., 30., 3), p['k'], 0.5 * p['k'], 2. * p['k']]
        out = check(p, betas)
        n += 1
        if out is None:
            continue
        # C is drawn at random; when it is nearly rank deficient the metric H = C^T M^-1 C is ill conditioned and
        # C H^-1 C^T = M holds only to cond(H) * round-off.  Such instances are counted and left out of the statistics.
        if np.linalg.cond(p['H']) > 1e6:
            worst['ill'] = worst.get('ill', 0) + 1
            worst['kin_ill'] = max(worst.get('kin_ill', 0.), float(np.max(np.abs(p['C'] @ np.linalg.solve(p['H'], p['C'].T) - p['M'])) / np.max(np.abs(p['M']))))
            worst['T_ill'] = max(worst.get('T_ill', -np.inf), out['worst_T'] / max(1., abs(out['Pf'])))
            worst['F_ill'] = max(worst.get('F_ill', -np.inf), out['worst_F'] / max(1., abs(out['Pf']))) if np.isfinite(out['worst_F']) else worst.get('F_ill', -np.inf)
            continue
        n_feas += 1
        acts[min(out['active'], 7)] += 1
        scale = max(1., abs(out['Pf']))
        worst['T'] = max(worst['T'], out['worst_T'] / scale)
        worst['F'] = max(worst['F'], out['worst_F'] / scale) if np.isfinite(out['worst_F']) else worst['F']
        worst['E'] = max(worst['E'], out['worst_E'])
        worst['Fw'] = max(worst['Fw'], out['worst_F_well'] / scale) if np.isfinite(out['worst_F_well']) else worst['Fw']
        worst['Ew'] = max(worst['Ew'], out['worst_E_well'])
        worst['cond'] = max(worst['cond'], out['cond'])
        worst['nwell'] += int(out['gram'] and out['cond'] < 1e4)
        cnt['bounds' if out['bounds'] else 'nobounds'] += 1
        cnt['nobounds_active'] += int(out['gram'])
    print(f"instances {n}, feasible with cond(H) <= 1e6: {n_feas}; active rows histogram {acts.tolist()};"
          f" left out for cond(H) > 1e6: {worst.get('ill', 0)} (there: (T) {worst.get('T_ill', 0.):.2e}, (F) {worst.get('F_ill', 0.):.2e};"
          f" largest relative residual of C H^-1 C^T = M {worst.get('kin_ill', 0.):.1e})")
    print(f"(T) largest  P_f - bound  (relative): {worst['T']:.3e}   (must be <= 0 up to round-off)")
    print(f"(F) largest  P_f - fine bound (relative): {worst['F']:.3e}")
    print(f"(E) largest relative error of the exact identity: {worst['E']:.3e}")
    print(f"    (F) and (E) are evaluated on the counted instances without effort bounds: {cnt['nobounds']} of {n_feas} ({cnt['nobounds_active']} of them with a binding row);"
          f" the {cnt['bounds']} counted instances with effort bounds enter (T) only")
    print(f"    instances without effort bounds and with binding rows whose normalised Gram matrix has a condition number below 1e4: {worst['nwell']}:  (F) {worst['Fw']:.3e}   (E) {worst['Ew']:.3e};"
          f"   largest condition number met {worst['cond']:.2e}")
    json.dump(dict(instances=n, counted_instances=n_feas, left_out_ill_conditioned=worst.get('ill', 0), rates_per_instance=6,
                   port_a_largest_relative_excess=worst['T'], fine_bound_largest_relative_excess=worst['F'], identity_largest_relative_error=worst['E'],
                   counts=cnt, active_rows_histogram=acts.tolist()), open(paths.NUM / 't1_results.json', 'w'), indent=1, default=float)
    return 0 if worst['T'] < 1e-7 and worst['F'] < 1e-7 and worst['E'] < 1e-6 else 1


if __name__ == '__main__':
    raise SystemExit(main())
