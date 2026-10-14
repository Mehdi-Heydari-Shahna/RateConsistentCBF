"""The projection lemma and the port inequality for carriers of the correction (constructed instances).

Instances are constructed so that the projection, its binding rows and one multiplier vector are known exactly, which
allows degenerate cases: linearly dependent binding gradients, positively dependent gradients, binding rows with zero
multiplier, more binding rows than dimensions, effort bounds next to barrier rows.

  M (d x d) s.p.d., C (d x m) of full row rank, kinetic metric H = C^T M^-1 C + R (R >= 0 on the null space of C);
  rows G phi' >= r:  barrier rows G_i = J_i M^-1 C (some covectors J_i are combinations of others) and effort bounds;
  phi (the filtered efforts), binding rows A_b, multipliers mu >= 0 on A_b (several of them zero),
  phi_r := phi - H^-1 G_Ab^T mu,   r_i := G_i phi on A_b,   r_i < G_i phi elsewhere      (KKT: phi is the projection of phi_r).

Checked for every carrier A (the binding set A_b, the support of mu, an independent carrier obtained from mu by the
Caratheodory reduction) and several rates beta, with s_hat = C^# M s, phi^(beta) = phi_r - (k - beta) s_hat:

  (F)  P_f <= (k - min(k, beta)) |s_hat_A|_H^2 + |s_hat_A|_H d_A,    d_A = min{|delta|_H : G_i delta >= -Pi_i, i in A},
  (E)  for an independent carrier of barrier rows:  P_f = (k - beta) |s_par|_M^2 + w_A^T Lam_A^-1 Pi_A,
  (N)  for a subset of the binding rows that is not a carrier the inequality (F) can fail.  Tested subset: the support of the
       multipliers without its last row; with dependent gradients it can still be a carrier, which is tested and counted.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np      # noqa: E402
from scipy.optimize import nnls      # noqa: E402

from ldp import Infeasible, independent_carrier, min_norm      # noqa: E402

rng = np.random.default_rng(20261007)


def spd(n, lo=1., hi=30.):
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    return (Q * rng.uniform(lo, hi, n)) @ Q.T


def proj_on_span(H, V, x):
    """H-orthogonal projection of x on the span of the columns of V."""
    if V.shape[1] == 0:
        return np.zeros_like(x)
    Lc = np.linalg.cholesky(H)
    W = Lc.T @ V
    U, sv, _ = np.linalg.svd(W, full_matrices=False)
    U = U[:, sv > 1e-10 * sv[0]]
    return np.linalg.solve(Lc.T, U @ (U.T @ (Lc.T @ x)))


def instance():
    d = int(rng.integers(2, 6))
    m = d + int(rng.integers(0, 3))
    M = spd(d)
    C = rng.standard_normal((d, m))
    if np.linalg.cond(C @ C.T) > 1e4:
        return None
    W = C.T @ np.linalg.solve(M, C)
    H = W.copy()
    if m > d:
        Z = np.linalg.svd(C)[2][d:].T
        H = H + Z @ spd(m - d, 0.05, 2.) @ Z.T * np.trace(W) / d
    # barrier covectors with planted dependence
    n0 = int(rng.integers(1, d + 1))
    J0 = rng.standard_normal((n0, d))
    extra = int(rng.integers(0, 4))
    rows = [J0]
    for _ in range(extra):
        kind = rng.integers(0, 3)
        if kind == 0:                                       # duplicate direction
            rows.append(J0[[rng.integers(0, n0)]] * rng.uniform(0.3, 3.))
        elif kind == 1:                                     # positive combination
            rows.append((rng.uniform(0., 1., n0) @ J0)[None, :])
        else:                                               # opposite direction (positively dependent pair)
            rows.append(-J0[[rng.integers(0, n0)]] * rng.uniform(0.3, 3.))
    J = np.vstack(rows)
    Rb = len(J)
    G = [J @ np.linalg.solve(M, C)]
    nb = int(rng.integers(0, 3)) if rng.random() < 0.35 else 0     # effort bounds as further rows
    for _ in range(nb):
        e = np.zeros(m)
        e[rng.integers(0, m)] = rng.choice([-1., 1.])
        G.append(e[None, :])
    G = np.vstack(G)
    R = len(G)
    phi = rng.standard_normal(m) * 3.
    # binding rows: a random subset that contains no pair of exactly opposite rows binding with positive multipliers only
    Ab = np.flatnonzero(rng.random(R) < 0.7)
    if len(Ab) == 0:
        return None
    mu = np.zeros(R)
    carry = Ab[rng.random(len(Ab)) < 0.7]
    mu[carry] = rng.uniform(0.2, 3., len(carry))
    Delta = np.linalg.solve(H, G[Ab].T @ mu[Ab])
    phi_r = phi - Delta
    r = G @ phi - rng.uniform(0.1, 1., R)
    r[Ab] = G[Ab] @ phi
    s = rng.standard_normal(d)
    k = rng.uniform(5., 20.)
    return dict(d=d, m=m, M=M, C=C, H=H, J=J, G=G, Rb=Rb, phi=phi, phi_r=phi_r, r=r, Ab=Ab, mu=mu, s=s, k=k, Delta=Delta)


def main(n_inst=20000):
    worst = dict(F_Ab=-np.inf, F_supp=-np.inf, F_ind=-np.inf, E=0., E_cond=1., E_wc=0., ldp=0.)
    cnt = dict(inst=0, dep=0, zero_mu=0, bounds=0, evals=0, E=0, noncarrier_fail=0, noncarrier=0, dAb_inf=0, dind_fin=0, tests_fin=0, reduced=0, inf_F_Ab=0, inf_F_supp=0,
               sub=0, sub_carrier=0, sub_carrier_fail=0)
    for _ in range(n_inst):
        I = instance()
        if I is None:
            continue
        M, C, H, G, r, s, k = I['M'], I['C'], I['H'], I['G'], I['r'], I['s'], I['k']
        Ab, mu, Rb = I['Ab'], I['mu'], I['Rb']
        cnt['inst'] += 1
        Hi = np.linalg.inv(H)
        s_hat = Hi @ (C.T @ s)                                 # C^# M s
        Pf = -float(s @ C @ I['Delta'])
        grads = Hi @ G.T                                       # columns: H-gradients of the rows
        sv = np.linalg.svd(G[Ab], compute_uv=False)
        dependent = len(Ab) > G.shape[1] or sv[-1] < 1e-9 * sv[0]
        cnt['dep'] += int(dependent)
        cnt['zero_mu'] += int(np.any(mu[Ab] == 0.))
        cnt['bounds'] += int(np.any(Ab >= Rb))
        mu_ind = independent_carrier(H, G, mu)
        assert np.max(np.abs(Hi @ (G.T @ mu_ind) - I['Delta'])) < 1e-8 * (1. + np.max(np.abs(I['Delta'])))
        A_ind = np.flatnonzero(mu_ind > 0.)
        cnt['reduced'] += int(len(A_ind) < np.sum(mu > 0.))
        carriers = dict(F_Ab=Ab, F_supp=np.flatnonzero(mu > 0.), F_ind=A_ind)
        for beta in (-3., 0., 0.4 * k, k, 1.5 * k):
            mu_r = min(k, beta)
            phi_b = I['phi_r'] - (k - beta) * s_hat
            Pi = G @ phi_b - r
            for key, A in carriers.items():
                if len(A) == 0:
                    bound = 0.
                else:
                    sA = proj_on_span(H, grads[:, A], s_hat)
                    nsA = float(np.sqrt(max(sA @ H @ sA, 0.)))
                    try:
                        delta, _, info = min_norm(H, G[A], -Pi[A])
                        dA = float(np.sqrt(max(delta @ H @ delta, 0.)))
                        worst['ldp'] = max(worst['ldp'], info.get('feas', 0.), info.get('stat', 0.), info.get('comp', 0.))
                    except Infeasible:                         # the defect of this carrier is +infinity: nothing is claimed
                        assert key != 'F_ind', 'an independent carrier always has a finite defect'
                        cnt['inf_' + key] += 1
                        cnt['evals'] += 1
                        continue
                    bound = (k - mu_r) * nsA ** 2 + nsA * dA
                worst[key] = max(worst[key], (Pf - bound) / max(1., abs(Pf)))
                cnt['evals'] += 1
            # (E) identity on an independent carrier of barrier rows
            if len(A_ind) and np.all(A_ind < Rb):
                JA = I['J'][A_ind]
                gA = np.linalg.solve(M, JA.T)
                Lam = JA @ gA
                w = JA @ s
                s_par = gA @ np.linalg.solve(Lam, w)
                rhs = (k - beta) * float(s_par @ M @ s_par) + float(w @ np.linalg.solve(Lam, Pi[A_ind]))
                err = abs(Pf - rhs) / max(1., abs(Pf))
                if err > worst['E']:
                    worst['E'], worst['E_cond'] = err, float(np.linalg.cond(Lam))
                if np.linalg.cond(Lam) < 1e4:
                    worst['E_wc'] = max(worst['E_wc'], err)
                cnt['E'] += 1
        # finiteness of the defect functional for an arbitrary right-hand side: full binding set against independent carrier
        barr = Ab[Ab < Rb]
        if len(barr) and len(A_ind) and np.all(A_ind < Rb):
            rr = rng.uniform(0.1, 1., len(I['J']))
            cnt['tests_fin'] += 1
            try:
                min_norm(M, I['J'][barr], rr[barr])
            except Infeasible:
                cnt['dAb_inf'] += 1
            try:
                min_norm(M, I['J'][A_ind], rr[A_ind])
                cnt['dind_fin'] += 1
            except Infeasible:
                pass
        # (N) a subset of the binding rows that does not carry the correction
        supp = np.flatnonzero(mu > 0.)
        if len(supp) >= 2:
            A = supp[:-1]
            beta = 0.4 * k
            phi_b = I['phi_r'] - (k - beta) * s_hat
            Pi = G @ phi_b - r
            sA = proj_on_span(H, grads[:, A], s_hat)
            nsA = float(np.sqrt(max(sA @ H @ sA, 0.)))
            try:
                delta, _, _ = min_norm(H, G[A], -Pi[A])
                bound = (k - beta) * nsA ** 2 + nsA * float(np.sqrt(max(delta @ H @ delta, 0.)))
                fail = int((Pf - bound) / max(1., abs(Pf)) > 1e-8)
                # with dependent gradients the support without its last row can still carry the correction: test whether
                # H Delta = G_supp^T mu_supp is a non-negative combination of the gradients of A (no random numbers are drawn)
                target = G[supp].T @ mu[supp]
                _, res_nn = nnls(G[A].T, target)
                still = int(res_nn <= 1e-9 * (1. + np.linalg.norm(target)))
                cnt['sub'] += 1
                cnt['sub_carrier'] += still
                cnt['sub_carrier_fail'] += still * fail
                cnt['noncarrier'] += 1 - still
                cnt['noncarrier_fail'] += (1 - still) * fail
            except Infeasible:
                pass
    print(f"instances {cnt['inst']}: binding gradients linearly dependent in {cnt['dep']}, a binding row with zero multiplier in {cnt['zero_mu']}, "
          f"an effort bound binding in {cnt['bounds']}; the Caratheodory reduction removed rows in {cnt['reduced']}")
    print(f"(F) largest (P_f - bound) / max(1, |P_f|) over {cnt['evals']} evaluations:  full binding set {worst['F_Ab']:+.2e},  support of the multiplier "
          f"{worst['F_supp']:+.2e},  independent carrier {worst['F_ind']:+.2e}")
    print(f"    defect infinite (nothing claimed): full binding set {cnt['inf_F_Ab']}, support of the multiplier {cnt['inf_F_supp']}, independent carrier 0; "
          f"largest residual of the conditions of the least-distance problems {worst['ldp']:.1e}")
    print(f"(E) identity on an independent carrier of barrier rows: {cnt['E']} evaluations, largest relative error {worst['E']:.2e} "
          f"(condition number of Lam_A there {worst['E_cond']:.1e}; {worst['E_wc']:.2e} over the instances with condition number below 1e4)")
    print(f"    defect functional d_A[r] for a positive right-hand side r: infinite for the full binding set in {cnt['dAb_inf']} of {cnt['tests_fin']} instances, "
          f"finite for the independent carrier in {cnt['dind_fin']} of {cnt['tests_fin']}")
    print(f"(N) support of the multipliers with its last row removed, {cnt['sub']} instances: it does not carry the correction in {cnt['noncarrier']}, and inequality (F) is "
          f"violated in {cnt['noncarrier_fail']} of these ({100. * cnt['noncarrier_fail'] / max(cnt['noncarrier'], 1):.1f} %); it still carries the correction in {cnt['sub_carrier']} "
          f"(dependent gradients), and (F) is violated in {cnt['sub_carrier_fail']} of these")
    json.dump(dict(counts=cnt, worst=worst, port_b_largest_relative_excess=max(worst['F_Ab'], worst['F_supp'], worst['F_ind'])),
              open(Path(__file__).with_name('t7_results.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
