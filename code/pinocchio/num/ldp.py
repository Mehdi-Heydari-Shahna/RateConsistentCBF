"""Least-distance problems with inequality rows, independent of the solver of the releases.

    project(H, x0, G, r)   argmin 1/2 |x - x0|_H^2   s.t.  G x >= r        (H symmetric positive definite)
    min_norm(H, G, r)      argmin |delta|_H           s.t.  G delta >= r

Method: Lawson-Hanson least-distance programming.  With H = L L^T and y = L^T (x - x0) the problem is
min |y|_2 s.t. E y >= f,  E = G L^-T,  f = r - G x0.  The non-negative least-squares problem
min_{u >= 0} | [E^T; f^T] u - e_{n+1} |  has the residual rho; the problem is infeasible iff rho = 0, and otherwise
y = -rho[:n] / rho[n] with the multipliers mu = u / (-rho[n]) >= 0.  The rows with mu > 0 bind and have linearly
independent gradients (property of the NNLS active set), so the support of mu is an independent carrier of the
correction.  The point is recomputed from the equality system of that support, and the conditions

    G x >= r,   mu >= 0,   mu_i (G_i x - r_i) = 0,   H (x - x0) = G^T mu

are returned as residuals.
"""
import numpy as np
from scipy.optimize import nnls


class Infeasible(RuntimeError):
    pass


def min_norm(H, G, r, tol=1e-11):
    """Returns delta, mu (multipliers, support = an independent carrier), info."""
    G = np.atleast_2d(np.asarray(G, float))
    r = np.asarray(r, float).ravel()
    n = H.shape[0]
    if G.shape[0] == 0 or np.all(r <= 0.):
        return np.zeros(n), np.zeros(G.shape[0]), dict(res=0., support=[])
    Lc = np.linalg.cholesky(H)
    E = np.linalg.solve(Lc, G.T).T                         # rows E_i = (L^-1 G_i^T)^T
    scale = max(1., float(np.max(np.abs(r))))
    A = np.vstack([E.T, r[None, :] / scale])
    b = np.zeros(n + 1)
    b[n] = 1.
    # At a minimiser of the non-negative least-squares problem |rho|^2 = -rho[n]; the routine can stop elsewhere when
    # columns are dependent, so this is tested and the problem is solved again with rescaled, permuted columns
    # (multiplying a row of G delta >= r by a positive number does not change the problem).
    gen = np.random.default_rng(12345)
    for trial in range(8):
        w = np.ones(A.shape[1]) if trial == 0 else gen.uniform(0.3, 3., A.shape[1])
        perm = np.arange(A.shape[1]) if trial == 0 else gen.permutation(A.shape[1])
        try:
            v, _ = nnls(A[:, perm] * w[perm], b, maxiter=50 * A.shape[1] + 200)
        except RuntimeError:
            continue
        u = np.zeros(A.shape[1])
        u[perm] = v * w[perm]
        rho = A @ u - b
        if abs(rho @ rho + rho[n]) <= 1e-9:
            break
    else:
        return _enumerate(H, Lc, G, E, r, scale)
    if np.linalg.norm(rho) < 1e-9 or -rho[n] <= 1e-12:
        raise Infeasible('rows inconsistent')
    supp = [int(i) for i in np.flatnonzero(u > tol * max(1., u.max()))]
    # exact solve on the support (rows binding, independent gradients)
    for _ in range(len(r) + 2):
        if not supp:
            y, lam = np.zeros(n), np.zeros(0)
        else:
            Es = E[supp]
            lam = np.linalg.lstsq(Es @ Es.T, r[supp], rcond=1e-13)[0]
            y = Es.T @ lam
        if len(lam) and lam.min() < -1e-9 * max(1., np.abs(lam).max()):
            del supp[int(np.argmin(lam))]
            continue
        viol = r - E @ y
        if supp:
            viol[supp] = -np.inf
        p = int(np.argmax(viol))
        if viol[p] > 1e-9 * scale:
            supp.append(p)
            continue
        break
    mu = np.zeros(len(r))
    mu[supp] = np.maximum(lam, 0.)
    delta = np.linalg.solve(Lc.T, y)
    res = dict(support=supp,
               feas=float(np.max(np.maximum(r - G @ delta, 0.))) / scale,
               stat=float(np.max(np.abs(H @ delta - G.T @ mu))) / max(1., float(np.max(np.abs(H @ delta)))),
               comp=float(np.max(np.abs(mu * (G @ delta - r)))) / max(1., scale * float(np.max(mu)) if len(mu) else 1.))
    return delta, mu, res


def _enumerate(H, Lc, G, E, r, scale, tol=1e-9):
    """Exact solution by enumeration of the independent sets of rows (used when the least-squares routine stalls on
    dependent rows; small problems only).  The conditions  lam >= 0,  E y >= r  with  y = E_T^T lam,  E_T y = r_T  are
    sufficient for the minimum, which is unique; if no independent set satisfies them the rows are inconsistent."""
    from itertools import combinations
    R, n = E.shape
    if R > 16:
        raise RuntimeError('non-negative least squares did not reach a minimiser (too many rows for the enumeration)')
    rank = int(np.linalg.matrix_rank(E, tol=1e-10 * max(1., np.linalg.norm(E, 2))))
    for size in range(1, rank + 1):
        for T in combinations(range(R), size):
            ET = E[list(T)]
            Gm = ET @ ET.T
            ev = np.linalg.eigvalsh(Gm)
            if ev[0] <= 1e-12 * ev[-1]:
                continue
            lam = np.linalg.solve(Gm, r[list(T)])
            if lam.min() < -tol * max(1., np.abs(lam).max()):
                continue
            y = ET.T @ lam
            if np.max(r - E @ y) > tol * scale:
                continue
            mu = np.zeros(R)
            mu[list(T)] = np.maximum(lam, 0.)
            delta = np.linalg.solve(Lc.T, y)
            res = dict(support=list(T), enumerated=True,
                       feas=float(np.max(np.maximum(r - G @ delta, 0.))) / scale,
                       stat=float(np.max(np.abs(H @ delta - G.T @ mu))) / max(1., float(np.max(np.abs(H @ delta)))),
                       comp=float(np.max(np.abs(mu * (G @ delta - r)))) / max(1., scale * float(np.max(mu))))
            return delta, mu, res
    raise Infeasible('rows inconsistent (enumeration)')


def project(H, x0, G, r):
    """argmin 1/2 |x - x0|_H^2 s.t. G x >= r; returns x, mu, info."""
    G = np.atleast_2d(np.asarray(G, float))
    delta, mu, info = min_norm(H, G, np.asarray(r, float).ravel() - G @ x0)
    return x0 + delta, mu, info


def independent_carrier(H, G, mu, tol=1e-10):
    """Caratheodory reduction: from multipliers mu >= 0 of rows G (correction Delta = H^-1 G^T mu) to multipliers with the
    same Delta whose support has linearly independent gradients."""
    mu = np.array(mu, float)
    G = np.atleast_2d(G)
    for _ in range(len(mu) + 1):
        supp = np.flatnonzero(mu > tol * max(1., mu.max() if len(mu) else 1.))
        if len(supp) == 0:
            break
        Gs = G[supp]
        sv = np.linalg.svd(Gs, compute_uv=False)
        if sv[-1] > 1e-9 * sv[0] and len(supp) <= G.shape[1]:
            break
        # a null vector of Gs^T: sum_i c_i G_i = 0
        _, _, Vt = np.linalg.svd(Gs.T)
        c = Vt[-1]
        if c.max() < -c.min():                              # the entry of largest modulus is made positive
            c = -c
        pos = c > 1e-10 * np.abs(c).max()
        ratios = np.full(len(c), np.inf)
        ratios[pos] = mu[supp][pos] / c[pos]
        j = int(np.argmin(ratios))
        mu[supp] = mu[supp] - ratios[j] * c
        mu[supp[j]] = 0.
        mu = np.maximum(mu, 0.)
    return mu
