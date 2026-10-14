"""The dual active-set routine rcx.qp.solve_qp as it was before the correction of its partial step (kept only for the
comparison in t13_qp_dependent.py; the simulations use rcx.qp).

Difference to rcx.qp.solve_qp: when the violated row p is rated linearly dependent on the active rows (squared H^-1 norm
of its projected normal below 1e-9 of that of the normal), a partial step here updates only the dual iterate and skips
the primal part  x <- x - t1 z,  and the full step that adds p is never taken.  The primal part is zero for exactly
dependent rows but not for nearly dependent ones, and the returned point is then feasible but not the minimiser.
"""
from __future__ import annotations

import numpy as np

_EPS = np.finfo(float).eps


class QPResult:
    __slots__ = ('x', 'mu', 'feasible', 'active', 'kkt', 'unconstrained', 'ok', 'iterations')


def solve_qp(H, g, A=None, b=None, max_iter=None, tol=1e-10):
    """Solve the strictly convex QP.  ``feasible`` is False when  {A x <= b}  is empty (x, mu are then None)."""
    H = np.asarray(H, float)
    g = np.asarray(g, float)
    n = len(g)
    out = QPResult()
    out.iterations = 0
    c, low = _cho(H)
    x = -_cho_solve(c, g)
    out.unconstrained = x.copy()
    out.ok = True
    if A is None or len(A) == 0:
        out.x, out.mu, out.feasible, out.active = x, np.zeros(0), True, np.zeros(0, bool)
        out.kkt = dict(primal=0., stationarity=float(np.max(np.abs(H @ x + g))), dual=0., complementarity=0.)
        return out
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m = len(b)
    HiAt = _cho_solve(c, A.T)                               # H^{-1} A^T  (n x m)
    dnorm = np.sqrt(np.maximum(np.einsum('ij,ji->i', A, HiAt), 1e-300))     # |A_i| in the H^{-1} metric
    xs = max(1., float(np.sqrt(x @ H @ x)))
    W = []                                                  # active rows (ordered)
    u = np.zeros(0)                                         # their multipliers
    max_iter = max_iter or 20 * (m + n) + 50
    feasible = True
    while out.iterations < max_iter:
        viol = (A @ x - b) / dnorm
        if W:
            viol[W] = -np.inf
        p = int(np.argmax(viol))
        if viol[p] <= tol * xs:
            break
        up = 0.
        np_ = A[p]
        while True:
            out.iterations += 1
            if W:
                N = A[W].T                                  # n x k
                HiN = HiAt[:, W]
                S = N.T @ HiN
                r = _solve_sym(S, HiN.T @ np_)              # dual step of the active rows
                z = HiAt[:, p] - HiN @ r                    # primal step direction
            else:
                r = np.zeros(0)
                z = HiAt[:, p].copy()
            zn = float(np_ @ z)                             # = |P n|^2 in the H^{-1} metric, >= 0
            dependent = zn <= 1e-9 * dnorm[p] ** 2
            pos = r > 1e-14 * max(1., float(np.max(np.abs(r))) if len(r) else 1.)
            if pos.any():
                ratios = np.where(pos, u / np.where(pos, r, 1.), np.inf)
                l = int(np.argmin(ratios))
                t1 = float(ratios[l])
            else:
                l, t1 = -1, np.inf
            t2 = np.inf if dependent else float(A[p] @ x - b[p]) / zn
            if not np.isfinite(t1) and not np.isfinite(t2):
                feasible = False
                break
            if t2 <= t1:                                    # full step: row p becomes active
                x = x - t2 * z
                u = np.r_[u - t2 * r, up + t2]
                W.append(p)
                break
            if not dependent:                               # partial step in primal and dual space
                x = x - t1 * z
            u = u - t1 * r
            up += t1
            del W[l]
            u = np.delete(u, l)
            if out.iterations >= max_iter:
                break
        if not feasible:
            break
    if not feasible:
        out.x, out.mu, out.feasible, out.active, out.kkt = None, None, False, None, None
        return out
    mu = np.zeros(m)
    if W:
        mu[W] = np.maximum(u, 0.)
    out.x, out.mu, out.feasible = x, mu, True
    out.active = mu > 0.
    res = A @ x - b
    out.kkt = dict(primal=float(np.max(res / dnorm)) / xs, stationarity=float(np.max(np.abs(H @ x + g + A.T @ mu))),
                   dual=float(max(0., -np.min(u))) if len(u) else 0., complementarity=float(np.max(np.abs(mu * res))))
    out.ok = bool(out.kkt['primal'] <= 1e-7 and out.iterations < max_iter)
    if not out.ok:                                          # numerically empty feasible set (nearly dependent active rows)
        out.x, out.mu, out.feasible, out.active = None, None, False, None
    return out


def _cho(H):
    return np.linalg.cholesky(H), True


def _cho_solve(c, rhs):
    y = np.linalg.solve(c, rhs)
    return np.linalg.solve(c.T, y)


def _solve_sym(S, rhs):
    try:
        return np.linalg.solve(S, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(S, rhs, rcond=None)[0]
