"""Small dense strictly convex quadratic programs by the dual active-set method of Goldfarb and Idnani (1983).

    minimise   1/2 x^T H x + g^T x      subject to   A x <= b,          H symmetric positive definite.

The iteration starts at the unconstrained minimiser x_u, adds the most violated row and keeps dual feasibility by
partial steps that drop blocking rows; it terminates at the optimum or with a certificate that  {A x <= b}  is
empty (a violated row whose negated normal is a non-negative combination of the active normals: no step can reduce
its violation without violating an active row).  It stops when no row is violated by more than tol max(1, |x_u|_H)
(rows normalized by their H^-1 norms, tol = 1e-10); when |x|_H is small compared with max(1, |x_u|_H), a row can
therefore remain violated by more than 1e-8 of |A_i| |x| + |b_i|.  The problems solved here have at most a few unknowns
(one per actuator channel plus slacks), so every step re-solves its small dense systems instead of updating
factorizations.  Every returned solution carries its KKT residuals; the tests compare the solver with SciPy's SLSQP and
with the HiGHS LP feasibility test on random problems.

A violated row p is rated dependent on the active rows when the squared H^-1 norm  A_p z  of its projected normal z is
below 1e-9 of that of the normal.  For such a row
  - the steps are those of the exact method: a partial step (a blocking active row is dropped before p becomes active)
    moves the primal and the dual iterate together, x <- x - t1 z and u <- u - t1 r, and the full step
    t2 = (A_p x - b_p) / (A_p z)  is taken whenever t2 <= t1 (for such a row only if its violation is very small, and
    only if A_p z exceeds 1e-14 of the squared norm);
  - a computed z whose H norm is below ten times its estimated round-off error,  eps sqrt(cond S) (1 + rho) |A_p|  (S:
    Gram matrix of the active normals in the H^-1 metric, rho = sum_i |A_i| |r_i| / |A_p|, H^-1 norms), is
    indistinguishable from round-off and is treated as zero, as for an exactly dependent row: the partial step moves only
    the dual iterate, and the row is not added;
  - when no active row blocks (t1 = inf; the normal of p is then nearly opposite to a non-negative combination of the
    active normals), the set {A x <= b} is rated numerically empty: the full step, with the multiplier
    t2 >= 1e9 (A_p x - b_p) / |A_p|^2  (H^-1 norm), would be computed from the nearly vanishing A_p z and be inaccurate;
  - after any step for such a row, the returned point is accepted only if the residuals of the active rows are below
    1e-8 of |A_i| (|x| + |x - x_u|) + |b_i| and the stationarity residual below 1e-8 of |H| |x| + |g| + |A|^T mu
    (normwise backward errors); otherwise the set is rated numerically empty.
The empty ratings are conservative (they can be given to a set that is not empty); the controllers count them and use
their fallback.  (An earlier version skipped the primal part of the partial step for every row rated dependent and
never added such rows; for rows whose normals make an angle of about 1e-6 to 1e-4 with the span of the active normals
the returned point was then feasible but not the minimiser, with relative distances up to about 3e-4.
pinocchio/num/t13_qp_dependent.py reproduces the earlier behaviour and checks the present one against high-precision
references.)

``solve_lexicographic`` is the fallback for inconsistent rows: smallest violations first, then the original cost
(no penalty weight); ``solve_relaxed`` is the weighted-slack variant (kept for comparison in the tests).
"""
from __future__ import annotations

import numpy as np

_EPS = np.finfo(float).eps
_ROUNDOFF_FACTOR = 10.     # a computed z of a dependent row below this multiple of its round-off error is treated as zero
_DEP_FLOOR = 1e-14          # a nearly dependent row is added only if A_p z exceeds this fraction of |A_p|^2 (H^-1 norm)


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
    used_dependent = False                                  # a row rated dependent was met (acceptance test at the end)
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
            viol_p = float(A[p] @ x - b[p])
            exact = False
            if dependent:
                used_dependent = True
                if W:
                    # the computed z carries an error of about eps sqrt(cond S) (1 + rho) |A_p| in the H norm (rho = sum_i
                    # |A_i| |r_i| / |A_p|, H^-1 norms); a z below _ROUNDOFF_FACTOR times that is indistinguishable from
                    # round-off and is treated as zero
                    zh = float(np.sqrt(max(float(z @ H @ z), 0.)))
                    noise = _EPS * np.sqrt(np.linalg.cond(S)) * (1. + float(np.abs(r) @ dnorm[W]) / dnorm[p]) * dnorm[p]
                    exact = not zh > _ROUNDOFF_FACTOR * noise
                else:
                    exact = True                            # a zero row
            if not dependent:
                t2 = viol_p / zn
            elif not exact and np.isfinite(t1) and zn > _DEP_FLOOR * dnorm[p] ** 2 and viol_p <= t1 * zn:
                # nearly dependent row whose violation the primal step removes before the multiplier of an active row
                # reaches zero: full step, the row becomes active (a partial step would carry x past the row)
                t2 = viol_p / zn
            else:
                t2 = np.inf
            if not np.isfinite(t1) and not np.isfinite(t2):
                feasible = False
                break
            if t2 <= t1:                                    # full step: row p becomes active
                x = x - t2 * z
                u = np.r_[u - t2 * r, up + t2]
                W.append(p)
                break
            # partial step in primal and dual space (both parts, also for a nearly dependent row: skipping the primal part
            # leaves x off the dual path); for a row treated as exactly dependent only the dual part
            if not exact:
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
    Hx, Atmu = H @ x, A.T @ mu
    stat = float(np.max(np.abs(Hx + g + Atmu)))
    out.kkt = dict(primal=float(np.max(res / dnorm)) / xs, stationarity=stat,
                   stationarity_relative=stat / max(1., float(np.max(np.abs(g))), float(np.max(np.abs(Hx))), float(np.max(np.abs(Atmu)))),
                   dual=float(max(0., -np.min(u))) if len(u) else 0., complementarity=float(np.max(np.abs(mu * res))))
    out.ok = bool(out.kkt['primal'] <= 1e-7 and out.iterations < max_iter)
    if out.ok and used_dependent:
        # after steps for rows rated dependent: residuals of the active rows below 1e-8 of |A_i| (|x| + |x - x_u|) + |b_i| and
        # of the stationarity condition below 1e-8 of |H| |x| + |g| + |A|^T mu (normwise backward errors), else rated empty
        act = mu > 0.
        xr = np.abs(x) + np.abs(x - out.unconstrained)
        rows = float(np.max(np.abs(res[act]) / np.maximum(np.abs(A[act]) @ xr + np.abs(b[act]), 1e-300))) if act.any() else 0.
        size = float(np.max(np.abs(H) @ np.abs(x) + np.abs(g) + np.abs(A).T @ mu))
        out.ok = max(rows, float(np.max(np.abs(Hx + g + Atmu))) / max(size, 1e-300)) <= 1e-8
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


def solve_relaxed(H, g, A_hard, b_hard, A_soft, b_soft, weight):
    """Hard rows kept, soft rows relaxed by non-negative slacks with a quadratic penalty.

        min 1/2 x^T H x + g^T x + 1/2 sum_i weight_i s_i^2    s.t.   A_hard x <= b_hard,   A_soft x - s <= b_soft,  s >= 0.

    Used only after ``solve_qp`` reported an empty feasible set (the event is counted and reported by the caller)."""
    n, ms = len(g), len(b_soft)
    weight = np.broadcast_to(np.asarray(weight, float), (ms,))
    Hx = np.zeros((n + ms, n + ms))
    Hx[:n, :n] = H
    Hx[n:, n:] = np.diag(weight)
    gx = np.r_[g, np.zeros(ms)]
    rows, rhs = [], []
    if A_hard is not None and len(A_hard):
        rows.append(np.hstack([A_hard, np.zeros((len(A_hard), ms))]))
        rhs.append(b_hard)
    rows.append(np.hstack([A_soft, -np.eye(ms)]))
    rhs.append(b_soft)
    rows.append(np.hstack([np.zeros((ms, n)), -np.eye(ms)]))
    rhs.append(np.zeros(ms))
    res = solve_qp(Hx, gx, np.vstack(rows), np.concatenate(rhs))
    return res, (None if not res.feasible else res.x[:n]), (None if not res.feasible else res.x[n:])


def solve_lexicographic(H, g, A_hard, b_hard, A_soft, b_soft, eps=(1e-6, 1e-4), slack_tol=1e-6):
    """Two-stage relaxation of a problem whose rows  A_hard x <= b_hard,  A_soft x <= b_soft  are inconsistent.

      stage 1   s* = argmin 1/2 |s|^2    s.t.   A_hard x <= b_hard,   A_soft x - s <= b_soft,   s >= 0
                the smallest sum of squared violations of the soft rows that the hard rows allow (s* is unique).  The soft
                rows must be given in the units in which their violations are to be compared, with entries of order
                one; the term  eps / 2 |x - x_u|^2_R  (x_u: unconstrained minimiser of the cost) makes the problem strictly
                convex and changes s* by about eps |x - x_u|.  R = n H / trace(H) + 1e-3 I is the cost metric itself
                (scaled to unit mean diagonal, so that the term transforms like the cost under a change of the
                unknowns) with a small isotropic part that bounds its condition number.  eps = 1e-6; 1e-4 is tried if
                the first solve fails numerically (a smaller eps would push the slack part of a row below the dependence
                test of the solver).
      stage 2   min 1/2 x^T H x + g^T x    s.t.   A_hard x <= b_hard,   A_soft x <= b_soft + s* + slack_tol
                among all points with the smallest violations, the minimiser of the original cost.

    No penalty weight relates violations to the cost, so the outcome does not depend on the scale or the conditioning of
    H.  Returns (result, x, s): ``result`` is the QPResult of stage 2 over the rows [A_hard; A_soft] (feasible = False if
    the hard rows alone are inconsistent; ok = False if stage 2 failed numerically and the stage-1 point is returned),
    s = max(0, A_soft x - b_soft) at the returned point."""
    H = np.asarray(H, float)
    g = np.asarray(g, float)
    n, ms = len(g), len(b_soft)
    A_soft, b_soft = np.asarray(A_soft, float).reshape(ms, n), np.asarray(b_soft, float)
    hard = A_hard is not None and len(A_hard) > 0
    A_hard = np.asarray(A_hard, float).reshape(-1, n) if hard else np.zeros((0, n))
    b_hard = np.asarray(b_hard, float) if hard else np.zeros(0)
    mh = len(b_hard)
    c, _ = _cho(H)
    x_u = -_cho_solve(c, g)
    A1 = np.vstack([np.hstack([A_hard, np.zeros((mh, ms))]), np.hstack([A_soft, -np.eye(ms)]), np.hstack([np.zeros((ms, n)), -np.eye(ms)])])
    b1 = np.r_[b_hard, b_soft, np.zeros(ms)]
    Rm = H * (n / np.trace(H)) + 1e-3 * np.eye(n)
    H1 = np.eye(n + ms)
    for e in np.atleast_1d(eps):
        H1[:n, :n] = e * Rm
        res1 = solve_qp(H1, np.r_[-e * (Rm @ x_u), np.zeros(ms)], A1, b1)
        if res1.feasible:
            break
    if not res1.feasible:
        return res1, None, None
    s_star = np.maximum(res1.x[n:], 0.)
    res2 = solve_qp(H, g, np.vstack([A_hard, A_soft]), np.r_[b_hard, b_soft + s_star + slack_tol])
    if not res2.feasible:                                   # numerically empty after the relaxation: keep the stage-1 point
        res1.ok = False                                     # (no multipliers of the original cost: callers must not use mu)
        res1.mu = np.zeros(mh + ms)
        res1.active = np.zeros(mh + ms, bool)
        x = res1.x[:n]
        res1.x = x
        return res1, x, np.maximum(A_soft @ x - b_soft, 0.)
    return res2, res2.x, np.maximum(A_soft @ res2.x - b_soft, 0.)
