"""The quadratic-program routine of the excavator controllers (rcx.qp.solve_qp, read-only) on rows that are nearly, but
not exactly, linearly dependent, before and after the correction of its steps for such rows.

The routine rates a violated row as dependent on the active ones when the squared H^-1 norm of its projected normal is
below 1e-9 of the squared norm of the normal.  Before the correction (qp_before_fix.py) it then skipped the primal part
of a partial step and never added the row.  For exactly dependent rows the primal part is zero; for a row whose normal
makes an angle of about 1e-6 to 1e-4 with the span of the active normals it is not, and the returned point was feasible
but not the minimiser.  The corrected routine (rcx.qp) takes the steps of the exact method for such rows (both parts of
a partial step, and the full step that adds the row whenever it is not longer than the partial step), treats a
projected normal below ten times its estimated round-off error as zero, and after any step for such a row accepts the
result only with relative residuals below 1e-8 (see rcx/qp.py).

Part 1, rows violated by amounts of order one:

    min 1/2 |x - x0|_H^2   s.t.  A x <= b,     one row of A = a combination of two other rows + eps * (unit vector).

Reference: the least-distance solver ldp.py (independent of rcx.qp).  Wherever a routine and ldp.py differ by more than
1e-9 (relative H-distance), the instance is solved once more by enumeration of the active sets in 50-digit arithmetic
(mpmath), and that point is the reference.  Reported for both routines: how often the returned point differs from the
minimiser, the largest relative distance, the largest relative stationarity residual |H x + g + A^T mu|, the largest
violation of a row, and the number of instances that a routine reports as infeasible although they are feasible, with
the H-distance of their minimiser from x0.

Part 2, small violations: two unit rows whose normals make an angle of 10^-7.5 to 10^-4.5, nearly parallel or nearly
opposite; the first is violated at x0, the second is violated (or met) by 1e-11 to 1e-7 at the minimiser over the first
row; with or without 1 to 3 further random rows.  In this regime a nearly dependent row is added to the active set by a
full step.  Reference for every instance: enumeration in 50-digit arithmetic.  Reported for both routines: points more
than 1e-8 (relative H-distance) from the minimiser; the backward error of the returned point x and multipliers mu,
evaluated in 50-digit arithmetic for the data passed to the routines (H, g = -H x0, A, b): the largest row residual
(violation of any row, |A_i x - b_i| of the rows with positive multipliers) relative to |A_i| (|x| + |x - x_u|) + |b_i|
(x_u: unconstrained minimiser), and the stationarity residual |H x + g + A^T mu| relative to |H| |x| + |g| + |A|^T |mu|
(componentwise magnitudes, largest component); (x, mu) is exact for a problem whose data differ by about that fraction.
Also reported: the stationarity residual relative to |H| |x - x_u| + |A|^T |mu| (the correction of the unconstrained
minimiser, taken as at least 1e-6 of the data), which shows the error of the earlier version in its own terms.
Nearly dependent rows make the
minimiser itself sensitive to the data, so a point with a backward error near round-off can still be farther than 1e-8
from the minimiser; with nearly opposite active rows the multipliers reach 1e8 and more.  Also reported: feasible sets
rated empty, empty sets rated feasible, the largest number of iterations and the instances that reach the iteration
limit.

Part 3, adversarial families (1 to 6 unknowns, up to 10 rows; H the identity or with eigenvalues spread over up to 7
decades): rows nearly parallel or nearly opposite to rows already present, near (the negative of) positive combinations
of 2 to 4 rows, exact, offset and scaled duplicates, narrow boxes, mixtures of these, rows and problems scaled by up to
1e4 and 1e5, tiny violations, unconstrained minimisers 1e2 to 1e6 times farther away, and nearly opposite active pairs
with a third nearly dependent row; the near rows make angles of 1e-13 to 10^-1.5 with their targets and are violated or
met by 1e-15 to 0.1 of their size at the minimiser of the rows already present.  Reference: the KKT systems of candidate
active sets in 60-digit arithmetic (the active sets suggested by the routines, then all).  Reported as in part 2; points
with a row backward error above 1e-8 are split off where an inactive row is violated by less than the stopping tolerance
of the routine (1e-10 max(1, |x0|_H) in H^-1-normalized units, large relative to |A_i| |x| + |b_i| when |x0|_H is large
compared with |x|).  An empty set rated feasible is a set that a change of the data below the backward error of the
returned point makes non-empty.

In every part, the instances on which the corrected routine fails (no point, a point off the minimiser in part 1, a
backward error above 1e-8 in parts 2 and 3) while the earlier version succeeds are counted.  Both versions rate a
feasible set as numerically empty when a violated row is rated dependent and no active row blocks its step (its normal
is then nearly opposite to a non-negative combination of the active normals; see rcx/qp.py).  That rating is
conservative: in part 1 such sets are rare and their minimisers lie far from x0; in parts 2 and 3 they are frequent in
the nearly opposite families, and their minimisers lie at H-distances from x0 of the same order as those of the other
instances.  The controllers count a sample rated empty and use the lexicographic fallback of rcx.qp; no sample of the
runs of the paper was rated empty.

    python t13_qp_dependent.py [instances per eps] [instances per family of part 2] [instances per family of part 3]
                               (defaults 20000, 2000 and 300; writes t13_results.json)
"""
import json
import sys
from itertools import combinations

sys.dont_write_bytecode = True
import mpmath as mp      # noqa: E402
import numpy as np      # noqa: E402

import paths            # noqa: E402

sys.path.insert(0, paths.RCX_HEAP)
from rcx.qp import solve_qp      # noqa: E402
from ldp import Infeasible, project      # noqa: E402
from qp_before_fix import solve_qp as solve_qp_before_fix      # noqa: E402

rng = np.random.default_rng(20261010)
ROUTINES = (('corrected', solve_qp), ('before_fix', solve_qp_before_fix))


def spd(n, gen=None):
    gen = rng if gen is None else gen
    Q = np.linalg.qr(gen.standard_normal((n, n)))[0]
    return (Q * gen.uniform(1., 20., n)) @ Q.T


def exact_projection(H, x0, A, b, digits=50):
    """argmin 1/2 |x - x0|_H^2 s.t. A x <= b by enumeration of the independent active sets in high precision (small
    problems only).  Returns x (float) or None if the rows are inconsistent."""
    with mp.workdps(digits):
        n, R = H.shape[0], A.shape[0]
        Hm, Am, bm, x0m = mp.matrix(H.tolist()), mp.matrix(A.tolist()), mp.matrix(b.tolist()), mp.matrix(x0.tolist())
        tol = mp.mpf(10) ** (-(digits // 2))
        Ax = lambda x, i: mp.fsum(Am[i, j] * x[j] for j in range(n))
        if all(Ax(x0m, i) - bm[i] <= tol for i in range(R)):
            return x0.copy()
        for size in range(1, min(R, n) + 1):
            for S in combinations(range(R), size):
                k = len(S)
                K, rhs = mp.zeros(n + k, n + k), mp.zeros(n + k, 1)
                for i in range(n):
                    for j in range(n):
                        K[i, j] = Hm[i, j]
                    rhs[i] = mp.fsum(Hm[i, j] * x0m[j] for j in range(n))
                for a, i in enumerate(S):
                    for j in range(n):
                        K[n + a, j] = K[j, n + a] = Am[i, j]
                    rhs[n + a] = bm[i]
                if abs(mp.det(K)) < mp.mpf(10) ** (-(digits - 5)):
                    continue
                sol = mp.lu_solve(K, rhs)
                x, lam = sol[:n], sol[n:]
                if any(lam[a] < -tol for a in range(k)) or any(Ax(x, i) - bm[i] > tol for i in range(R)):
                    continue
                return np.array([float(v) for v in x])
        return None


def nearly_dependent_rows(n_inst):
    """Part 1."""
    out = {}
    for eps in (0., 1e-6, 1e-5, 3e-5, 1e-4, 1e-3):
        st = {name: dict(off_1e7=0, off_1e9=0, worst_distance=0., cost_excess_there=0., worst_stationarity=0., worst_row_violation=0.,
                         reported_infeasible=0, reported_infeasible_minimiser_distance=[]) for name, _ in ROUTINES}
        n_ok, skipped, arbitrated, ldp_off, regressions = 0, 0, 0, 0, 0
        for _ in range(n_inst):
            n = int(rng.integers(3, 6))
            R = int(rng.integers(3, 6))
            H = spd(n)
            A = rng.standard_normal((R, n))
            A /= np.linalg.norm(A, axis=1)[:, None]
            i, j, p = rng.choice(R, 3, replace=False)
            v = rng.standard_normal(n)
            a = rng.uniform(0.2, 1.) * A[i] + rng.uniform(0.2, 1.) * A[j]
            A[p] = a / np.linalg.norm(a) + eps * v / np.linalg.norm(v)
            x0 = rng.standard_normal(n) * 2.
            b = A @ x0 - rng.uniform(-0.2, 1., R)             # most rows violated at x0
            try:
                xe, _, _ = project(H, x0, -A, -b)
            except Infeasible:
                skipped += 1
                continue
            step = float(np.sqrt((xe - x0) @ H @ (xe - x0)))
            if step < 1e-3:                                    # no row binds (or hardly): nothing to compare
                continue
            res = {name: f(H, -H @ x0, A, b) for name, f in ROUTINES}
            hdist = lambda x, y: float(np.sqrt((x - y) @ H @ (x - y)) / step)
            if any(r.feasible and hdist(r.x, xe) > 1e-9 for r in res.values()):
                xm = exact_projection(H, x0, A, b)             # arbitration in 50-digit arithmetic
                arbitrated += 1
                if xm is None:
                    skipped += 1
                    continue
                ldp_off += int(hdist(xe, xm) > 1e-9)
                xe = xm
                step = float(np.sqrt((xe - x0) @ H @ (xe - x0)))
            n_ok += 1
            good = {name: r.feasible and hdist(r.x, xe) <= 1e-7 for name, r in res.items()}
            regressions += int(good['before_fix'] and not good['corrected'])
            for name, r in res.items():
                s = st[name]
                if not r.feasible:
                    # the routine rates the feasible set as numerically empty; recorded with the H-distance of the minimiser from x0
                    s['reported_infeasible'] += 1
                    s['reported_infeasible_minimiser_distance'].append(step)
                    continue
                dist = hdist(r.x, xe)
                cost = float(((r.x - x0) @ H @ (r.x - x0) - step ** 2) / step ** 2)
                s['off_1e7'] += int(dist > 1e-7)
                s['off_1e9'] += int(dist > 1e-9)
                if dist > s['worst_distance']:
                    s['worst_distance'], s['cost_excess_there'] = dist, cost
                gr = H @ (r.x - x0) + A.T @ r.mu
                s['worst_stationarity'] = max(s['worst_stationarity'], float(np.max(np.abs(gr))) / max(1., float(np.max(np.abs(H @ (xe - x0))))))
                s['worst_row_violation'] = max(s['worst_row_violation'], float(np.max(A @ r.x - b)))
        out[str(eps)] = dict(instances=n_ok, skipped_infeasible=skipped, high_precision_arbitrations=arbitrated,
                             ldp_reference_off_by_more_than_1e9=ldp_off, corrected_worse_than_before_fix=regressions,
                             **{name: st[name] for name, _ in ROUTINES})
        print(f'eps = {eps:7.0e}: {n_ok} feasible instances with a binding row ({arbitrated} decided in 50-digit arithmetic, '
              f'in {ldp_off} of them the double-precision reference was off by more than 1e-9); instances where only the earlier '
              f'version is within 1e-7 of the minimiser: {regressions}')
        for name, _ in ROUTINES:
            s = st[name]
            print(f'    {name:10s}: differs from the minimiser by more than 1e-7 (relative) in {s["off_1e7"]}, by more than 1e-9 in {s["off_1e9"]}; '
                  f'largest relative distance {s["worst_distance"]:.1e} (relative excess of the cost there {s["cost_excess_there"]:.1e}); '
                  f'largest relative stationarity residual {s["worst_stationarity"]:.1e}; largest row violation {s["worst_row_violation"]:.1e}; '
                  f'reported infeasible {s["reported_infeasible"]}'
                  + (f' (H-distance of the minimiser from x0: {", ".join(f"{d:.1e}" for d in s["reported_infeasible_minimiser_distance"])})'
                     if s['reported_infeasible'] else ''), flush=True)
    return out


def unit(v):
    return v / np.linalg.norm(v)


def turned(a, theta, gen):
    """A unit vector at the angle theta from the unit vector a."""
    v = gen.standard_normal(len(a))
    v = unit(v - (v @ a) * a)
    return unit(np.cos(theta) * a + np.sin(theta) * v)


def small_violation_instance(gen, opposite, extra):
    n = int(gen.integers(3, 6)) if extra else 3
    H = spd(n, gen)
    x0 = gen.standard_normal(n)
    a1 = unit(gen.standard_normal(n))
    b1 = a1 @ x0 - gen.uniform(0.2, 1.)                                    # first row violated at x0
    Hi = np.linalg.inv(H)
    xp = x0 - (a1 @ x0 - b1) * (Hi @ a1) / (a1 @ Hi @ a1)                  # minimiser over the first row
    a2 = turned(-a1 if opposite else a1, 10 ** gen.uniform(-7.5, -4.5), gen)
    b2 = a2 @ xp - gen.choice([-1., 1.]) * 10 ** gen.uniform(-11, -7)     # second row violated or met there by 1e-11..1e-7
    A, b = [a1, a2], [b1, b2]
    if extra:
        for _ in range(int(gen.integers(1, 4))):
            c = unit(gen.standard_normal(n))
            A.append(c)
            b.append(c @ x0 - gen.uniform(-0.5, 1.))
    return H, x0, np.array(A), np.array(b)


def backward_errors(H, g, A, b, r, digits=50):
    """(stationarity, rows): normwise backward errors of the returned point x and multipliers mu for the data passed to
    the routine (H, g, A, b), evaluated in high precision (see the module docstring)."""
    with mp.workdps(digits):
        n, R = H.shape[0], A.shape[0]
        Hm, Am, bm = mp.matrix(H.tolist()), mp.matrix(A.tolist()), mp.matrix(b.tolist())
        gm = mp.matrix([float(v) for v in g])
        xu = mp.lu_solve(Hm, -gm)                                       # unconstrained minimiser
        mu = [mp.mpf(float(v)) for v in r.mu]
        xm = [mp.mpf(float(v)) for v in r.x]
        res = [abs(mp.fsum([Hm[i, j] * xm[j] for j in range(n)] + [gm[i]] + [Am[k, i] * mu[k] for k in range(R)])) for i in range(n)]
        data = [mp.fsum([abs(Hm[i, j] * xm[j]) for j in range(n)] + [abs(gm[i])] + [abs(Am[k, i] * mu[k]) for k in range(R)]) for i in range(n)]
        step = [mp.fsum([abs(Hm[i, j] * (xm[j] - xu[j])) for j in range(n)] + [abs(Am[k, i] * mu[k]) for k in range(R)]) for i in range(n)]
        tiny = mp.mpf(10) ** -300
        e_stat = max(res) / max(max(data), tiny)
        e_stat_correction = max(res) / max(max(step), mp.mpf(10) ** -6 * max(data), tiny)
        e_rows = mp.mpf(0)
        for i in range(R):
            ri = mp.fsum(Am[i, j] * xm[j] for j in range(n)) - bm[i]
            e = abs(ri) if r.mu[i] > 0. else max(ri, mp.mpf(0))
            scale = mp.fsum(abs(Am[i, j]) * (abs(xm[j]) + abs(xm[j] - xu[j])) for j in range(n)) + abs(bm[i])
            e_rows = max(e_rows, e / max(scale, tiny))
        return float(e_stat), float(e_rows), float(e_stat_correction)


def small_violations(n_inst):
    """Part 2."""
    gen = np.random.default_rng(20261013)
    out = {}
    for fam, opposite, extra in (('nearly_parallel', False, False), ('nearly_opposite', True, False),
                                 ('nearly_parallel_extra_rows', False, True), ('nearly_opposite_extra_rows', True, True)):
        st = {name: dict(off_1e8=0, backward_error_over_1e8=0, worst_backward_error_stationarity=0., worst_backward_error_rows=0.,
                         stationarity_over_1e8_relative_to_correction=0, worst_stationarity_relative_to_correction=0.,
                         worst_distance=0., worst_row_violation=0., reported_infeasible=0,
                         reported_infeasible_minimiser_distance_range=None, reported_feasible_though_empty=0, max_iterations=0,
                         iteration_limit_reached=0) for name, _ in ROUTINES}
        n_feas = n_empty = regressions = 0
        for _ in range(n_inst):
            H, x0, A, b = small_violation_instance(gen, opposite, extra)
            xe = exact_projection(H, x0, A, b)
            limit = 20 * (len(b) + len(x0)) + 50
            res = {name: f(H, -H @ x0, A, b) for name, f in ROUTINES}
            for name, r in res.items():
                st[name]['max_iterations'] = max(st[name]['max_iterations'], r.iterations)
                st[name]['iteration_limit_reached'] += int(r.iterations >= limit)
            if xe is None:
                n_empty += 1
                for name, r in res.items():
                    st[name]['reported_feasible_though_empty'] += int(r.feasible)
                continue
            n_feas += 1
            step = max(float(np.sqrt((xe - x0) @ H @ (xe - x0))), 1e-12)
            good = {}
            for name, r in res.items():
                s = st[name]
                good[name] = False
                if not r.feasible:
                    s['reported_infeasible'] += 1
                    lo, hi = s['reported_infeasible_minimiser_distance_range'] or (np.inf, 0.)
                    s['reported_infeasible_minimiser_distance_range'] = [min(lo, step), max(hi, step)]
                    continue
                dist = float(np.sqrt((r.x - xe) @ H @ (r.x - xe))) / step
                s['off_1e8'] += int(dist > 1e-8)
                s['worst_distance'] = max(s['worst_distance'], dist)
                s['worst_row_violation'] = max(s['worst_row_violation'], float(np.max(A @ r.x - b)))
                e_stat, e_rows, e_corr = backward_errors(H, -H @ x0, A, b, r)
                s['worst_backward_error_stationarity'] = max(s['worst_backward_error_stationarity'], e_stat)
                s['worst_backward_error_rows'] = max(s['worst_backward_error_rows'], e_rows)
                s['backward_error_over_1e8'] += int(max(e_stat, e_rows) > 1e-8)
                s['worst_stationarity_relative_to_correction'] = max(s['worst_stationarity_relative_to_correction'], e_corr)
                s['stationarity_over_1e8_relative_to_correction'] += int(e_corr > 1e-8)
                good[name] = max(e_stat, e_rows) <= 1e-8
            regressions += int(good['before_fix'] and not good['corrected'])
        out[fam] = dict(feasible_instances=n_feas, empty_instances=n_empty, corrected_worse_than_before_fix=regressions,
                        **{name: st[name] for name, _ in ROUTINES})
        print(f'{fam}: {n_feas} feasible and {n_empty} empty instances (50-digit reference); instances where only the earlier version '
              f'returns a point with a backward error below 1e-8: {regressions}')
        for name, _ in ROUTINES:
            s = st[name]
            rng_d = s['reported_infeasible_minimiser_distance_range']
            print(f'    {name:10s}: more than 1e-8 from the minimiser in {s["off_1e8"]}, largest relative distance {s["worst_distance"]:.1e}; '
                  f'largest backward error {s["worst_backward_error_stationarity"]:.1e} (stationarity), {s["worst_backward_error_rows"]:.1e} (rows), '
                  f'above 1e-8 in {s["backward_error_over_1e8"]}; stationarity relative to the correction above 1e-8 in '
                  f'{s["stationarity_over_1e8_relative_to_correction"]} (largest {s["worst_stationarity_relative_to_correction"]:.1e}); '
                  f'largest row violation {s["worst_row_violation"]:.1e}; '
                  f'feasible set rated empty {s["reported_infeasible"]}'
                  + (f' (H-distance of the minimiser from x0 {rng_d[0]:.1e} to {rng_d[1]:.1e})' if rng_d else '')
                  + f'; empty set rated feasible {s["reported_feasible_though_empty"]}; at most {s["max_iterations"]} iterations, '
                  f'iteration limit reached {s["iteration_limit_reached"]}', flush=True)
    return out


# ------------------------------------------------------------------------------------------------ part 3
FAMILIES = ('parallel', 'opposite', 'cone', 'negative_cone', 'duplicates', 'box', 'mixed', 'scaled', 'tiny_violations',
            'far_minimiser', 'opposite_active')


def random_metric(n, gen):
    kind = int(gen.integers(5))
    if kind == 0:
        return np.eye(n)
    Q = np.linalg.qr(gen.standard_normal((n, n)))[0]
    e = (gen.uniform(1., 20., n), 10 ** gen.uniform(0, 4, n), 10 ** gen.uniform(-3, 3, n), 10 ** gen.uniform(-1, 6, n))[kind - 1]
    H = (Q * e) @ Q.T
    return 0.5 * (H + H.T)


def adversarial_instance(gen, fam):
    """Rows nearly dependent on rows already present (angles 1e-13 to 10^-1.5), placed at the minimiser of the rows
    already present and violated or met there by 1e-15 to 0.1 of the size of the row; see the module docstring."""
    n = int(gen.integers(1, 7))
    H = random_metric(n, gen)
    sx = 10 ** gen.uniform(-1, 1)
    x0 = gen.standard_normal(n) * sx
    sv = sx * 10 ** gen.uniform(-3, 0.5)
    k = int(gen.integers(1, min(n, 4) + 1)) if fam != 'box' else int(gen.integers(0, 3))
    A = [unit(gen.standard_normal(n)) for _ in range(k)]
    b = [a @ x0 - gen.uniform(-0.3, 1.) * sv for a in A]

    def near(direction, sign):
        Ac, bc = np.array(A).reshape(-1, n), np.array(b)
        xp = x0
        if len(bc):
            for f in (solve_qp, solve_qp_before_fix):
                r = f(H, -H @ x0, Ac, bc)
                if r.feasible:
                    xp = r.x
                    break
        v = gen.standard_normal(n)
        v = v - (v @ direction) / (direction @ direction) * direction
        th = 10 ** gen.uniform(-13, -1.5)
        a = unit(sign * direction) if np.linalg.norm(v) < 1e-300 else np.cos(th) * unit(sign * direction) + np.sin(th) * unit(v)
        A.append(a)
        b.append(float(a @ xp) - gen.choice([-1., 1.]) * 10 ** gen.uniform(-15, -1) * (float(np.abs(a) @ np.abs(xp)) + 1.))

    def combination(kk):
        J = gen.choice(len(A), kk, replace=False)
        return sum(10 ** gen.uniform(-2, 1) * A[j] for j in J)

    if fam in ('parallel', 'opposite'):
        for _ in range(int(gen.integers(1, 4))):
            near(A[int(gen.integers(len(A)))], 1. if fam == 'parallel' else -1.)
    elif fam in ('cone', 'negative_cone'):
        for _ in range(int(gen.integers(1, 3))):
            if len(A) < 2:
                A.append(unit(gen.standard_normal(n)))
                b.append(A[-1] @ x0 - gen.uniform(-0.3, 1.) * sv)
            c = combination(int(gen.integers(2, min(len(A), 4) + 1)))
            if np.linalg.norm(c) > 1e-12:
                near(unit(c), 1. if fam == 'cone' else -1.)
    elif fam == 'duplicates':
        for _ in range(int(gen.integers(1, 4))):
            i, mode = int(gen.integers(len(A))), int(gen.integers(3))
            if mode == 0:
                A.append(A[i].copy()); b.append(b[i])
            elif mode == 1:
                A.append(A[i].copy()); b.append(b[i] + gen.choice([-1., 1.]) * 10 ** gen.uniform(-16, -6) * (abs(b[i]) + 1))
            else:
                c = 10 ** gen.uniform(-3, 3)
                A.append(c * A[i]); b.append(c * b[i])
        if gen.random() < 0.5:
            near(A[int(gen.integers(len(A)))], gen.choice([-1., 1.]))
    elif fam == 'box':
        lo = x0 + gen.uniform(-1.5, 0.5, n) * sv
        hi = lo + 10 ** gen.uniform(-6, 0.5, n) * sv
        for j in gen.permutation(n)[:min(n, 4)]:
            e = np.zeros(n)
            e[j] = 1.
            A.append(e.copy()); b.append(hi[j])
            A.append(-e); b.append(-lo[j])
        if gen.random() < 0.7:
            near(A[int(gen.integers(len(A)))], gen.choice([-1., 1.]))
    elif fam == 'mixed':
        for _ in range(int(gen.integers(2, 5))):
            if len(A) >= 9:
                break
            op = int(gen.integers(4))
            if op == 0 or len(A) < 2:
                near(A[int(gen.integers(len(A)))], gen.choice([-1., 1.]))
            elif op == 1:
                c = combination(int(gen.integers(2, min(len(A), 4) + 1)))
                if np.linalg.norm(c) > 1e-12:
                    near(unit(c), gen.choice([-1., 1.]))
            elif op == 2:
                i = int(gen.integers(len(A)))
                A.append(A[i].copy()); b.append(b[i])
            else:
                A.append(unit(gen.standard_normal(n))); b.append(A[-1] @ x0 - gen.uniform(-0.3, 1.) * sv)
    elif fam in ('scaled', 'far_minimiser'):
        if fam == 'far_minimiser':
            x0 = x0 * 10 ** gen.uniform(2, 6)
        for _ in range(int(gen.integers(1, 4 if fam == 'scaled' else 3))):
            near(A[int(gen.integers(len(A)))], gen.choice([-1., 1.]))
    elif fam == 'tiny_violations':
        for _ in range(int(gen.integers(1, 5))):
            a = unit(gen.standard_normal(n))
            A.append(a); b.append(a @ x0 - gen.choice([-1., 1.]) * 10 ** gen.uniform(-14, -6) * (np.abs(a) @ np.abs(x0) + 1.))
        if gen.random() < 0.5:
            near(A[int(gen.integers(len(A)))], gen.choice([-1., 1.]))
    elif fam == 'opposite_active':
        a1 = unit(gen.standard_normal(n))
        A.append(a1); b.append(a1 @ x0 - gen.uniform(0.2, 1.) * sv)
        near(a1, -1.)
        if n >= 2:
            c = 10 ** gen.uniform(-2, 1) * A[-1] + 10 ** gen.uniform(-2, 1) * A[-2]
            near(unit(c) if np.linalg.norm(c) > 1e-14 else A[-1], gen.choice([-1., 1.]))
    A, b = np.array(A).reshape(-1, n)[:10], np.array(b)[:10]
    if fam == 'scaled':
        sr = 10 ** gen.uniform(-4, 4, len(b))
        t = 10 ** gen.uniform(-5, 5)
        A, b, x0 = A * sr[:, None], b * sr * t, x0 * t
    return H, x0, A, b


class Reference:
    """Minimiser by the KKT system of candidate active sets in high precision: first the active sets suggested by the
    routines (and subsets of the rows nearly active at their points), then all sets."""

    def __init__(self, H, x0, A, b, digits=60):
        self.digits, self.n, self.m = digits, len(x0), len(b)
        with mp.workdps(digits):
            self.H, self.A, self.b = mp.matrix(H.tolist()), mp.matrix(A.tolist()), mp.matrix(b.tolist())
            self.Hx0 = [mp.fsum(self.H[i, j] * mp.mpf(float(x0[j])) for j in range(self.n)) for i in range(self.n)]
        self.cache = {}

    def kkt(self, S):
        S = tuple(sorted(S))
        if S in self.cache:
            return self.cache[S]
        n, k = self.n, len(S)
        with mp.workdps(self.digits):
            K, rhs = mp.zeros(n + k, n + k), mp.zeros(n + k, 1)
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.H[i, j]
                rhs[i] = self.Hx0[i]
            for a, i in enumerate(S):
                for j in range(n):
                    K[n + a, j] = K[j, n + a] = self.A[i, j]
                rhs[n + a] = self.b[i]
            try:
                sol = mp.lu_solve(K, rhs)
            except ZeroDivisionError:
                self.cache[S] = None
                return None
            tol = mp.mpf(10) ** (-(self.digits // 2))
            lam = [sol[n + a] for a in range(k)]
            ok = all(v >= -tol * max([abs(w) for w in lam] + [mp.mpf(1)]) for v in lam)
            if ok:
                for i in range(self.m):
                    ri = mp.fsum(self.A[i, j] * sol[j] for j in range(n)) - self.b[i]
                    if ri > tol * (mp.fsum(abs(self.A[i, j] * sol[j]) for j in range(n)) + abs(self.b[i]) + mp.mpf(10) ** -300):
                        ok = False
                        break
            out = np.array([float(sol[j]) for j in range(n)]) if ok else None
        self.cache[S] = out
        return out

    def solve(self, hints):
        for S in list(hints) + [c for size in range(0, min(self.m, self.n) + 1) for c in combinations(range(self.m), size)]:
            if len(S) <= self.n:
                x = self.kkt(S)
                if x is not None:
                    return x
        return None


def adversarial(n_per_family):
    """Part 3."""
    gen = np.random.default_rng(20261014)
    out = {}
    for fam in FAMILIES:
        st = {name: dict(rated_feasible=0, rated_empty=0, rated_empty_though_feasible=0, rated_feasible_though_empty=0,
                         backward_error_over_1e8=0, of_which_below_stopping_tolerance=0, worst_backward_error_stationarity=0.,
                         stationarity_over_1e8_relative_to_correction=0, max_iterations=0, iteration_limit_reached=0) for name, _ in ROUTINES}
        n_feas = n_empty = regressions = 0
        for _ in range(n_per_family):
            H, x0, A, b = adversarial_instance(gen, fam)
            res = {name: f(H, -H @ x0, A, b) for name, f in ROUTINES}
            hints = []
            for r in res.values():
                if r.feasible:
                    hints.append(tuple(int(i) for i in np.flatnonzero(r.mu > 0.)))
                    rr = (A @ r.x - b) / (np.abs(A) @ np.abs(r.x) + np.abs(b) + 1e-300)
                    nearly = [int(i) for i in np.flatnonzero(rr > -1e-7)]
                    if len(nearly) <= 8:
                        hints += [c for size in range(0, min(len(nearly), len(x0)) + 1) for c in combinations(nearly, size)]
            xe = Reference(H, x0, A, b).solve(hints)
            n_feas += int(xe is not None)
            n_empty += int(xe is None)
            limit = 20 * (len(b) + len(x0)) + 50
            Hi = np.linalg.inv(H)
            dn = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', A, Hi, A), 1e-300))
            xs = max(1., float(np.sqrt(x0 @ H @ x0)))
            good = {}
            for name, r in res.items():
                s = st[name]
                s['max_iterations'] = max(s['max_iterations'], r.iterations)
                s['iteration_limit_reached'] += int(r.iterations >= limit)
                good[name] = False
                if not r.feasible:
                    s['rated_empty'] += 1
                    s['rated_empty_though_feasible'] += int(xe is not None)
                    continue
                s['rated_feasible'] += 1
                s['rated_feasible_though_empty'] += int(xe is None)
                e_stat, e_rows, e_corr = backward_errors(H, -H @ x0, A, b, r)
                s['worst_backward_error_stationarity'] = max(s['worst_backward_error_stationarity'], e_stat)
                s['stationarity_over_1e8_relative_to_correction'] += int(e_corr > 1e-8)
                if max(e_stat, e_rows) > 1e-8:
                    s['backward_error_over_1e8'] += 1
                    # an inactive row violated by less than the stopping tolerance of the routine (1e-10 max(1, |x0|_H), in
                    # H^-1-normalized units): large relative to |A_i| |x| + |b_i| when |x0|_H is large compared with |x|
                    viol = (A @ r.x - b) / dn
                    act = r.mu > 0.
                    act_res = float(np.max(np.abs(A[act] @ r.x - b[act]) / (np.abs(A[act]) @ np.abs(r.x) + np.abs(b[act])))) if act.any() else 0.
                    if (e_stat <= 1e-8 and act_res <= 1e-8 and np.all(act | (viol <= 1e-10 * xs * 1.0001))
                            and np.max(np.where(act, -np.inf, viol)) > 0.):
                        s['of_which_below_stopping_tolerance'] += 1
                else:
                    good[name] = True
            regressions += int(good['before_fix'] and not good['corrected'])
        out[fam] = dict(instances=n_per_family, feasible_instances=n_feas, empty_instances=n_empty, corrected_worse_than_before_fix=regressions,
                        **{name: st[name] for name, _ in ROUTINES})
        print(f'{fam}: {n_feas} feasible and {n_empty} empty instances (60-digit reference); instances where only the earlier version '
              f'returns a point with a backward error below 1e-8: {regressions}')
        for name, _ in ROUTINES:
            s = st[name]
            print(f'    {name:10s}: rated feasible {s["rated_feasible"]}, empty {s["rated_empty"]} (feasible set rated empty {s["rated_empty_though_feasible"]}, '
                  f'empty set rated feasible {s["rated_feasible_though_empty"]}); backward error above 1e-8 in {s["backward_error_over_1e8"]} '
                  f'(of them rows violated below the stopping tolerance {s["of_which_below_stopping_tolerance"]}); largest stationarity backward '
                  f'error {s["worst_backward_error_stationarity"]:.1e}, relative to the correction above 1e-8 in '
                  f'{s["stationarity_over_1e8_relative_to_correction"]}; at most {s["max_iterations"]} iterations, iteration limit reached '
                  f'{s["iteration_limit_reached"]}', flush=True)
    return out


def main(n_inst=20000, n_small=2000, n_adversarial=300):
    out = dict(nearly_dependent_rows=nearly_dependent_rows(n_inst), small_violations=small_violations(n_small),
               adversarial=adversarial(n_adversarial))
    json.dump(out, open(paths.NUM / 't13_results.json', 'w'), indent=1)
    # The corrected routine must return the minimiser (part 1) or a point with a backward error below 1e-8 (part 2) wherever it
    # returns a point, must never reach the iteration limit, must not rate an empty set feasible (parts 1-2), must have
    # stationarity backward errors below 1e-8 (part 3), and must not fail on an instance on which the earlier version
    # succeeds (all parts); in part 1 the minimisers of the sets it rates empty lie far from x0.
    bad = [f'eps = {e}' for e, r in out['nearly_dependent_rows'].items()
           if r['corrected']['off_1e7'] or r['corrected_worse_than_before_fix']
           or any(d < 1e3 for d in r['corrected']['reported_infeasible_minimiser_distance'])]
    bad += [f for f, r in out['small_violations'].items()
            if r['corrected']['backward_error_over_1e8'] or r['corrected']['iteration_limit_reached'] or r['corrected']['reported_feasible_though_empty']
            or r['corrected_worse_than_before_fix']]
    bad += [f for f, r in out['adversarial'].items()
            if r['corrected']['iteration_limit_reached'] or r['corrected_worse_than_before_fix'] or r['corrected']['worst_backward_error_stationarity'] > 1e-8]
    if bad:
        raise SystemExit(f'corrected routine fails the check for: {", ".join(bad)}')
    print('corrected routine: all checks passed')


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:4]))
