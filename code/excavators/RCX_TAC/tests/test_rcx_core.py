"""Machine-independent tests of the extension (rcx): QP solver against independent solvers, signed-distance primitives,
coordinated effort filter and its fallbacks, required-motion shaping, actuator response fractions, plant-side row
monitor, one-step row prediction and the sampled-data safety statement on a one-degree-of-freedom system whose
assumptions hold by construction.

    python tests/test_rcx_core.py          or          python -m pytest tests/test_rcx_core.py -q

The test functions are those of the earlier single-machine release that use no machine, and three tests of the parts
of the support interface that need no machine (contact rows, filter metric on redundant channels, hull of the contact
points); the tests that need a compiled machine are in bench/tests_machine.py and run with  python run.py <machine> tests.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcx.actuators import Channel, response_fractions
from rcx.barriers import joint_angle
from rcx.filter import (FilterOptions, barrier_rows, delivered_rows, filter_forces, filter_uncoordinated, row_margins,
                        shape_required_velocity)
from rcx.geometry import Obstacle
from rcx.monitor import RowMonitor
from rcx.qp import solve_lexicographic, solve_qp, solve_relaxed

RNG = np.random.default_rng(11)


def expm_so3(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def test_qp_against_independent_solvers():
    from scipy.optimize import linprog, minimize
    rng = np.random.default_rng(0)
    disagreements, worst_gap, worst_primal = 0, 0., 0.
    for trial in range(250):
        n, m = rng.integers(2, 7), rng.integers(1, 30)
        Lm = rng.standard_normal((n, n))
        H = (Lm @ Lm.T + 0.05 * np.eye(n)) * 10 ** rng.uniform(-3, 3)
        g = rng.standard_normal(n) * 10 ** rng.uniform(-2, 2)
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m) + rng.uniform(-0.5, 2.0)
        if trial % 3 == 0:                                   # duplicated rows and box bounds (degenerate cases)
            A = np.vstack([A, A[:2], np.eye(n), -np.eye(n)])
            b = np.r_[b, b[:2], np.full(2 * n, 3.)]
            m = len(b)
        res = solve_qp(H, g, A, b)
        margin = -linprog(np.r_[np.zeros(n), -1.], A_ub=np.c_[A, np.ones(m)], b_ub=b, bounds=[(None, None)] * n + [(None, 10.)],
                          method='highs').fun
        if abs(margin) < 1e-7:
            continue                                         # feasibility undecidable at solver tolerance
        if (not res.feasible) != (margin < 0):
            disagreements += 1
            continue
        if not res.feasible:
            continue
        x0 = linprog(np.zeros(n), A_ub=A, b_ub=b, bounds=[(None, None)] * n, method='highs').x
        ref = minimize(lambda x: 0.5 * x @ H @ x + g @ x, x0, jac=lambda x: H @ x + g,
                       constraints=[dict(type='ineq', fun=lambda x: b - A @ x, jac=lambda x: -A)], method='SLSQP',
                       options=dict(maxiter=1000, ftol=1e-15))
        f1 = 0.5 * res.x @ H @ res.x + g @ res.x
        if np.max(A @ ref.x - b) < 1e-9:
            worst_gap = max(worst_gap, (f1 - ref.fun) / max(1., abs(ref.fun)))
        worst_primal = max(worst_primal, res.kkt['primal'])
    assert disagreements == 0 and worst_gap < 1e-6 and worst_primal < 1e-7


def test_relaxed_qp_keeps_hard_rows():
    rng = np.random.default_rng(1)
    done = 0
    for _ in range(200):
        n, m = 4, 12
        Lm = rng.standard_normal((n, n))
        H, g = Lm @ Lm.T + 0.05 * np.eye(n), rng.standard_normal(n)
        A, b = rng.standard_normal((m, n)), rng.standard_normal(m) - 1.0
        if solve_qp(H, g, A, b).feasible:
            continue
        Ab, bb = np.vstack([np.eye(n), -np.eye(n)]), np.full(2 * n, 5.)
        r2, x, s = solve_relaxed(H, g, Ab, bb, A, b, 1e6)
        assert r2.feasible and np.all(Ab @ x <= bb + 1e-8) and np.all(A @ x - s <= b + 1e-7) and s.min() >= -1e-9
        done += 1
    assert done > 20


def test_signed_distance_primitives():
    recs = [dict(id='p', type='plane', normal=[0.2, -0.3, 1.], offset=0.4),
            dict(id='b', type='box', center=[0.3, -0.2, 0.5], half=[0.6, 0.4, 0.3], rpy=[0.2, -0.3, 0.5]),
            dict(id='c', type='capsule', a=[-1., 0.2, 0.3], b=[1.5, -0.4, 0.8], radius=0.25),
            dict(id='s', type='sphere', center=[0.1, 0.2, -0.3], radius=0.5)]
    e = 1e-6
    for rec in recs:
        obs = Obstacle(rec)
        X = RNG.uniform(-2.5, 2.5, (300, 3))
        d, n, H = obs.evaluate(X)
        keep = d > 0.05                                           # outside the primitive
        X, d, n, H = X[keep], d[keep], n[keep], H[keep]
        g_fd = np.stack([(obs.evaluate(X + e * np.eye(3)[k])[0] - obs.evaluate(X - e * np.eye(3)[k])[0]) / (2 * e) for k in range(3)], axis=1)
        assert np.max(np.abs(g_fd - n)) < 1e-6
        assert np.max(np.abs(np.linalg.norm(n, axis=1) - 1.)) < 1e-9
        if rec['type'] == 'plane':                               # allowed side normal . x >= offset, for a normal of any length
            nrm = np.asarray(rec['normal'], float)
            assert np.max(np.abs(d - (X @ nrm - rec['offset']) / np.linalg.norm(nrm))) < 1e-12
        H_fd = np.stack([(obs.evaluate(X + e * np.eye(3)[k])[1] - obs.evaluate(X - e * np.eye(3)[k])[1]) / (2 * e) for k in range(3)], axis=2)
        smooth = np.max(np.abs(H_fd - H), axis=(1, 2)) < 1e-3     # the Hessian is piecewise continuous: skip the seams
        assert smooth.mean() > 0.97


def test_filter_gives_virtual_constraint_forces():
    """Kinetic metric, equal response fractions, inactive bounds:  C Theta (phi - phi_r)  lies in span{J_i^T}."""
    rng = np.random.default_rng(3)
    opt = FilterOptions()
    for _ in range(20):
        Lm = rng.standard_normal((4, 4))
        Mr = Lm @ Lm.T + 0.5 * np.eye(4)
        C = rng.standard_normal((4, 4)) + 2. * np.eye(4)
        J = rng.standard_normal((2, 4))
        hdot, gamma, h = rng.standard_normal(2), rng.standard_normal(2), rng.uniform(0.05, 0.3, 2)
        a1, a2 = np.full(2, 6.), np.full(2, 16.)
        a, drift, b, MiJ = barrier_rows(Mr, np.zeros(4), C, J, hdot, gamma, h, a1, a2, np.zeros(4))
        theta = np.full(4, 0.33)
        f = rng.standard_normal(4)
        c, r0, psi2, need, hdd = delivered_rows(a, drift, b, hdot, a1, a2, f, theta, np.full(4, 0.18), np.full(4, 0.06), 0.01)
        phi_r = f + rng.standard_normal(4) * 5.
        big = np.full(4, 1e6)
        phi, rec = filter_forces(opt, phi_r, Mr, C, c, r0 + 1.0, -big, big, np.ones(4), theta)
        assert rec['feasible']
        if not rec['active'].any():
            continue
        dev = C @ (theta * (phi - phi_r))
        coef = np.linalg.lstsq(J.T, dev, rcond=None)[0]
        assert np.max(np.abs(J.T @ coef - dev)) < 1e-6 * max(1., np.max(np.abs(dev)))
        assert np.all(c @ phi >= r0 + 1.0 - 1e-7)


def test_infeasible_effort_filter_is_lexicographic_and_scale_free():
    """Infeasible hard problem (a row that the effort bounds cannot meet).  The fallback keeps the bounds, attains the
    smallest sum of squared violations (each relative to the change of its row over the effort range) and, among the
    commands that do, the one closest to the nominal efforts; the result does not depend on the scale of the inertia
    (i.e. of the filter metric), however ill-conditioned the metric is."""
    from scipy.optimize import minimize
    opt = FilterOptions()
    rng = np.random.default_rng(4)
    checked_cost = 0
    for trial in range(40):
        Lm = rng.standard_normal((3, 3))
        Mr = (Lm @ Lm.T + 0.5 * np.eye(3)) * 10 ** rng.uniform(-2, 4)
        C = (rng.standard_normal((3, 3)) + 2. * np.eye(3)) * 10 ** rng.uniform(-1, 1)
        scale = 10 ** rng.uniform(2, 6, 3)
        lo, hi = -scale, scale
        theta = np.full(3, 0.33)
        c = rng.standard_normal((2, 3)) / scale[None, :]
        phi_r = rng.uniform(-0.5, 0.5, 3) * scale
        reach = np.abs(c) @ scale                                   # largest value of each row inside the bounds
        rhs = np.array([1.5 * reach[0], c[1] @ phi_r + 0.3 * (reach[1] - abs(c[1] @ phi_r))])
        phi, rec = filter_forces(opt, phi_r, Mr, C, c, rhs, lo, hi, scale, theta)
        assert not rec['feasible'] and rec['numerically_ok']
        assert np.all(phi >= lo * (1. + 1e-9)) and np.all(phi <= hi * (1. + 1e-9))
        x, x_r = phi / scale, phi_r / scale
        A = c * scale[None, :]
        rho = 2. * reach                                            # change of each row over the whole effort range

        def V(y):                                                   # stage 1: violation measure and its gradient
            v = np.maximum(0., rhs - A @ y) / rho
            return 0.5 * v @ v, -(A / rho[:, None]).T @ v
        best = min((minimize(V, y0, jac=True, bounds=[(-1., 1.)] * 3, method='L-BFGS-B', options=dict(ftol=1e-16, gtol=1e-13, maxiter=5000))
                    for y0 in (x, np.zeros(3), np.sign(A[0]), -np.sign(A[0]))), key=lambda r_: r_.fun)
        assert V(x)[0] <= best.fun * (1. + 1e-4) + 1e-12
        assert np.max(np.abs(rec['slack'] - np.maximum(0., rhs - c @ phi))) < 1e-9 * np.max(rho)
        # stage 2: among the commands with these violations, the closest to the nominal one in the filter metric
        H = (C.T @ np.linalg.solve(Mr, C)) * np.outer(scale * theta, scale * theta)
        H = H / np.max(np.abs(H))
        cost = lambda y: 0.5 * (y - x_r) @ H @ (y - x_r)
        v_allowed = rec['slack'] + 1e-9 * rho
        ref = minimize(cost, x, jac=lambda y: H @ (y - x_r), bounds=[(-1., 1.)] * 3, method='SLSQP',
                       constraints=[dict(type='ineq', fun=lambda y: A @ y - rhs + v_allowed, jac=lambda y: A)], options=dict(maxiter=500, ftol=1e-15))
        if ref.success and np.all(A @ ref.x - rhs + v_allowed >= -1e-9 * rho):
            assert cost(x) <= ref.fun * (1. + 1e-4) + 1e-10
            checked_cost += 1
        phi2, rec2 = filter_forces(opt, phi_r, 1e3 * Mr, C, c, rhs, lo, hi, scale, theta)
        assert np.max(np.abs(phi2 - phi) / scale) < 1e-5
    assert checked_cost > 20
    # rows on different channels do not compete: the reachable one is met
    Mr, C, scale, theta = np.diag([2., 3., 5.]), np.eye(3), np.array([10., 20., 30.]), np.full(3, 0.5)
    c = np.array([[1., 0., 0.], [0., 1., 0.]])
    phi, rec = filter_forces(opt, np.zeros(3), Mr, C, c, np.array([15., 8.]), -scale, scale, scale, theta)
    assert not rec['feasible'] and abs(phi[0] - 10.) < 1e-4 and abs(phi[1] - 8.) < 1e-4 and abs(phi[2]) < 1e-9
    assert abs(rec['slack'][0] - 5.) < 1e-3 and rec['slack'][1] < 1e-4


def test_rows_without_authority_and_fallbacks_of_the_required_motion():
    W = np.diag([1., 4.])
    J = np.array([[1., 0.], [0., 0.]])                             # second row: no velocity changes it
    u, info = shape_required_velocity(np.zeros(2), W, J, np.array([0.5, -1.]))
    assert not info['relaxed'] and abs(u[0] - 0.5) < 1e-9 and abs(u[1]) < 1e-12 and not info['no_authority'].any()
    u, info = shape_required_velocity(np.zeros(2), W, J, np.array([0.5, 1.]))      # violated and without authority: reported only
    assert info['level'] == 0 and abs(u[0] - 0.5) < 1e-9 and info['no_authority'].tolist() == [False, True]
    assert info['active'].tolist() == [True, False]
    # conflicting barrier rows without any capability row: least-squares relaxation, not the nominal velocity
    u, info = shape_required_velocity(np.array([0.3, 0.]), W, np.array([[1., 0.], [-1., 0.]]), np.array([1., 1.]))
    assert info['relaxed'] and info['level'] == 1 and abs(u[0]) < 1e-5 and abs(info['slack'] - 1.) < 1e-5
    # effort filter: a row on which no effort acts is not passed to the problem
    Mr, C, S = np.diag([2., 3.]), np.eye(2), np.array([10., 10.])
    c = np.array([[0., 0.], [1., 0.]])
    phi, rec = filter_forces(FilterOptions(), np.zeros(2), Mr, C, c, np.array([1., 4.]), -S, S, S, np.ones(2))
    assert rec['feasible'] and rec['no_authority'].tolist() == [True, False] and abs(phi[0] - 4.) < 1e-9
    # per-channel ablation: the same row must not drive the channels to their bounds
    out = filter_uncoordinated(np.array([1., -2.]), c, np.array([-1., 4.]), -S, S, S)
    assert np.allclose(out, [4., -2.])
    out = filter_uncoordinated(np.array([1., -2.]), c, np.array([1., 4.]), -S, S, S)
    assert np.allclose(out, [4., -2.])


def test_row_monitor_uses_same_instant_derivatives():
    """Plant-side monitor on an analytic h(t): psi1, psi2 and the weighted sample mean of psi2 against closed forms."""
    a1, a2, dt, Ts = 6., 16., 1e-3, 0.01
    h = lambda t: 0.4 + 0.3 * np.sin(2.3 * t) + 0.05 * t * t
    hd = lambda t: 0.69 * np.cos(2.3 * t) + 0.1 * t
    hdd = lambda t: -1.587 * np.sin(2.3 * t) + 0.1

    class Barriers:
        R, alpha1, alpha2 = 1, np.array([a1]), np.array([a2])
        ids, cls = ['row'], np.array(['test'])
        pair_rows, class_rows = {'row': np.array([0])}, {'test': np.array([0])}

        @staticmethod
        def values(R, p, qt):
            return np.array([h(float(qt[0]))])

    class Compiled:
        barriers, Ts = Barriers(), 0.01

    mon = RowMonitor(Compiled(), dt)
    N = 2000
    for i in range(N + 1):
        mon.update_values(i * dt, None, None, np.array([i * dt]))
    t = np.arange(N) * dt                                         # the last step has no centred difference
    psi1 = hd(t) + a1 * h(t)
    psi2 = hdd(t) + (a1 + a2) * hd(t) + a1 * a2 * h(t)
    out = mon.summary()['pairs']['row']
    assert abs(out['h_min'] - np.min(h(np.arange(N + 1) * dt))) < 1e-15
    assert abs(out['psi1_min'] - psi1.min()) < 1e-5 and abs(out['row_function_physics_rate_min'] - psi2.min()) < 1e-4
    tk = np.arange(0, N - 20 + 1, 10) * dt                         # control samples whose following sample is recorded too
    p0, p1 = hd(tk) + a1 * h(tk), hd(tk + Ts) + a1 * h(tk + Ts)
    mean = a2 * (p1 - np.exp(-a2 * Ts) * p0) / (1. - np.exp(-a2 * Ts))
    assert abs(out['row_function_sample_mean_min'] - mean.min()) < 2e-3
    # the closed form used above is the weighted integral of psi2 over the sample
    sgrid = np.linspace(0., Ts, 2001)
    w = np.exp(-a2 * (Ts - sgrid))
    f2 = lambda tt: hdd(tt) + (a1 + a2) * hd(tt) + a1 * a2 * h(tt)
    assert abs(np.trapezoid(w * f2(tk[7] + sgrid), sgrid) / np.trapezoid(w, sgrid) - mean[7]) < 1e-6
    # a quadratic h whose psi1 and psi2 increase with time: all minima belong to step 0 and to the first control sample,
    # where the differences are exact, so the first window, its alignment and the time stamps are tested at round-off
    h = lambda t: 0.5 + 0.2 * t + 0.3 * t * t
    hd = lambda t: 0.2 + 0.6 * t
    mon = RowMonitor(Compiled(), dt)
    for i in range(101):
        mon.update_values(i * dt, None, None, np.array([i * dt]))
    out = mon.summary()['pairs']['row']
    psi1_0 = hd(0.) + a1 * h(0.)
    psi2_0 = 0.6 + (a1 + a2) * hd(0.) + a1 * a2 * h(0.)
    mean_0 = a2 * ((hd(Ts) + a1 * h(Ts)) - np.exp(-a2 * Ts) * psi1_0) / (1. - np.exp(-a2 * Ts))
    assert abs(out['psi1_min'] - psi1_0) < 1e-9 and out['psi1_min_at_s'] == 0. and abs(out['row_function_physics_rate_min'] - psi2_0) < 1e-7
    assert abs(out['row_function_sample_mean_min'] - mean_0) < 1e-7


def test_filter_multipliers_split_the_command_deviation():
    """H_phi (phi - phi_r) = sum_i mu_i c_i - (mu_hi - mu_lo) / S  for the feasible problem and for the two-stage fallback
    (the runtime splits the power of the command deviation into rows and bounds with these multipliers); the arrays of
    the record are indexed by the original rows, also with a row without authority and a row that the bounds make
    redundant in the problem."""
    opt = FilterOptions()
    rng = np.random.default_rng(8)
    relaxed = feasible = 0
    for trial in range(60):
        Lm = rng.standard_normal((3, 3))
        Mr = (Lm @ Lm.T + 0.5 * np.eye(3)) * 10 ** rng.uniform(-1, 3)
        C = rng.standard_normal((3, 3)) + 2. * np.eye(3)
        scale = 10 ** rng.uniform(2, 5, 3)
        lo, hi = -scale, scale
        theta = np.full(3, 0.33)
        c = rng.standard_normal((4, 3)) / scale[None, :]
        c[1] = 0.                                                   # a row on which no effort acts
        phi_r = rng.uniform(-0.9, 0.9, 3) * scale
        reach = np.abs(c) @ scale
        rhs = np.array([c[0] @ phi_r + rng.uniform(0.2, 1.4) * (reach[0] - c[0] @ phi_r),     # binding; unreachable beyond factor 1
                        rng.choice([-1., 1.]),                                              # dead row, satisfied or violated
                        -1.5 * reach[2],                                                    # cannot be violated inside the bounds
                        c[3] @ phi_r + 0.1 * (reach[3] - abs(c[3] @ phi_r))])
        phi, rec = filter_forces(opt, phi_r, Mr, C, c, rhs, lo, hi, scale, theta)
        assert rec['numerically_ok'] and rec['H'] is not None
        assert rec['no_authority'].tolist() == [False, bool(rhs[1] > 0.), False, False]
        assert not rec['active'][1] and not rec['active'][2] and rec['mu'][1] == 0. and rec['mu'][2] == 0. and rec['slack'][1] == 0.
        lhs = rec['H'] @ (phi - phi_r)
        rhs_kkt = rec['mu'] @ c - (rec['bound_mu'][:3] - rec['bound_mu'][3:]) / scale
        assert np.max(np.abs(lhs - rhs_kkt)) < 1e-6 * max(np.max(np.abs(lhs)), 1e-12)
        assert np.all(rec['mu'] >= 0.) and np.all(rec['bound_mu'] >= 0.)
        if rec['feasible']:
            feasible += 1
            assert np.all(c[[0, 2, 3]] @ phi >= rhs[[0, 2, 3]] - 1e-7 * reach[[0, 2, 3]]) and np.all(rec['slack'] == 0.)
        else:
            relaxed += 1
            assert rec['slack'][2] == 0. and np.max(np.abs(rec['slack'] - np.where([True, False, True, True], np.maximum(0., rhs - c @ phi), 0.))) == 0.
    assert relaxed > 10 and feasible > 10
    # hard rows that are inconsistent on their own cannot be repaired by relaxing the soft ones
    res, x, sl = solve_lexicographic(np.eye(2), np.zeros(2), np.array([[1., 0.], [-1., 0.]]), np.array([-1., -1.]), np.array([[0., 1.]]), np.array([-3.]))
    assert not res.feasible and x is None
    res, x, sl = solve_lexicographic(np.eye(2), np.zeros(2), np.array([[1., 0.]]), np.array([1.]), np.array([[-1., 0.], [0., 1.]]), np.array([-2., 5.]))
    assert res.feasible and res.ok and abs(x[0] - 1.) < 1e-5 and abs(x[1]) < 1e-9 and abs(sl[0] - 1.) < 1e-5 and sl[1] == 0.


def simulate_toy(margins, seed, steps=1500, Ts=0.01, tau=0.025, sub=40):
    """One mass (1000 kg) with a first-order force loop driven towards a limit  x <= 1  by a nominal PD law.

    The force-loop error (|f - f_model| <= e over every sample) and the load (|w| <= w_bar, unknown to the controller)
    are adversarial: both act towards the limit and switch at random instants.  Returns the minima of h, psi1 and psi2
    over all integration substeps."""
    rng = np.random.default_rng(seed)
    m, e, w_bar, fmax = 1000., 300., 400., 2e4
    a1, a2 = np.array([6.]), np.array([16.])
    opt = FilterOptions(margin_effectiveness=False) if margins else FilterOptions(margin_force=False, margin_load=False, margin_sampling=False,
                                                                                    margin_effectiveness=False)
    ch = Channel(dict(id='c', type='rotary_torque', ports=['p'], torque_limit_Nm=fmax, speed_limit_rad_s=10., response_time=tau), [0])
    theta, theta_bar, theta2 = (np.array([v]) for v in ch.response(Ts))
    Mr, C, J = np.array([[m]]), np.array([[1.]]), np.array([[-1.]])          # h = 1 - x
    x, v, f = 0., 0., 0.
    dt = Ts / sub
    out = dict(h=np.inf, psi1=np.inf, psi2=np.inf)
    eps_sign, w_sign = 1., 1.
    for k in range(steps):
        if rng.random() < 0.05:
            eps_sign = rng.choice([-1., 1., 1.])
        if rng.random() < 0.05:
            w_sign = rng.choice([-1., 1., 1.])
        h, hdot = np.array([1. - x]), np.array([-v])
        a, drift, b, MiJ = barrier_rows(Mr, np.zeros(1), C, J, hdot, np.zeros(1), h, a1, a2, np.zeros(1))
        fm = np.array([f])
        c, r0, psi2, need, hdd = delivered_rows(a, drift, b, hdot, a1, a2, fm, theta, theta_bar, theta2, Ts)
        margin, comp = row_margins(opt, a, MiJ, J, need, w_bar / np.sqrt(m), np.array([e]), Ts, 0., 0., a1, a2)
        phi_r = np.array([np.clip(m * (30. * (1.6 - x) - 8. * v), -fmax, fmax)])     # reference beyond the limit
        phi, rec = filter_forces(opt, phi_r, Mr, C, c, r0 + margin, np.array([-fmax]), np.array([fmax]), np.array([fmax]), theta)
        assert rec['feasible']
        f0 = f
        for i in range(1, sub + 1):
            s = i * dt
            f_model = phi[0] + (f0 - phi[0]) * np.exp(-s / tau)
            f = f_model + eps_sign * e                          # force-loop error inside its bound, towards the limit
            w = w_sign * w_bar                                  # unknown load inside its bound
            acc = (f + w) / m
            x += dt * v + 0.5 * dt * dt * acc
            v += dt * acc
            hh, hd, hdd_true = 1. - x, -v, -acc
            out['h'] = min(out['h'], hh)
            out['psi1'] = min(out['psi1'], hd + a1[0] * hh)
            out['psi2'] = min(out['psi2'], hdd_true + (a1[0] + a2[0]) * hd + a1[0] * a2[0] * hh)
    return out


def test_sampled_data_safety_statement_on_a_toy_system():
    worst_with, worst_without = dict(h=np.inf, psi1=np.inf, psi2=np.inf), dict(h=np.inf, psi1=np.inf, psi2=np.inf)
    for seed in range(6):
        r = simulate_toy(True, seed)
        for k in r:
            worst_with[k] = min(worst_with[k], r[k])
        r = simulate_toy(False, seed)
        for k in r:
            worst_without[k] = min(worst_without[k], r[k])
    # with the generated margins the row function stays non-negative at all times (not only at the samples)
    assert worst_with['psi2'] >= -1e-6 and worst_with['psi1'] >= 0. and worst_with['h'] >= 0.
    # without them the same disturbances drive the row function negative
    assert worst_without['psi2'] < -0.1


def test_one_step_row_prediction_is_exact_for_frozen_coefficients():
    rng = np.random.default_rng(5)
    for _ in range(20):
        Ts = rng.choice([0.01, 0.02, 0.04])
        taus = np.array([0.05, 0.025, 0.025])
        a = rng.standard_normal((1, 3))
        d, h0, hd0 = rng.standard_normal(), rng.uniform(0.1, 0.5), rng.standard_normal()
        a1, a2 = np.array([rng.uniform(1, 8)]), np.array([rng.uniform(6, 20)])
        f, phi = rng.standard_normal(3), rng.standard_normal(3) * 3.
        th = -np.expm1(-Ts / taus)
        r = taus / Ts
        b = -np.array([d]) - (a1 + a2) * hd0 - a1 * a2 * h0
        c, r0, psi2, need, hdd = delivered_rows(a, np.array([d]), b, np.array([hd0]), a1, a2, f, th, 1. - r * th, 0.5 - r + r * r * th, Ts)
        pred = float((c @ (phi - f) - need)[0])
        # reference: fine integration of  h_ddot = a f(t) + d  with first-order force loops
        n = 20000
        s = np.linspace(0., Ts, n + 1)
        ft = phi[None, :] + (f - phi)[None, :] * np.exp(-s[:, None] / taus[None, :])
        acc = ft @ a[0] + d
        vel = hd0 + np.concatenate([[0.], np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(s))])
        pos = h0 + np.concatenate([[0.], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * np.diff(s))])
        exact = acc[-1] + (a1[0] + a2[0]) * vel[-1] + a1[0] * a2[0] * pos[-1]
        assert abs(pred - exact) < 1e-6 * max(1., abs(exact))


def test_required_motion_shaping_precedence():
    """Capability rows stay hard when the barrier rows are inconsistent with them."""
    W = np.eye(2)
    J, rhs = np.array([[1., 0.]]), np.array([2.])                # barrier row asks  u_0 >= 2
    A_cap, b_cap = np.vstack([np.eye(2), -np.eye(2)]), np.ones(4)  # capability |u| <= 1
    u, info = shape_required_velocity(np.zeros(2), W, J, rhs, A_cap, b_cap, n_speed=4)
    assert info['relaxed'] and info['level'] == 1 and abs(u[0] - 1.) < 1e-5 and abs(u[1]) < 1e-9
    u, info = shape_required_velocity(np.zeros(2), W, J, np.array([0.5]), A_cap, b_cap, n_speed=4)
    assert not info['relaxed'] and abs(u[0] - 0.5) < 1e-9
    # effort rows (after the speed rows) stay hard when they conflict with the speed rows: level 2
    A_two = np.array([[1., 0.], [-1., 0.]])                         # speed row  u_0 <= 1,  effort row  u_0 >= 2
    u, info = shape_required_velocity(np.zeros(2), W, np.zeros((0, 2)), np.zeros(0), A_two, np.array([1., -2.]), n_speed=1)
    assert info['relaxed'] and info['level'] == 2 and abs(u[0] - 2.) < 1e-5 and info['speed_active'] and info['effort_active']
    assert abs(info['slack'] - 1.) < 1e-5
    # the same with a barrier row  u_0 <= 1  that conflicts with the effort row as well: it shares the violation with the speed row
    u, info = shape_required_velocity(np.zeros(2), W, np.array([[-1., 0.]]), np.array([-1.]), A_two, np.array([1., -2.]), n_speed=1)
    assert info['level'] == 2 and abs(u[0] - 2.) < 1e-5 and info['active'].tolist() == [True]

def test_response_fractions_and_rotary_response():
    """Response fractions of a first-order effort loop against 50-digit arithmetic (also across the switch to the series
    branch), and the response of a channel without a declared response time."""
    assert Channel(dict(id='x', type='rotary_torque', ports=['swing'], torque_limit_Nm=1e5, speed_limit_rad_s=1.), [0]).response(0.01) == (1., 1., 0.5)
    import mpmath as mp
    mp.mp.dps = 50
    for x in (1e-6, 1e-3, 0.03, 0.0999, 0.1001, 0.4, 3.):
        th, thb, th2 = response_fractions(1., x)
        X = mp.mpf(x)
        e_th = 1 - mp.e ** (-X)
        assert abs(th - float(e_th)) < 1e-15 and abs(thb - float(1 - e_th / X)) < 1e-13 * x and abs(th2 - float(mp.mpf(1) / 2 - 1 / X + e_th / X ** 2)) < 1e-13 * x


def test_degenerate_geometry():
    """Query points on the axis of a capsule, at the centre of a sphere, and a capsule of zero length: finite values and
    unit normals."""
    for rec, pts in ((dict(id='c', type='capsule', a=[0., 0., 0.], b=[1., 0., 0.], radius=0.1), [[0.5, 0., 0.], [0.2, 0., 0.]]),
                     (dict(id='s', type='sphere', center=[0.1, 0.2, 0.3], radius=0.2), [[0.1, 0.2, 0.3]]),
                     (dict(id='z', type='capsule', a=[0.3, 0., 0.], b=[0.3, 0., 0.], radius=0.1), [[0.3, 0., 0.], [0.5, 0.1, 0.]])):
        d, n, H = Obstacle(rec).evaluate(np.array(pts, float))
        assert np.all(np.isfinite(d)) and np.all(np.isfinite(n)) and np.all(np.isfinite(H))
        assert np.max(np.abs(np.linalg.norm(n, axis=1) - 1.)) < 1e-12
        assert abs(d[0] + rec['radius']) < 1e-12 and np.max(np.abs(H[0])) == 0.


def test_cut_joint_angle_branch():
    """The angle of a cut joint is taken on the branch within pi of the given centre (its seed value)."""
    axis = np.array([0., -1., 0.])
    for angle, center in ((3.5, 3.0), (-3.5, -3.0), (0.4, 0.), (3.0, 0.), (-2.9, 3.0)):
        got = joint_angle(np.eye(3), expm_so3(angle * axis), axis, center)
        assert abs((got - angle + np.pi) % (2 * np.pi) - np.pi) < 1e-12 and abs(got - center) <= np.pi + 1e-12
    assert abs(joint_angle(np.eye(3), expm_so3(3.5 * axis), axis, 3.0) - 3.5) < 1e-12


def test_hull_order_of_contact_points():
    """Convex hull of the contact points, counter-clockwise: interior and collinear points are no vertices, every point lies
    on the inner side of every edge, and the normals that the support rows form from the edges point inwards."""
    from rcx.barriers import hull_order
    P = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.], [0.2, 0.1], [0., 1.], [1., 0.]])      # square, interior point, two on edges
    order = hull_order(P)
    assert sorted(order) == [0, 1, 2, 3]
    V = P[order]
    area = 0.5 * sum(V[i][0] * V[(i + 1) % 4][1] - V[(i + 1) % 4][0] * V[i][1] for i in range(4))
    assert abs(area - 4.) < 1e-12                                # positive area: counter-clockwise
    rng = np.random.default_rng(5)
    for _ in range(20):
        Q = rng.normal(size=(7, 2))
        W = Q[hull_order(Q)]
        k = len(W)
        assert k >= 3
        for i in range(k):
            e = W[(i + 1) % k] - W[i]
            inward = np.array([-e[1], e[0]])
            assert np.all((Q - W[i]) @ inward >= -1e-12) and inward @ (Q.mean(axis=0) - W[i]) > 0.


def test_contact_rows_keep_the_forces_inside_the_friction_cone():
    """rcx.filter.contact_rows on a random affine force map lam = A f + b: commands that satisfy the rows give, for every
    force-loop error inside its bound, contact forces with a normal part above the declared minimum plus the reserve and
    tangential parts inside the pyramid, hence inside the friction cone of the declared coefficient."""
    from types import SimpleNamespace
    from rcx.filter import contact_rows
    rng = np.random.default_rng(3)
    m, n = 7, 3
    sup = SimpleNamespace(contacts=[dict(rows=np.arange(3 * k, 3 * k + 3), friction=0.4 + 0.2 * k, normal_min=500. * k) for k in range(n)],
                          reserve=150.)
    checked = 0
    for _ in range(20):
        A = rng.normal(size=(3 * n, m))
        f = 100. * rng.normal(size=m)
        b = -A @ f + np.tile([0., 0., 5000.], n) + 300. * rng.normal(size=3 * n)       # loaded contacts at the measured efforts
        theta, e = rng.uniform(0.2, 1., m), rng.uniform(0., 20., m)
        cc, rc, margin = contact_rows(sup, A, b, f, theta, e)
        assert cc.shape == (5 * n, m) and np.all(margin >= sup.reserve)
        for _ in range(100):
            phi = f + 300. * rng.normal(size=m)
            if not np.all(cc @ phi >= rc):
                continue
            checked += 1
            delivered = f + theta * (phi - f) + rng.uniform(-1., 1., m) * e
            for c, (lx, ly, lz) in zip(sup.contacts, (A @ delivered + b).reshape(-1, 3)):
                kf = c['friction'] / np.sqrt(2.)
                assert lz >= c['normal_min'] + sup.reserve - 1e-9
                assert abs(lx) <= kf * lz - sup.reserve + 1e-9 and abs(ly) <= kf * lz - sup.reserve + 1e-9
                assert np.hypot(lx, ly) <= c['friction'] * lz + 1e-9
    assert checked > 50, checked


def test_filter_metric_on_redundant_channels():
    """More channels than coordinates: the kinetic filter metric is completed on the null space of C, so the effort problem
    is strictly convex, and a row that internal efforts alone can change is met by them, without a change of the reduced
    effort that the delivered command produces."""
    rng = np.random.default_rng(8)
    d, m = 3, 5
    for _ in range(10):
        C = rng.normal(size=(d, m))
        X = rng.normal(size=(d, d))
        Mr = X @ X.T + d * np.eye(d)
        S, theta = rng.uniform(0.5, 2., m), np.full(m, 0.6)
        phi_r = rng.normal(size=m)
        lo, hi = np.full(m, -1e6), np.full(m, 1e6)
        Z = np.linalg.svd(C * (S * theta)[None, :])[2][d:].T        # internal efforts in the scaled command coordinates
        c = (Z[:, 0] / S)[None, :]
        rhs = np.array([float(c[0] @ phi_r) + 1.0])
        phi, rec = filter_forces(FilterOptions(), phi_r, Mr, C, c, rhs, lo, hi, S, theta)
        assert rec['feasible'] and rec['active'][0] and abs(float(c[0] @ phi) - rhs[0]) < 1e-8
        assert np.min(np.linalg.eigvalsh(rec['H'])) > 0.
        assert np.max(np.abs(C @ (theta * (phi - phi_r)))) < 1e-8
        # a general row changes the reduced effort as well, and is met
        c2 = rng.normal(size=(1, m))
        rhs2 = np.array([float(c2[0] @ phi_r) + 1.0])
        phi2, rec2 = filter_forces(FilterOptions(), phi_r, Mr, C, c2, rhs2, lo, hi, S, theta)
        assert rec2['feasible'] and abs(float(c2[0] @ phi2) - rhs2[0]) < 1e-8


def _exact_minimiser(H, x0, A, b):
    """argmin 1/2 |x - x0|_H^2 s.t. A x <= b by enumeration of the active sets in 40-digit arithmetic (small problems);
    None if the rows are inconsistent."""
    from itertools import combinations
    import mpmath as mp
    with mp.workdps(40):
        n, R = len(x0), len(b)
        Hm, Am, bm, xm = mp.matrix(H.tolist()), mp.matrix(A.tolist()), mp.matrix(b.tolist()), mp.matrix(x0.tolist())
        tol = mp.mpf(10) ** -20
        Ax = lambda x, i: mp.fsum(Am[i, j] * x[j] for j in range(n))
        if all(Ax(xm, i) - bm[i] <= tol for i in range(R)):
            return x0.copy()
        for S in (c for k in range(1, min(R, n) + 1) for c in combinations(range(R), k)):
            K, rhs = mp.zeros(n + len(S), n + len(S)), mp.zeros(n + len(S), 1)
            for i in range(n):
                for j in range(n):
                    K[i, j] = Hm[i, j]
                rhs[i] = mp.fsum(Hm[i, j] * xm[j] for j in range(n))
            for a, i in enumerate(S):
                for j in range(n):
                    K[n + a, j] = K[j, n + a] = Am[i, j]
                rhs[n + a] = bm[i]
            if abs(mp.det(K)) < mp.mpf(10) ** -35:
                continue
            sol = mp.lu_solve(K, rhs)
            if all(sol[n + a] >= -tol for a in range(len(S))) and all(Ax(sol, i) - bm[i] <= tol for i in range(R)):
                return np.array([float(sol[i]) for i in range(n)])
    return None


def test_qp_nearly_dependent_rows_give_the_minimiser():
    """A row whose normal makes an angle of about 1e-5 with the span of two other rows (instances of
    pinocchio/num/t13_qp_dependent.py): the returned point is the minimiser to 1e-8 (relative H-distance; reference:
    enumeration of the active sets in 40-digit arithmetic).  Before the correction of the partial step the routine
    returned, on a few per cent of such instances, feasible points about 1e-5 to 1e-4 away from the minimiser."""
    rng = np.random.default_rng(7)
    exact = _exact_minimiser
    worst, checked = 0., 0
    for trial in range(200):
        n, R = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        H = (Q * rng.uniform(1., 20., n)) @ Q.T
        A = rng.standard_normal((R, n))
        A /= np.linalg.norm(A, axis=1)[:, None]
        i, j, p = rng.choice(R, 3, replace=False)
        v = rng.standard_normal(n)
        a = rng.uniform(0.2, 1.) * A[i] + rng.uniform(0.2, 1.) * A[j]
        A[p] = a / np.linalg.norm(a) + (1e-5 if trial % 2 else 3e-5) * v / np.linalg.norm(v)
        x0 = 2. * rng.standard_normal(n)
        b = A @ x0 - rng.uniform(-0.2, 1., R)
        xe = exact(H, x0, A, b)
        if xe is None:
            continue
        step = float(np.sqrt((xe - x0) @ H @ (xe - x0)))
        if step < 1e-3:
            continue
        res = solve_qp(H, -H @ x0, A, b)
        assert res.feasible
        worst = max(worst, float(np.sqrt((res.x - xe) @ H @ (res.x - xe))) / step)
        checked += 1
    assert checked > 150 and worst < 1e-8, (checked, worst)


def test_qp_small_violations_of_nearly_dependent_rows():
    """A nearly dependent row violated by a very small amount is added by the full step of the exact method.
    (1) x1 <= -1 and x1 + 2e-5 x2 <= -1 - 2e-10: minimiser (-1, -1e-5) with multipliers (0.5, 0.5); a partial step alone
    carried x past the second row, and the routine cycled to its iteration limit.  (2) Instances of part 2 of
    pinocchio/num/t13_qp_dependent.py (two rows at an angle of 10^-7.5 to 10^-4.5, nearly parallel or nearly opposite, the
    second violated or met by 1e-11 to 1e-7 at the minimiser over the first, with and without further rows): no instance
    reaches the iteration limit, no empty set is rated feasible (reference: 40-digit enumeration), and every returned
    point and its multipliers have normwise backward errors below 1e-8 (row and stationarity residuals relative to the
    magnitudes of their terms).  Feasible sets with nearly opposite rows may be rated empty (conservative rating, see
    rcx/qp.py); nearly parallel ones never are."""
    res = solve_qp(np.eye(2), np.zeros(2), np.array([[1., 0.], [1., 2e-5]]), np.array([-1., -1. - 2e-10]))
    assert res.feasible and res.iterations <= 3, (res.feasible, res.iterations)
    assert np.max(np.abs(res.x - np.array([-1., -1e-5]))) < 1e-11 and np.max(np.abs(res.mu - 0.5)) < 1e-4, (res.x, res.mu)
    rng = np.random.default_rng(23)

    def unit(v):
        return v / np.linalg.norm(v)
    counts = dict(feasible=0, rated_empty_parallel=0, rated_empty_opposite=0)
    for trial in range(400):
        opposite, extra = trial % 2 == 1, trial % 4 >= 2
        n = int(rng.integers(3, 6)) if extra else 3
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        H = (Q * rng.uniform(1., 20., n)) @ Q.T
        x0 = rng.standard_normal(n)
        a1 = unit(rng.standard_normal(n))
        b1 = a1 @ x0 - rng.uniform(0.2, 1.)
        Hi = np.linalg.inv(H)
        xp = x0 - (a1 @ x0 - b1) * (Hi @ a1) / (a1 @ Hi @ a1)
        v = rng.standard_normal(n)
        v = unit(v - (v @ a1) * a1)
        theta = 10 ** rng.uniform(-7.5, -4.5)
        a2 = unit(np.cos(theta) * (-a1 if opposite else a1) + np.sin(theta) * v)
        A, b = [a1, a2], [b1, a2 @ xp - rng.choice([-1., 1.]) * 10 ** rng.uniform(-11, -7)]
        if extra:
            for _ in range(int(rng.integers(1, 4))):
                c = unit(rng.standard_normal(n))
                A.append(c)
                b.append(c @ x0 - rng.uniform(-0.5, 1.))
        A, b = np.array(A), np.array(b)
        res = solve_qp(H, -H @ x0, A, b)
        assert res.iterations < 20 * (len(b) + n) + 50
        xe = _exact_minimiser(H, x0, A, b)
        if xe is None:
            assert not res.feasible
            continue
        if not res.feasible:
            counts['rated_empty_opposite' if opposite else 'rated_empty_parallel'] += 1
            continue
        counts['feasible'] += 1
        # normwise backward errors: residuals relative to the magnitudes of their terms
        d, r = res.x - x0, A @ res.x - b
        rows = np.where(res.mu > 0., np.abs(r), np.maximum(r, 0.)) / (np.abs(A) @ np.abs(res.x) + np.abs(b))
        stat = np.max(np.abs(H @ d + A.T @ res.mu)) / np.max(np.abs(H) @ np.abs(d) + np.abs(A).T @ res.mu)
        assert np.max(rows) <= 1e-8 and stat <= 1e-8, (trial, np.max(rows), stat)
    assert counts['rated_empty_parallel'] == 0 and counts['feasible'] > 250, counts


def test_qp_rows_dependent_to_round_off():
    """Two instances of adversarial testing with ill-conditioned active sets.  (1) Two pairs of duplicated rows (0, 3) and
    (1, 5), rows 0 and 1 at an angle of about 1e-4: the minimiser has the active rows 0, 2, 5 and multipliers of about
    4e8.  A partial step along the computed z of a duplicated row (round-off, multiplied by a step of 1e7) made the
    routine cycle to its iteration limit; the routine must return the minimiser.  (2) A row nearly in the cone of active
    rows whose Gram matrix has a condition number of about 5e9: a full step along a round-off z returned a point far from
    the minimiser; the routine must return a point with normwise backward errors below 1e-8 or rate the set empty
    (conservative rating)."""
    H1 = np.array([[1278.9773958356384, -521.846632959113, -8.12360048461337, 563.1241314371621, -645.034574906761],
                   [-521.846632959113, 545.3386987121097, 220.4475908748479, 89.5234607289657, 79.6233110270269],
                   [-8.12360048461337, 220.4475908748479, 650.536338071077, -600.0945272155781, -757.4748458520619],
                   [563.1241314371621, 89.5234607289657, -600.0945272155781, 2040.410647784476, 1095.4963003160824],
                   [-645.034574906761, 79.6233110270269, -757.4748458520619, 1095.4963003160824, 2710.156660801421]])
    g1 = np.array([-795.6029411394278, 665.019762108046, 892.1659847326598, -1125.0341612212594, -764.484080673148])
    a0 = [0.2530245399843731, -0.058713439581243206, -0.42144743850753297, 0.7677272282323637, 0.4068270809377461]
    a1 = [0.7581246637975918, 0.20879766693719157, 0.238518957857777, -0.3096752863663557, -0.47839361636598965]
    A1 = np.array([a0, a1, [-0.28581947175548783, 0.05138300314659053, 0.41765543759416635, -0.7657416839711478, -0.39353605353271964], a0,
                   [0.2595438088088156, -0.05726334230836673, -0.42074726165583276, 0.7674110644108868, 0.404239929949474], a1])
    b1 = np.array([0.7087563557993407, 0.21615165939884018, -0.7284777554249532, 0.7087563557993407, 0.7122760574642172, 0.21615165939884018])
    H2 = np.array([[25.733900275374207, 15.732548788276912, 11.065915799592668, -4.403459815242855, -5.933471982830149],
                   [15.732548788276912, 14.246901731476992, 3.6876501145230707, 9.817315099542569, -6.404157897239937],
                   [11.065915799592668, 3.6876501145230707, 6.8170173654475485, -10.235188408531185, -0.6955487835638424],
                   [-4.403459815242855, 9.817315099542569, -10.235188408531185, 34.628026751383, -6.519760287596658],
                   [-5.933471982830149, -6.404157897239937, -0.6955487835638424, -6.519760287596658, 3.1306970164594428]])
    g2 = np.array([21.512448482671847, -4.472389467292095, 21.01130154937389, -51.40780061161802, 5.668430178067532])
    A2 = np.array([[-0.35054548947446684, -0.6489700994640091, -0.011014539408246187, -0.16307859915912443, 0.6551638880682469],
                   [-0.1342647485527377, -0.15144427850955544, 0.6392041878169983, -0.009755104844426073, -0.7418628256015233],
                   [0.27595647107986016, 0.7479450738650913, -0.21576796493422018, 0.14811876588316342, -0.5439955965368535],
                   [0.4543196044454827, 0.024283907886486888, -0.22767604712604866, -0.8400519064402449, 0.18836241898379863],
                   [-0.21134773570671098, -0.7308885923210977, 0.18640979107784642, -0.1343972719739478, 0.6068960069919762],
                   [0.2479699540204779, 0.5731053252068555, -0.7033998823185325, 0.08825314962314905, 0.32787371846260444]])
    b2 = np.array([-0.5800191716164613, -1.4287985158034096, 1.1602248972957943, -0.0724038174645382, -1.0965279533477277, 1.9750376666564253])

    def backward_error(H, g, A, b, res):
        r = A @ res.x - b
        rows = np.where(res.mu > 0., np.abs(r), np.maximum(r, 0.)) / (np.abs(A) @ np.abs(res.x) + np.abs(b))
        return max(float(np.max(rows)), float(np.max(np.abs(H @ res.x + g + A.T @ res.mu)) / np.max(np.abs(H) @ np.abs(res.x) + np.abs(g) + np.abs(A).T @ res.mu)))
    res = solve_qp(H1, g1, A1, b1)
    assert res.feasible and res.iterations < 20 * (6 + 5) + 50, (res.feasible, res.iterations)
    assert sorted(np.flatnonzero(res.mu > 0.).tolist()) == [0, 2, 5] and backward_error(H1, g1, A1, b1, res) < 1e-8
    res = solve_qp(H2, g2, A2, b2)
    assert not res.feasible or backward_error(H2, g2, A2, b2, res) < 1e-8


def test_qp_degenerate_rows():
    """(1) A violated zero row (0 <= -1) makes the set empty, alone and together with other rows.  (2) Minimiser at the
    origin: rows A x <= 0 with two normals at an angle of 10^-5.5 to 10^-3.5 and H x_u in their cone, so that both rows
    are active at x* = 0 with zero right-hand sides; the acceptance test after the step for the nearly dependent row must
    accept the point (its residuals are compared with |A_i| (|x| + |x - x_u|) + |b_i|, not with |A_i| |x| + |b_i| alone,
    which vanishes at the origin)."""
    assert not solve_qp(np.eye(2), np.zeros(2), np.array([[0., 0.]]), np.array([-1.])).feasible
    assert not solve_qp(np.eye(2), np.zeros(2), np.array([[0., 0.], [1., 0.]]), np.array([-1., -1.])).feasible
    rng = np.random.default_rng(31)
    worst, solved = 0., 0
    for _ in range(300):
        n = int(rng.integers(2, 6))
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        H = (Q * rng.uniform(1., 20., n)) @ Q.T
        a1 = rng.standard_normal(n)
        a1 /= np.linalg.norm(a1)
        v = rng.standard_normal(n)
        v -= (v @ a1) * a1
        v /= np.linalg.norm(v)
        th = 10 ** rng.uniform(-5.5, -3.5)
        a2 = np.cos(th) * a1 + np.sin(th) * v
        xu = np.linalg.solve(H, rng.uniform(0.2, 1.) * a1 + rng.uniform(0.2, 1.) * a2) * 10 ** rng.uniform(0., 2.)
        res = solve_qp(H, -H @ xu, np.array([a1, a2]), np.zeros(2))
        assert res.feasible
        solved += 1
        worst = max(worst, float(np.sqrt(res.x @ H @ res.x / (xu @ H @ xu))))
    # the stopping tolerance 1e-10 max(1, |x_u|_H) leaves the nearly parallel row unenforced when its violation at the
    # minimiser over the first row is below it: |x|_H / |x_u|_H is then of the order of the angle, at most about 3e-4
    assert solved == 300 and worst < 1e-3, worst


def run_all(namespace=None):
    """Runs every test_ function of the namespace; returns (results, failures) with results = [(name, 'PASS' | message, seconds)]."""
    import time
    import traceback
    ns = globals() if namespace is None else namespace
    tests = [(k, v) for k, v in sorted(ns.items()) if k.startswith('test_') and callable(v)]
    results, failed = [], 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            results.append((name, 'PASS', time.perf_counter() - t0))
            print(f'PASS {name}  ({time.perf_counter() - t0:.1f} s)', flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            traceback.print_exc()
            results.append((name, f'FAIL: {exc}', time.perf_counter() - t0))
            print(f'FAIL {name}: {exc}', flush=True)
    print(f'{len(tests) - failed}/{len(tests)} passed')
    return results, failed


if __name__ == '__main__':
    raise SystemExit(1 if run_all()[1] else 0)
