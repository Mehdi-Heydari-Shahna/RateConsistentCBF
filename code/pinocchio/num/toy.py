"""Continuous-time two-layer safe VDC on a 6R manipulator with ideal effort sources (Pinocchio sample model).

Plant            q_dot = u,   M(q) u_dot + N(q,u) u + g(q) = tau (+ M e_m),             M_dot = N + N^T
Barriers         h_i(q): joint limits and half spaces for the end effector; shifted functions  hs_i = h_i - sigma_i
Kinematic layer  u_r = U_r(t,q) = argmin 1/2 |v - u_n|_W^2   s.t.  J_i v >= -alpha1_i hs_i,     u_n = qd_dot + lam (qd - q)
VDC              tau_r = M a_r + N u_r + g + K (u_r - u),   K = k M (uniform) or a declared non-uniform gain
                 a_r = exact derivative of U_r along the motion (analytic, from the active set), or the nominal
                 acceleration plus a first-order filtered derivative of the correction (feedforward='filtered')
Effort layer     a = argmin 1/2 |a - a_nom|_G^2   s.t.  J_i a + Jdot_i u + (alpha1_i + alpha2_i) J_i u + alpha1_i alpha2_i hs_i >= 0
                 G = M: kinetic metric (tau - tau_r = M (a - a_nom) = sum_i lambda_i J_i^T);  G = M^2: Euclidean metric of the efforts

Everything that the analysis uses is logged: storage S = 1/2 s^T M s, filter power P_f = -s^T M (a - a_nom), the active
rows of both layers, s_par (M-orthogonal projection of s on the span of the active gradients), the row values Pi^(beta)_i
at the beta-VDC acceleration, the least-constraint norm of their negative part, and the terms of the row identity

    Pi^(beta)_i = zs_dot_i + beta zs_i + (alpha2_i - beta) psi1s_i - c_i - J_i e_a,
    zs_i = J_i u_r + alpha1_i hs_i,   psi1s_i = J_i u + alpha1_i hs_i,   c_i = Jdot_i s - J_i M^-1 N s,   e_a = u_r_dot - a_r.
"""
import sys

import numpy as np
import pinocchio as pin

import paths                    # noqa: E402  (locations of the release trees)

sys.path.insert(0, paths.RCX_HEAP)
from rcx.qp import solve_qp      # noqa: E402

ACT_TOL = 1e-9


class ProjectionError(RuntimeError):
    pass


def project(G, x0, J, rhs, tol=1e-9):
    """argmin 1/2 |x - x0|_G^2  s.t.  J x >= rhs   (G symmetric positive definite).  Returns x and the multipliers.

    The active set of the release solver is used as a start; the point is then recomputed from the equality system of
    the active rows and the set is corrected (most negative multiplier out, most violated row in) until the KKT conditions
    hold to round-off.  (This refinement was introduced because the release solver used to skip the primal part of a
    partial step for a row that it rates as dependent on the active ones, which left a stationarity residual of order
    1e-4 when the rows are nearly, but not exactly, dependent.  The corrected solver takes that part; the refinement is
    kept as an independent check of the KKT conditions and normally returns the solver's point unchanged.)"""
    R = len(rhs)
    mu = np.zeros(R)
    if R == 0:
        return x0.copy(), mu
    slack0 = J @ x0 - rhs
    if np.all(slack0 >= 0.):
        return x0.copy(), mu
    res = solve_qp(G, -G @ x0, -J, -rhs)
    if not res.feasible:
        raise ProjectionError('rows inconsistent')
    Y = np.linalg.solve(G, J.T)                               # G^-1 J^T
    Lam = J @ Y
    dn = np.sqrt(np.maximum(np.diag(Lam), 1e-300))
    act = list(np.flatnonzero(res.mu > 0.))
    scale = 1. + float(np.max(np.abs(slack0) / dn))
    for _ in range(4 * R + 8):
        if act:
            LA = Lam[np.ix_(act, act)]
            lam = np.linalg.lstsq(LA, -slack0[act], rcond=1e-12)[0]
            x = x0 + Y[:, act] @ lam
        else:
            lam, x = np.zeros(0), x0.copy()
        if len(lam) and np.min(lam * dn[act]) < -tol * scale:
            del act[int(np.argmin(lam * dn[act]))]
            continue
        viol = (rhs - J @ x) / dn
        if act:
            viol[act] = -np.inf
        p = int(np.argmax(viol))
        if viol[p] > tol * scale:
            act.append(p)
            continue
        mu[act] = np.maximum(lam, 0.)
        return x, mu
    raise ProjectionError('active set did not settle')


class Arm:
    def __init__(self):
        self.model = pin.buildSampleModelManipulator()
        self.data = self.model.createData()
        self.n = self.model.nv
        self.fid = self.model.getFrameId('effector_body')

    def dyn(self, q, v):
        m, d = self.model, self.data
        M = pin.crba(m, d, q)
        M = np.triu(M) + np.triu(M, 1).T
        N = pin.computeCoriolisMatrix(m, d, q, v).copy()
        g = pin.computeGeneralizedGravity(m, d, q).copy()
        return M, N, g

    def ee(self, q):
        m, d = self.model, self.data
        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        p = d.oMf[self.fid].translation.copy()
        J = pin.computeFrameJacobian(m, d, q, self.fid, pin.LOCAL_WORLD_ALIGNED)[:3].copy()
        return p, J


class Barriers:
    """Rows: joint limits (lower, upper) and half spaces  n . p_ee >= d  for the end effector."""

    def __init__(self, arm, limits=(), planes=(), sigma=0.):
        self.arm = arm
        self.rows = []
        for j, lo, hi, a1, a2 in limits:
            if lo is not None:
                self.rows.append(('lo', j, lo, a1, a2))
            if hi is not None:
                self.rows.append(('hi', j, hi, a1, a2))
        for nvec, d0, a1, a2 in planes:
            self.rows.append(('plane', np.asarray(nvec, float) / np.linalg.norm(nvec), d0, a1, a2))
        self.R = len(self.rows)
        self.a1 = np.array([r[3] for r in self.rows], float)
        self.a2 = np.array([r[4] for r in self.rows], float)
        self.sigma = np.broadcast_to(np.asarray(sigma, float), (self.R,)).copy()
        self.names = [f'{r[0]}{r[1] if r[0] != "plane" else ""}' for r in self.rows]

    def hJ(self, q):
        """Shifted barrier values hs = h - sigma and their gradients (rows of J)."""
        n = self.arm.n
        h, J = np.zeros(self.R), np.zeros((self.R, n))
        p, Jp = self.arm.ee(q)
        for i, r in enumerate(self.rows):
            if r[0] == 'lo':
                h[i], J[i, r[1]] = q[r[1]] - r[2], 1.
            elif r[0] == 'hi':
                h[i], J[i, r[1]] = r[2] - q[r[1]], -1.
            else:
                h[i], J[i] = r[1] @ p - r[2], r[1] @ Jp
        return h - self.sigma, J

    def Jdot(self, q, v, eps=1e-6):
        """d/dt J(q(t)) for q_dot = v (central difference of the analytic Jacobian)."""
        return (self.hJ(q + eps * v)[1] - self.hJ(q - eps * v)[1]) / (2 * eps)


class Controller:
    """feedforward: 'exact' (a_r = derivative of U_r along the motion) or 'filtered' (state c_f, time constant tau_c)."""

    def __init__(self, arm, bar, ref, lam=12., k=12., W='identity', metric='kinetic', effort_layer=True, kinematic_layer=True,
                 k_matrix=None, feedforward='exact', tau_c=0.03):
        self.arm, self.bar, self.ref = arm, bar, ref
        self.lam, self.k, self.W, self.metric = lam, k, W, metric
        self.effort_layer, self.kinematic_layer = effort_layer, kinematic_layer
        self.k_matrix = k_matrix                         # optional non-uniform gain: function M -> K (necessity example)
        self.feedforward, self.tau_c = feedforward, tau_c

    # ------------------------------------------------------------------ kinematic layer and its derivative along the motion
    def required(self, t, q, u, M, N, h, J, Jd):
        """u_n, u_n_dot, u_r, u_r_dot (derivative along the actual motion), active rows of the kinematic layer."""
        bar = self.bar
        qd, qd_dot, qd_ddot = self.ref(t)
        u_n = qd_dot + self.lam * (qd - q)
        un_dot = qd_ddot + self.lam * (qd_dot - u)
        act = np.zeros(bar.R, bool)
        if not self.kinematic_layer or bar.R == 0:
            return u_n, un_dot, u_n.copy(), un_dot.copy(), act
        if self.W == 'kinetic':
            Wm = M
            Y = np.linalg.inv(M)
            Yd = -Y @ (N + N.T) @ Y
        else:
            Wm = np.eye(self.arm.n)
            Y, Yd = np.eye(self.arm.n), np.zeros((self.arm.n, self.arm.n))
        x, mu = project(Wm, u_n, J, -bar.a1 * h)                  # J v >= -alpha1 hs
        act = mu > ACT_TOL * (1. + float(np.max(mu)))
        if not act.any():
            return u_n, un_dot, x, un_dot.copy(), act
        JA, JdA = J[act], Jd[act]
        G = JA @ Y @ JA.T
        nu = np.linalg.solve(G, -bar.a1[act] * h[act] - JA @ u_n)
        Gd = JdA @ Y @ JA.T + JA @ Yd @ JA.T + JA @ Y @ JdA.T
        nud = np.linalg.solve(G, -bar.a1[act] * (JA @ u) - JdA @ u_n - JA @ un_dot - Gd @ nu)
        u_r = u_n + Y @ JA.T @ nu
        ur_dot = un_dot + Yd @ JA.T @ nu + Y @ JdA.T @ nu + Y @ JA.T @ nud
        return u_n, un_dot, u_r, ur_dot, act

    def U_r(self, t, q):
        """The required velocity alone (for finite-difference checks)."""
        M, N, _ = self.arm.dyn(q, np.zeros(self.arm.n))
        h, J = self.bar.hJ(q)
        return self.required(t, q, np.zeros(self.arm.n), M, N, h, J, np.zeros_like(J))[2]

    # ------------------------------------------------------------------ control law
    def __call__(self, t, q, u, cf=None, log=False):
        arm, bar, k = self.arm, self.bar, self.k
        M, N, g = arm.dyn(q, u)
        h, J = bar.hJ(q)
        Jd = bar.Jdot(q, u) if bar.R else np.zeros_like(J)
        u_n, un_dot, u_r, ur_dot, act_k = self.required(t, q, u, M, N, h, J, Jd)
        if self.feedforward == 'exact':
            a_r, cf_dot = ur_dot, None
        else:                                               # nominal acceleration + filtered derivative of the correction
            cf_dot = ((u_r - u_n) - cf) / self.tau_c
            a_r = un_dot + cf_dot
        s = u_r - u
        Ks = k * (M @ s) if self.k_matrix is None else self.k_matrix(M) @ s
        a_nom = a_r + np.linalg.solve(M, N @ s + Ks)
        a1, a2 = bar.a1, bar.a2
        gamma = Jd @ u
        rhs = -(gamma + (a1 + a2) * (J @ u) + a1 * a2 * h)        # rows  J a >= rhs
        if self.effort_layer and bar.R:
            Gm = M if self.metric == 'kinetic' else M @ M
            a, lam_f = project(Gm, a_nom, J, rhs)
        else:
            a, lam_f = a_nom.copy(), np.zeros(bar.R)
        act = lam_f > ACT_TOL * (1. + float(np.max(lam_f))) if bar.R else np.zeros(0, bool)
        if not log:
            # also the right-hand side of the storage balance without the mismatch term, and M s for that term
            Ms = M @ s
            return a, cf_dot, (act_k, act), float(-s @ Ks + Ms @ (ur_dot - a_r) - Ms @ (a - a_nom)), Ms
        Mi = np.linalg.inv(M)
        c = Jd @ s - J @ (Mi @ (N @ s))
        zs = J @ u_r + a1 * h
        zs_dot = J @ ur_dot + Jd @ u_r + a1 * (J @ u)
        psi1 = J @ u + a1 * h
        e_a = ur_dot - a_r
        out = dict(M=M, N=N, s=s, u_r=u_r, u_n=u_n, a_r=a_r, ur_dot=ur_dot, e_a=e_a, a_nom=a_nom, a=a, h=h, J=J, Jd=Jd, rhs=rhs, lam=lam_f,
                   act=act, act_k=act_k, S=0.5 * float(s @ M @ s), Pf=-float(s @ M @ (a - a_nom)), psi1=psi1, zs=zs, zs_dot=zs_dot, c=c,
                   psi2=J @ a - rhs, psi2_nom=J @ a_nom - rhs, w=J @ s, Ks=Ks, dissipation=float(s @ Ks),
                   P_ea=float(s @ M @ e_a))
        return a, cf_dot, out


def least_norm(M, Ja, r):
    """min |d|_M  s.t.  J_i d >= r_i  for the rows of Ja (inf when the rows are inconsistent)."""
    if len(r) == 0 or np.all(r <= 0.):
        return 0.
    try:
        x, _ = project(M, np.zeros(M.shape[0]), Ja, r)
    except ProjectionError:
        return np.inf
    return float(np.sqrt(max(x @ M @ x, 0.)))


def analysis(out, k, beta, bar):
    """Quantities of the dissipation theorem for one logged sample; beta: scalar comparison rate."""
    M, s, J, act = out['M'], out['s'], out['J'], out['act']
    mu = min(k, beta)
    res = dict(spar2=0., dA=0., bound=0., Pi_min=np.inf, star_err=0., dA_slack=0., dA_curv=0., dA_ea=0., bound_split=0.)
    a_beta = out['a_nom'] + (beta - k) * s
    Pi_all = J @ a_beta - out['rhs']
    # row identity for all rows (not only the active ones)
    star = out['zs_dot'] + beta * out['zs'] + (bar.a2 - beta) * out['psi1'] - out['c'] - J @ out['e_a']
    if bar.R:
        res['star_err'] = float(np.max(np.abs(star - Pi_all) / (1. + np.abs(Pi_all))))
    if not act.any():
        return res
    Ja = J[act]
    Lc = np.linalg.cholesky(M)
    Gg = np.linalg.solve(M, Ja.T)
    U, sv, _ = np.linalg.svd(Lc.T @ Gg, full_matrices=False)
    U = U[:, sv > 1e-9 * sv[0]]
    spar2 = float(np.sum((U.T @ (Lc.T @ s)) ** 2))
    Pi = Pi_all[act]
    dA = least_norm(M, Ja, -Pi)
    # split of the defect: slack part, curvature part and feedforward part (subadditivity of the least-constraint norm)
    zeta = np.maximum(0., -(out['zs_dot'] + beta * out['zs'])[act])
    dA_slack = least_norm(M, Ja, zeta)
    dA_curv = least_norm(M, Ja, out['c'][act])
    dA_ea = least_norm(M, Ja, (J @ out['e_a'])[act])
    res.update(spar2=spar2, dA=dA, bound=(k - mu) * spar2 + np.sqrt(spar2) * dA, Pi_min=float(np.min(Pi)), dA_slack=dA_slack,
               dA_curv=dA_curv, dA_ea=dA_ea, bound_split=(k - mu) * spar2 + np.sqrt(spar2) * (dA_slack + dA_curv + dA_ea))
    return res


def rk4_step(f, t, y, h):
    """One classical Runge-Kutta step of  y' = f(t, *y)  for a state y given as a tuple of arrays."""
    k1 = f(t, *y)
    k2 = f(t + h / 2, *[yi + h / 2 * ki for yi, ki in zip(y, k1)])
    k3 = f(t + h / 2, *[yi + h / 2 * ki for yi, ki in zip(y, k2)])
    k4 = f(t + h, *[yi + h * ki for yi, ki in zip(y, k3)])
    return tuple(yi + h / 6 * (a + 2 * b + 2 * c + d) for yi, a, b, c, d in zip(y, k1, k2, k3, k4))


def advance_with_events(f, signature, t, y, h, counts=None, events=True, max_events=12):
    """One grid step of  y' = f(t, *y)  (y: tuple of arrays) by RK4, with location of the changes of the active sets.

    ``signature(t, *y)`` identifies the active sets of the layers.  If it differs at the end of the step, the change is
    located by bisection on the length of the step (to 1e-12 s) and the step is split there.  A stage evaluated on the far
    side of the change enters the step with weight h / 6.  The step therefore ends 1e-7 s before the change (all stages on
    the near side), and one Euler step of 1e-7 s crosses it (error of the crossing: 1e-9 s times the jump of the
    right-hand side); the rest of the grid step is integrated in the same way (at most ``max_events`` changes per step).
    The same routine integrates every arm loop (toy.simulate, t14_consistent.simulate_consistent); ``counts['events']``
    counts the located changes."""
    y = tuple(y)
    t_end, events_here = t + h, 0
    while t_end - t > 1e-13:
        h = t_end - t
        sig0 = signature(t, *y)
        new = rk4_step(f, t, y, h)
        if not events or events_here >= max_events or signature(t + h, *new) == sig0:
            return new
        lo, hi = 0., h
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if signature(t + mid, *rk4_step(f, t, y, mid)) == sig0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        h1 = max(lo - 1e-7, 0.)
        if h1 > 0.:
            y = rk4_step(f, t, y, h1)
        d = min(hi - h1 + 1e-9, t_end - t - h1)
        k1 = f(t + h1, *y)
        y = tuple(yi + d * ki for yi, ki in zip(y, k1))
        t += h1 + d
        events_here += 1
        if counts is not None:
            counts['events'] = counts.get('events', 0) + 1
    return y


def simulate(ctl, q0, u0, T, dt=1e-3, e_m=None, betas=None, log_every=1, events=True):
    """RK4 on a fixed grid; inside a step the changes of an active set are located by bisection and the step is split there
    (the right-hand side jumps where the kinematic layer changes its active set; advance_with_events).  e_m(t, q, u):
    acceleration mismatch added to the plant.  Returns the log (dict of arrays).
    betas: dict name -> function(out) -> beta (comparison rates evaluated at every logged sample)."""
    arm, bar = ctl.arm, ctl.bar
    q, u = np.array(q0, float), np.array(u0, float)
    filt = ctl.feedforward != 'exact'
    cf = np.zeros(arm.n + 1)
    n = int(round(T / dt))
    betas = betas or {'k': lambda o: ctl.k}
    scal = ['t', 'balance', 'S', 'Pf', 'hmin', 'psi1min', 'psi2min', 'nact', 'nact_k', 'snorm', 'e', 'zmin', 'psi2nom_min', 'dissipation', 'P_ea', 'P_em',
            'ea_norm', 'em_norm', 'subset']
    log = {kk: [] for kk in scal}
    for name in betas:
        for kk in ('beta', 'bound', 'spar2', 'dA', 'star_err', 'dA_slack', 'dA_curv', 'dA_ea', 'bound_split', 'Pi_min'):
            log[f'{kk}:{name}'] = []
    for kk in ('q', 'u', 'act', 'act_k', 'h', 'psi1', 'zs', 'w', 's'):
        log[kk] = []

    def f(t, q, u, cf):
        # cf carries the state of the feedforward filter (n entries) and the integral of the storage balance (last entry)
        a, cfd, _, rate, Ms = ctl(t, q, u, cf[:-1])
        if e_m is not None:
            em = e_m(t, q, u)
            a = a + em
            rate -= float(Ms @ em)
        return u, a, np.r_[cfd if filt else np.zeros(arm.n), rate]

    def signature(t, q, u, cf):
        ak, ae = ctl(t, q, u, cf[:-1])[2]
        return ak.tobytes() + ae.tobytes()

    counts = dict(events=0)
    for i in range(n + 1):
        t = i * dt
        if i % log_every == 0:
            _, _, o = ctl(t, q, u, cf[:-1], log=True)
            em = e_m(t, q, u) if e_m is not None else np.zeros(arm.n)
            qd = ctl.ref(t)[0]
            big = np.inf
            vals = dict(t=t, balance=float(cf[-1]), S=o['S'], Pf=o['Pf'], hmin=float(np.min(o['h'])) if bar.R else big, psi1min=float(np.min(o['psi1'])) if bar.R else big,
                        psi2min=float(np.min(o['psi2'])) if bar.R else big, nact=int(o['act'].sum()), nact_k=int(o['act_k'].sum()),
                        snorm=float(np.sqrt(2 * o['S'])), e=float(np.max(np.abs(qd - q))), zmin=float(np.min(o['zs'])) if bar.R else big,
                        psi2nom_min=float(np.min(o['psi2_nom'])) if bar.R else big, dissipation=o['dissipation'], P_ea=o['P_ea'],
                        P_em=-float(o['s'] @ o['M'] @ em), ea_norm=float(np.sqrt(o['e_a'] @ o['M'] @ o['e_a'])),
                        em_norm=float(np.sqrt(em @ o['M'] @ em)), subset=bool(np.all(o['act_k'][o['act']])))
            for kk in scal:
                log[kk].append(vals[kk])
            for name, fn in betas.items():
                b = float(fn(o))
                an = analysis(o, ctl.k, b, bar)
                log[f'beta:{name}'].append(b)
                for kk, v in an.items():
                    log[f'{kk}:{name}'].append(v)
            for kk in ('act', 'act_k', 'h', 'psi1', 'zs', 'w', 's'):
                log[kk].append(o[kk].copy())
            log['q'].append(q.copy()); log['u'].append(u.copy())
        if i == n:
            break
        q, u, cf = advance_with_events(f, signature, t, (q, u, cf), dt, counts, events)
    log['events'] = [counts['events']]
    return {kk: np.array(v) for kk, v in log.items()}


def smooth_step(t, T):
    """Quintic step from 0 to 1 over [0, T] with its first two derivatives."""
    x = float(np.clip(t / T, 0., 1.))
    inside = 0. < t < T
    return (10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T * inside,
            (60 * x - 180 * x ** 2 + 120 * x ** 3) / T ** 2 * inside)


def ramp_reference(q_start, q_end, T):
    q_start, q_end = np.asarray(q_start, float), np.asarray(q_end, float)

    def ref(t):
        s, sd, sdd = smooth_step(t, T)
        return q_start + s * (q_end - q_start), sd * (q_end - q_start), sdd * (q_end - q_start)
    return ref


def constant_reference(q_end):
    q_end = np.asarray(q_end, float)
    z = np.zeros_like(q_end)
    return lambda t: (q_end, z, z)
