"""Coordinated safety layer of the extended RoboCompiler: required-motion shaping and the actuator-force filter.

Notation (compiled independent speeds u, channel efforts, reduced model  M_r u_dot + h_r = C f + tau_ext  with the
*delivered* channel efforts f and C = L^T G):

  barrier row i        h_i,   h_dot_i = hb_i + J_i u,   h_ddot_i = a_i^T f + d_i
                       a_i = C^T M_r^{-1} J_i^T,        d_i = J_i M_r^{-1}(tau_ext - h_r) + gamma_i
  row function         psi2_i(f) = h_ddot_i + (alpha1_i + alpha2_i) h_dot_i + alpha1_i alpha2_i h_i
                       (psi2_i >= 0 for all t keeps  psi1_i = h_dot_i + alpha1_i h_i >= 0  and  h_i >= 0)

Required-motion shaping (kinematic layer; RoboCompiler Eq. (17) with inequality rows):

  u_r = argmin 1/2 |u_r - u_r0|^2_{W_v}
        s.t.  hb_i + J_i u_r >= -alpha1_i (h_i - sigma_i)              first-order barrier rows (shifted by the standoff)
              lo_s <= L u_r <= hi_s                                   actuator port-speed capability, with headroom gain g:
                                                                      hi_s = cap_hi + g (cap_hi - L u), lo_s = cap_lo - g (L u - cap_lo)
              lo_n <= phi_r(u_r) <= hi_n                              the nominal VDC effort stays inside the effort capability
  W_v = L^T diag(1 / cap^2) L  (port-speed metric), M_r (kinetic) or Jx^T Jx (task rates);  phi_r(u_r) = P u_r + p0 is the
  affine map of the generated VDC evaluator, in which the required task acceleration is the nominal one plus the filtered
  derivative of the task-rate correction  Jx (u_r - u_r0);
  sigma_i = m_i^state / (alpha1_i alpha2_i) is the standoff equivalent to the state-dependent margin of the effort row.
  When the rows are inconsistent the barrier rows are relaxed first, then the port-speed rows as well; the effort rows
  are kept (every relaxation is counted).
  With g > 0 a stalled actuator still receives the velocity-error effort (the required port speed may exceed the capability
  in proportion to the distance of the actual port speed from it), while a moving one is not driven past its capability.

Coordinated actuator-force filter (second-order rows on the efforts the declared force loops deliver):

  the force loops are first-order,  f(t_k + Ts) = f + Theta (phi - f)  (Theta, Theta_bar, Theta_2: rcx.actuators);
  one-step prediction of the row function for a held command phi (exact for coefficients frozen over the sample):

      psi2_i^+ = psi2_i(f) + a_i^T (Theta + (alpha1_i + alpha2_i) Ts Theta_bar + alpha1_i alpha2_i Ts^2 Theta_2) (phi - f)
                 + Ts [ (alpha1_i + alpha2_i) h_ddot_i(f) + alpha1_i alpha2_i h_dot_i ] + alpha1_i alpha2_i Ts^2 / 2 h_ddot_i(f)

  phi = argmin 1/2 |Theta (phi - phi_r)|^2_W      s.t.   psi2_i^+ >= m_i,     lo <= phi <= hi,       W = C^T M_r^{-1} C
        (kinetic metric on the delivered-effort deviation: with inactive bounds and equal response fractions
         C Theta (phi - phi_r) = sum_i lambda_i g_i J_i^T with lambda_i, g_i >= 0, a set of virtual constraint forces along
         the barrier gradients)

  m_i = (1 + 2 kappa_i) m_L,i     m_L,i = |J_i|_{M^-1} ( eps |tau_ext_hat|_{M^-1} + w0 ):  load-estimate error in the kinetic
                                  dual norm (Cauchy-Schwarz);  w0 = worst tool wrench with |F| <= Wf, |M| <= Wm
      + 2 kappa_i m_F,i           m_F,i = sum_j |a_ij| e_j:  force-loop model error  |f_j - f_j,model| <= e_j
      + 2 Ts jbar                 unmodelled change of the row function over one sample (declared)
      + rho/(1-rho) max(0, push_i)    control-effectiveness (inertia, response-time) uncertainty rho on the increase of
                                  psi2 that the command has to produce
  kappa_i = 1 + (alpha1_i + alpha2_i) Ts + alpha1_i alpha2_i Ts^2 / 2  is the growth of an acceleration error in the row
  function over one sample; every error is counted once for the one-step prediction and once for the excursion between
  the samples (and the load error once more for the estimate at the sample instant), which is what keeps psi2 >= 0 at
  all times and not only at the sample instants.

With ideal effort sources (tau = 0: Theta = Theta_bar = I, Theta_2 = I / 2) and Ts -> 0 the row is the usual second-order row
on the command.

Fallback.  If the hard problem is infeasible (conflicting rows, or rows that the effort bounds cannot meet) the command
is determined in two stages (rcx.qp.solve_lexicographic), without a penalty weight:

  1. the smallest violations  s* = argmin sum_i (s_i / rho_i)^2   s.t.  c_i^T phi + s_i >= r0_i + m_i,  lo <= phi <= hi,  s >= 0,
     each violation measured against  rho_i = sum_j |c_ij| (hi_j - lo_j),  the change of that row over the whole effort
     range (a row that does not compete with a violated one keeps its right-hand side up to the tolerance of the
     fallback, 1e-6 of rho_i);
  2. among all commands with those violations, the one closest to the nominal efforts in the filter metric.

Both stages are independent of units, of the scale of the metric and of the coordinate partition.  The event and the
slack are returned and counted by the caller; the safety statement does not cover such a sample.
Rows on which the efforts have no authority (dual norm zero, e.g. a body that no channel can move along the row
gradient; or, in the fallback, rho_i = 0: every channel that acts on the row has coinciding bounds) are not passed to
either problem; they are reported when their right-hand side is violated.
"""
from __future__ import annotations

import numpy as np

from .qp import solve_lexicographic, solve_qp

DEAD_ROW = 1e-9                 # a row whose dual norm is below this fraction of the largest one has no authority


class FilterOptions:
    def __init__(self, **kw):
        self.metric = 'kinetic'            # 'kinetic' (W = C^T M_r^-1 C) or 'identity' (normalised channel efforts)
        self.velocity_metric = 'port_speed'  # 'port_speed' (L^T diag(1/cap^2) L), 'kinetic' (M_r) or 'task' (Jx^T Jx)
        self.velocity_shaping = True       # barrier rows in the required-motion problem
        self.speed_rows = True             # port-speed capability rows in the required-motion problem
        self.effort_rows = True            # nominal-effort capability rows in the required-motion problem
        self.force_filter = True
        self.coordinated = True            # False: per-channel (decentralised) correction, ablation only
        self.actuator_model = True         # rows on the efforts delivered by the declared force loops (False: on the command)
        self.sampled_prediction = True     # one-step prediction of the row function over the sample time
        self.margin_force = True
        self.margin_load = True
        self.margin_sampling = True
        self.margin_effectiveness = True
        self.enforce_bounds = True
        self.contact_rows = True           # rows on the contact forces of a mechanism on ground contacts (rcx.compiler.Support)
        self.null_weight = 1e-3            # weight of the effort deviation that changes contact forces only (redundant actuation)
        # first layer of the safety layer (paper: Theorem on consistent required motion)
        #   'velocity'      released: first-order barrier rows shape the required velocity (u_r is an output of a projection)
        #   'acceleration'  consistent: the required velocity is a controller state, its derivative a_r is shaped by the rows
        #                   zs_i' + beta_i zs_i >= 0 on the slack zs_i = hb_i + J_i u_r + alpha1_i (h_i - sigma_i), beta_i = min(beta_bar, alpha2_i)
        self.first_layer = 'velocity'
        self.beta_bar = 16.0               # rate of the rows of the acceleration-level first layer
        self.beta_cap = True               # paper: beta_i = min(beta_bar, alpha2_i); False: beta_i = beta_bar (rate test above alpha2)
        self.kappa_r = 20.0                # gain of the nominal required acceleration toward the nominal required velocity
        self.conditioning = False          # conditioned required velocity: u_r = U + u_D,  u_D' = -kappa_c u_D + M^-1 C (phi - phi_r)
        self.kappa_c = 12.0                # decay rate of the conditioning state (paper: kappa_r = k makes it the loop of the releases)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f'unknown filter option {k}')
            setattr(self, k, v)

    def as_dict(self):
        return dict(self.__dict__)


# ---------------------------------------------------------------------------------------------- required motion
def shape_required_velocity(u_r0, W, J, rhs, A_cap=None, b_cap=None, n_speed=0):
    """Projection of the nominal required velocity onto the admissible set in the metric W (positive definite).

    Barrier rows  J u >= rhs;  capability rows  A_cap u <= b_cap  (the first ``n_speed`` are port-speed rows, the rest
    nominal-effort rows).  When the rows are inconsistent the barrier rows are relaxed first (level 1: capability rows
    hard), then the port-speed rows as well (level 2: effort rows hard; speed and barrier rows relaxed together);
    level 3 returns the nominal velocity.  A relaxation is lexicographic (rcx.qp.solve_lexicographic): the smallest sum
    of squared violations first, then the velocity closest to the nominal one.  Every row is normalised by its dual
    norm sqrt(a W^-1 a^T) (violations are compared in these units), so that the result, also the relaxed one, does not
    depend on the coordinate partition.  Barrier rows without authority (dual norm zero up to round-off: no admissible
    velocity changes them) are left out and reported in ``no_authority`` when their right-hand side is violated."""
    d = len(u_r0)
    D = 1. / np.sqrt(np.diag(W))                        # u = D y
    H = (W * D[:, None]) * D[None, :]
    H = 0.5 * (H + H.T)
    Hi = np.linalg.inv(H)
    y0 = u_r0 / D
    g = -H @ y0
    nb_all = len(rhs)
    info = dict(feasible=True, relaxed=False, active=np.zeros(nb_all, bool), mu=np.zeros(nb_all), speed_active=False, effort_active=False,
                slack=0., level=0, no_authority=np.zeros(nb_all, bool), numerically_ok=True)

    def dual_norm(A):
        return np.sqrt(np.maximum(np.einsum('ri,ij,rj->r', A, Hi, A), 0.))

    keep = np.zeros(0, int)
    if nb_all:
        A_all, b_all = -(np.asarray(J, float) * D[None, :]), -np.asarray(rhs, float)
        rn = dual_norm(A_all)
        dead = rn <= DEAD_ROW * max(float(np.max(rn)), 1e-300)
        info['no_authority'] = dead & (b_all < 0.)
        keep = np.flatnonzero(~dead)
        A_b, b_b = A_all[keep] / rn[keep, None], b_all[keep] / rn[keep]
    else:
        A_b, b_b = np.zeros((0, d)), np.zeros(0)
    nb = len(b_b)
    if A_cap is None or len(A_cap) == 0:
        A_c, b_c = np.zeros((0, d)), np.zeros(0)
    else:
        A_c, b_c = np.asarray(A_cap, float) * D[None, :], np.asarray(b_cap, float)
        rc = np.maximum(dual_norm(A_c), 1e-300)
        A_c, b_c = A_c / rc[:, None], b_c / rc
    nc = len(b_c)

    def barrier(values, fill):
        out = np.full(nb_all, fill)
        out[keep] = values
        return out

    if nb + nc == 0:
        return u_r0.copy(), info
    res = solve_qp(H, g, np.vstack([A_b, A_c]), np.r_[b_b, b_c])
    if res.feasible:
        info.update(active=barrier(res.mu[:nb] > 0., False), mu=barrier(res.mu[:nb], 0.), speed_active=bool(np.any(res.mu[nb:nb + n_speed] > 0.)),
                    effort_active=bool(np.any(res.mu[nb + n_speed:] > 0.)))
        return D * res.x, info
    # level 1: capability rows hard, barrier rows soft (the effort filter enforces the second-order rows)
    info['relaxed'] = True
    if nb:
        res2, x, s = solve_lexicographic(H, g, A_c if nc else None, b_c if nc else None, A_b, b_b)
        if res2.feasible:
            info.update(active=barrier((s > 1e-9) | (res2.mu[nc:nc + nb] > 0.), False), mu=barrier(res2.mu[nc:nc + nb], 0.),
                        speed_active=bool(np.any(res2.mu[:n_speed] > 0.)), effort_active=bool(np.any(res2.mu[n_speed:nc] > 0.)),
                        slack=float(np.max(s)), level=1, numerically_ok=bool(res2.ok))
            return D * x, info
    # level 2: effort rows hard (consistent on their own while the affine effort map is nonsingular), speed and barrier rows soft
    if nc > n_speed:
        A_s = np.vstack([A_b, A_c[:n_speed]])
        b_s = np.r_[b_b, b_c[:n_speed]]
        if len(b_s):
            ne = nc - n_speed
            res3, x, s = solve_lexicographic(H, g, A_c[n_speed:], b_c[n_speed:], A_s, b_s)
            if res3.feasible:
                info.update(active=barrier((s[:nb] > 1e-9) | (res3.mu[ne:ne + nb] > 0.), False), mu=barrier(res3.mu[ne:ne + nb], 0.),
                            speed_active=bool(np.any(s[nb:] > 1e-9) or np.any(res3.mu[ne + nb:] > 0.)),
                            effort_active=bool(np.any(res3.mu[:ne] > 0.)), slack=float(np.max(s)), level=2, numerically_ok=bool(res3.ok))
                return D * x, info
        else:
            res3 = solve_qp(H, g, A_c[n_speed:], b_c[n_speed:])
            if res3.feasible:
                info.update(effort_active=True, level=2)
                return D * res3.x, info
    info.update(feasible=False, level=3)
    return u_r0.copy(), info


# ---------------------------------------------------------------------------------------------- effort rows
def barrier_rows(Mr, hr, C, J, hdot, gamma, h, alpha1, alpha2, tau_ext):
    """Coefficients a (R x m), drift d (R) and the unmargined right-hand side b = -(psi2 - a f) (R) of the rows."""
    MiJ = np.linalg.solve(Mr, J.T)                      # d x R
    a = MiJ.T @ C                                       # R x m
    drift = MiJ.T @ (tau_ext - hr) + gamma
    b = -drift - (alpha1 + alpha2) * hdot - alpha1 * alpha2 * h
    return a, drift, b, MiJ


def delivered_rows(a, drift, b, hdot, alpha1, alpha2, f, theta, theta_bar, theta2, Ts):
    """Rows  c_i^T phi >= r0_i + m_i  on the command: one-step prediction of the row function for first-order force loops.

    With the coefficients a_i, d_i frozen over the sample the prediction is exact:
        h_ddot^+ = h_ddot + a (Theta (phi - f))
        h_dot^+  = h_dot + Ts h_ddot + Ts a (Theta_bar (phi - f))
        h^+      = h + Ts h_dot + Ts^2 / 2 h_ddot + Ts^2 a (Theta_2 (phi - f))
    Returns c (R x m), r0 (R), psi2 at the measured efforts, the required increase of psi2 before margins and h_ddot.
    Ts = 0 drops the prediction of the state-driven change: the row is then imposed on the efforts delivered at the end of
    the sample (fractions theta of the caller's sample time) with h and h_dot of the present sample; with ideal effort
    sources (theta = 1) that is the continuous-time row on the command."""
    s12, p12 = alpha1 + alpha2, alpha1 * alpha2
    c = a * (theta[None, :] + (s12 * Ts)[:, None] * theta_bar[None, :] + (p12 * Ts * Ts)[:, None] * theta2[None, :])
    hdd = a @ f + drift
    psi2 = a @ f - b                                    # = h_ddot + (alpha1 + alpha2) h_dot + alpha1 alpha2 h
    known = Ts * (s12 * hdd + p12 * hdot) + 0.5 * Ts * Ts * p12 * hdd
    need = -psi2 - known                                # required increase of psi2 before margins
    return c, c @ f + need, psi2, need, hdd


def contact_rows(sup, A, b, f, theta, e_force):
    """Rows  c phi >= r  of the declared ground contacts on the contact forces  lam = A f + b  (root-frame components per
    contact: x, y tangential, z normal) of the efforts delivered at the end of the sample, f + theta (phi - f):

        normal      lam_z >= N_min + margin
        friction    mu / sqrt(2) lam_z - lam_x >= margin,  mu / sqrt(2) lam_z + lam_x >= margin,  and the same for y
                    (pyramid inside the friction cone of the declared coefficient)

    margin = sum_j |row A|_j e_j + reserve  (force-loop model error and the declared force reserve).  Returns the row
    matrix, the right-hand sides and the margins."""
    rows = []
    rhs0 = []
    for c in sup.contacts:
        ix, iy, iz = c['rows']
        k = c['friction'] / np.sqrt(2.)
        rows.append(A[iz])
        rhs0.append(c['normal_min'] - b[iz])
        for i, sgn in ((ix, 1.), (ix, -1.), (iy, 1.), (iy, -1.)):
            rows.append(k * A[iz] - sgn * A[i])
            rhs0.append(-(k * b[iz] - sgn * b[i]))
    Arow = np.array(rows)
    margin = np.abs(Arow) @ e_force + sup.reserve
    c_phi = Arow * theta[None, :]
    r = np.array(rhs0) - Arow @ ((1. - theta) * f) + margin
    return c_phi, r, margin


def load_error_bound(Mr, tau_ext, TF, TM, Wf, Wm, eps_load):
    """Declared bound on the load-estimate error in the kinetic dual norm:
    |tau_ext - tau_ext_hat|_{M_r^-1} <= eps |tau_ext_hat|_{M_r^-1} + w0,  w0 = worst tool wrench (|F| <= Wf, |M| <= Wm)."""
    Mi = np.linalg.inv(Mr)
    tn = float(np.sqrt(max(tau_ext @ Mi @ tau_ext, 0.)))
    w0 = Wf * float(np.sqrt(max(np.linalg.eigvalsh(TF.T @ Mi @ TF)[-1], 0.))) + Wm * float(np.sqrt(max(np.linalg.eigvalsh(TM.T @ Mi @ TM)[-1], 0.)))
    return eps_load * tn + w0, tn, w0


def row_margins(opt, a, MiJ, J, need, load_bound, e_force, Ts, jerk, rho, alpha1, alpha2):
    """Robust tightening of every row; returns the total margin and its components (m/s^2 or coordinate/s^2).

    With  kappa_i = 1 + (alpha1_i + alpha2_i) Ts + alpha1_i alpha2_i Ts^2 / 2  (growth of an acceleration error in the row
    function over one sample),  m_F = sum_j |a_ij| e_j  and  m_L = |J_i|_{M^-1} * (bound on the load error norm):

        force      2 kappa m_F            prediction over the sample and excursion between the samples
        load       (1 + 2 kappa) m_L      the same two, plus the error of the estimate at the sample instant
        sampling   2 Ts jbar              declared bound on the unmodelled change of the row function (coefficient
                                          variation, different response times of the channels)
        standoff   force + load + sampling    (state-dependent part of the margin; sigma = standoff / (alpha1 alpha2))
        effectiveness  rho / (1 - rho) max(0, need + standoff)    need + standoff = increase of psi2 that the command
                                          has to produce; with a true effectiveness in [1 - rho, 1] of the modelled one
                                          the delivered increase is then at least need + standoff
        allowance  kappa m_F + (1 + kappa) m_L + Ts jbar    largest one-step prediction residual covered by the margin
    """
    R = len(need)
    comp = dict(force=np.zeros(R), load=np.zeros(R), sampling=np.zeros(R), effectiveness=np.zeros(R), standoff=np.zeros(R),
                allowance=np.zeros(R), load_error=np.zeros(R))
    if R == 0:
        return np.zeros(0), comp
    kappa = 1. + (alpha1 + alpha2) * Ts + 0.5 * alpha1 * alpha2 * Ts * Ts if opt.sampled_prediction else np.ones(R)
    if opt.margin_force:
        mF = np.abs(a) @ e_force
        comp['force'] = 2. * kappa * mF
        comp['allowance'] += kappa * mF
    if opt.margin_load:
        mL = np.sqrt(np.maximum(np.einsum('ri,ir->r', J, MiJ), 0.)) * load_bound       # |J_i|_{M^-1} * bound on |tau_err|_{M^-1}
        comp['load'] = (1. + 2. * kappa) * mL
        comp['load_error'] = mL
        comp['allowance'] += (1. + kappa) * mL
    if opt.margin_sampling:
        comp['sampling'] = np.full(R, 2. * Ts * jerk)
        comp['allowance'] += Ts * jerk
    comp['standoff'] = comp['force'] + comp['load'] + comp['sampling']                           # state-dependent part
    if opt.margin_effectiveness and rho > 0.:
        comp['effectiveness'] = rho / (1. - rho) * np.maximum(0., need + comp['standoff'])
    return comp['standoff'] + comp['effectiveness'], comp


def filter_forces(opt, phi_r, Mr, C, c, rhs, lo, hi, scale, theta):
    """Coordinated effort filter.  ``c`` (R x m), ``rhs`` (R): rows  c phi >= rhs.  Returns phi and a record."""
    m = len(phi_r)
    S = np.asarray(scale, float)
    if opt.metric == 'kinetic':
        W = C.T @ np.linalg.solve(Mr, C)
        ST = S * theta
        H = (W * ST[:, None]) * ST[None, :]
        H = 0.5 * (H + H.T)
        if C.shape[0] < m:
            # redundant actuation: the kinetic metric does not see effort deviations in the null space of C (they change
            # contact forces, not the motion); there the metric is the normalised effort, scaled to a fraction of the mean
            # kinetic weight, so that a row which internal forces can meet is met mainly by them.  It is a weight, not a
            # priority: where the kinetic metric has directions that are as cheap, a part of such a row is met by motion.
            Cs = C * ST[None, :]
            Z = np.linalg.svd(Cs)[2][C.shape[0]:].T
            H += opt.null_weight * np.trace(H) / C.shape[0] * (Z @ Z.T)
        H += 1e-9 * np.trace(H) / m * np.eye(m)
    else:
        H = np.eye(m)
    x_r = phi_r / S
    g = -H @ x_r
    R = len(rhs)
    As = -(np.asarray(c, float).reshape(R, m) * S[None, :])
    bs = -np.asarray(rhs, float)
    # dual norm of every row in the filter metric: the largest change of the row per unit of the effort cost
    rn = np.sqrt(np.maximum(np.einsum('ri,ir->r', As, np.linalg.solve(H, As.T)), 0.)) if R else np.zeros(0)
    # rows without actuator authority cannot be influenced: they are reported, not passed to the QP (relative test on the
    # dual norm; absolute floor: a full-scale effort changes the row by less than 1e-12 of its unit)
    dead = (rn <= DEAD_ROW * max(float(np.max(rn)) if R else 0., 1e-300)) | (np.linalg.norm(As, axis=1) <= 1e-12)
    no_authority = dead & (bs < 0.)
    live = np.flatnonzero(~dead)
    An, bn, rl = As[live] / rn[live, None], bs[live] / rn[live], rn[live]
    nl = len(live)
    if opt.enforce_bounds:
        Ab = np.vstack([np.eye(m), -np.eye(m)])
        bb = np.r_[hi / S, -lo / S]
    else:
        Ab, bb = np.zeros((0, m)), np.zeros(0)
    nbd = len(bb)
    rec = dict(feasible=True, relaxed=False, slack=np.zeros(R), mu=np.zeros(R), active=np.zeros(R, bool), bound_active=np.zeros(2 * m, bool),
               bound_mu=np.zeros(2 * m), kkt=None, iterations=0, numerically_ok=True, no_authority=no_authority, H=H / np.outer(S, S))
    res = solve_qp(H, g, np.vstack([An, Ab]), np.r_[bn, bb])
    if res.feasible:
        x = res.x
        rec['mu'][live] = res.mu[:nl] / rl
        rec['active'][live] = res.mu[:nl] > 0.
        rec.update(kkt=res.kkt, iterations=res.iterations, numerically_ok=bool(res.ok))
        if opt.enforce_bounds:
            rec.update(bound_active=res.mu[nl:] > 0., bound_mu=res.mu[nl:])
    else:
        rec['feasible'], rec['relaxed'] = False, True
        # rows that cannot be violated inside the effort box are dropped from the relaxed problem (exact culling)
        if opt.enforce_bounds:
            worst = np.where(An > 0., An * (hi / S)[None, :], An * (lo / S)[None, :]).sum(axis=1)    # max of An x over the box
            keep = worst > bn
        else:
            keep = np.ones(nl, bool)
        sub = np.flatnonzero(keep)
        # violations are compared as fractions of the change of each row over the whole effort range (unit rows without bounds)
        rho = (np.abs(An[sub]) @ ((hi - lo) / S)) if opt.enforce_bounds else np.ones(len(sub))
        frozen = rho <= 1e-12 * np.abs(An[sub]).sum(axis=1)      # no channel that acts on the row can move inside its bounds
        rec['no_authority'][live[sub[frozen]]] = True
        sub, rho = sub[~frozen], rho[~frozen]
        idx = live[sub]
        res2, x, s = solve_lexicographic(H, g, Ab if nbd else None, bb if nbd else None, An[sub] / rho[:, None], bn[sub] / rho)
        if not res2.feasible:
            x = np.clip(x_r, lo / S, hi / S) if opt.enforce_bounds else x_r.copy()
            rec['numerically_ok'] = False
        else:
            rec['active'][idx] = (s > 1e-9) | (res2.mu[nbd:nbd + len(sub)] > 0.)
            rec['mu'][idx] = res2.mu[nbd:nbd + len(sub)] / (rho * rl[sub])
            if nbd:
                rec.update(bound_active=res2.mu[:nbd] > 0., bound_mu=res2.mu[:nbd])
            rec['kkt'], rec['iterations'], rec['numerically_ok'] = res2.kkt, res2.iterations, bool(res2.ok)
        if not rec['numerically_ok']:                       # no multipliers of the effort cost: the caller splits the command
            rec['H'] = None                                 # deviation into bounds and rows without them
    if opt.enforce_bounds:                                  # the solver meets the bounds to its tolerance; the command meets them exactly
        x = np.clip(x, lo / S, hi / S)
    phi = x * S
    if rec['relaxed']:                                      # violations of the returned command, in the units of the rows
        rec['slack'] = np.where(dead, 0., np.maximum(0., np.asarray(rhs, float) - np.asarray(c, float).reshape(R, m) @ phi))
    rec['phi'] = phi
    return phi, rec


def filter_uncoordinated(phi_r, c, rhs, lo, hi, scale=None, enforce_bounds=True):
    """Ablation: every channel corrects on its own, assuming the other channels keep their nominal efforts.

    A channel acts on a row only if a full-scale effort of that channel changes the row by at least 1e-3 of what the
    strongest channel does (``scale``: effort scales); rows that no channel can change are skipped."""
    m = len(phi_r)
    S = np.ones(m) if scale is None else np.asarray(scale, float)
    phi = np.clip(phi_r, lo, hi) if enforce_bounds else np.array(phi_r, float)
    box_lo, box_hi = (lo, hi) if enforce_bounds else (np.full(m, -np.inf), np.full(m, np.inf))
    reach = np.abs(c) * S[None, :] if len(rhs) else np.zeros((0, m))
    strongest = reach.max(axis=1) if len(rhs) else np.zeros(0)
    out = phi.copy()
    for j in range(m):
        lo_j, hi_j = box_lo[j], box_hi[j]
        for i in range(len(rhs)):
            if strongest[i] <= 0. or reach[i, j] < 1e-3 * strongest[i]:
                continue
            cij = c[i, j]
            need = (rhs[i] - c[i] @ phi + cij * phi[j]) / cij
            if cij > 0:
                lo_j = max(lo_j, need)
            else:
                hi_j = min(hi_j, need)
        if lo_j > hi_j:                                     # rows of this channel conflict with each other or with its bounds
            lo_j = hi_j = float(np.clip(0.5 * (lo_j + hi_j), box_lo[j], box_hi[j])) if np.isfinite(lo_j + hi_j) else phi[j]
        out[j] = min(max(phi[j], lo_j), hi_j)
    return out
