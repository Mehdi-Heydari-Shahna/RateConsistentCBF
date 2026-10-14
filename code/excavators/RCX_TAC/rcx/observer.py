"""Generalized-momentum observer of the external load in the compiled independent coordinates.

With  p = M_r u  and the reduced model  M_r u_dot + h_r = tau_act + tau_ext,

    p_dot = tau_act + tau_ext + M_r_dot u - h_r,
    r(t)  = K [ p(t) - p(0) - int_0^t (tau_act - h_r + M_r_dot u + r) ds ]     =>     r_dot = K (tau_ext - r).

No acceleration measurement is used.  ``tau_act`` is the reduced effort of the *measured* port efforts
(L^T f_meas), so the estimate is not affected by the difference between commanded and delivered efforts, nor by
the safety filter.  M_r_dot is evaluated from the generated body modules (M_dot = C + C^T) and the lift derivative.
Friction and unmodelled payload appear in the estimate as external load.

The runtime applies the observer to the momentum conjugate to the declared task coordinates,
p_x = Jx^-T M_r u  with  p_x_dot = Jx^-T (tau_act + tau_ext + M_r_dot u - h_r) - Jx^-T Jx_dot^T p_x,
so that the two filter states (estimate and its slow part) do not depend on the coordinate partition; the
estimate in the independent coordinates is  tau_ext = Jx^T r_x.
"""
from __future__ import annotations

import numpy as np


class MomentumObserver:
    def __init__(self, n, gain, slow_rate=None):
        self.n = n
        self.K = np.broadcast_to(np.asarray(gain, float), (n,)).copy()
        self.slow_rate = slow_rate
        self.reset()

    def reset(self):
        self.p0 = None
        self.integral = np.zeros(self.n)
        self.r = np.zeros(self.n)
        self.slow = np.zeros(self.n)
        self.beta_prev = None
        self.t_prev = None

    def update(self, t, Mr, Mr_dot, u, hr, tau_act):
        return self.update_momentum(t, Mr @ u, tau_act - hr + Mr_dot @ u)

    def update_momentum(self, t, p, beta):
        """Observer on a generalized momentum p with  p_dot = beta + (external load);  returns the load estimate."""
        if self.p0 is None:
            self.p0, self.beta_prev, self.t_prev = p.copy(), beta, t
            return self.r.copy()
        dt = t - self.t_prev
        if dt > 0.:
            self.integral += dt * (0.5 * (beta + self.beta_prev) + self.r)
            self.r = self.K * (p - self.p0 - self.integral)
            if self.slow_rate is not None:
                self.slow += -np.expm1(-self.slow_rate * dt) * (self.r - self.slow)
        self.beta_prev, self.t_prev = beta, t
        return self.r.copy()

    def state(self):
        return dict(p0=np.zeros(self.n) if self.p0 is None else self.p0, integral=self.integral, r=self.r, slow=self.slow,
                    beta_prev=np.zeros(self.n) if self.beta_prev is None else self.beta_prev,
                    t_prev=np.array(-1. if self.t_prev is None else self.t_prev), started=np.array(self.p0 is not None))

    def restore(self, st):
        if bool(st['started']):
            self.p0, self.beta_prev, self.t_prev = st['p0'].copy(), st['beta_prev'].copy(), float(st['t_prev'])
        else:
            self.p0 = self.beta_prev = self.t_prev = None
        self.integral, self.r, self.slow = st['integral'].copy(), st['r'].copy(), st['slow'].copy()
