"""Plant-side monitor of the generated safety functions (independent of the controller's shadow configuration).

``RowMonitor.update_values`` takes the poses of the graph frames and the tree coordinates of a *plant* at its own
integration rate (step dt) and records, per barrier,

  h            the value at every plant step (its minimum decides whether a limit was violated);
  psi1, psi2   psi1 = h_dot + alpha1 h  and the row function  psi2 = h_ddot + (alpha1 + alpha2) h_dot + alpha1 alpha2 h
               from centred differences of h, so that h, h_dot and h_ddot belong to the same instant t_n:
                   h_dot(t_n) = (h_{n+1} - h_{n-1}) / (2 dt),      h_ddot(t_n) = (h_{n+1} - 2 h_n + h_{n-1}) / dt^2
               (second-order accurate; evaluated one step later, when h_{n+1} is known; at the first step h_dot uses
               the one-sided second-order formula and h_ddot the second difference of the first three steps, which
               is first-order accurate there).  Only poses enter: the Jacobians of the generated evaluators are not used.
  weighted sample mean of psi2
               From  psi1_dot + alpha2 psi1 = psi2  it follows over one control sample Ts that
                   psi1(t_k + Ts) = exp(-alpha2 Ts) psi1(t_k) + int_0^Ts exp(-alpha2 (Ts - s)) psi2(t_k + s) ds,
               so the mean of psi2 with the weights exp(-alpha2 (Ts - s)),
                   psi2_mean = alpha2 (psi1(t_k + Ts) - exp(-alpha2 Ts) psi1(t_k)) / (1 - exp(-alpha2 Ts)),
               is the quantity that decides whether psi1 >= 0 is kept from one sample to the next; the raw psi2 at the
               plant rate contains the plant's fastest dynamics (valves, contact), which this mean averages over a sample.
"""
from __future__ import annotations

import numpy as np


class RowMonitor:
    def __init__(self, xc, dt, window_s=None):
        self.xc, self.dt = xc, float(dt)
        bs = xc.barriers
        self.R = bs.R
        self.h_min = np.full(self.R, np.inf)
        self.psi_min = np.full(self.R, np.inf)
        self.row_min = np.full(self.R, np.inf)
        self.row_mean_min = np.full(self.R, np.inf)
        self.t_hmin = np.zeros(self.R)
        self.t_psimin = np.zeros(self.R)
        self._h1 = self._h2 = None                              # h_{n-1}, h_{n-2} of the present call
        self._t1 = None
        self._psi_w = None
        self.count = 0
        T = xc.Ts if window_s is None else float(window_s)
        self.window = max(1, int(round(T / self.dt)))
        self._decay = np.exp(-bs.alpha2 * self.window * self.dt)
        self._gain = bs.alpha2 / (1. - self._decay)
        self.h = None

    def _record(self, t, index, h, hd, hdd):
        """psi1 and psi2 of the plant step ``index`` (time t) from same-instant h, h_dot, h_ddot."""
        bs = self.xc.barriers
        psi = hd + bs.alpha1 * h
        lower = psi < self.psi_min
        self.t_psimin[lower] = t
        np.minimum(self.psi_min, psi, out=self.psi_min)
        np.minimum(self.row_min, hdd + (bs.alpha1 + bs.alpha2) * hd + bs.alpha1 * bs.alpha2 * h, out=self.row_min)
        if index % self.window == 0:                            # a control sample instant
            if self._psi_w is not None:
                np.minimum(self.row_mean_min, (psi - self._decay * self._psi_w) * self._gain, out=self.row_mean_min)
            self._psi_w = psi

    def update_values(self, t, R, p, qt, com=None):
        """Poses R (F, 3, 3), p (F, 3) of the plan frames and tree coordinates of the plant at time t (one call per step);
        ``com``: the plant's own centre of mass for the support rows."""
        bs = self.xc.barriers
        h = bs.values(R, p, qt) if com is None else bs.values(R, p, qt, com)
        dt = self.dt
        better = h < self.h_min
        self.t_hmin[better] = t
        np.minimum(self.h_min, h, out=self.h_min)
        n = self.count                                          # index of this step
        if n >= 2:
            h1, h2 = self._h1, self._h2
            hdd = (h - 2. * h1 + h2) / (dt * dt)
            if n == 2:                                          # step 0: one-sided second-order difference
                self._record(self._t1 - dt, 0, h2, (-3. * h2 + 4. * h1 - h) / (2. * dt), hdd)
            self._record(self._t1, n - 1, h1, (h - h2) / (2. * dt), hdd)
        self._h2, self._h1, self._t1 = self._h1, h, t
        self.count += 1
        self.h = h
        return h

    def pair_values(self, h=None):
        h = self.h if h is None else h
        return np.array([float(np.min(h[idx])) for idx in self.xc.barriers.pair_rows.values()])

    def summary(self):
        bs = self.xc.barriers
        out = {}
        for pid, idx in bs.pair_rows.items():
            k = idx[int(np.argmin(self.h_min[idx]))]
            kp = idx[int(np.argmin(self.psi_min[idx]))]
            out[pid] = dict(h_min=float(self.h_min[k]), at_s=float(self.t_hmin[k]), row=bs.ids[k], cls=str(bs.cls[k]),
                            psi1_min=float(self.psi_min[kp]), psi1_min_at_s=float(self.t_psimin[kp]), psi1_min_row=bs.ids[kp],
                            row_function_sample_mean_min=float(np.min(self.row_mean_min[idx])),
                            row_function_physics_rate_min=float(np.min(self.row_min[idx])))
        cls = {}
        for c, idx in bs.class_rows.items():
            cls[c] = dict(h_min=float(np.min(self.h_min[idx])), psi1_min=float(np.min(self.psi_min[idx])),
                          row_function_sample_mean_min=float(np.min(self.row_mean_min[idx])),
                          row_function_physics_rate_min=float(np.min(self.row_min[idx])), rows=int(len(idx)))
        return dict(pairs=out, classes=cls, samples=self.count)
