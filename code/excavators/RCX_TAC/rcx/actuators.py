"""Actuator capability models generated from the actuator specification records.

A *channel* is one command that drives one or more actuator ports of the mechanism graph with the same effort
(e.g. two boom cylinders on one valve section).  From the declared capabilities the compiler generates, per
channel, the effort bounds at the current port speed and the port-speed limits:

  hydraulic_cylinder   bore/rod areas A, B, supply pressure ps, valve flow q (per cylinder), minimum pressure p0
      extension  (v >= 0):  f_max = A ps (1 - (A v / q)^2) - B p_out(B v),     f_min = A p0 - B ps
      retraction (v <  0):  f_min = A p_out(A |v|) - B ps (1 - (B |v| / q)^2),  f_max = A ps - B p0
      p_out(Q) = min(p0 + kappa ps (Q / q)^2, cap ps)         meter-out back pressure of the discharging chamber
      (the driving side is limited by the orifice law of the supply edge, the braking side only by relief pressure)
      speed limits: the supply edge passes at most q, and the discharging chamber needs the back pressure
      ps (Q / q)^2 <= p_out, which the declared meter-out model provides up to  Q / q <= x_cap = sqrt((cap - p0/ps) / kappa);
      beyond it the force loop cannot hold its command (the chamber pressure leaves the commanded value), hence
          extension   v <=  sigma min(q / A, x_cap q / B),        retraction  -v <= sigma min(q / B, x_cap q / A)
  rotary_torque        |tau| <= tau_max,   |w| <= sigma w_max

``derate`` moves both effort bounds inwards by (1 - derate) of their magnitude (reserve for friction, leakage and
pressure-control error): for lo < 0 < hi this scales them towards zero.

Force-loop model.  The actuator force controllers are not ideal effort sources: every channel declares the response
time tau of its closed force loop and the bound e on the error of that model,

      f_dot = (phi - f) / tau,        | f(t) - f_model(t) | <= e          (f delivered, phi commanded effort).

For a command held over one sample Ts the model gives, with r = tau / Ts,

      effort at the end of the sample        f + theta (phi - f),            theta     = 1 - exp(-Ts / tau)
      mean effort over the sample            f + theta_bar (phi - f),        theta_bar = 1 - r theta
      double time integral over the sample   Ts^2 (f / 2 + theta2 (phi - f)),  theta2  = 1/2 - r + r^2 theta

(tau = 0: theta = theta_bar = 1, theta2 = 1/2).  The safety rows are imposed on these *delivered* efforts (rcx.filter).
"""
from __future__ import annotations

import numpy as np


class Channel:
    def __init__(self, record, port_index):
        self.id = record['id']
        self.type = record['type']
        self.ports = list(record['ports'])
        self.port_index = np.asarray(port_index, int)
        self.derate = float(record.get('derate', 1.0))
        self.sigma = float(record.get('speed_fraction', 1.0))
        self.force_error = float(record.get('force_error', 0.))
        self.response_time = float(record.get('response_time', 0.))
        if self.type == 'hydraulic_cylinder':
            self.A = np.pi / 4. * float(record['bore_m']) ** 2
            self.B = self.A - np.pi / 4. * float(record['rod_m']) ** 2
            self.ps = float(record['supply_pressure_Pa'])
            self.q = float(record['valve_flow_m3_s'])
            self.p0 = float(record.get('min_pressure_Pa', 0.))
            self.kappa = float(record.get('back_pressure_factor', 1.0))
            self.cap = float(record.get('back_pressure_cap', 1.0))
            self.scale = self.A * self.ps
            x_cap = np.sqrt(max(self.cap - self.p0 / self.ps, 0.) / self.kappa) if self.kappa > 0. else 1.
            self.speed_ext = self.sigma * min(self.q / self.A, x_cap * self.q / self.B)
            self.speed_ret = self.sigma * min(self.q / self.B, x_cap * self.q / self.A)
            self.speed = min(self.speed_ext, self.speed_ret)
        elif self.type == 'rotary_torque':
            self.limit = float(record['torque_limit_Nm'])
            self.scale = self.limit
            self.speed = self.speed_ext = self.speed_ret = self.sigma * float(record['speed_limit_rad_s'])
        else:
            raise ValueError(f'{self.id}: unsupported channel type {self.type}')

    def _cylinder(self, v):
        A, B, ps, q, p0 = self.A, self.B, self.ps, self.q, self.p0
        if v >= 0.:
            xa, xb = min(A * v / q, 1.), min(B * v / q, 1.)
            hi = A * ps * (1. - xa * xa) - B * min(p0 + self.kappa * ps * xb * xb, self.cap * ps)
            lo = A * p0 - B * ps
        else:
            xa, xb = min(-A * v / q, 1.), min(-B * v / q, 1.)
            lo = A * min(p0 + self.kappa * ps * xa * xa, self.cap * ps) - B * ps * (1. - xb * xb)
            hi = A * ps - B * p0
        return lo, hi

    def bounds(self, port_velocity):
        """Effort-command bounds (lo, hi) at the current speeds of this channel's ports."""
        if self.type == 'rotary_torque':
            return -self.derate * self.limit, self.derate * self.limit
        lo, hi = -np.inf, np.inf
        for v in np.atleast_1d(port_velocity):
            a, b = self._cylinder(float(v))
            lo, hi = max(lo, a), min(hi, b)
        # reserve: both bounds move inwards by (1 - derate) of their magnitude (for lo < 0 < hi this scales them towards
        # zero; in overrun, where both bounds have the same sign, the interval must not become wider than the model)
        lo, hi = lo + (1. - self.derate) * abs(lo), hi - (1. - self.derate) * abs(hi)
        if hi < lo:                                         # flow-saturated in both directions: keep a valid interval
            lo = hi = 0.5 * (lo + hi)
        return lo, hi

    def response(self, Ts):
        """Fractions (theta, theta_bar, theta2) of a command step that the first-order force loop delivers over one sample."""
        return response_fractions(self.response_time, Ts)

    def describe(self):
        d = dict(id=self.id, type=self.type, ports=self.ports, derate=self.derate, speed_limits=[-float(self.speed_ret), float(self.speed_ext)], effort_scale=self.scale,
                 force_error=self.force_error, response_time_s=self.response_time)
        if self.type == 'hydraulic_cylinder':
            d.update(area_A_m2=self.A, area_B_m2=self.B, static_bounds_N=[self.derate * (self.A * self.p0 - self.B * self.ps),
                                                                         self.derate * (self.A * self.ps - self.B * self.p0)])
        return d


def response_fractions(tau, Ts):
    """theta = 1 - exp(-x), theta_bar = 1 - theta / x, theta2 = 1/2 - 1/x + theta / x^2 with x = Ts / tau: the fraction of a
    command step delivered at the end of a sample, its time average over the sample and its double time integral (unit
    Ts^2).  The closed forms cancel for a slow loop (x << 1); there the series in x is used."""
    if tau <= 0.:
        return 1., 1., 0.5
    x = Ts / tau
    theta = -np.expm1(-x)
    if x < 0.1:                                             # alternating series, terms up to x^9: relative error < 1e-12
        tb = t2 = 0.
        term = 1.                                           # x^n / n!
        for n in range(1, 10):
            term *= x / n
            sign = 1. if n % 2 else -1.
            tb += sign * term / (n + 1)                     # theta_bar = sum_{n>=1} (-1)^(n+1) x^n / (n+1)!
            t2 += sign * term / ((n + 1) * (n + 2))         # theta2    = sum_{n>=1} (-1)^(n+1) x^n / (n+2)!
        return float(theta), float(tb), float(t2)
    return float(theta), float(1. - theta / x), float(0.5 - 1. / x + theta / (x * x))


def build_channels(spec_actuators, port_ids, group_ids, group_ports=None):
    """Channel models in the order of the compiled actuator groups.  ``group_ports`` (group id -> actuated ports of the
    graph) must agree with the ports declared by the channel of the same id: the force map of the graph and the
    capability model of the specification then describe the same actuators."""
    records = {r['id']: r for r in spec_actuators['channels']}
    if sorted(records) != sorted(group_ids):
        raise ValueError(f'Actuator channels {sorted(records)} do not match the actuator groups of the graph {sorted(group_ids)}')
    out = []
    for g in group_ids:
        rec = records[g]
        if group_ports is not None and sorted(rec['ports']) != sorted(group_ports[g]):
            raise ValueError(f"Channel {g}: declared ports {sorted(rec['ports'])} differ from the ports of the actuator group "
                             f"in the graph {sorted(group_ports[g])}")
        out.append(Channel(rec, [port_ids.index(p) for p in rec['ports']]))
    return out
