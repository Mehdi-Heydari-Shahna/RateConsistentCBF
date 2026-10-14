"""Synthetic double-acting hydraulic actuators of a plant; not identified hardware hydraulics.

The model of the earlier single-machine release (itself a port of RoboCompiler ``hydraulic_actuators.HydraulicBank``)
for any number of cylinders and rotary drives: two pressure states per cylinder (compressible chambers, valve flow
limited by the orifice law of the supply and return edges, leakage, relief at supply pressure, meter-out back pressure
of the discharging chamber), Coulomb and viscous friction, and first-order torque-limited rotary drives.

Cylinder coordinate: extension; chamber A (area A) extends, chamber B (area B) retracts.  The coordinate window
[lower, upper] gives the chamber volumes; the plant has no end stops, a coordinate more than ``overtravel`` outside the
window is an error (chamber volume would vanish).
"""
from __future__ import annotations

import numpy as np


class HydraulicBank:
    def __init__(self, cylinders, rotary, params, initial_force, initial_torque, *, capacity_scale=1., backpressure_factor=1.3,
                 backpressure_cap=0.35, overtravel=0.12, friction_smoothing=0.01):
        """``cylinders``: list of records (area_A_m2, area_B_m2, lower_m, upper_m, valve_flow_m3_s, coulomb_N,
        viscous_N_s_m); ``rotary``: list of records (limit_Nm, response_s, coulomb_Nm, viscous_Nm_s_rad); ``params``:
        supply_pressure_Pa, bulk_modulus_Pa, dead_volume_m3, pressure_response_s, leakage_m3_s_Pa,
        minimum_command_pressure_Pa."""
        self.n, self.nr = len(cylinders), len(rotary)
        g = lambda key, recs: np.array([float(r[key]) for r in recs])
        self.A, self.B = g('area_A_m2', cylinders), g('area_B_m2', cylinders)
        self.lower, self.upper = g('lower_m', cylinders), g('upper_m', cylinders)
        self.stroke = self.upper - self.lower
        self.qmax = g('valve_flow_m3_s', cylinders)
        self.coulomb, self.viscous = g('coulomb_N', cylinders), g('viscous_N_s_m', cylinders)
        self.ps = float(params['supply_pressure_Pa']) * capacity_scale
        self.beta = float(params['bulk_modulus_Pa'])
        self.dead = float(params['dead_volume_m3'])
        self.tau_p = float(params['pressure_response_s'])
        self.leak = float(params['leakage_m3_s_Pa'])
        self.low = float(params['minimum_command_pressure_Pa'])
        self.backpressure_factor, self.backpressure_cap = float(backpressure_factor), float(backpressure_cap)
        self.overtravel, self.eps = float(overtravel), float(friction_smoothing)
        self.rot_limit = g('limit_Nm', rotary) * capacity_scale
        self.rot_tau = g('response_s', rotary)
        self.rot_coulomb, self.rot_viscous = g('coulomb_Nm', rotary), g('viscous_Nm_s_rad', rotary)
        self.pressure = self.targets(np.asarray(initial_force, float))
        self.rotary = np.clip(np.asarray(initial_torque, float), -self.rot_limit, self.rot_limit)
        self.saturation_count = 0
        self.relief_count = 0

    def targets(self, force, velocity=None):
        """Chamber pressure targets (n, 2) that deliver ``force`` with the smallest pressures above the minimum pressure,
        raised by the meter-out back pressure of the discharging chamber when the cylinder moves."""
        low, A, B = self.low, self.A, self.B
        pa = np.where(force >= 0, (force + B * low) / A, low)
        pb = np.where(force >= 0, low, (A * low - force) / B)
        if velocity is not None:
            # discharging Q from a chamber through the orifice law Q = q_max sqrt(p / p_s) needs p >= p_s (Q / q_max)^2;
            # the target uses factor * p_s (Q / q_max)^2, capped at cap * p_s so that fast motion cannot cancel the command
            va = np.minimum(A * velocity, 0)
            vb = np.minimum(-B * velocity, 0)
            ra = np.minimum(low + self.backpressure_factor * self.ps * (va / self.qmax) ** 2, self.backpressure_cap * self.ps)
            rb = np.minimum(low + self.backpressure_factor * self.ps * (vb / self.qmax) ** 2, self.backpressure_cap * self.ps)
            preload = np.maximum.reduce([A * (ra - pa), B * (rb - pb), np.zeros(self.n)])
            pa = pa + preload / A
            pb = pb + preload / B
        return np.clip(np.column_stack((pa, pb)), 0, self.ps)

    def volumes(self, q):
        x = np.asarray(q, float) - self.lower                       # extension from the retracted end
        if np.any(x < -self.overtravel) or np.any(x > self.stroke + self.overtravel):
            raise ValueError('Cylinder left its stroke domain: extension ' + str(x) + ' of ' + str(self.stroke))
        x = np.clip(x, 0., self.stroke)
        return self.dead + np.column_stack((self.A * x, self.B * (self.stroke - x)))

    def pressure_force(self):
        return self.A * self.pressure[:, 0] - self.B * self.pressure[:, 1]

    def evaluate(self, q, v, force_command, torque_command, rotary_velocity, dt):
        v = np.asarray(v, float)
        p = self.pressure
        V = self.volumes(q)
        beta = self.beta
        dv = np.column_stack((self.A * v, -self.B * v))
        target = self.targets(np.asarray(force_command, float), v)
        leak = self.leak * (p[:, 0] - p[:, 1])
        L = np.column_stack((leak, -leak))
        request = dv + L + V / beta * (target - p) / self.tau_p
        capacity = self.qmax[:, None] * np.sqrt(np.maximum(np.where(request >= 0, self.ps - p, p), 0) / self.ps)
        flow = np.clip(request, -capacity, capacity)
        unbounded = p + dt * beta / V * (flow - dv - L)
        bounded = np.clip(unbounded, 0, self.ps)
        relief = (unbounded - bounded) * V / (dt * beta)
        dp = beta / V * (flow - dv - L - relief)
        fric = self.coulomb * np.tanh(v / self.eps) + self.viscous * v
        w = np.asarray(rotary_velocity, float)
        rf = self.rot_coulomb * np.tanh(w / self.eps) + self.rot_viscous * w
        force = self.A * p[:, 0] - self.B * p[:, 1] - fric
        torque = self.rotary - rf
        dr = (np.clip(np.asarray(torque_command, float), -self.rot_limit, self.rot_limit) - self.rotary) / self.rot_tau
        saturated = bool(np.any(np.abs(request) > capacity + 1e-15))
        limited = bool(np.any(target <= 0) | np.any(target >= self.ps))
        self.saturation_count += int(saturated)
        self.relief_count += int(np.any(relief != 0))
        supply = self.ps * np.maximum(flow, 0).sum()
        power = dict(supply=float(supply), throttle=float(supply - np.sum(p * flow)), friction=float(fric @ v + rf @ w),
                     mechanical=float(force @ v + torque @ w))
        return dict(force=force, torque=torque, pressure_derivative=dp, rotary_derivative=dr, flow=flow, power=power,
                    valve_saturated=saturated, pressure_limit_active=limited, pressure_max=float(p.max()) if self.n else 0.)

    def advance(self, item, dt):
        self.pressure = self.pressure + dt * item['pressure_derivative']
        self.rotary = self.rotary + dt * item['rotary_derivative']
        if self.n and (self.pressure.min() < -1e-6 or self.pressure.max() > self.ps + 1e-6):
            raise ValueError('Pressure bound violated')
        self.pressure = np.clip(self.pressure, 0, self.ps)


class TrackDrive:
    """First-order torque response per side with a traction-derived limit (rigid-roller surrogate of a track)."""

    def __init__(self, torque_limit, rollers_per_side, response=0.05, viscous=40.):
        self.limit, self.tau, self.viscous, self.n = float(torque_limit), float(response), float(viscous), int(rollers_per_side)
        self.torque = np.zeros(2)                  # per roller, per side (L, R)

    def evaluate(self, command_per_roller2, roller_speeds, dt):
        cmd = np.clip(np.asarray(command_per_roller2, float), -self.limit, self.limit)
        dtau = (cmd - self.torque) / self.tau
        effort = np.r_[np.full(self.n, self.torque[0]), np.full(self.n, self.torque[1])] - self.viscous * np.asarray(roller_speeds)
        return dict(effort=effort, derivative=dtau, saturated=bool(np.any(np.abs(command_per_roller2) > self.limit)))

    def advance(self, item, dt):
        self.torque = np.clip(self.torque + dt * item['derivative'], -self.limit, self.limit)


class TravelController:
    """Bounded base-pose tracking with a PI roller-speed servo (adapted from RoboCompiler v26)."""

    def __init__(self, radius, gauge, *, forward_bound=0.55, yaw_rate_bound=0.20, kp=3000., ki=2500., ibound=3.0):
        self.radius, self.gauge = float(radius), float(gauge)
        self.forward_bound, self.yaw_rate_bound = forward_bound, yaw_rate_bound
        self.kp, self.ki, self.ibound = kp, ki, ibound
        self.integral = np.zeros(2)
        self.last = {}

    def evaluate(self, pose, yaw_rate, speeds, target, rate, dt, *, update_outer=True):
        """``pose`` (x, y, yaw) and yaw rate of the machine frame, mean roller speeds (left, right) -> torque per roller (L, R)."""
        if update_outer or not self.last:
            yaw = pose[2]
            f = np.array([np.cos(yaw), np.sin(yaw)])
            lateral = np.array([-np.sin(yaw), np.cos(yaw)])
            error = target[:2] - pose[:2]
            heading = np.arctan2(np.sin(target[2] - yaw), np.cos(target[2] - yaw))
            ref_forward = np.array([np.cos(target[2]), np.sin(target[2])])
            ref_speed = float(rate[:2] @ ref_forward)
            v = np.clip(rate[:2] @ f + 0.65 * (error @ f), -self.forward_bound, self.forward_bound)
            steering = 3.0 * ref_speed * float(error @ lateral)
            damping = 0.8 * (float(rate[2]) - yaw_rate)
            omega = np.clip(rate[2] + 1.3 * heading + steering + damping, -self.yaw_rate_bound, self.yaw_rate_bound)
            desired = np.array([v - 0.5 * self.gauge * omega, v + 0.5 * self.gauge * omega]) / self.radius
            self.last = dict(pose=np.array(pose, float), target=np.asarray(target, float).copy(), desired=desired, error=error,
                             heading_error=heading, yaw_rate=yaw_rate)
        err = self.last['desired'] - np.asarray(speeds, float)
        self.integral = np.clip(self.integral + dt * err, -self.ibound, self.ibound)
        torque = self.kp * err + self.ki * self.integral
        self.last.update(speed=np.asarray(speeds, float), command=torque)
        return torque
