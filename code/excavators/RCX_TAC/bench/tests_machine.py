"""Offline tests of the generated interfaces on one machine (no plant).

    python run.py <machine> tests

The machine-dependent tests of the earlier single-machine release, written for any machine adapter: identities of
the generated body plan against the ported compiler oracle, derivatives against finite differences, loop consistency
of the required motion at the tangent lift and at every cut joint, the VDC assembly against the reduced dynamics, the
barrier rows, one runtime step against its own rows, covariance under a change of the coordinate partition, the
actuator capability models, the validation of the specification, limits on a cut joint and the broad phase.  The
machine-independent tests (tests/test_rcx_core.py) are run as well; the report lists both.

Random configurations are drawn inside the windows that the adapter declares for references (bench.reference.Planner),
with a moving base (random pose, twist and acceleration of the graph root frame).

A machine whose graph root is the ground (a mechanism on declared ground contacts, specification section ``support``)
is tested at rest of the root frame, with the commanded coordinates drawn at random and the held coordinates (contact
anchors) at their seed values; its support interface has tests of its own: the contact forces against the momentum
balance of the whole machine and against the reduced dynamics of all independent coordinates, the distribution of the
nominal efforts against a direct minimisation, and the contact rows against the contact forces of the delivered
efforts.
"""
from __future__ import annotations

import copy
import importlib.util
import time
import traceback
from pathlib import Path

import numpy as np

from core.reduced import floating_rows
from rcx import spec as spec_mod
from rcx.actuators import Channel
from rcx.compiler import compile_extended
from rcx.filter import FilterOptions
from rcx.lift import extended_lift
from rcx.runtime import SafeVDC

from .reference import Planner, closed_configuration
from .world import code_hash, save_json

ROOT = Path(__file__).resolve().parents[1]
NOT_APPLICABLE = 'N/A'          # returned by a test of an interface that the machine does not have (ground support)


def expm_so3(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


class MachineTests:
    def __init__(self, machine, seed=11):
        self.machine = machine
        self.rng = np.random.default_rng(seed)
        self.phys = machine.physical()
        self.site = machine.test_site()
        self.spec = machine.spec(self.phys, **self.site)
        self.xc = compile_extended(self.phys, self.spec)
        self.ci = self.xc.ci
        self.n = self.xc.d                                    # commanded independent coordinates
        self.moving_root = not machine.ground_root            # the graph root is a body of the machine (it moves)
        planner = Planner(self.phys, self.xc, machine)
        lo, hi, _ = planner.task_box()
        self.lo = np.where(np.isfinite(lo), lo, -1.2)
        self.hi = np.where(np.isfinite(hi), hi, 1.2)
        self.narrow = np.zeros(self.n, bool)                   # coordinates with a test range of their own (pose of a chassis on its legs)
        for jid, (a, b) in machine.test_ranges(self.phys).items():
            if jid in planner.task_ids:
                self.lo[planner.index(jid)], self.hi[planner.index(jid)] = a, b
                self.narrow[planner.index(jid)] = True
        # the planner's coordinates are the task coordinates; the compiled interface of the default partition uses the same
        if list(planner.task_ids) != list(self.xc.free_ids):
            raise ValueError('the default partition must be the task partition')

    # ------------------------------------------------------------------ states
    def random_u(self):
        return self.rng.uniform(self.lo, self.hi)

    @staticmethod
    def rates(ud, xc):
        """Speeds of all independent coordinates from the commanded ones (held coordinates at rest)."""
        if not len(xc.held):
            return np.asarray(ud, float)
        out = np.zeros(xc.d_all)
        out[xc.free] = ud
        return out

    def closed(self, u, xc=None, start=None):
        """Closed configuration at the commanded coordinates u (held coordinates at their seed values)."""
        xc = self.xc if xc is None else xc
        return closed_configuration(xc.ci, xc.compose(u), start=start)

    def random_state(self, moving_base=True):
        rng, xc = self.rng, self.xc
        u = self.random_u()
        ud = rng.uniform(-0.3, 0.3, self.n)
        q = self.closed(u)
        lift = xc.lift(q, self.rates(ud, xc))
        if moving_base and self.moving_root:
            base = dict(p=rng.uniform(-1, 1, 3), R=expm_so3(rng.uniform(-0.15, 0.15, 3)), v=rng.uniform(-0.3, 0.3, 3), w=rng.uniform(-0.2, 0.2, 3),
                        a=rng.uniform(-0.5, 0.5, 6))
        else:
            base = dict(p=np.zeros(3), R=np.eye(3), v=np.zeros(3), w=np.zeros(3), a=np.zeros(6))
        return u, ud, q, lift, base

    def kin_at(self, q, lift, base, dynamics=True, xc=None):
        xc = self.xc if xc is None else xc
        return xc.plan.evaluate(q[self.ci.tree_rows], lift.vT, base['p'], base['R'], base['v'], base['w'], dynamics=dynamics)

    def advance(self, u, ud, base, dt, start=None):
        """State after dt along the current velocities (base twist and independent rates constant)."""
        b2 = dict(base)
        b2['p'] = base['p'] + dt * base['v']
        b2['R'] = expm_so3(dt * base['w']) @ base['R']
        q2 = self.closed(u + dt * ud, start=start)
        return q2, self.xc.lift(q2, self.rates(ud, self.xc)), b2

    def measured_state(self, u, ud, base, xc=None):
        xc = self.xc if xc is None else xc
        ci = xc.ci
        q = self.closed(u, xc)
        lift = xc.lift(q, self.rates(ud, xc))
        kin = xc.plan.evaluate(q[ci.tree_rows], lift.vT, base['p'], base['R'], base['v'], base['w'])
        Mr, hr, L, C, _ = xc.reduce(lift, kin, base['a'])
        return q, lift, kin, Mr, hr, L, C

    @staticmethod
    def holding_efforts(xc, lift, kin, base, hr, C):
        """Channel efforts with  C phi = h_r  (they hold the state); with more channels than commanded coordinates: the
        nominal distribution of the generated support interface."""
        if not xc.redundant:
            return np.linalg.solve(C, hr)
        Mxf, hx, Cx = xc.support.held_maps(lift, kin, base['a'], xc.S_port, xc.G)
        D, D_lam = xc.nominal_map(C, Cx)
        return D @ hr + D_lam @ hx

    @staticmethod
    def controller_base(base):
        return dict(position=base['p'], rotation=base['R'], velocity=np.r_[base['v'], base['w']], acceleration=base['a'])

    # ------------------------------------------------------------------ generated body plan
    def test_body_plan_against_ported_oracle(self):
        ci = self.ci
        for _ in range(4):
            u, ud, q, lift, base = self.random_state()
            kin = self.kin_at(q, lift, base)
            qt = q[ci.tree_rows]
            local = ci.dyn.evaluate(qt, lift.vT, np.zeros(3))
            fl = floating_rows(local, ci.comp.cmg, qt, base['p'], base['R'], np.r_[base['v'], base['w'], lift.vT], ci.gravity)
            assert np.max(np.abs(kin.M - fl['mass_matrix'])) < 1e-9 * np.max(np.abs(kin.M))
            assert np.max(np.abs(kin.h - fl['bias_forces'])) < 1e-9 * np.max(np.abs(kin.h))

    def test_jacobians_and_derivatives_by_finite_differences(self):
        dt = 1e-5
        for _ in range(3):
            u, ud, q, lift, base = self.random_state()
            kin = self.kin_at(q, lift, base)
            qp, lp, bp = self.advance(u, ud, base, dt, q)
            qm, lm, bm = self.advance(u, ud, base, -dt, q)
            kp, km = self.kin_at(qp, lp, bp), self.kin_at(qm, lm, bm)
            v_fd = (kp.p - km.p) / (2 * dt)
            assert np.max(np.abs(v_fd - kin.J[:, :3] @ kin.nu)) < 1e-7
            D_fd = (kp.J - km.J) / (2 * dt)
            assert np.max(np.abs(D_fd - kin.D)) < 2e-6
            M_fd = (kp.M - km.M) / (2 * dt)
            assert np.max(np.abs(M_fd - (kin.C + kin.C.T))) < 2e-5 * np.max(np.abs(kin.M))

    # ------------------------------------------------------------------ loop consistency
    def test_lift_derivative_and_loop_consistency_of_required_motion(self):
        ci, rng, n = self.ci, self.rng, self.n
        for _ in range(4):
            u, ud, q, lift, base = self.random_state()
            ref = ci.lift(q, self.rates(ud, self.xc))
            assert np.max(np.abs(lift.Edot @ ud - ref['c'])) < 1e-10                       # curvature of the paper
            u_r, ur_dot = rng.uniform(-0.5, 0.5, n), rng.uniform(-1, 1, n)
            Jc = lift.closure_J
            assert np.max(np.abs(Jc @ (lift.E @ u_r))) < 1e-10                             # required velocity closes the loops
            eps = 1e-6
            Jdot = (ci.graph.residual(q + eps * lift.v)[1] - ci.graph.residual(q - eps * lift.v)[1]) / (2 * eps)
            acc = Jc @ (lift.E @ ur_dot + lift.Edot @ u_r) + Jdot @ (lift.E @ u_r)
            assert np.max(np.abs(acc)) < 1e-7                                              # and so does its derivative

    def test_required_twists_are_compatible_at_every_cut_joint(self):
        """Body-level statement: the required twists of the two bodies of a cut (loop-closing) revolute joint give the same
        velocity of the joint point and a relative angular velocity along the joint axis."""
        ci, plan = self.ci, self.xc.plan
        joints = {j['id']: j for j in ci.comp.cmg['joints']}
        assert len(ci.cut_ids) > 0
        for _ in range(3):
            u, ud, q, lift, base = self.random_state()
            kin = self.kin_at(q, lift, base, dynamics=False)
            nu_r = kin.nu.copy()
            nu_r[6:] = lift.ET @ self.rng.uniform(-0.5, 0.5, self.n)
            for jid in ci.cut_ids:
                j = joints[jid]
                A, B = plan.index[j['base_body']], plan.index[j['follower_body']]
                TA, TB = np.asarray(j['T_BJ']), np.asarray(j['T_FJ'])
                pA, JA, _ = plan.point(kin, A, TA[:3, 3])
                pB, JB, _ = plan.point(kin, B, TB[:3, 3])
                assert np.max(np.abs(pA - pB)) < 1e-9                                      # the loop is closed
                assert np.max(np.abs((JA - JB) @ nu_r)) < 1e-9                             # required point velocities agree
                axis = kin.R[A] @ TA[:3, :3] @ np.asarray(j['axis'], float)
                w_rel = (kin.J[A, 3:] - kin.J[B, 3:]) @ nu_r
                assert np.max(np.abs(np.cross(w_rel, axis))) < 1e-9                        # relative rotation about the pin

    # ------------------------------------------------------------------ VDC and storage
    def test_vdc_assembly_equals_reduced_dynamics_and_affine_map(self):
        xc, rng, n = self.xc, self.rng, self.n
        for _ in range(4):
            u, ud, q, lift, base = self.random_state()
            kin = self.kin_at(q, lift, base)
            Mr, hr, L, C, Mr_dot = xc.reduce(lift, kin, base['a'])
            ET, EdT = lift.ET, lift.EdotT
            # zero feedback gain, required motion = actual motion: the assembled required force is the reduced inverse dynamics
            udd = rng.uniform(-1, 1, n)
            nu_r = kin.nu.copy()
            mod = xc.plan.vdc(kin, nu_r, np.r_[base['a'], ET @ udd + EdT @ ud], 0.)
            assert np.max(np.abs(ET.T @ mod['tau'][6:] - (Mr @ udd + hr))) < 1e-8 * np.max(np.abs(hr))
            # general required motion with body gains: affine map used by the effort rows
            k = np.full(len(xc.plan.rigid_ids), 12.)
            k[::2] = 20.
            u_r, ur_dot = rng.uniform(-0.5, 0.5, n), rng.uniform(-1, 1, n)
            nu_r[6:] = ET @ u_r
            mod = xc.plan.vdc(kin, nu_r, np.r_[base['a'], ET @ ur_dot + EdT @ u_r], k)
            Kr = ET.T @ xc.plan.gain_matrix(kin, k)[6:, 6:] @ ET
            Mqq, Mqb, Cqq, Cqb = kin.M[6:, 6:], kin.M[6:, :6], kin.C[6:, 6:], kin.C[6:, :6]
            N = ET.T @ (Mqq @ EdT + Cqq @ ET) + Kr
            c0 = ET.T @ (Mqb @ base['a'] + Cqb @ kin.nu[:6] + kin.grav[6:]) - Kr @ ud
            tau = ET.T @ mod['tau'][6:]
            assert np.max(np.abs(tau - (Mr @ ur_dot + N @ u_r + c0))) < 1e-9 * np.max(np.abs(tau))
            # storage: sum of the body accompanying functions = 1/2 s^T M_r s;  dissipation = s^T K_r s
            s = u_r - ud
            assert abs(mod['storage'].sum() - 0.5 * s @ Mr @ s) < 1e-10 * max(1., mod['storage'].sum())
            assert abs(mod['dissipation'].sum() - s @ Kr @ s) < 1e-9 * max(1., mod['dissipation'].sum())
            # passivity structure behind the storage balance:  M_r_dot - 2 E^T (M Edot + C E)  is skew-symmetric
            S = Mr_dot - 2. * ET.T @ (Mqq @ EdT + Cqq @ ET)
            assert np.max(np.abs(S + S.T)) < 1e-8 * np.max(np.abs(Mr_dot) + 1.)

    # ------------------------------------------------------------------ barriers
    def test_barrier_rows_by_finite_differences(self):
        xc, ci = self.xc, self.ci
        dt = 1e-5
        classes = set(xc.barriers.cls.tolist())
        assert {'stroke', 'efence'} <= classes                    # the test site declares planes; the machine its strokes
        for _ in range(3):
            u, ud, q, lift, base = self.random_state()
            kin = self.kin_at(q, lift, base)
            rows = xc.barriers.evaluate(kin, q[ci.tree_rows], broad_phase=False)
            qp, lp, bp = self.advance(u, ud, base, dt, q)
            qm, lm, bm = self.advance(u, ud, base, -dt, q)
            kp, km = self.kin_at(qp, lp, bp), self.kin_at(qm, lm, bm)
            rp = xc.barriers.evaluate(kp, qp[ci.tree_rows], broad_phase=False)
            rm = xc.barriers.evaluate(km, qm[ci.tree_rows], broad_phase=False)
            hd = rows.G @ kin.nu
            hd_fd = (rp.h - rm.h) / (2 * dt)
            assert np.max(np.abs(hd - hd_fd)) < 1e-6
            # second derivative for constant velocities:  d/dt (G nu) = G nu_dot + beta  with  nu_dot = [0; Edot_T u]
            hdd_fd = (rp.G @ kp.nu - rm.G @ km.nu) / (2 * dt)
            hdd = rows.G[:, 6:] @ lift.cT + rows.beta
            ok = np.abs(hdd - hdd_fd) < 1e-4 * (1. + np.abs(hdd))
            assert ok.mean() > 0.995                                   # rows on a Hessian seam of a box or segment are skipped
            assert np.all(rows.h == xc.barriers.values(kin.R, kin.p, q[ci.tree_rows]))
            if xc.barriers.support is not None:                        # the static-support rows: every one of them, both derivatives
                k0 = xc.barriers.support['start']
                sl = slice(k0, k0 + len(xc.barriers.support['edges']))
                assert 'support' in classes and sl.stop > sl.start
                assert np.all(ok[sl]) and np.all(np.isfinite(rows.G[sl])) and np.max(np.abs(hd - hd_fd)[sl]) < 1e-6

    # ------------------------------------------------------------------ runtime
    def test_runtime_step_satisfies_its_rows(self):
        xc, rng, n = self.xc, self.rng, self.n
        ctl = SafeVDC(xc, FilterOptions())
        feasible = exact = 0
        for k in range(24):                                         # random states can lie beyond a limit: such samples are counted out
            if k >= 4 and feasible >= 3 and exact >= 3:
                break
            ctl.reset()
            u, ud, q, lift, base = self.random_state()
            _, _, kin, Mr, hr, L, C = self.measured_state(u, ud, base)
            f_meas = xc.G @ self.holding_efforts(xc, lift, kin, base, hr, C)       # efforts that hold the state
            ref = (u + self.reference_offset(0.3), rng.uniform(-0.2, 0.2, n), np.zeros(n))
            phi = ctl.step(0., xc.compose(u), self.rates(ud, xc), self.controller_base(base), f_meas, ref)
            last = ctl.last
            assert ctl.stats['vdc_affine_max'] < 1e-10 and ctl.stats['storage_identity_max'] < 1e-10
            lo, hi = last['lo'], last['hi']
            if last['feasible']:
                feasible += 1
                assert np.all(last['c'] @ phi >= last['rhs'] - 1e-6 * (1. + np.abs(last['rhs'])))
                assert np.all(phi >= lo - 1e-6 * xc.scale) and np.all(phi <= hi + 1e-6 * xc.scale)
                if xc.support is not None:                            # contact rows of the same problem
                    cc, rc = last['contact_c'], last['contact_rhs']
                    assert len(rc) == 5 * xc.support.n and np.all(cc @ phi >= rc - 1e-6 * (1. + np.abs(rc)))
                # the command deviation is the sum of its parts: barrier rows, contact rows, effort bounds
                assert abs(last['P_cmd'] - (last['P_rows'] + last['P_contact'] + last['P_bounds'])) < 1e-6 * (1. + abs(last['P_cmd']))
            exact += int(last['velocity_level'] == 0)
            if last['velocity_level'] == 0:
                ur = last['u_r']
                assert np.all(last['hdot_r'] >= -last['rows'].alpha1[last['on']] * (last['h'] - last['sigma']) - 1e-7)
                g = xc.speed_headroom
                pv = last['L'] @ ud
                assert np.all(last['L'] @ ur <= xc.port_speed_hi + g * (xc.port_speed_hi - pv) + 1e-9)
                assert np.all(last['L'] @ ur >= xc.port_speed_lo - g * (pv - xc.port_speed_lo) - 1e-9)
                assert np.all(last['phi_r'] <= hi + 1e-6 * xc.scale) and np.all(last['phi_r'] >= lo - 1e-6 * xc.scale)
        assert feasible >= 3 and exact >= 3, f'only {feasible} feasible and {exact} exact samples were checked'

    def reference_offset(self, size):
        """Random offset of a task reference: ``size`` in the units of the task coordinates, a tenth of it for the
        coordinates with a test range of their own (pose of a chassis that stands on its legs)."""
        return self.rng.uniform(-size, size, self.n) * np.where(self.narrow, 0.1, 1.)

    def test_partition_covariance_of_generated_interfaces(self):
        """The same physical motion and the same task reference give the same channel efforts, row coefficients, row
        functions, margins and storage for the joint-space and the cylinder-space partition of the same graph, also over
        several samples (the filter states of the runtime are formed in the declared task coordinates)."""
        xc, ci, rng, n = self.xc, self.ci, self.rng, self.n
        other = next(k for k in self.phys['partition_requests'] if k != 'joint_space')
        xc_c = compile_extended(self.phys, self.spec, other)
        assert list(xc_c.ci.independent_ids) != list(ci.independent_ids)
        ctl_j, ctl_c = SafeVDC(xc, FilterOptions()), SafeVDC(xc_c, FilterOptions())
        ids_c = [ci.coordinate_ids.index(k) for k in xc_c.ci.independent_ids]
        scale = xc.scale
        rate = np.resize(np.array([0.02, -0.03, 0.05, 0.04]), n)
        exact = 0
        for _ in range(6):
            ctl_j.reset()
            ctl_c.reset()
            u0, ud, q, lift, base0 = self.random_state()
            ref0 = u0 + self.reference_offset(0.05)
            for k in range(4):
                t = 0.01 * k
                u = u0 + t * ud
                base = dict(base0, p=base0['p'] + t * base0['v'], R=expm_so3(t * base0['w']) @ base0['R'])
                q, lift, kin, Mr, hr, L, C = self.measured_state(u, ud, base)
                f_meas = xc.G @ (self.holding_efforts(xc, lift, kin, base, hr, C) + rng.uniform(-2e3, 2e3, xc.m))
                ref = (ref0 + t * rate, rate, np.zeros(n))
                pj = ctl_j.step(t, xc.compose(u), self.rates(ud, xc), self.controller_base(base), f_meas, ref)
                pc = ctl_c.step(t, q[ids_c], lift.v[ids_c], self.controller_base(base), f_meas, ref)
                a, b = ctl_j.last, ctl_c.last
                assert a['feasible'] == b['feasible'] and a['velocity_level'] == b['velocity_level']
                # A state whose velocity rows are inconsistent (a random state can lie far inside a limit) goes through the
                # relaxed shaping problem, and a sample whose effort rows are inconsistent through the two-stage fallback.
                # Their stages carry a tolerance of 1e-6, which the staged problem amplifies at states far beyond their
                # limits (largest difference seen on the machines of this workspace: 1.3e-5 of the effort scale at a state
                # 1.9 m inside a limit); such samples are compared at 1e-4, the exact ones at 1e-7 and 1e-6.
                relaxed = a['velocity_level'] > 0
                assert np.max(np.abs(a['phi_r'] - b['phi_r']) / scale) < (1e-4 if relaxed else 1e-7)
                exact += int(not relaxed and a['feasible'])
                assert np.max(np.abs(pj - pc) / scale) < (1e-6 if a['feasible'] and not relaxed else 1e-4)
                assert np.array_equal(a['on'], b['on'])
                assert np.max(np.abs(a['a'] - b['a']) * scale[None, :]) < 1e-6 * max(1., np.max(np.abs(a['a']) * scale[None, :]))
                assert np.max(np.abs(a['psi2'] - b['psi2'])) < 1e-6 * max(1., np.max(np.abs(a['psi2'])))
                # quantities that depend on the required velocity inherit the tolerance of the shaping problem
                loose = 1000. if relaxed else 1.
                assert abs(a['storage'] - b['storage']) < loose * 1e-7 * max(1., a['storage'])
                assert np.max(np.abs(a['margin'] - b['margin'])) < loose * 1e-6 * max(1., np.max(np.abs(a['margin'])))
                # the load estimate is a covector: compare it through the row drift it produces
                assert np.max(np.abs(a['hdd'] - b['hdd'])) < loose * 1e-6 * max(1., np.max(np.abs(a['hdd'])))
        assert exact >= 4                                           # the tight tolerances were exercised

    # ------------------------------------------------------------------ actuator models and specification
    def test_actuator_capability_models(self):
        xc = self.xc
        kinds = set()
        for ch in xc.channels:
            kinds.add(ch.type)
            th, thb, th2 = ch.response(xc.Ts)
            if ch.response_time > 0:
                r = ch.response_time / xc.Ts
                s = np.linspace(0., xc.Ts, 20001)
                y = 1. - np.exp(-s / ch.response_time)
                assert abs(th - y[-1]) < 1e-12
                assert abs(thb - np.trapezoid(y, s) / xc.Ts) < 1e-8
                inner = np.concatenate([[0.], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(s))])
                assert abs(th2 - np.trapezoid(inner, s) / xc.Ts ** 2) < 1e-8
                assert abs(th2 - (0.5 - r + r * r * th)) < 1e-12
            if ch.type == 'hydraulic_cylinder':
                for v in np.linspace(-0.3, 0.3, 13):
                    lo, hi = ch.bounds(np.full(len(ch.port_index), v))
                    assert lo <= hi
                lo0, hi0 = ch.bounds(np.zeros(len(ch.port_index)))
                assert ch.bounds(np.full(len(ch.port_index), 0.1))[1] < hi0        # less pushing force when extending
                assert ch.bounds(np.full(len(ch.port_index), -0.1))[0] > lo0       # less pulling force when retracting
                x_cap = np.sqrt((ch.cap - ch.p0 / ch.ps) / ch.kappa)
                assert abs(ch.speed_ret - ch.sigma * min(ch.q / ch.B, x_cap * ch.q / ch.A)) < 1e-12
        assert kinds                                                 # every machine has at least one channel type
        # the reserve (derate) moves both bounds inwards, also when they have the same sign (overrun of a flow-limited cylinder)
        recs = [c for c in xc.spec['actuators']['channels'] if c['type'] == 'hydraulic_cylinder' and len(c['ports']) == 1]
        if recs:
            rec = dict(recs[-1])
            full, reserve = Channel(dict(rec, derate=1.0), [0]), Channel(dict(rec, derate=0.9), [0])
            for v in np.linspace(-0.6, 0.6, 61):
                (lo1, hi1), (lo9, hi9) = full.bounds(np.array([v])), reserve.bounds(np.array([v]))
                assert lo9 >= lo1 - 1e-9 and hi9 <= hi1 + 1e-9

    def test_specification_validation(self):
        PHYS, SPEC = self.phys, self.spec
        bad = copy.deepcopy(SPEC)
        bad['safety']['clearance'][0]['jacobian'] = [[1, 0, 0]]
        try:
            spec_mod.validate(bad, PHYS)
        except spec_mod.SpecError:
            pass
        else:
            raise AssertionError('pre-authored generated structure accepted')
        bad = copy.deepcopy(SPEC)
        bad['geometry']['spheres'][0]['body'] = 'no_such_body'
        try:
            compile_extended(PHYS, bad)
        except (spec_mod.SpecError, ValueError):
            pass
        else:
            raise AssertionError('unknown body accepted')
        bad = copy.deepcopy(SPEC)
        bad['actuators']['channels'] = bad['actuators']['channels'][:-1]
        try:
            compile_extended(PHYS, bad)
        except (spec_mod.SpecError, ValueError):
            pass
        else:
            raise AssertionError('missing actuator channel accepted')

        def rejected(change, what, compile_too=False):
            broken = copy.deepcopy(SPEC)
            change(broken)
            try:
                spec_mod.validate(broken, PHYS)
                if compile_too:
                    compile_extended(PHYS, broken)
            except spec_mod.SpecError:
                return
            raise AssertionError(what + ' accepted')

        assert spec_mod.validate(copy.deepcopy(SPEC), PHYS)
        pair = SPEC['safety']['clearance'][0]
        rejected(lambda sp: sp['safety']['clearance'][0].update(dmin=0.2), 'misspelt key')                      # closed records
        rejected(lambda sp: sp.update(partition='cylinder_space'), 'unknown top-level key')
        rejected(lambda sp: sp['geometry']['obstacles'][0].update(sdf='x'), 'unknown obstacle key')
        rejected(lambda sp: sp['safety']['clearance'].append(dict(pair)), 'duplicate clearance id')
        rejected(lambda sp: sp['safety']['coordinate_limits'][0].update(lower=None, upper=None), 'limit without a bound')
        rejected(lambda sp: sp['safety']['coordinate_limits'][0].update(margin=-0.01), 'negative margin')
        rejected(lambda sp: sp['uncertainty'].pop('tool_body'), 'tool wrench bound without a tool body')
        multi = [c for c in SPEC['actuators']['channels'] if len(c['ports']) >= 2]
        if multi:                                                   # channel on a subset of its actuator group
            m0 = multi[0]
            rejected(lambda sp: next(c for c in sp['actuators']['channels'] if c['id'] == m0['id']).update(ports=m0['ports'][:1]),
                     'channel on a subset of its actuator group')
        rejected(lambda sp: sp['geometry']['obstacles'].append(dict(id='flat', type='plane', normal=[0., 0., 0.], offset=0.)), 'plane without a normal')
        rejected(lambda sp: sp['geometry']['obstacles'].append(dict(id='thin', type='box', center=[0., 0., 0.], half=[1., 0., 1.])), 'box without volume')
        # spheres on the owner body of their obstacle
        owner = {o['id']: o.get('frame', 'world') for o in SPEC['geometry']['obstacles']}
        own = next(((s['group'], oid) for oid, fr in owner.items() for s in SPEC['geometry']['spheres'] if s['body'] == fr), None)
        if own is not None:
            rejected(lambda sp: sp['safety']['clearance'].append(dict(id='own', groups=[own[0]], obstacle=own[1], alpha=[6., 16.])),
                     'spheres on the owner body of their obstacle')
        # task coordinates that do not determine the motion: a task coordinate replaced by a coordinate that depends on another one
        ci = self.ci
        E = ci.N
        dep = None
        task = list(self.xc.task.ids)
        for k, jid in enumerate(ci.coordinate_ids):
            if jid in ci.independent_ids:
                continue
            nz = np.flatnonzero(np.abs(E[k]) > 1e-9)
            if len(nz) == 1 and self.n >= 2 and ci.independent_ids[int(nz[0])] in task:
                dep = (jid, task.index(ci.independent_ids[int(nz[0])]))
                break
        if dep is not None:
            jid, i = dep
            ids = list(task)
            ids[(i + 1) % self.n] = jid                             # jid and task coordinate i are one degree of freedom
            rejected(lambda sp: sp['control'].update(task_ids=ids), 'dependent task coordinates', compile_too=True)
        # ground support (schema 1.1): closed records, anchors that are the x, y, z translation of the root frame, a support
        # polygon; redundant actuation is generated only together with declared contacts
        if 'support' in SPEC:
            con = SPEC['support']['contacts'][0]
            rejected(lambda sp: sp['support']['contacts'][0].update(mu=0.5), 'misspelt contact key')
            rejected(lambda sp: sp['support']['contacts'][0].update(friction=0.), 'contact without friction')
            rejected(lambda sp: sp['support']['contacts'][0].update(anchor=con['anchor'][::-1]), 'anchor that is not an x, y, z chain')
            rejected(lambda sp: sp['support']['contacts'][0].update(anchor=con['anchor'][:2]), 'anchor of two joints')
            rejected(lambda sp: sp['support']['contacts'][1].update(anchor=con['anchor']), 'anchor joints used by two contacts')
            rejected(lambda sp: sp['support']['contacts'].__delitem__(slice(2, None)), 'support polygon of two contacts')
            rejected(lambda sp: sp['support']['centroid'].update(margin=-0.1), 'negative support margin')
            rejected(lambda sp: sp.update(schema=spec_mod.SCHEMA), 'section support under schema 1.0')
            rejected(lambda sp: sp['support'].update(contacts=[]), 'support without a contact')
            # records that the declaration of a contact does not fit: an anchor whose coordinates are not the position of the
            # contact point (attachment frame moved), and an anchor that is not joined to the mechanism by a point contact
            # (one of the three revolute joints behind it fixed; its axis moved off the contact point)
            jrec = {j['id']: j for j in PHYS['joints']}
            body = jrec[con['anchor'][2]]['body_b']
            wrist = next(j['id'] for j in PHYS['joints'] if j['id'] != con['anchor'][2] and body in (j['body_a'], j['body_b']))

            def records(change):
                bad = copy.deepcopy(PHYS)
                change({j['id']: j for j in bad['joints']})
                return bad

            def moved(j, key):
                T = np.asarray(j[key], float).copy()
                T[0, 3] += 0.5
                j[key] = T.tolist()
            for what, change, text in (
                    ('anchor frame that is not at the identity', lambda J: moved(J[con['anchor'][1]], 'T_AJ'), 'attachment frames at the identity'),
                    ('contact that transmits a moment', lambda J: J[wrist].update(type='fixed'), 'point contact'),
                    ('contact joint off the contact point', lambda J: moved(J[wrist], 'T_AJ' if J[wrist]['body_a'] == body else 'T_BJ'),
                     'pass through the contact point')):
                try:
                    spec_mod.validate(copy.deepcopy(SPEC), records(change))
                except spec_mod.SpecError as exc:
                    assert text in str(exc), f'{what}: rejected for another reason ({exc})'
                else:
                    raise AssertionError(f'{what} accepted')
            bad = copy.deepcopy(SPEC)
            del bad['support']
            bad['schema'] = spec_mod.SCHEMA
            try:
                compile_extended(PHYS, bad)
            except (spec_mod.SpecError, ValueError):
                pass
            else:
                raise AssertionError('held coordinates without declared contacts accepted')

    def test_limits_on_cut_joints(self):
        """A limit declared on a joint that the compiler cut (a loop-closing pin) gives rows that agree with the tangent lift
        in value, rate and acceleration bias and with finite differences along the motion; the plant-side value, computed
        from the body poses alone, is the same."""
        ci, machine = self.ci, self.machine
        joints = {j['id']: j for j in ci.comp.cmg['joints']}
        pin = next(j for j in ci.cut_ids if joints[j]['type'] == 'revolute')
        assert pin in ci.cut_ids and pin not in ci.tree_ids
        k = ci.coordinate_ids.index(pin)
        seed = float(ci.seed[k])
        lower, upper = seed - 1.0, seed + 0.6
        xc = compile_extended(self.phys, machine.spec(self.phys, joint_limits={pin: (lower, upper)}, joint_margin=0.03))
        rows_of = [i for i, pid in enumerate(xc.barriers.row_pair) if pid == f'{pin}_joint']
        assert len(rows_of) == 2
        plans = [r for r in xc.manifest['barriers']['plans'] if r.get('cut_joint') == pin]
        assert len(plans) == 2 and set(plans[0]['bodies']) == {joints[pin]['base_body'], joints[pin]['follower_body']}
        dt = 1e-5
        for _ in range(4):
            u, ud, q, lift, base = self.random_state()
            kin = xc.plan.evaluate(q[ci.tree_rows], lift.vT, base['p'], base['R'], base['v'], base['w'])
            rows = xc.barriers.evaluate(kin, q[ci.tree_rows], broad_phase=False)
            for i, sign, ref in ((rows_of[0], 1., lower + 0.03), (rows_of[1], -1., -(upper - 0.03))):
                assert abs(rows.h[i] - (sign * q[k] - ref)) < 1e-9
                assert np.max(np.abs(rows.G[i, :6])) < 1e-12                                   # no authority of the base motion
                assert np.max(np.abs(rows.G[i, 6:] @ lift.ET - sign * lift.E[k])) < 1e-9
                assert abs(rows.G[i, 6:] @ lift.cT + rows.beta[i] - sign * (lift.Edot[k] @ ud)) < 1e-6
            assert np.max(np.abs(xc.barriers.values(kin.R, kin.p, q[ci.tree_rows])[rows_of] - rows.h[rows_of])) < 1e-12
            qp, lp, bp = self.advance(u, ud, base, dt, q)
            qm, lm, bm = self.advance(u, ud, base, -dt, q)
            vals = []
            for qq, ll, bb in ((qp, lp, bp), (qm, lm, bm)):
                kk = xc.plan.evaluate(qq[ci.tree_rows], ll.vT, bb['p'], bb['R'], bb['v'], bb['w'], dynamics=False)
                vals.append(xc.barriers.values(kk.R, kk.p, qq[ci.tree_rows]))
            assert np.max(np.abs((rows.G @ kin.nu)[rows_of] - (vals[0] - vals[1])[rows_of] / (2 * dt))) < 1e-6
        # limits outside (-pi, pi) of the seed value on a cut joint have no unique angle
        try:
            compile_extended(self.phys, machine.spec(self.phys, joint_limits={pin: (seed - 4.0, None)}))
        except ValueError as exc:
            assert 'within pi of its seed value' in str(exc)
        else:
            raise AssertionError('limit of a cut joint beyond the branch of its seed accepted')

    def test_broad_phase_distance_covers_one_sample(self):
        SPEC, xc = self.spec, self.xc
        mb, Ts = SPEC['safety']['motion_bounds'], SPEC['actuators']['sample_time_s']
        alpha = {c['id']: c['alpha'] for c in SPEC['safety']['clearance']}
        assert len(xc.barriers.blocks) > 0
        for b in xc.barriers.blocks:
            a1, a2 = alpha[b['pair']]
            D = (mb['point_acceleration_m_s2'] + mb['row_margin_m_s2'] + (a1 + a2) * mb['point_speed_m_s']) / (a1 * a2)
            assert abs(b['activation'] - (D + Ts * mb['point_speed_m_s'])) < 1e-12
            # one sample after h = activation the row function still exceeds the declared margin bound for the declared motion bounds
            h_next = b['activation'] - Ts * mb['point_speed_m_s']
            assert a1 * a2 * h_next - (a1 + a2) * mb['point_speed_m_s'] - mb['point_acceleration_m_s2'] >= mb['row_margin_m_s2'] - 1e-9

    def test_runtime_options(self):
        """Nominal-effort fraction below one and no filter on the derivative of the shaping correction: the runtime runs and
        the nominal efforts stay inside the reduced capability."""
        spec = self.machine.spec(self.phys, nominal_effort_fraction=0.8, correction_filter=0., **self.site)
        xc = compile_extended(self.phys, spec)
        ctl = SafeVDC(xc, FilterOptions())
        n = self.n
        checked = 0
        for state in range(8):                                      # a random state beyond a limit relaxes the shaping problem
            if state >= 1 and checked >= 3:
                break
            ctl.reset()
            u, ud, q, lift, base = self.random_state()
            _, lift, kin, Mr, hr, L, C = self.measured_state(u, ud, base, xc)
            f_meas = xc.G @ self.holding_efforts(xc, lift, kin, base, hr, C)
            for k in range(3):
                ref = (u + np.where(self.narrow, 0.02, 0.2), np.full(n, 0.05) * np.where(self.narrow, 0.1, 1.), np.zeros(n))
                ctl.step(0.01 * k, xc.compose(u), self.rates(ud, xc), self.controller_base(base), f_meas, ref)
                last = ctl.last
                if last['velocity_level'] == 0:
                    checked += 1
                    lo, hi = last['lo'], last['hi']
                    assert np.all(last['phi_r'] <= hi - 0.2 * np.abs(hi) + 1e-6 * xc.scale) and np.all(last['phi_r'] >= lo + 0.2 * np.abs(lo) - 1e-6 * xc.scale)
            assert ctl.stats['vdc_affine_max'] < 1e-9
        assert checked >= 3, f'only {checked} samples without relaxation were checked'

    def test_static_support_centre_of_pressure(self):
        """bench.stability: about the returned centre of pressure the weight of the machine and the given loads have no
        moment about a horizontal axis (the defining property), without loads it is the ground projection of the centre of
        mass, and the edge margins are its signed distances from the edges of the support polygon (positive inside)."""
        from .stability import G, Stability
        poly = self.machine.support_polygon(self.phys)
        if poly is None:
            return NOT_APPLICABLE
        st = Stability(self.phys, self.xc)
        assert abs(st.total - sum(b['mass_kg'] for b in self.phys['bodies'])) < 1e-9 * st.total       # every body is a frame of the plan
        P = np.asarray(poly, float)
        for _ in range(3):
            u = self.random_u()
            m = st.margins(u, poly)
            assert np.allclose(m['cop'], m['com'][:2], atol=1e-10)
            loads = [(np.array([900., -400., -6000.]), np.array([4.0, 1.5, 0.8])), (np.array([0., 0., -2500.]), np.array([-1.0, 2.0, 1.5]))]
            ml = st.margins(u, poly, loads=loads)
            c = np.r_[ml['cop'], 0.]
            forces = [(np.array([0., 0., -m['mass'] * G]), m['com'])] + loads
            moment = sum(np.cross(p - c, f) for f, p in forces)
            assert np.max(np.abs(moment[:2])) < 1e-6 * m['mass'] * G
            for k, (a, b) in enumerate(zip(P, np.roll(P, -1, axis=0))):
                e = b - a
                d = (e[0] * (ml['cop'][1] - a[1]) - e[1] * (ml['cop'][0] - a[0])) / np.linalg.norm(e)     # left of a -> b: inside
                assert abs(d - ml['margins'][k]) < 1e-10
            assert abs(ml['min_margin'] - min(ml['margins'])) < 1e-15

    # ------------------------------------------------------------------ support interface (mechanism on ground contacts)
    def _support_state(self):
        """Random state with random delivered efforts: (u, ud, q, lift, kin, base, Mr, hr, C, f, udd, lam) with the commanded
        accelerations of the reduced model and the contact forces of the generated force map (no external load)."""
        xc, sup = self.xc, self.xc.support
        u, ud, q, lift, base = self.random_state()
        kin = self.kin_at(q, lift, base)
        Mr, hr, L, C, _ = xc.reduce(lift, kin, base['a'])
        f = self.holding_efforts(xc, lift, kin, base, hr, C) + self.rng.uniform(-0.05, 0.05, xc.m) * xc.scale
        Mxf, hx, Cx = sup.held_maps(lift, kin, base['a'], xc.S_port, xc.G)
        A, b = sup.force_map(Mr, hr, C, Mxf, hx, Cx, np.zeros(xc.d), np.zeros(len(xc.held)))
        udd = np.linalg.solve(Mr, C @ f - hr)
        return u, ud, q, lift, kin, base, Mr, hr, C, f, udd, A @ f + b

    def test_contact_forces_balance_the_momentum_of_the_machine(self):
        """Independent statement of mechanics for the generated contact-force map: the forces of the ground on the declared
        contacts, together with the weight, equal the rate of change of the linear and of the angular momentum of all bodies
        (about the root origin).  The efforts are internal and do not appear; body accelerations come from the generated
        Jacobians, the momenta from the mass records.

        The bodies of the virtual chains (those that the plant graph does not have: frames of the chassis pose and of the
        contact anchors) are held partly by the joint constraints of the root frame, which are no contact forces; their
        weight and inertia force (their total mass times the largest |a - g|) is the tolerance of the balance."""
        xc = self.xc
        if xc.support is None:
            return NOT_APPLICABLE
        plan, sup = xc.plan, xc.support
        bodies = {b['id']: b for b in self.phys['bodies']}
        g = np.asarray(xc.ci.gravity, float)
        real = {b['id'] for b in self.machine.physical(plant=True)['bodies']}
        m_virtual = sum(float(b['mass_kg']) for b in self.phys['bodies'] if b['id'] not in real)
        assert m_virtual < 1e-4 * sup.weight / np.linalg.norm(g)     # the virtual bodies are negligible against the machine
        for _ in range(3):
            u, ud, q, lift, kin, base, Mr, hr, C, f, udd, lam = self._support_state()
            nud = np.r_[base['a'], lift.ET @ udd + lift.cT]         # accelerations of the tree coordinates (held ones at rest)
            force, moment = np.zeros(3), np.zeros(3)
            a_max, reach = 0., 0.
            for bid in plan.rigid_ids:
                rec = bodies[bid]
                i = plan.index[bid]
                m, Ib = float(rec['mass_kg']), np.asarray(rec['inertia_com_kg_m2'], float)
                pc, Jc, Dc = plan.point(kin, bid, np.asarray(rec['com_m'], float))
                ac = Jc @ nud + Dc @ kin.nu                          # acceleration of the centre of mass
                R = kin.R[i]
                w = kin.J[i, 3:] @ kin.nu
                al = kin.J[i, 3:] @ nud + kin.D[i, 3:] @ kin.nu
                Iw = R @ Ib @ R.T
                force += m * (ac - g)
                moment += np.cross(pc, m * (ac - g)) + Iw @ al + np.cross(w, Iw @ w)
                a_max, reach = max(a_max, float(np.linalg.norm(ac - g))), max(reach, float(np.linalg.norm(pc)))
            lam3 = lam.reshape(-1, 3)
            P = np.array([kin.p[plan.index[c['frame']]] for c in sup.contacts])
            tol = 1e-7 * sup.weight + m_virtual * a_max
            assert np.max(np.abs(lam3.sum(axis=0) - force)) < tol
            assert np.max(np.abs(np.cross(P, lam3).sum(axis=0) - moment)) < tol * max(1., reach)

    def test_contact_forces_against_the_reduced_dynamics_of_all_coordinates(self):
        """The same contact forces from the reduced dynamics written for all independent coordinates of the partition (the
        anchors included, with zero speed and acceleration): rows of the anchors, without the generated split."""
        xc = self.xc
        if xc.support is None:
            return NOT_APPLICABLE
        ci = xc.ci
        for _ in range(3):
            u, ud, q, lift, kin, base, Mr, hr, C, f, udd, lam = self._support_state()
            full = extended_lift(ci, q, self.rates(ud, xc))          # all columns of the tangent lift
            ET = full.ET
            Mqq = kin.M[6:, 6:]
            M_all = ET.T @ Mqq @ ET
            h_all = ET.T @ (kin.h[6:] + Mqq @ full.cT + kin.M[6:, :6] @ base['a'])
            C_all = (xc.S_port.T @ ET).T @ xc.G
            acc = np.zeros(xc.d_all)
            acc[xc.free] = udd
            res = M_all @ acc + h_all - C_all @ f                    # = generalized constraint forces: zero on the commanded rows
            assert np.max(np.abs(res[xc.free])) < 1e-7 * xc.support.weight
            assert np.max(np.abs(res[xc.held] - lam)) < 1e-7 * xc.support.weight

    def test_effort_distribution_of_redundant_actuation(self):
        """phi = D tau + D_lam lam_0 satisfies C phi = tau and minimises the declared cost: compared with a least-squares
        solution on the null space of C."""
        xc = self.xc
        if xc.support is None or not xc.redundant:
            return NOT_APPLICABLE
        sup = xc.support
        for _ in range(3):
            u, ud, q, lift, kin, base, Mr, hr, C, f, udd, lam = self._support_state()
            Mxf, hx, Cx = sup.held_maps(lift, kin, base['a'], xc.S_port, xc.G)
            D, D_lam = xc.nominal_map(C, Cx)
            assert np.max(np.abs(C @ D - np.eye(xc.d))) < 1e-9 and np.max(np.abs(C @ D_lam)) < 1e-9 * np.max(np.abs(D_lam))
            tau = hr + self.rng.uniform(-0.02, 0.02, xc.d) * np.abs(hr).max()
            lam0 = hx + Mxf @ np.linalg.solve(Mr, tau - hr)          # contact forces of the nominal model at zero efforts
            phi = D @ tau + D_lam @ lam0
            # direct minimisation of  sum (phi / scale)^2 + w sum |lam_t / weight|^2,  lam_t = St (lam0 - Cx phi),  over C phi = tau
            Z = np.linalg.svd(C)[2][xc.d:].T
            phi_p = np.linalg.lstsq(C, tau, rcond=None)[0]
            a1 = np.diag(1. / xc.scale)
            a2 = np.sqrt(sup.tangential_weight) / sup.weight * (sup.St @ Cx)
            b2 = np.sqrt(sup.tangential_weight) / sup.weight * (sup.St @ lam0)
            Als = np.vstack([a1 @ Z, a2 @ Z])
            bls = np.r_[-a1 @ phi_p, b2 - a2 @ phi_p]
            y = np.linalg.lstsq(Als, bls, rcond=None)[0]
            assert np.max(np.abs(phi - (phi_p + Z @ y)) / xc.scale) < 1e-8
            # and the tangential forces are indeed smaller than with the efforts of smallest weighted norm
            phi_min = (xc.scale[:, None] ** 2 * C.T) @ np.linalg.solve(C @ (xc.scale[:, None] ** 2 * C.T), tau)
            t_opt = np.linalg.norm(sup.St @ (lam0 - Cx @ phi))
            t_min = np.linalg.norm(sup.St @ (lam0 - Cx @ phi_min))
            assert t_opt <= t_min + 1e-9 * sup.weight

    def test_contact_rows_bound_the_contact_forces_of_the_delivered_efforts(self):
        """Commands that satisfy the generated contact rows give contact forces inside the declared limits at the end of the
        sample for every force-loop error inside its declared bound: normal force above its minimum plus the reserve,
        tangential components inside the friction pyramid."""
        from rcx.filter import contact_rows
        xc = self.xc
        if xc.support is None:
            return NOT_APPLICABLE
        sup, rng = xc.support, self.rng
        checked = 0
        for _ in range(3):
            u, ud, q, lift, kin, base, Mr, hr, C, f, udd, lam = self._support_state()
            Mxf, hx, Cx = sup.held_maps(lift, kin, base['a'], xc.S_port, xc.G)
            A, b = sup.force_map(Mr, hr, C, Mxf, hx, Cx, np.zeros(xc.d), np.zeros(len(xc.held)))
            cc, rc, margin = contact_rows(sup, A, b, f, xc.theta, xc.e_force)
            assert cc.shape == (5 * sup.n, xc.m) and np.all(margin >= sup.reserve)
            for _ in range(200):
                phi = f + rng.uniform(-0.2, 0.2, xc.m) * xc.scale
                if not np.all(cc @ phi >= rc):
                    continue
                checked += 1
                delivered = f + xc.theta * (phi - f) + rng.uniform(-1., 1., xc.m) * xc.e_force
                l3 = (A @ delivered + b).reshape(-1, 3)
                for c, (lx, ly, lz) in zip(sup.contacts, l3):
                    k = c['friction'] / np.sqrt(2.)
                    assert lz >= c['normal_min'] + sup.reserve - 1e-6
                    assert abs(lx) <= k * lz - sup.reserve + 1e-6 and abs(ly) <= k * lz - sup.reserve + 1e-6
                    assert np.hypot(lx, ly) <= c['friction'] * lz + 1e-6      # inside the friction cone
        assert checked > 0

    def test_contact_rows_stay_when_the_barrier_rows_are_off(self):
        """Variant without barrier rows in the effort problem: a mechanism on ground contacts still commands efforts that
        satisfy its contact rows (one QP with the contact rows only); with the contact rows switched off as well the
        command is the nominal effort inside the effort bounds."""
        xc, rng, n = self.xc, self.rng, self.n
        if xc.support is None:
            return NOT_APPLICABLE
        keep = SafeVDC(xc, FilterOptions(force_filter=False, velocity_shaping=False))
        none = SafeVDC(xc, FilterOptions(force_filter=False, velocity_shaping=False, contact_rows=False))
        each = SafeVDC(xc, FilterOptions(coordinated=False))          # barrier rows channel by channel, then the contact rows
        checked = binding = checked_each = 0
        for k in range(12):
            if k >= 3 and checked >= 3 and binding >= 2 and checked_each >= 2:
                break
            u, ud, q, lift, base = self.random_state()
            _, _, kin, Mr, hr, L, C = self.measured_state(u, ud, base)
            f_meas = xc.G @ self.holding_efforts(xc, lift, kin, base, hr, C)
            # efforts that are not those of rest, so that the contact rows have something to correct
            f_meas = f_meas + xc.G @ (rng.uniform(-0.1, 0.1, xc.m) * xc.scale)
            ref = (u + self.reference_offset(0.3), rng.uniform(-0.2, 0.2, n), np.zeros(n))
            for ctl in (keep, none, each):
                ctl.reset()
            args = (0., xc.compose(u), self.rates(ud, xc), self.controller_base(base), f_meas, ref)
            phi = keep.step(*args)
            last = keep.last
            assert not np.any(last['active']) and last['P_rows'] == 0.        # no barrier row in the effort problem
            if last['feasible']:
                checked += 1
                binding += int(np.any(last['contact_active']))
                assert np.all(last['contact_c'] @ phi >= last['contact_rhs'] - 1e-6 * (1. + np.abs(last['contact_rhs'])))
                assert np.all(phi >= last['lo'] - 1e-6 * xc.scale) and np.all(phi <= last['hi'] + 1e-6 * xc.scale)
                assert abs(last['P_cmd'] - (last['P_contact'] + last['P_bounds'])) < 1e-6 * (1. + abs(last['P_cmd']))
            phi0 = none.step(*args)
            assert np.max(np.abs(phi0 - np.clip(none.last['phi_r'], none.last['lo'], none.last['hi'])) / xc.scale) < 1e-12
            assert len(none.last['contact_rhs']) == 0 and none.last['P_contact'] == 0.
            phi1 = each.step(*args)
            last = each.last
            if last['feasible']:
                checked_each += 1
                assert np.all(last['contact_c'] @ phi1 >= last['contact_rhs'] - 1e-6 * (1. + np.abs(last['contact_rhs'])))
                assert np.all(phi1 >= last['lo'] - 1e-6 * xc.scale) and np.all(phi1 <= last['hi'] + 1e-6 * xc.scale)
                assert abs(last['P_cmd'] - (last['P_rows'] + last['P_contact'] + last['P_bounds'])) < 1e-6 * (1. + abs(last['P_cmd']))
        assert checked >= 3 and binding >= 2 and checked_each >= 2, f'checked {checked}, with a binding contact row {binding}, per channel {checked_each}'

    # ------------------------------------------------------------------ runner
    def run_all(self):
        names = sorted(k for k in dir(self) if k.startswith('test_'))
        results, failed, skipped = [], 0, 0
        for name in names:
            t0 = time.perf_counter()
            try:
                if getattr(self, name)() == NOT_APPLICABLE:
                    skipped += 1
                    results.append((name, NOT_APPLICABLE, time.perf_counter() - t0))
                    print(f'N/A  {name}  (the machine has no such interface)', flush=True)
                    continue
                results.append((name, 'PASS', time.perf_counter() - t0))
                print(f'PASS {name}  ({time.perf_counter() - t0:.1f} s)', flush=True)
            except Exception as exc:                                   # noqa: BLE001
                failed += 1
                traceback.print_exc()
                results.append((name, f'FAIL: {exc}', time.perf_counter() - t0))
                print(f'FAIL {name}: {exc}', flush=True)
        print(f'{len(names) - failed - skipped}/{len(names) - skipped} passed' + (f', {skipped} not applicable' if skipped else ''))
        return results, failed


def core_tests():
    spec = importlib.util.spec_from_file_location('test_rcx_core', ROOT / 'tests' / 'test_rcx_core.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_all(vars(mod))


def run(machine):
    """All offline tests for one machine; writes results/<machine>/tests/report.json."""
    sha = code_hash(machine.name)
    out = machine.results_dir() / 'tests'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    t0 = time.perf_counter()
    print('--- machine-independent tests (tests/test_rcx_core.py)')
    core, core_failed = core_tests()
    print(f'--- tests on the compiled interfaces of {machine.name}')
    suite = MachineTests(machine)
    mach, mach_failed = suite.run_all()
    mach_na = sum(r == NOT_APPLICABLE for _, r, _ in mach)
    ci = suite.ci
    report = dict(status='COMPLETED' if core_failed + mach_failed == 0 else 'FAILED', machine=machine.name, code_sha256=sha,
                  wall_s=time.perf_counter() - t0,
                  counts=dict(core=len(core), core_failed=core_failed, machine=len(mach) - mach_na, machine_failed=mach_failed,
                              machine_not_applicable=mach_na),
                  core=[dict(name=n, result=r, seconds=s) for n, r, s in core], machine_tests=[dict(name=n, result=r, seconds=s) for n, r, s in mach],
                  compiled=dict(mobility=len(ci.independent_ids), commanded=int(suite.xc.d), held_ids=list(suite.xc.held_ids),
                                task_ids=list(suite.xc.task.ids), tree_coordinates=len(ci.tree_ids),
                                cut_joints=list(ci.cut_ids), channels=[c.id for c in suite.xc.channels], rows=int(suite.xc.barriers.R),
                                classes=sorted(set(suite.xc.barriers.cls.tolist())), site=suite.site))
    save_json(out / 'report.json', report)
    print(f"tests of {machine.name}: {len(core) - core_failed}/{len(core)} machine-independent, {len(mach) - mach_na - mach_failed}/"
          f"{len(mach) - mach_na} on the compiled interfaces" + (f' ({mach_na} not applicable)' if mach_na else '') + f"; status {report['status']}")
    return report
