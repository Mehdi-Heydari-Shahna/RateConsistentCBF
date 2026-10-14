"""Reference generation on the compiled graph (machine independent): joint segments, timelines, tool-path segments
by numerical inverse kinematics of the serial tool chain, and pacing against the declared actuator port speeds.

Only references are produced; the plant and the soil evolve natively.

Tool pose convention ("soil frame" of the bucket, see env.soil): x from the bucket pivot to the lip, y lateral, z towards
the belly.  With the lateral axis horizontal the pose is described by the position of a bucket point and the *pitch*
phi of the x axis in the vertical plane of the arm (phi = 0: lip pointing away from the machine, horizontal; phi = -pi/2:
lip pointing down; phi = -pi: lip pointing to the machine, mouth up), measured from the heading  y_s x e_z  of the tool.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from core.reduced import CompiledInterface


def smooth(t, start, end):
    """Quintic (C2) time scaling and its derivatives on [start, end]."""
    d = end - start
    s = np.clip((t - start) / d, 0., 1.)
    f = s ** 3 * (10 - 15 * s + 6 * s * s)
    df = 30 * s * s * (1 - s) ** 2 / d
    ddf = 60 * s * (1 - 3 * s + 2 * s * s) / d ** 2
    return f, df, ddf


def closed_configuration(ci, u, step=0.05, start=None):
    """Closed configuration of a compiled graph for all its independent coordinates ``u``; the continuation state of
    ``ci`` is left as it is.  Bounded continuation along the straight line in the independent coordinates from the
    present state of ``ci`` (or from the closed configuration ``start``); if that fails, the homotopy acquisition of the
    compiler from the same state.  (The acquisition deforms an artificial defect of the cut joints; for a target far from
    its start, e.g. a turned chassis on four ground contacts, it can stagnate although the configuration exists.)  Raises
    ValueError if both fail."""
    u = np.asarray(u, float)
    state = (ci.seed, ci.N, ci.mapping)
    stats = dict(ci.stats)
    try:
        if start is None:
            begin = (state[0].copy(), state[1].copy(), dict(state[2]))
        else:
            start = np.array(start, float)
            N, mapping = ci.pacdm.mapping(start)
            if N is None:
                raise ValueError(f'the start configuration is not a regular closed configuration: {mapping}')
            begin = (start, N, mapping)
        ci.seed, ci.N, ci.mapping = begin[0].copy(), begin[1].copy(), dict(begin[2])
        try:
            return ci.assemble(u, step=step)
        except ValueError:
            ci.seed, ci.N, ci.mapping = begin
            return ci.reconstruct(u)
    finally:
        ci.seed, ci.N, ci.mapping = state
        ci.stats.clear()
        ci.stats.update(stats)


class Segment:
    """Quintic joint-space segment between two task-coordinate vectors (C2), or a hold."""
    kind = 'arm'

    def __init__(self, name, duration, u0, u1=None, guard=None, timeout_extension=8.):
        self.name, self.duration = name, float(duration)
        self.u0 = np.asarray(u0, float).copy()
        self.u1 = self.u0.copy() if u1 is None else np.asarray(u1, float).copy()
        self.guard, self.timeout_extension = guard, timeout_extension

    def reference(self, tau):
        f, df, ddf = smooth(tau, 0., self.duration)
        delta = self.u1 - self.u0
        return self.u0 + f * delta, df * delta, ddf * delta


class PathSegment:
    """Tool path: task coordinates as a C2 spline of the path parameter s in [0, 1] (knots from inverse kinematics along a
    straight tool path), run with the quintic time scaling s = f(tau)."""
    kind = 'arm'

    def __init__(self, name, duration, knots, guard=None, timeout_extension=8.):
        self.name, self.duration = name, float(duration)
        K = np.asarray(knots, float)
        self.spline = CubicSpline(np.linspace(0., 1., len(K)), K, axis=0, bc_type='natural')
        self.d1, self.d2 = self.spline.derivative(1), self.spline.derivative(2)
        self.u0, self.u1 = K[0].copy(), K[-1].copy()
        self.guard, self.timeout_extension = guard, timeout_extension

    def reference(self, tau):
        f, df, ddf = smooth(tau, 0., self.duration)
        u1, u2 = self.d1(f), self.d2(f)
        return self.spline(f), u1 * df, u2 * df * df + u1 * ddf


class TravelSegment(Segment):
    """Base travel between two site poses (x, y, yaw) with the front held."""
    kind = 'travel'

    def __init__(self, name, duration, u_hold, pose0, pose1, guard=None, timeout_extension=8.):
        super().__init__(name, duration, u_hold, None, guard, timeout_extension)
        self.pose0, self.pose1 = np.asarray(pose0, float), np.asarray(pose1, float)

    def base_reference(self, tau):
        f, df, _ = smooth(tau, 0., self.duration)
        delta = self.pose1 - self.pose0
        return self.pose0 + f * delta, df * delta


class Timeline:
    """Fixed-duration phase sequence (no guards): the references do not react to the plant."""
    guarded = False

    def __init__(self, phases):
        self.phases = list(phases)
        self.names = [p.name for p in self.phases]
        self.starts = np.cumsum([0.] + [p.duration for p in self.phases])
        self.duration = float(self.starts[-1])
        self.phase = 0
        self.done, self.failures, self.events = False, [], []
        for a, b in zip(self.phases[:-1], self.phases[1:]):
            if np.max(np.abs(a.reference(a.duration)[0] - b.reference(0.)[0])) > 1e-6:
                raise ValueError(f'Reference discontinuity entering {b.name}')

    def references(self, t):
        k = int(np.clip(np.searchsorted(self.starts, t, side='right') - 1, 0, len(self.phases) - 1))
        self.phase = k
        return self.phases[k].reference(min(t - self.starts[k], self.phases[k].duration))

    def update(self, t, obs):
        if t >= self.duration:
            self.done = True


# ---------------------------------------------------------------------------------------------------- tool chain
def _rot(axis, angle):
    x, y, z = axis
    K = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    return np.eye(3) + np.sin(angle) * K + (1. - np.cos(angle)) * (K @ K)


class ToolChain:
    """Forward and inverse kinematics of the serial chain from the graph root to the tool body.

    The chain is the tree path of the compiled graph; every moving joint on it must be a task (independent) coordinate,
    so the pose of the tool depends on the task coordinates only and no loop closure is involved."""

    def __init__(self, ci: CompiledInterface, tool_body, R_bs, p_bs, task_ids=None):
        s = ci.comp.structure
        task_ids = list(ci.independent_ids) if task_ids is None else list(task_ids)
        by_child = {e['child']: e for e in s['edges']}
        path, b = [], tool_body
        while b != s['root']:
            e = by_child[b]
            path.append(e)
            b = e['parent']
        path.reverse()
        self.steps = []
        self.bodies = [e['child'] for e in path]
        ids = ci.coordinate_ids
        for e in path:
            k = None
            if e['coord'] is not None:
                jid = ids[e['coord']]
                if jid not in task_ids:
                    raise ValueError(f'joint {jid} on the tool chain is not a task coordinate')
                k = task_ids.index(jid)
            self.steps.append((np.asarray(e['enter'], float), np.asarray(e['leave'], float), e['kind'], np.asarray(e['axis'], float), k))
        self.task_ids = task_ids
        self.chain_index = [st[4] for st in self.steps if st[4] is not None]
        self.R_bs, self.p_bs = np.asarray(R_bs, float), np.asarray(p_bs, float)
        self.n = len(self.task_ids)

    def body_poses(self, u):
        """{body: (R, p)} of the bodies on the chain in the graph root frame."""
        R, p = np.eye(3), np.zeros(3)
        out = {}
        for (enter, leave, kind, axis, k), body in zip(self.steps, self.bodies):
            p = p + R @ enter[:3, 3]
            R = R @ enter[:3, :3]
            if kind == 'revolute':
                R = R @ _rot(axis, u[k])
            elif kind == 'prismatic':
                p = p + R @ (axis * u[k])
            p = p + R @ leave[:3, 3]
            R = R @ leave[:3, :3]
            out[body] = (R, p)
        return out

    def frame(self, u):
        """Rotation and origin of the soil frame of the tool in the graph root frame."""
        R, p = np.eye(3), np.zeros(3)
        for enter, leave, kind, axis, k in self.steps:
            p = p + R @ enter[:3, 3]
            R = R @ enter[:3, :3]
            if kind == 'revolute':
                R = R @ _rot(axis, u[k])
            elif kind == 'prismatic':
                p = p + R @ (axis * u[k])
            p = p + R @ leave[:3, 3]
            R = R @ leave[:3, :3]
        return R @ self.R_bs, p + R @ self.p_bs

    def pose(self, u, point=(0., 0.)):
        """Root-frame position of the soil-frame point (x, z) and the pitch of the tool."""
        R, p = self.frame(u)
        xs, ys = R[:, 0], R[:, 1]
        heading = np.array([ys[1], -ys[0], 0.])                 # y_s x e_z
        hn = np.linalg.norm(heading)
        heading = heading / hn if hn > 1e-9 else np.array([1., 0., 0.])
        return p + R @ np.array([point[0], 0., point[1]]), float(np.arctan2(xs[2], xs @ heading))

    def lateral_tilt(self, u):
        """Elevation of the lateral axis of the tool (zero when the section plane of the bucket is vertical)."""
        R, _ = self.frame(u)
        return float(np.arcsin(np.clip(R[2, 1], -1., 1.)))

    def solve(self, target, pitch, free, guess, point=(0., 0.), tol=1e-10, max_iter=60):
        """Task coordinates that put ``point`` at ``target`` (root frame) with pitch ``pitch``; the four coordinates named in
        ``free`` are solved (the others keep the values of ``guess``).  Damped Newton iteration with a numerical Jacobian."""
        u = np.array(guess, float)
        idx = [self.task_ids.index(k) for k in free]
        if len(idx) != 4:
            raise ValueError('four free coordinates are needed for position and pitch')
        target = np.asarray(target, float)

        def residual(v):
            pos, phi = self.pose(v, point)
            return np.r_[pos - target, (phi - pitch + np.pi) % (2 * np.pi) - np.pi]
        r = residual(u)
        for _ in range(max_iter):
            if np.max(np.abs(r)) < tol:
                return u
            J = np.empty((4, 4))
            for c, k in enumerate(idx):
                up, um = u.copy(), u.copy()
                up[k] += 1e-6
                um[k] -= 1e-6
                J[:, c] = (residual(up) - residual(um)) / 2e-6
            try:
                step = np.linalg.solve(J.T @ J + 1e-12 * np.eye(4), -J.T @ r)
            except np.linalg.LinAlgError:
                break
            scale = min(1., 0.35 / max(1e-12, float(np.max(np.abs(step)))))       # at most 0.35 rad (or m) per iteration
            for _ls in range(12):
                trial = u.copy()
                trial[idx] += scale * step
                rt = residual(trial)
                if np.linalg.norm(rt) < np.linalg.norm(r) or scale < 1e-4:
                    break
                scale *= 0.5
            u, r = trial, rt
        if np.max(np.abs(r)) < 1e-7:
            return u
        raise ValueError(f'inverse kinematics did not converge: residual {r}, target {target}, pitch {pitch:.3f}')


class Planner:
    """Reference planning on a private copy of the compiled controller graph: validation of knots (closure, declared
    coordinate windows), actuator port speeds along a segment and pacing of segment durations."""

    def __init__(self, physical, xc, machine, allowance=0.65, limit_margin=None):
        self.ci = CompiledInterface(physical, 'joint_space')
        self.ci_run = CompiledInterface(physical, 'joint_space')       # continuation along segments
        # commanded independent coordinates (task coordinates of the references); a mechanism on ground contacts has, in
        # addition, held coordinates (contact anchors), which keep their seed values in every reference
        held = set(getattr(xc, 'held_ids', []))
        self.free = np.array([i for i, k in enumerate(self.ci.independent_ids) if k not in held], int)
        self.task_ids = [self.ci.independent_ids[i] for i in self.free]
        self._seed_all = self.ci.seed[self.ci.active].copy()
        self.seed = self._seed_all[self.free].copy()
        self.chain = ToolChain(self.ci, physical['machine']['tool_body'], *machine.soil_frame(), task_ids=self.task_ids)
        self.port_rows = np.array([self.ci.coordinate_ids.index(p) for p in xc.ci.port_ids], int)
        self.speed_lo, self.speed_hi = xc.port_speed_lo.copy(), xc.port_speed_hi.copy()
        self.allowance = allowance
        self.windows = []                                            # (coordinate index, lower, upper, id) to be respected by knots
        ids = self.ci.coordinate_ids
        for jid, c in physical['machine']['cylinders'].items():
            if jid in ids:
                # distance of nominal references from the stroke ends: record of the cylinder, else the default of the machine
                m = c.get('reference_margin_m', machine.stroke_reference_margin) if limit_margin is None else limit_margin
                self.windows.append((ids.index(jid), c['lower_m'] + m, c['upper_m'] - m, jid))
        for jid, (lo, hi) in machine.joint_windows(physical).items():
            self.windows.append((ids.index(jid), -np.inf if lo is None else lo, np.inf if hi is None else hi, jid))

        self._box = None

    def index(self, joint):
        return self.task_ids.index(joint)

    def full(self, u):
        """All independent coordinates of the compiled graph for the task coordinates u (held coordinates at their seed)."""
        out = self._seed_all.copy()
        out[self.free] = u
        return out

    def closed(self, u):
        """Closed configuration of the graph at task coordinates u (ValueError if it cannot be assembled)."""
        return closed_configuration(self.ci, self.full(np.asarray(u, float)))

    def scan(self, joint, step=0.03, max_steps=180, stop=None):
        """Sweep of one task coordinate from the seed in both directions (others at their seed values) by continuation on
        the compiled graph, until ``stop(q)`` is true, the assembly fails or a branch guard is reached.  Returns the
        values of the task coordinate (ascending), the closed configurations (rows) and the columns dq/du_joint of the
        tangent map.  The last sample of either direction may lie outside the region accepted by ``stop``."""
        i = self.index(joint)
        col = self.free[i]
        ci = CompiledInterface(self.ci.physical, 'joint_space')
        seed_u = self.seed.copy()
        seed_q, seed_N, seed_map = ci.seed.copy(), ci.N.copy(), dict(ci.mapping)
        out = {0: (seed_u[i], seed_q.copy(), seed_N[:, col].copy())}
        for sign in (1, -1):
            ci.seed, ci.N, ci.mapping = seed_q.copy(), seed_N.copy(), dict(seed_map)
            for k in range(1, max_steps + 1):
                u = seed_u.copy()
                u[i] += sign * k * step
                try:
                    q = ci.assemble(self.full(u), step=step)
                except ValueError:
                    break
                out[sign * k] = (u[i], q.copy(), ci.N[:, col].copy())
                if stop is not None and stop(q):
                    break
        keys = sorted(out)
        return np.array([out[k][0] for k in keys]), np.array([out[k][1] for k in keys]), np.array([out[k][2] for k in keys])

    def task_box(self):
        """Windows of the task coordinates implied by the declared coordinate windows.  A windowed coordinate that depends
        on one task coordinate only (a cylinder on one joint) bounds that task coordinate: the task coordinate is swept
        from the seed until a windowed coordinate leaves its window or the graph can no longer be assembled.  Returns
        (lower, upper, exact); ``exact`` is False if some windowed coordinate depends on several task coordinates (the
        box is then necessary but not sufficient, and only ``validate`` decides)."""
        if self._box is not None:
            return self._box
        ci = self.ci
        n = len(self.task_ids)
        lo, hi, exact = np.full(n, -np.inf), np.full(n, np.inf), True
        E = ci.N[:, self.free]
        seed_u = self.seed
        dependents = {}
        for k, wlo, whi, jid in self.windows:
            if jid in self.task_ids:
                i = self.index(jid)
                lo[i], hi[i] = max(lo[i], wlo), min(hi[i], whi)
                continue
            dep = np.flatnonzero(np.abs(E[k]) > 1e-9)
            if len(dep) != 1:
                exact = False
                continue
            dependents.setdefault(int(dep[0]), []).append((k, wlo, whi, jid))
        for i, wins in dependents.items():
            def outside(q, i=i, wins=wins):
                return any(not wlo <= q[k] <= whi for k, wlo, whi, _ in wins) or not lo[i] <= q[self.ci.active[self.free[i]]] <= hi[i]
            uu, Q, _ = self.scan(self.task_ids[i], stop=outside)
            k0 = int(np.argmin(np.abs(uu - seed_u[i])))
            if outside(Q[k0]):
                raise ValueError(f'a declared window is violated at the seed (task coordinate {self.task_ids[i]})')
            for sign, end in ((-1, 0), (1, len(uu) - 1)):
                bound = uu[end]                               # end of the sweep: assembly limit or first sample outside
                if end != k0 and outside(Q[end]):
                    inner = end - sign
                    # first crossing of a window end between the last inside and the first outside sample (linear interpolation)
                    fr = 1.
                    for k, wlo, whi, _ in wins:
                        c_in, c_out = Q[inner, k], Q[end, k]
                        for target in (wlo, whi):
                            if np.isfinite(target) and (c_in - target) * (c_out - target) < 0.:
                                fr = min(fr, (target - c_in) / (c_out - c_in))
                    bound = uu[inner] + fr * (uu[end] - uu[inner])
                if sign < 0:
                    lo[i] = max(lo[i], bound)
                else:
                    hi[i] = min(hi[i], bound)
        self._box = (lo, hi, exact)
        return self._box

    def inside_box(self, u, margin=0.):
        lo, hi, _ = self.task_box()
        return bool(np.all(u >= lo + margin) and np.all(u <= hi - margin))

    def validate(self, u):
        """Closed configuration of a knot inside the declared coordinate windows (raises ValueError otherwise)."""
        q = self.closed(u)
        for k, lo, hi, jid in self.windows:
            if not lo <= q[k] <= hi:
                raise ValueError(f'{jid} = {q[k]:.4f} outside [{lo:.4f}, {hi:.4f}] at u = {np.round(u, 4).tolist()}')
        return q

    def inside(self, u):
        try:
            self.validate(u)
            return True
        except ValueError:
            return False

    def port_speed_ratio(self, phase, samples=25):
        """Largest ratio of a demanded port speed to its declared capability along the phase (>1: too fast)."""
        worst = 0.
        for tau in np.linspace(0., phase.duration, samples):
            u, ud, _ = phase.reference(tau)
            try:
                q = self.ci_run.assemble(self.full(u))
            except ValueError:
                continue                                   # beyond the assembly range: a deliberately unreachable target
            E = self.ci_run.N
            v = E[self.port_rows][:, self.free] @ ud
            worst = max(worst, float(np.max(np.where(v >= 0., v / self.speed_hi, v / self.speed_lo))))
        return worst / self.allowance

    def paced(self, make, duration, validate=True):
        """Instantiate ``make(duration)``; stretch the duration until every port speed is within the allowance of its
        declared capability, then validate knots on the compiled graph."""
        for _ in range(14):
            phase = make(duration)
            factor = self.port_speed_ratio(phase)
            if factor <= 1.0:
                break
            duration = float(np.ceil(duration * factor * 10.) / 10.)
        else:
            raise ValueError(f'{phase.name}: could not satisfy the port-speed capabilities')
        if validate:
            for tau in np.linspace(0., phase.duration, 7):
                self.validate(phase.reference(tau)[0])
        return phase

    def move(self, name, u0, u1, minimum=2., validate=False, **kw):
        """Paced quintic joint segment (the target may be deliberately inadmissible: no validation by default)."""
        return self.paced(lambda T: Segment(name, T, u0, u1, **kw), minimum, validate=validate)

    def path(self, name, duration, u_start, target0, target1, pitch0, pitch1, free, point, knots=41, **kw):
        """Paced straight tool path of ``point`` from target0 to target1 (root frame) with a linear pitch schedule; the task
        coordinates of the first knot continue from ``u_start``."""
        target0, target1 = np.asarray(target0, float), np.asarray(target1, float)
        K, u = [], np.array(u_start, float)
        for s in np.linspace(0., 1., knots):
            u = self.chain.solve(target0 + s * (target1 - target0), pitch0 + s * (pitch1 - pitch0), free, u, point)
            if not self.inside_box(u):
                raise ValueError(f'{name}: the path leaves the task box at s = {s:.2f} (u = {np.round(u, 4).tolist()})')
            K.append(u.copy())
        if np.max(np.abs(K[0] - np.asarray(u_start, float))) > 1e-6:
            raise ValueError(f'{name}: the path does not start at the given task coordinates')
        return self.paced(lambda T: PathSegment(name, T, K, **kw), duration)

    def guesses(self, free):
        """Starting points of the inverse kinematics: the seed and a lattice of the task box in the free coordinates."""
        lo, hi, _ = self.task_box()
        seed = self.seed.copy()
        bounded = np.isfinite(lo) & np.isfinite(hi)
        with np.errstate(invalid='ignore'):
            mid = np.where(bounded, 0.5 * (lo + hi), seed)
            half = np.where(bounded, 0.5 * (hi - lo), 0.)
        idx = [self.index(k) for k in free if bounded[self.index(k)]]
        out = [seed, mid]
        import itertools
        for combo in itertools.product((-0.65, 0., 0.65), repeat=len(idx)):
            g = mid.copy()
            for i, c in zip(idx, combo):
                g[i] += c * half[i]
            out.append(g)
        return out

    def turn(self, free, point=(0., 0.)):
        """(index, sense) of the free task coordinate that turns the tool about a vertical axis without a window (the slew):
        sense = d(azimuth of the tool point) / d(coordinate) at the seed."""
        lo, hi, _ = self.task_box()
        idx = [self.index(k) for k in free if not (np.isfinite(lo[self.index(k)]) and np.isfinite(hi[self.index(k)]))]
        if not idx:
            return None, 0.
        seed = self.seed
        p0, _ = self.chain.pose(seed, point)
        up = seed.copy()
        up[idx[0]] += 1e-4
        p1, _ = self.chain.pose(up, point)
        return idx[0], float(np.sign(np.arctan2(p1[1], p1[0]) - np.arctan2(p0[1], p0[0])) or 1.)

    def slew(self, u, azimuth, free, point=(0., 0.)):
        """Task coordinates of u with the slew coordinate changed so that the tool point has the given azimuth."""
        i, sense = self.turn(free, point)
        if i is None:
            raise ValueError('no slew coordinate among the free task coordinates')
        u1 = np.array(u, float)
        for _ in range(8):                                    # the slew axis need not pass through the origin of the root frame
            p0, _ = self.chain.pose(u1, point)
            err = (azimuth - np.arctan2(p0[1], p0[0]) + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < 1e-12:
                break
            u1[i] += sense * err
        if abs(err) > 1e-9:
            raise ValueError('the slew coordinate does not rotate the tool point about a vertical axis')
        return u1

    def solve_pose(self, target, pitch, free, point=(0., 0.), guess=None, fixed=None, margin=0.):
        """Task coordinates that put the bucket point at ``target`` with the given pitch, inside the task box.  The free
        coordinate whose window is unbounded (the slew) starts at the azimuth of the target; ``fixed`` = {joint: value}
        sets other task coordinates.  Raises ValueError if no starting point leads into the box."""
        target = np.asarray(target, float)
        starts = ([np.asarray(guess, float)] if guess is not None else []) + self.guesses(free)
        lo, hi, _ = self.task_box()
        seed = self.seed
        ti, sense = self.turn(free, point)
        turn = [] if ti is None else [ti]
        if turn:                                              # slew: rotate the seed heading to the azimuth of the target
            p0, _ = self.chain.pose(seed, point)
            turn_value = seed[ti] + sense * (np.arctan2(target[1], target[0]) - np.arctan2(p0[1], p0[0]))
        last = None
        for k, g in enumerate(starts):
            g = np.array(g, float)
            for jid, val in (fixed or {}).items():
                g[self.index(jid)] = val
            if turn and not (k == 0 and guess is not None):
                g[turn[0]] = turn_value
            try:
                u = self.chain.solve(target, pitch, free, g, point, max_iter=40)
            except ValueError as exc:
                last = exc
                continue
            if self.inside_box(u, margin):
                return u
            last = ValueError(f'solution outside the task box: {np.round(u, 4).tolist()}')
        raise ValueError(f'no admissible pose for target {np.round(target, 3).tolist()}, pitch {np.degrees(pitch):.1f} deg: {last}')

    def joint_for_coordinate(self, u_base, joint, coordinate, target, step=0.02, count=200):
        """Value of the task coordinate ``joint`` (others as in u_base) at which graph coordinate ``coordinate`` equals
        ``target``: the task coordinate is swept from u_base in both directions by continuation on the compiled graph
        until the coordinate crosses the target, then bisected.  A target beyond a stroke end is found as long as the
        graph can be assembled there."""
        k = self.index(joint)
        col = self.ci.coordinate_ids.index(coordinate)
        ci = self.ci_run
        u_base = np.array(u_base, float)
        q0 = self.closed(u_base)
        N0, map0 = self.ci.pacdm.mapping(q0)

        def value(angle):
            u = u_base.copy()
            u[k] = angle
            return ci.assemble(self.full(u), step=step)[col] - target
        a0 = float(u_base[k])
        f0 = q0[col] - target
        for sign in (1., -1.):
            ci.seed, ci.N, ci.mapping = q0.copy(), N0.copy(), dict(map0)
            a, fa = a0, f0
            for _ in range(count):
                b = a + sign * step
                try:
                    fb = value(b)
                except ValueError:
                    break
                if fa * fb <= 0.:
                    lo, flo, hi = a, fa, b
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        fm = value(mid)
                        if flo * fm <= 0.:
                            hi = mid
                        else:
                            lo, flo = mid, fm
                    return 0.5 * (lo + hi)
                a, fa = b, fb
        raise ValueError(f'{coordinate} = {target} is not reachable from {joint} = {a0}')
