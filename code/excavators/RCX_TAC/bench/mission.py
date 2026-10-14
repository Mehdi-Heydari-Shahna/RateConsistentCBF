"""Guarded task references of the site mission on the continuum soil (machine independent), and the mission runner.

    travel -> level the pile -> dig natural ground -> dump on the levelled site -> level -> dig -> dump -> level -> return

The structure is that of the earlier single-machine release: three levelling stages
and two digging cycles, every phase a C2 reference in the task coordinates with a guard on the plant state.  What
differs between machines is data: reach, depths, bucket pitches and durations come from the machine adapter
(``machine.mission_parameters()``); the tool paths are solved by numerical inverse kinematics on the compiled graph
(bench.reference), every knot is validated on that graph (closure, declared coordinate windows) and every phase is
stretched until the demanded actuator port speeds stay inside their declared capabilities.

  levelling lane (azimuth a, grade g = absolute lip height), style 'push' (default):
      push pass   - outward with the back of the open bucket as a blade, lip at g: soil above g is carried ahead and fills
                    the low ground along the lane, the excess ends as a ridge beyond the far edge
      rake pass   - inward from beyond that ridge with the mouth facing the machine, lip at g: the excess is collected
      pour        - the collected soil is poured back over the pad (or, after the last stage, on a spoil heap)
  levelling lane, style 'rake' (for a bucket whose floor cannot be set steeply at short reach):
      rake pass   - inward with the lip leading at g; what arrives at the near edge is curled in, carried out and poured
                    over the far part of the lane (after the last stage: on a spoil heap)
  digging cycle (natural ground at the dig area, lanes side by side):
      penetrate with the teeth leading, drag at depth, curl about the bucket pin, lift, slew, pour while moving over the
      site, return.

Phase guards.  A phase ends when its time has elapsed and its guard holds: tracking error below the tolerance (three
times the tolerance for soil-contact phases), bucket loaded / emptied, soil settled, base arrived.  A reference that
leads into a declared keep-out cannot be tracked to its end point, so a phase whose time has elapsed is also accepted
when the safety layer is holding the machine (barrier rows active) and the motion has settled; such a phase is recorded
in ``holds`` only if the unmodified tracking guard would have refused it at that moment.  Every phase has a timeout.

Only references are produced; the plant and the soil evolve natively.
"""
from __future__ import annotations

import traceback
from types import SimpleNamespace

import numpy as np

from env.scene import RenderRecorder
from rcx.compiler import compile_extended
from rcx.filter import FilterOptions
from rcx.runtime import SafeVDC

from .hazards import VARIANTS, build_report, clearance_table, variant_spec
from .loop import run_closed_loop
from .reference import PathSegment, Planner, Segment, TravelSegment
from .world import PlantWorld, TruthMonitor, code_hash, save_json, work_metrics

D2R = np.pi / 180.


class SiteMission:
    guarded = True                                   # phases end on guards: the run is complete only when the mission is done

    def __init__(self, machine, planner: Planner, layout, ground_height, par, *, tracking_tolerance=0.035, soil_speed_settled=0.08,
                 mass_quantum=None, hold_speed=0.02, dig_only=False):
        self.machine, self.pl, self.layout, self.ground, self.par = machine, planner, layout, ground_height, par
        self.tol = tracking_tolerance
        self.soil_speed_settled = soil_speed_settled
        self.mass_quantum = mass_quantum if mass_quantum is not None else par.get('mass_quantum', 40.)   # guard quantum (kg)
        self.hold_speed = float(hold_speed)          # largest task-coordinate speed of a settled hold
        self.bucket = machine.bucket_geometry()
        self.lip = self.bucket.lip
        self.site_center = np.asarray(layout.site_center, float)
        self.dig_center = np.asarray(layout.dig_center, float)
        self.work_half = np.asarray(layout.work_footprint, float) / 2
        self.travel_distance = float(par.get('travel_distance', 0.))
        self.fixed = dict(par.get('fixed', {}))      # task coordinates that are not part of a tool pose (tilt, rotator, ...)
        self.free = list(par['free'])
        self.n = len(planner.task_ids)
        # azimuth of the levelling site seen from the machine (the machine faces +x; most machines work straight ahead)
        self.site_azimuth = float(np.arctan2(self.site_center[1], self.site_center[0]))
        self.u_transport = self.pose(self.site_azimuth, *par['transport'], fixed=par.get('transport_fixed'))
        self.phases, self.events, self.failures, self.holds = [], [], [], []
        self.phase, self.entered, self.done, self.finished_at = 0, 0., False, None
        self.last = (self.u_transport.copy(), np.zeros(self.n), np.zeros(self.n))
        self.base_target = np.array([-self.travel_distance, 0., 0.])
        self.base_rate = np.zeros(3)
        self.dig_lanes = tuple(par['dig_lanes'])
        if dig_only:
            self._build_dig_only()
        else:
            self._build()
        self.names = [p.name for p in self.phases]
        self.duration_nominal = float(sum(p.duration for p in self.phases))

    # ------------------------------------------------------------------ geometry helpers
    def _g(self, r, a):
        """Pristine ground height below the point at radius r, azimuth a (machine frame at the work position)."""
        return float(self.ground(r * np.cos(a), r * np.sin(a)))

    @staticmethod
    def _xyz(a, r, z):
        return np.array([r * np.cos(a), r * np.sin(a), z])

    def pose(self, a, r, z, pitch_deg, guess=None, point=None, fixed=None):
        """Task coordinates of a tool pose; ``fixed`` = {task coordinate: value} overrides the values of the coordinates that
        are not solved (telescope, chassis pose, ... of a phase group)."""
        return self.pl.solve_pose(self._xyz(a, r, z), pitch_deg * D2R, self.free, self.lip if point is None else point, guess,
                                  fixed=dict(self.fixed, **(fixed or {})), margin=0.005)

    def _end(self):
        prev = self.phases[-1]
        return prev.reference(prev.duration)[0]

    # ------------------------------------------------------------------ phase constructors
    def _append(self, phase):
        if self.phases:
            if np.max(np.abs(self._end() - phase.reference(0.)[0])) > 1e-6:
                raise ValueError(f'Reference discontinuity entering {phase.name}')
        self.phases.append(phase)
        return phase

    def hold(self, name, duration, guard=None, **kw):
        u = self._end() if self.phases else self.u_transport
        return self._append(Segment(name, duration, u, None, guard=guard, **kw))

    def joint_to(self, name, u1, duration, guard=None, **kw):
        u0 = self._end()
        return self._append(self.pl.paced(lambda T: Segment(name, T, u0, u1, guard=guard, **kw), duration))

    def path_to(self, name, a, rz1, pitch1_deg, duration, guard=None, point=None, **kw):
        """Straight tool path of the lip (or ``point``) in the vertical plane at azimuth a from the present pose to (r, z)
        with a linear pitch schedule."""
        u0 = self._end()
        point = self.lip if point is None else point
        p0, phi0 = self.pl.chain.pose(u0, point)
        phi1 = pitch1_deg * D2R
        phi1 = phi0 + ((phi1 - phi0 + np.pi) % (2 * np.pi) - np.pi)          # shortest rotation
        return self._append(self.pl.path(name, duration, u0, p0, self._xyz(a, *rz1), phi0, phi1, self.free, point, guard=guard, **kw))

    def curl_to(self, name, pitch1_deg, duration, guard=None, **kw):
        """Rotation of the bucket about its pin (pin held in place) to the given pitch."""
        u0 = self._end()
        p0, phi0 = self.pl.chain.pose(u0, (0., 0.))
        phi1 = phi0 + ((pitch1_deg * D2R - phi0 + np.pi) % (2 * np.pi) - np.pi)
        return self._append(self.pl.path(name, duration, u0, p0, p0, phi0, phi1, self.free, (0., 0.), guard=guard, **kw))

    def slew_to(self, name, a, duration, guard=None, **kw):
        """Slew to azimuth a with the front held (the lip keeps radius, height and pitch)."""
        u0 = self._end()
        u1 = self.pl.slew(u0, a, self.free, self.lip)
        return self._append(self.pl.paced(lambda T: Segment(name, T, u0, u1, guard=guard, **kw), duration))

    def travel(self, name, pose0, pose1, duration):
        return self._append(TravelSegment(name, duration, self._end() if self.phases else self.u_transport, pose0, pose1, guard='base_arrived'))

    # ------------------------------------------------------------------ levelling
    def _leveling(self, index):
        P = self.par['level']
        if P.get('style', 'push') == 'rake':
            return self._leveling_rake(index)
        g = P['grades'][index - 1]
        lanes = P['lanes']
        Rc = float(np.hypot(*self.site_center))
        half = self.work_half[0]
        r_near, r_far = Rc - half - P.get('near_margin', 0.10), Rc + half + P.get('far_margin', 0.15)
        r_ridge = r_far + P.get('ridge', 0.65)
        lift = P.get('lift', 0.55)
        L = lambda name: f'level{index}: {name}'
        for k, a in enumerate(lanes):
            tag = f'lane {k + 1}/{len(lanes)}'
            self.joint_to(L(f'{tag} approach near edge'), self.pose(a, r_near, g + lift, P['push']), 3.5)
            self.path_to(L(f'{tag} lower blade'), a, (r_near, g), P['push'], 2.5, guard='tracking_loose')
            self.path_to(L(f'{tag} push pass'), a, (r_far, g), P['push'], 6., guard='tracking_loose')
            self.path_to(L(f'{tag} lift blade'), a, (r_far - 0.25, g + lift), P['push'], 2.)
        a = 0.5 * (min(lanes) + max(lanes))
        self.joint_to(L('reach beyond ridge'), self.pose(a, r_ridge, g + lift, P['rake']), 3.)
        self.path_to(L('lower rake'), a, (r_ridge, g), P['rake'], 2., guard='tracking_loose')
        self.path_to(L('rake pass'), a, (r_near, g), P['rake_end'], 7., guard='tracking_loose')
        self.curl_to(L('curl collected'), P['carry'], 2.5)
        if index < len(P['grades']):
            r0p, r1p, zp, ap = r_near + P['pour_from_near'], Rc + P['pour_to_centre'], g + P['pour_height'], a      # back over the pad
        else:
            (r0p, r1p), zp, ap = P['spoil_r'], P['spoil_z'], P['spoil_azimuth']                                    # spoil heap beside the pad
        self.joint_to(L('carry collected'), self.pose(ap, r0p, zp, P['carry']), 3.5)
        self.path_to(L('pour collected'), ap, (r1p, zp), P['pour'], 5., guard='tracking_loose')
        self.joint_to(L('retract to transport'), self.u_transport, 3.5)
        self.hold(L('settle'), 1.0, guard='soil_settled')

    def _leveling_rake(self, index):
        """Levelling for a bucket whose floor plate cannot be set steeply at short reach (a flat plate pushed over loose soil
        bears on it like a footing).  Every lane is graded by an inward pass with the lip leading at the grade: soil above
        the grade is drawn inward and fills the low ground along the lane.  What arrives at the near edge is curled in,
        carried out and poured over the far part of the lane (after the last stage: on a spoil heap)."""
        P = self.par['level']
        g = P['grades'][index - 1]
        lanes = P['lanes']
        Rc = float(np.hypot(*self.site_center))
        half = self.work_half[0]
        r_near, r_far = Rc - half - P.get('near_margin', 0.10), Rc + half + P.get('far_margin', 0.15)
        lift = P.get('lift', 0.55)
        last = index == len(P['grades'])
        L = lambda name: f'level{index}: {name}'
        fx, az0 = P.get('fixed'), self.site_azimuth
        for k, lane in enumerate(lanes):
            a = az0 + lane
            tag = f'lane {k + 1}/{len(lanes)}'
            self.joint_to(L(f'{tag} reach far edge'), self.pose(a, r_far, g + lift, P['rake'], fixed=fx), 3.5)
            self.path_to(L(f'{tag} lower rake'), a, (r_far, g), P['rake'], 2., guard='tracking_loose')
            self.path_to(L(f'{tag} rake pass'), a, (r_near, g), P['rake_end'], 7., guard='tracking_loose')
            self.curl_to(L(f'{tag} curl collected'), P['carry'], 2.5)
            if last:
                (r0p, r1p), zp, ap = P['spoil_r'], P['spoil_z'], az0 + P['spoil_azimuth']
            else:
                r0p, r1p, zp, ap = Rc + P['pour_from_centre'], r_far - P.get('pour_before_far', 0.2), g + P['pour_height'], a
            self.joint_to(L(f'{tag} carry collected'), self.pose(ap, r0p, zp, P['carry_out'], fixed=fx), 3.5)
            self.path_to(L(f'{tag} pour collected'), ap, (r1p, zp), P['pour'], 4., guard='tracking_loose')
        self.joint_to(L('retract to transport'), self.u_transport, 3.5)
        self.hold(L('settle'), 1.0, guard='soil_settled')

    # ------------------------------------------------------------------ digging
    def _dig(self, index, to_site=True):
        P = self.par['dig']
        D = lambda name: f'dig{index}: {name}'
        rz_c = float(np.hypot(*self.dig_center))
        ang = float(np.arctan2(self.dig_center[1], self.dig_center[0])) + self.dig_lanes[index - 1]
        r_start = rz_c + P['start_beyond']
        pen, drag, depth = P['penetration'], P['drag'], P['depth']
        g0, g1 = self._g(r_start, ang), self._g(r_start - pen, ang)
        z_cut = min(g0, g1) - depth
        fx = P.get('fixed')
        self.slew_to(D('slew to dig area'), ang, 6.)
        self.joint_to(D('reach out'), self.pose(ang, r_start, g0 + P['approach_height'], P['touch'], fixed=fx), 5.)
        self.path_to(D('set teeth on ground'), ang, (r_start, g0 + 0.01), P['touch'], 2.5, guard='tracking_loose')
        self.path_to(D('penetrate'), ang, (r_start - pen, z_cut), P['pen'], 4., guard='load_aware', timeout_extension=8.)
        self.path_to(D('drag at depth'), ang, (r_start - pen - drag, z_cut), P['drag_pitch'], 5., guard='load_aware', timeout_extension=8.)
        self.curl_to(D('curl'), P['curl'], P.get('curl_duration', 3.5), guard='load_aware', timeout_extension=8.)
        self.joint_to(D('lift clear'), self.pose(ang, *P['lift'], fixed=P.get('lift_fixed', self.par.get('transport_fixed'))), 4.)
        self.hold(D('hold load'), 1., guard='loaded')
        if not to_site:
            return
        az0 = self.site_azimuth
        self.slew_to(D('slew to site'), az0, 6.)
        S = self.par['dump']
        Rc = float(np.hypot(*self.site_center))
        z_d = S['height'] + self._g(Rc, az0)
        self.joint_to(D('position over site'), self.pose(az0, Rc + S['from'], z_d, S['pitch_from'], fixed=S.get('fixed')), 3.5)
        self.path_to(D('dump'), az0, (Rc + S['to'], z_d), S['pitch_to'], 5.)
        self.hold(D('wait for deposition'), 2., guard='emptied')
        if S.get('lift', 0.) > 0.:                                # raise the open bucket before it is curled in over the fresh heap
            self.path_to(D('lift from site'), az0, (Rc + S['to'], z_d + S['lift']), S['pitch_to'], 2.5)
        self.joint_to(D('retract to transport'), self.u_transport, 4.5)

    def _build(self):
        p0 = np.array([-self.travel_distance, 0., 0.])
        self.pl.validate(self.u_transport)
        self._append(Segment('settle', 3., self.u_transport, None, guard='soil_settled'))
        if self.travel_distance > 0.:
            self.travel('approach travel', p0, np.zeros(3), self.par.get('travel_duration', 12.))
        self._leveling(1)
        self._dig(1)
        self._leveling(2)
        self._dig(2)
        self._leveling(3)
        if self.travel_distance > 0.:
            self.travel('return travel', np.zeros(3), p0, self.par.get('travel_duration', 12.))
        self.hold('stop', 2.)

    def _build_dig_only(self):
        """First digging cycle up to the lifted load (hazard scenarios)."""
        self._append(Segment('hold', 2., self.u_transport))
        self._dig(1, to_site=False)
        self.hold('hold end', 2.)

    # ------------------------------------------------------------------ run-time interface
    def references(self, t):
        ph = self.phases[self.phase]
        tau = t - self.entered
        arm = ph.reference(min(tau, ph.duration))
        if ph.kind == 'travel':
            self.base_target, self.base_rate = ph.base_reference(min(tau, ph.duration))
        else:
            self.base_rate = np.zeros(3)
        self.last = arm
        return arm, self.base_target, self.base_rate

    def tracking_accepts(self, phase, obs):
        err = float(obs['tracking_error'])
        if phase.guard == 'tracking_loose':
            return err < 3 * self.tol
        if phase.guard == 'load_aware':
            # a completed cut reference may proceed once material is collected even if soil resistance prevents the precise
            # joint end point (RoboCompiler soil_mission policy)
            return err < 3 * self.tol or obs['bucket_mass'] >= 3 * self.mass_quantum
        return err < self.tol

    def update(self, t, obs):
        """Advance the phase machine; obs holds tracking_error, soil_speed, bucket_mass, base_pose, safety_hold, speed."""
        if self.done:
            return
        ph = self.phases[self.phase]
        tau = t - self.entered
        err = float(obs['tracking_error'])
        held = bool(obs.get('safety_hold', False)) and float(obs.get('speed', 0.)) < self.hold_speed
        tracked = self.tracking_accepts(ph, obs)
        ready = tau >= ph.duration and (tracked or held)
        if ph.guard == 'soil_settled':
            ready = ready and obs['soil_speed'] < self.soil_speed_settled
        if ph.guard == 'base_arrived':
            ready = ready and np.linalg.norm(obs['base_pose'][:2] - self.base_target[:2]) < 0.12 and abs(obs['base_pose'][2] - self.base_target[2]) < 0.05
        if ph.guard == 'emptied':
            ready = ready and obs['bucket_mass'] < 0.5 * self.mass_quantum and obs['soil_speed'] < 0.25
        if ph.guard == 'loaded':
            if obs['bucket_mass'] <= 0. and not any(f.startswith(ph.name) for f in self.failures):
                self.failures.append(f'{ph.name}: bucket empty after digging')
        if tau > ph.duration + ph.timeout_extension:
            self.failures.append(f'Phase timeout: {ph.name} (error {err:.4f}' + (', safety hold)' if held else ')'))
            self.done, self.finished_at = True, t
            return
        if not ready:
            return
        needed = held and not tracked
        self.events.append(dict(time_s=t, phase=ph.name, bucket_mass_kg=obs['bucket_mass'], site_mass_kg=obs.get('site_mass', 0.),
                                peak_site_height_m=obs.get('site_peak', 0.), site_rms_m=obs.get('site_rms', 0.), tracking_error=err,
                                safety_hold=bool(needed)))
        if needed:
            self.holds.append(dict(time_s=float(t), phase=ph.name, guard=str(ph.guard), tracking_error=err, pairs=list(obs.get('hold_pairs', []))))
        if self.phase == len(self.phases) - 1:
            self.done, self.finished_at = True, t
        else:
            self.phase += 1
            self.entered = t


def pipe_across_lane(mission: SiteMission, index=1, radius_along=None, depth=0.55, pipe_radius=0.08, half_length=1.5):
    """Horizontal pipe perpendicular to dig lane ``index``, crossing it ``radius_along`` metres from the machine axis."""
    ang = float(np.arctan2(mission.dig_center[1], mission.dig_center[0])) + mission.dig_lanes[index - 1]
    dvec = np.array([np.cos(ang), np.sin(ang)])
    pvec = np.array([-np.sin(ang), np.cos(ang)])
    c = radius_along * dvec
    z = float(mission.ground(c[0], c[1])) - depth
    return dict(a=[*(c - half_length * pvec), z], b=[*(c + half_length * pvec), z], radius=pipe_radius)


def dig_lane(mission: SiteMission, index=1):
    return float(np.arctan2(mission.dig_center[1], mission.dig_center[0])) + mission.dig_lanes[index - 1], float(np.hypot(*mission.dig_center))


# ---------------------------------------------------------------------------------------------------- runner
def site_hazards(mission, par):
    """Site of the mission run: design-depth floor in the dig area, buried pipe across dig lane 1, overhead limit, walls."""
    S = par['site']
    ang, r = dig_lane(mission, 1)
    grade = float(mission.ground(r * np.cos(ang), r * np.sin(ang)))
    return dict(floor_z=grade - S['floor_depth'], ceiling_z=S['ceiling'], wall_front_x=S['wall_front'], wall_side_y=S['wall_side'],
                pipe=pipe_across_lane(mission, 1, radius_along=S['pipe_radius_along'], depth=S['pipe_depth']))


def run(machine, variant='full', *, name=None, duration=900., observer_gain=100., payload_scale=1., render=False, export_spec=False):
    """Complete site mission under the generated interfaces; writes results/<machine>/<name>/{trace.npz, report.json,
    manifest.json, spec.json, phases.json, soil_final.npz[, render.npz]}."""
    name = name or f'mission_{variant}'
    out = machine.results_dir() / name
    sha = code_hash(machine.name)                                 # state of the sources when the run starts
    out.mkdir(parents=True, exist_ok=True)
    for stale in ('report.json', 'trace.npz', 'soil_final.npz', 'spec.json', 'manifest.json', 'phases.json', 'render.npz'):
        (out / stale).unlink(missing_ok=True)                     # a run that fails must not leave an older result in place
    par = machine.mission_parameters()
    world = PlantWorld(machine, out, base=(-par.get('travel_distance', 0.), 0., 0., 0.), payload_scale=payload_scale)
    xc0 = compile_extended(world.ctrl_phys, machine.spec(world.ctrl_phys))
    planner = Planner(world.ctrl_phys, xc0, machine)
    mission = SiteMission(machine, planner, world.layout, world.ground, par)
    site = site_hazards(mission, par)
    spec = variant_spec(variant, machine.spec(world.ctrl_phys, **site))
    xc = compile_extended(world.ctrl_phys, spec)
    save_json(out / 'spec.json', spec)
    if export_spec:                                               # the specification of this run as an input record
        save_json(machine.results_dir().parents[1] / 'machines' / machine.name / 'control_safety_spec.json', spec)
    save_json(out / 'manifest.json', xc.manifest)
    save_json(out / 'phases.json', dict(names=mission.names, durations=[p.duration for p in mission.phases],
                                        nominal_duration_s=mission.duration_nominal, transport=mission.u_transport, parameters=par_plain(par)))
    opt = FilterOptions(**VARIANTS[variant])
    ctl = SafeVDC(xc, opt, observer_gain=observer_gain)
    world.bind(xc)
    initial_gap = world.place(xc.ci, world.full_coordinates(xc, mission.u_transport))
    monitor = TruthMonitor(world, xc)
    recorder = RenderRecorder() if render else None
    status, error, tr, wall = run_closed_loop(world, ctl, mission, duration=duration, monitor=monitor, label=name, print_every=10., render=recorder)
    small = {k: (v.astype(np.float32) if v.dtype == np.float64 and k not in ('time', 'c_time') else v) for k, v in tr.items()}
    np.savez_compressed(out / 'trace.npz', **small, pair_ids=np.array(list(xc.barriers.pair_rows)), phase_names=np.array(mission.names))
    if recorder is not None:
        np.savez_compressed(out / 'render.npz', **recorder.arrays(), h_initial=world.sm.h_initial)
    sm = world.sm
    np.savez_compressed(out / 'soil_final.npz', h=sm.h, loose=sm.loose, h_initial=sm.h_initial, ground=sm.ground)
    truth = clearance_table(xc, monitor)
    starts = [0.] + [e['time_s'] for e in mission.events]
    shim = SimpleNamespace(names=mission.names, starts=np.array(starts))
    info = dict(machine=machine.name, scenario='site_mission', variant=variant, partition='joint_space', code_sha256=sha,
                arguments=dict(variant=variant, name=name, duration=duration, observer_gain=observer_gain, payload_scale=payload_scale,
                               render=render))
    desc = (f'complete site mission ({len(mission.phases)} phases) on a site with a design-depth floor, a buried pipe across the dig '
            'lanes, an overhead limit and e-fence walls')
    try:
        report = build_report(info, name, status, error, tr, wall, world, xc, ctl, opt, shim, truth, desc, site, xc.Ts, initial_gap)
    except Exception:                                             # noqa: BLE001  (a failed run still leaves a report with its status)
        report = dict(status='ERROR', error=(error or '') + '\nreport: ' + traceback.format_exc(), machine=machine.name,
                      scenario='site_mission', variant=variant, name=name, description=desc, code_sha256=sha, simulation_status=status,
                      truth=truth, violated_pairs=[p for p, r in truth['pairs'].items() if r['violated']])
        save_json(out / 'report.json', report)
        print(f"Saved {out}  status=ERROR (report could not be built)\n{report['error']}")
        return report
    digs = [dict(cycle=k + 1, bucket_mass_kg=e['bucket_mass_kg'], time_s=e['time_s'])
            for k, e in enumerate(ev for ev in mission.events if ev['phase'].endswith('hold load'))]
    report.update(
        mission=dict(phases_completed=len(mission.events), phase_count=len(mission.phases), failures=mission.failures,
                     events=mission.events, safety_holds=mission.holds, digs=digs, nominal_duration_s=mission.duration_nominal),
        soil_final=dict(metrics=sm.metrics(), work_area=work_metrics(sm, world.boxes['work']), stats=sm.stats, mass0_kg=sm.mass0,
                        mass_error_kg=sm.mass_error()),
        final_base_pose=world.machine_pose())
    save_json(out / 'report.json', report)
    v = report['violated_pairs']
    print(f"Saved {out}  status={status}  phases {len(mission.events)}/{len(mission.phases)}  failures={mission.failures}  "
          f"violated={v if v else 'none'}  sim={world.d.time:.1f}s  wall={wall:.0f}s")
    print('  digs:', [(g['cycle'], round(g['bucket_mass_kg'])) for g in digs], ' safety holds:', len(mission.holds))
    print('  class h_min:', {k: round(r['h_min'], 4) for k, r in truth['classes'].items()})
    A = report['assumptions']
    rnd = lambda x: None if x is None else round(x, 4)        # None: nothing to check (e.g. no intervention in an off run)
    print('  assumptions: force-model', rnd(A['force_model_coverage']), '(intervention', rnd(A['force_model_coverage_during_intervention']),
          ') row residual', rnd(A['row_residual_coverage']), ' load', rnd(A['load_bound_coverage']), ' infeasible', report['filter']['infeasible_steps'],
          ' holds:', [(h['phase'], h['pairs']) for h in mission.holds])
    return report


def par_plain(par):
    """Mission parameters as plain values (for the phase record)."""
    return {k: (par_plain(v) if isinstance(v, dict) else v) for k, v in par.items()}
