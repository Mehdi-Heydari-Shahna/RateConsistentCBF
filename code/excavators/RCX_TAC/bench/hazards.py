"""Hazard scenarios of the extended RoboCompiler on the native plant of a machine with the continuum soil
(machine-independent runner; scenario definitions come from the machine adapter).

Every scenario declares its site (e-fence planes, buried pipe) in the specification and drives references that violate
a safety limit on purpose; the extended compiler generates the VDC evaluator, the barrier/storage evaluators and the
coordinated actuator-force filter from the same physical graph.  Variants switch parts of the generated safety layer
off (ablations) while the plant, the references and the nominal VDC stay the same.

Outputs results/<machine>/hazards/<name>/{trace.npz, report.json, manifest.json, spec.json[, render.npz]}; truth values
of every safety function are evaluated from the native body poses at every physics step (bench.world.TruthMonitor).
An older result of the same name is removed when a run starts; a run whose simulation or report fails writes a report
with status ERROR.
"""
from __future__ import annotations

import traceback

import mujoco
import numpy as np

import core.native as native_module
from env.scene import RenderRecorder
from env.soil import SoilParams
from rcx.compiler import compile_extended
from rcx.filter import FilterOptions
from rcx.runtime import SafeVDC

from .loop import run_closed_loop
from .reference import Planner
from .world import PlantWorld, TruthMonitor, code_hash, save_json

VARIANTS = {
    'full': dict(),
    'off': dict(force_filter=False, velocity_shaping=False),          # nominal VDC; actuator capability rows kept (and the contact
                                                                       # rows of a mechanism on ground contacts: support interface)
    'kinematic_only': dict(force_filter=False, velocity_shaping=True),
    'force_only': dict(force_filter=True, velocity_shaping=False),
    'no_margins': dict(margin_force=False, margin_load=False, margin_sampling=False, margin_effectiveness=False),
    'no_actuator_model': dict(actuator_model=False),                  # rows on the command (ideal effort sources assumed)
    'no_prediction': dict(sampled_prediction=False),                  # no one-step prediction of the state-driven row change
    'no_effort_rows': dict(effort_rows=False),                        # required motion not limited by the effort capability
    'uncoordinated': dict(coordinated=False),
    'identity_metric': dict(metric='identity'),
    'kinetic_velocity_metric': dict(velocity_metric='kinetic'),
    'task_velocity_metric': dict(velocity_metric='task'),
    'ts20': dict(), 'ts40': dict(),
    'ts40_continuous': dict(sampled_prediction=False, actuator_model=False),   # continuous-time row on the command at 25 Hz
    # paper: first layers of the safety layer
    'consistent': dict(first_layer='acceleration'),                    # rows on the required acceleration, beta_i = min(16, alpha2_i)
    'consistent_b8': dict(first_layer='acceleration', beta_bar=8.0),    # the same with a slower first layer
    'conditioned': dict(conditioning=True),                            # released first layer + conditioning state (kappa_c = k)
    'consistent_force_only': dict(first_layer='acceleration', velocity_shaping=False),   # second-order first layer without barrier rows
    'accel_b40': dict(first_layer='acceleration', beta_bar=40.0, beta_cap=False),        # smooth but not consistent: beta_i = 40 > alpha2_i
    'full_ffraw': dict(),                                              # released first layer, feedforward by the backward difference
    # machines on declared ground contacts (specification section ``support``)
    'no_contact_rows': dict(contact_rows=False),                       # effort filter without the rows on the contact forces
    'min_norm_distribution': dict(),                                   # nominal efforts of smallest weighted norm (contact rows kept)
    'min_norm_no_contact_rows': dict(contact_rows=False),              # both
    'off_no_contact_rows': dict(force_filter=False, velocity_shaping=False, contact_rows=False),      # no row on the efforts at all
}
SAMPLE_TIME = {'ts20': 0.02, 'ts40': 0.04, 'ts40_continuous': 0.04}


def _min_norm(spec):
    spec['support']['tangential_weight'] = 0.
    return spec


def _gain(k):
    def change(spec):
        spec['control']['k'] = float(k)
        return spec
    return change


# paper: gain study (velocity-feedback gain k of the VDC law; the barrier rates stay as declared)
for _k in (2, 4, 6, 24):
    VARIANTS[f'full_k{_k}'] = dict()
    VARIANTS[f'consistent_k{_k}'] = dict(first_layer='acceleration')
    VARIANTS[f'kinematic_only_k{_k}'] = dict(force_filter=False, velocity_shaping=True)


# variants that change the specification instead of an option of the filter
SPEC_CHANGE = {'min_norm_distribution': _min_norm, 'min_norm_no_contact_rows': _min_norm}
for _k in (2, 4, 6, 24):
    SPEC_CHANGE[f'full_k{_k}'] = _gain(_k)
    SPEC_CHANGE[f'consistent_k{_k}'] = _gain(_k)
    SPEC_CHANGE[f'kinematic_only_k{_k}'] = _gain(_k)


def _ffraw(spec):
    spec['control']['correction_filter_s'] = 0.            # no filter: backward difference of the shaping correction
    return spec


SPEC_CHANGE['full_ffraw'] = _ffraw
SUPPORT_VARIANTS = ('no_contact_rows', 'min_norm_distribution', 'min_norm_no_contact_rows', 'off_no_contact_rows')


def variant_spec(variant, spec):
    """Specification of a variant: the variants of the support interface need a machine on declared ground contacts, and
    some of them change the specification."""
    if variant in SUPPORT_VARIANTS and 'support' not in spec:
        raise ValueError(f'variant {variant} needs a machine on declared ground contacts')
    return SPEC_CHANGE[variant](spec) if variant in SPEC_CHANGE else spec


# Largest inclination of the base that counts as standing on its support (rad).  Beyond it the base is tipping: the
# plant then has support impacts, which the assumptions of the generated interfaces exclude.
BASE_TILT_LIMIT = 0.005
# A machine on declared ground contacts keeps its support while every contact carries load and no contact point has
# moved by more than this distance (m) from where it stood at the start (contact assumption of the support interface).
CONTACT_SLIP_LIMIT = 0.02


def contact_report(world, xc, tr):
    """Ground contacts of the plant against the declared contact model: smallest normal force, largest ratio of tangential
    to normal force, largest displacement of every contact point and the range of the depth by which the contact bodies
    sink into the ground (plant, every physics step after the settling time), samples with a binding
    contact row, and the difference between the contact forces of the model (at the measured efforts) and of the plant at
    the 40 ms samples after the first second."""
    cp = world.contact_peak
    if cp is None or xc.support is None:
        return None
    out = dict(ids=list(xc.support.ids), normal_force_min_N=cp['normal_min_N'], friction_ratio_max=cp['friction_ratio_max'],
               slip_max_m=cp['slip_max_m'], steps_without_load=cp['lost_steps'], slip_limit_m=CONTACT_SLIP_LIMIT,
               penetration_min_m=cp['depth_min_m'], penetration_max_m=cp['depth_max_m'],
               declared_friction=[c['friction'] for c in xc.support.contacts],
               kept=bool(np.all(cp['lost_steps'] == 0) and np.all(cp['slip_max_m'] <= CONTACT_SLIP_LIMIT)),
               contact_row_fraction=float(np.mean(np.asarray(tr['c_contact_active']) > 0)) if len(tr.get('c_contact_active', [])) else 0.,
               contact_row_slack_max_N=float(np.max(tr['c_contact_slack'], initial=0.)) if len(tr.get('c_contact_slack', [])) else 0.)
    if len(tr.get('contact_true', [])) and np.any(tr['time'] > 1.0):
        sel = tr['time'] > 1.0
        err = tr['contact_model'][sel] - tr['contact_true'][sel]
        out.update(model_error_rms_N=np.sqrt(np.mean(err ** 2, axis=0)), model_error_max_N=np.max(np.abs(err), axis=0),
                   normal_force_mean_N=np.mean(tr['contact_true'][sel][:, :, 2], axis=0))
    return out


def clearance_table(xc, monitor):
    """Truth minima per pair with the geometric clearance (distance between surfaces) for the clearance pairs."""
    s = monitor.summary()
    dmin = {c['id']: float(c.get('d_min', 0.)) for c in xc.spec['safety']['clearance']}
    for pid, rec in s['pairs'].items():
        if pid in dmin:
            rec['surface_distance_min_m'] = rec['h_min'] + dmin[pid]
            rec['d_min'] = dmin[pid]
        rec['violated'] = bool(rec['h_min'] < 0.)
    return s


def energy_split(power, dt):
    return dict(injected_J=float(np.sum(np.maximum(power, 0.) * dt)), removed_J=float(np.sum(np.minimum(power, 0.) * dt)))


def storage_balance(tr):
    """Energy ledger of the storage balance over the run, from the controller samples:

        nu(t_end) - nu(0) = - dissipated + mismatch + load + acc + zoh + residual

    with the trapezoidal sums of the nodal powers  s^T K_r s,  -s^T (L^T f - C phi_r),  -s^T (tau_ext - tau_comp)  (tau_ext from
    the native acceleration of the sample instant),  s^T M_r (du_r / Ts - u_r_dot)  and the zero-order-hold term of the
    required-velocity steps (bench.loop).  The residual is what this sampled ledger does not account for; it is given
    in total, relative to the dissipated energy, and split into the part inside 0.5 s of the largest storage and the rest."""
    t, nu = tr['c_time'], tr['c_storage']

    def total(p):
        return np.concatenate([[0.], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))])
    diss, mis, load, acc = (total(tr[k]) for k in ('c_dissipation', 'c_P_mis', 'c_P_load', 'c_P_acc'))
    # hold term: one energy per sample (plain sum; the first sample has no step), plus the end term of the sample considered
    zoh = np.concatenate([[0.], np.cumsum(tr['c_P_zoh'][1:] * np.diff(t))]) + tr['c_E_step']
    res = (nu - nu[0]) + diss - mis - load - acc - zoh
    k = int(np.argmax(nu))
    i0, i1 = int(np.searchsorted(t, t[k] - 0.5)), min(len(t) - 1, int(np.searchsorted(t, t[k] + 0.5)))
    led = dict(storage_change_J=float(nu[-1] - nu[0]), dissipated_J=float(diss[-1]), effort_mismatch_J=float(mis[-1]),
               uncompensated_load_J=float(load[-1]), required_acceleration_error_J=float(acc[-1]), zero_order_hold_J=float(zoh[-1]),
               residual_J=float(res[-1]), residual_without_zero_order_hold_term_J=float(res[-1] + zoh[-1]),
               residual_relative=abs(float(res[-1])) / max(1., float(diss[-1])),
               residual_largest_excursion_J=float(np.max(np.abs(res))), storage_peak_at_s=float(t[k]),
               residual_within_half_second_of_peak_J=float(res[i1] - res[i0]),
               dissipated_within_half_second_of_peak_J=float(diss[i1] - diss[i0]))
    led['residual_elsewhere_J'] = led['residual_J'] - led['residual_within_half_second_of_peak_J']
    return led


def coverage(violations, checks):
    """Fraction of checks without violation; None when nothing was checked (a coverage of 100 % would be vacuous)."""
    return None if checks == 0 else 1. - violations / checks


def build_report(info, name, status, error, tr, wall, world, xc, ctl, opt, timeline, truth, desc, site, Ts, initial_gap, free_motion=False):
    """Report of one run.  Coverage statistics of the force-loop model and of the load bound exclude the first second
    (the force loops start from the static holding efforts) and are taken at the control samples.  ``info``: dict with
    machine, scenario, variant, partition, code_sha256 and the run arguments."""
    ct = tr['c_time']
    n = len(ct)
    e = tr['u'] - tr['u_des'] if len(tr['u']) else np.zeros((1, xc.d))
    settled = ct > 1.0 if n else np.zeros(0, bool)
    checked = bool(n and settled.any())                        # a run of one second or less has nothing to check
    act = tr['c_n_active'] > 0 if n else np.zeros(0, bool)
    dtc = np.diff(ct, append=ct[-1] + Ts) if n else np.zeros(0)
    cover = tr['c_load_err'][settled] <= tr['c_load_bound'][settled] if checked else np.zeros(0, bool)
    fme = tr['c_force_model_err'][settled] if checked else np.zeros((1, xc.m))
    st = ctl.diagnostics()
    contacts = contact_report(world, xc, tr)
    fm_ok = np.all(fme <= xc.e_force[None, :] + 1e-9, axis=1) if checked else np.zeros(0, bool)
    # samples with a binding row of the effort problem: a barrier row or, for a machine on ground contacts, a contact row
    act_any = act | (np.asarray(tr['c_contact_active']) > 0) if n and len(tr.get('c_contact_active', [])) == n else act
    after_intervention = np.r_[False, act_any[:-1]][settled] if checked else np.zeros(0, bool)  # the command of the previous sample was filtered
    binding = cover[act_any[settled]] if checked else np.zeros(0, bool)                         # load bound at samples with a binding effort row
    return dict(
        status=status, error=error, machine=info['machine'], scenario=info['scenario'], variant=info['variant'], name=name, description=desc,
        partition=info.get('partition', 'joint_space'), sample_time_s=Ts, duration_s=float(world.d.time), wall_s=wall, options=opt.as_dict(),
        site=site, code_sha256=info['code_sha256'], run_arguments=info.get('arguments', {}),
        task_ids=list(xc.task.ids), task_labels=list(world.ctrl_phys['machine'].get('task_labels', xc.task.ids)),
        task_units=list(world.ctrl_phys['machine'].get('task_units', ['rad'] * xc.d)),
        channel_ids=[c.id for c in xc.channels], truth=truth,
        violated_pairs=[p for p, r in truth['pairs'].items() if r['violated']],
        tracking=dict(rms=np.sqrt((e ** 2).mean(0)), max=np.abs(e).max(0)),
        filter=dict(intervention_fraction=float(act.mean()) if n else 0., bound_fraction=float(tr['c_bound'].mean()) if n else 0.,
                    infeasible_steps=int(np.sum(~tr['c_feasible'])) if n else 0, slack_max=float(tr['c_slack_max'].max()) if n else 0.,
                    margin_max=float(tr['c_margin_max'].max()) if n else 0., standoff_max_m=float(tr['c_sigma_max'].max()) if n else 0.,
                    has_effort_qp=bool((opt.force_filter and opt.coordinated) or (xc.support is not None and opt.contact_rows)),
                    velocity_slack_max=float(tr['c_velocity_slack'].max()) if n else 0.,
                    no_authority_steps=int(np.sum(tr['c_no_authority'])) if n else 0,
                    velocity_rows_active_fraction=float(np.mean(tr['c_velocity_active'] > 0)) if n else 0.,
                    velocity_relaxed_steps=int(np.sum(tr['c_velocity_level'] > 0)) if n else 0,
                    effort_rows_active_fraction=float(np.mean(tr['c_effort_active'])) if n else 0.,
                    qp_stationarity_relative_max=float(np.nanmax(tr['c_qp_stationarity'])) if n and len(tr.get('c_qp_stationarity', [])) else 0.,
                    speed_rows_active_fraction=float(np.mean(tr['c_speed_active'])) if n else 0.,
                    command_deviation_rms=np.sqrt(np.mean((tr['c_phi'] - tr['c_phi_r']) ** 2, axis=0)) if n else np.zeros(xc.m)),
        storage=dict(max=float(tr['c_storage'].max()) if n else 0., final=float(tr['c_storage'][-1]) if n else 0.,
                     mean=float(np.mean(tr['c_storage'])) if n else 0.,
                     p99=float(np.percentile(tr['c_storage'], 99)) if n else 0.,
                     dissipated_J=float(np.sum(tr['c_dissipation'] * dtc)),
                     effort_mismatch=energy_split(tr['c_P_mis'], dtc), command_deviation=energy_split(tr['c_P_cmd'], dtc),
                     barrier_rows=energy_split(tr['c_P_rows'], dtc), effort_bounds=energy_split(tr['c_P_bounds'], dtc),
                     **({'contact_rows': energy_split(tr['c_P_contact'], dtc)} if 'c_P_contact' in tr and len(tr['c_P_contact']) else {}),
                     balance=storage_balance(tr) if n else {}),
        assumptions=dict(
            force_model_error_p99=np.percentile(fme, 99, axis=0), force_model_error_max=fme.max(0),
            declared_force_model_error=xc.e_force, force_model_coverage=float(fm_ok.mean()) if checked else None,
            force_model_checks_during_intervention=int(after_intervention.sum()),
            force_model_coverage_during_intervention=coverage(int(np.sum(~fm_ok[after_intervention])), int(after_intervention.sum())),
            row_residual_coverage=coverage(st['row_residual_violations'], st['row_residual_checks']),
            row_residual_max=st['row_residual_max'], row_residual_checks=st['row_residual_checks'],
            row_psi2_min_after_active=st['row_psi2_min_after_active'],
            load_bound_coverage=float(cover.mean()) if len(cover) else None,
            load_bound_checks_during_intervention=int(len(binding)),
            load_bound_coverage_during_intervention=coverage(int(np.sum(~binding)), int(len(binding))),
            load_error_norm_p99=float(np.percentile(tr['c_load_err'][settled], 99)) if n and settled.any() else 0.,
            late_activation_rows=st['late_activation_rows'], initial_condition_rows=st['initial_condition_rows']),
        soil=dict(bucket_mass_end_kg=float(tr['bucket_mass'][-1]) if len(tr['bucket_mass']) else 0.,
                  bucket_mass_max_kg=float(tr['bucket_mass'].max()) if len(tr['bucket_mass']) else 0.,
                  pit_depth_max_m=float(tr['pit_depth'].max()) if len(tr['pit_depth']) else 0.,
                  pit_volume_end_m3=float(tr['pit_volume'][-1]) if len(tr['pit_volume']) else 0., soil_force_peak_N=world.peak['soil_force_N'],
                  mass_error_kg=float(tr['mass_error'].max()) if len(tr['mass_error']) else 0., free_motion_scenario=bool(free_motion),
                  unintended_soil_contact=bool(free_motion and world.peak['soil_force_N'] > 0.)),
        plant=dict(world.peak, base_tilt_limit_rad=BASE_TILT_LIMIT,
                   base_support_kept=bool(contacts['kept'] if contacts is not None else world.peak['base_tilt_rad'] <= BASE_TILT_LIMIT),
                   contacts=contacts,
                   initial_closure_gap_m=initial_gap, mujoco=mujoco.__version__, native_solve=dict(native_module.STATS),
                   valve_saturated_fraction=float(tr['valve_saturated'].mean()) if len(tr['valve_saturated']) else 0., **world.energy_report()),
        controller=st, phase_names=timeline.names, phase_starts=getattr(timeline, 'starts', None))


def print_summary(report, out, wall):
    truth = report['truth']
    v = report['violated_pairs']
    print(f"Saved {out}  status={report['status']}  violated={v if v else 'none'}  wall={wall:.0f}s")
    con = report['plant'].get('contacts')
    if con is not None:
        print('  contacts', con['ids'], 'normal min (kN)', np.round(np.asarray(con['normal_force_min_N']) / 1e3, 1), 'ratio max',
              np.round(con['friction_ratio_max'], 2), 'slip max (mm)', np.round(1e3 * np.asarray(con['slip_max_m']), 1),
              'contact rows active', round(con['contact_row_fraction'], 3),
              'model error rms (kN)', np.round(np.asarray(con.get('model_error_rms_N', np.zeros((1, 3)))) / 1e3, 2).tolist())
        if not con['kept']:
            print('  WARNING: a ground contact was lost or slipped beyond the limit; the support interface assumes held contacts')
    elif not report['plant']['base_support_kept']:
        print(f"  WARNING: the base left its support (peak inclination {1e3 * report['plant']['base_tilt_rad']:.1f} mrad); the generated "
              'interfaces treat the base motion as measured and smooth')
    if report['soil']['unintended_soil_contact']:
        print(f"  WARNING: soil contact in a free-motion scenario (peak force {report['soil']['soil_force_peak_N']:.0f} N)")
    cls = truth['classes']
    print('  class h_min:', {k: round(r['h_min'], 4) for k, r in cls.items()})
    print('  class psi1_min:', {k: round(r['psi1_min'], 4) for k, r in cls.items()})
    print('  class sample-mean row function min:', {k: round(r['row_function_sample_mean_min'], 3) for k, r in cls.items()})
    print('  tracking RMS:', np.round(report['tracking']['rms'], 4), ' interventions:', round(report['filter']['intervention_fraction'], 3),
          ' infeasible:', report['filter']['infeasible_steps'], ' velocity relaxed:', report['filter']['velocity_relaxed_steps'],
          ' storage max:', round(report['storage']['max'], 1), ' standoff max:', round(report['filter']['standoff_max_m'], 4))
    A = report['assumptions']
    rnd = lambda x, k=4: None if x is None else round(x, k)
    print('  assumptions: force-model coverage', rnd(A['force_model_coverage']), '(during intervention', rnd(A['force_model_coverage_during_intervention']),
          ') row-residual coverage', rnd(A['row_residual_coverage']), 'max', round(A['row_residual_max'], 2), ' load coverage', rnd(A['load_bound_coverage']),
          '(during intervention', rnd(A['load_bound_coverage_during_intervention']), ') late rows', A['late_activation_rows'],
          ' psi2 after active min', round(A['row_psi2_min_after_active'], 3))
    bal = report['storage']['balance']
    print('  storage energy J:', {k: (round(v['injected_J'], 1), round(v['removed_J'], 1)) for k, v in report['storage'].items() if isinstance(v, dict) and 'injected_J' in v},
          'dissipated', round(report['storage']['dissipated_J'], 1), ' ledger residual', round(bal.get('residual_J', 0.), 1), 'J (relative',
          round(bal.get('residual_relative', 0.), 4), ')')


def run(machine, scenario, variant='full', *, name=None, partition='joint_space', duration=None, observer_gain=100., payload_scale=1.,
        soil_strength=1., render=False, out_root=None):
    """One hazard run.  Returns the report (also written to the result folder)."""
    name = name or f'{scenario}__{variant}'
    out = (out_root or machine.results_dir() / 'hazards') / name
    sha = code_hash(machine.name)                             # state of the sources when the run starts
    out.mkdir(parents=True, exist_ok=True)
    for stale in ('report.json', 'trace.npz', 'spec.json', 'manifest.json', 'render.npz'):      # a run that fails must not leave an older result
        (out / stale).unlink(missing_ok=True)
    arguments = dict(scenario=scenario, variant=variant, name=name, partition=partition, duration=duration, observer_gain=observer_gain,
                     payload_scale=payload_scale, soil_strength=soil_strength, render=render)
    soil_params = SoilParams()
    if soil_strength != 1.:
        soil_params.c_bank *= soil_strength
        soil_params.k_winkler *= soil_strength
        soil_params.p_bearing_bank *= soil_strength
    world = PlantWorld(machine, out, payload_scale=payload_scale, soil_params=soil_params)
    xc0 = compile_extended(world.ctrl_phys, machine.spec(world.ctrl_phys))
    planner = Planner(world.ctrl_phys, xc0, machine)
    kw, timeline, desc, free_motion = machine.scenario(scenario, world, xc0, planner)
    Ts = SAMPLE_TIME.get(variant, 0.01)
    spec = machine.spec(world.ctrl_phys, sample_time=Ts, **kw)
    spec = variant_spec(variant, spec)
    xc = compile_extended(world.ctrl_phys, spec, partition)
    save_json(out / 'spec.json', spec)
    save_json(out / 'manifest.json', xc.manifest)
    opt = FilterOptions(**VARIANTS[variant])
    ctl = SafeVDC(xc, opt, observer_gain=observer_gain)
    world.bind(xc)
    x0 = np.asarray(timeline.phases[0].reference(0.)[0], float)          # task coordinates of the start
    if partition != 'joint_space':
        qj = planner.closed(x0)
        u0 = np.array([qj[planner.ci.coordinate_ids.index(k)] for k in xc.ci.independent_ids])
    else:
        u0 = world.full_coordinates(xc, x0)                  # held coordinates (contact anchors) at their seed values
    initial_gap = world.place(xc.ci, u0)
    monitor = TruthMonitor(world, xc)
    recorder = RenderRecorder() if render else None
    total = min(duration, timeline.duration) if duration else timeline.duration
    status, error, tr, wall = run_closed_loop(world, ctl, timeline, duration=total, monitor=monitor, label=name, render=recorder)
    np.savez_compressed(out / 'trace.npz', **tr, pair_ids=np.array(list(xc.barriers.pair_rows)), phase_names=np.array(timeline.names))
    if recorder is not None:
        np.savez_compressed(out / 'render.npz', **recorder.arrays(), h_initial=world.sm.h_initial)
    truth = clearance_table(xc, monitor)
    info = dict(machine=machine.name, scenario=scenario, variant=variant, partition=partition, code_sha256=sha, arguments=arguments)
    try:
        report = build_report(info, name, status, error, tr, wall, world, xc, ctl, opt, timeline, truth, desc, kw, Ts, initial_gap, free_motion)
    except Exception:                                         # noqa: BLE001  (a failed run still leaves a report with its status)
        report = dict(status='ERROR', error=(error or '') + '\nreport: ' + traceback.format_exc(), machine=machine.name, scenario=scenario,
                      variant=variant, name=name, description=desc, code_sha256=sha, simulation_status=status, truth=truth,
                      violated_pairs=[p for p, r in truth['pairs'].items() if r['violated']])
        save_json(out / 'report.json', report)
        print(f"Saved {out}  status=ERROR (report could not be built)\n{report['error']}")
        return report
    save_json(out / 'report.json', report)
    print_summary(report, out, wall)
    return report
