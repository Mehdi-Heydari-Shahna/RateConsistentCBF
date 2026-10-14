"""Closed loop of plant, generated controller and reference source (machine independent; from the earlier release).

    reference source -> controller (every Ts) -> hydraulics, soil and native solve (every physics step) -> truth monitor, traces

``ctl.step(t, u, ud, base, f_meas, ref)`` returns channel efforts (rcx.runtime.SafeVDC);
``source.references(t)`` returns (x_des, xd_des, xdd_des) in the task coordinates, or ((...), base_target, base_rate);
``source.update(t, obs)`` advances guarded missions.
"""
from __future__ import annotations

import time
import traceback

import numpy as np

from .world import MODES, PlantWorld, TruthMonitor, work_metrics

SAMPLE_KEYS = ['time', 'qpos', 'qvel', 'phase', 'u', 'u_des', 'ud', 'ud_des', 'lip', 'channels', 'phi_nominal', 'port_effort', 'f_meas',
               'pressure', 'tracking_error', 'closure_gap', 'force_residual', 'soil_force', 'soil_torque', 'soil_mode', 'cut_depth',
               'bucket_volume', 'bucket_mass', 'payload_mass', 'base_pose', 'valve_saturated', 'pressure_limited', 'pair_h', 'tau_ext_hat',
               'storage', 'n_active', 'feasible', 'intervening', 'energy', 'site_rms', 'pit_volume', 'pit_depth', 'work_mean', 'work_rms',
               'mass_error', 'airborne_volume']
CTRL_KEYS = ['c_time', 'c_storage', 'c_dissipation', 'c_P_cmd', 'c_P_mis', 'c_P_rows', 'c_P_bounds', 'c_s', 'c_u_r', 'c_u_r0', 'c_ud',
             'c_e', 'c_phi', 'c_phi_r', 'c_phi_meas', 'c_lo', 'c_hi', 'c_tau_ext', 'c_n_active', 'c_feasible', 'c_bound', 'c_hmin',
             'c_psi1_min', 'c_psi2_min', 'c_row_min', 'c_margin_max', 'c_slack_max', 'c_velocity_active', 'c_velocity_level',
             'c_effort_active', 'c_speed_active', 'c_wall', 'c_load_comp', 'c_pair_h', 'c_tau_ext_true', 'c_load_err', 'c_load_bound',
             'c_load_norm', 'c_udd', 'c_f_applied', 'c_pair_active', 'c_pair_velocity_active', 'c_x', 'c_force_model_err',
             'c_resid_max', 'c_psi2_after', 'c_sigma_max', 'c_P_acc', 'c_P_load', 'c_P_zoh', 'c_E_step', 'c_velocity_slack', 'c_no_authority',
             'c_zs_min', 'c_uD_norm', 'c_a_r', 'c_pair_zs', 'c_pair_w', 'c_qp_stationarity']


def pair_slack_log(ctl, L, npairs):
    """Per barrier pair: slack zs_i = hdot_r_i + alpha1_i (h_i - sigma_i) of the first-order row at the required velocity and
    w_i = J_i s = hdot_r_i - hdot_i, for the row of the pair (among the rows in the broad phase) with the smallest slack;
    NaN for a pair without such a row.  (Paper: Corollary on the error at an activation.)"""
    pz = np.full(npairs, np.nan)
    pw = np.full(npairs, np.nan)
    zs = np.asarray(L['zs'], float)
    if len(zs):
        pidx = ctl._row_pair_index[np.asarray(L['on'], int)]
        wv = np.asarray(L['hdot_r'], float) - np.asarray(L['hdot'], float)
        for j in np.argsort(zs)[::-1]:          # the smallest slack of a pair is written last
            pz[pidx[j]] = zs[j]
            pw[pidx[j]] = wv[j]
    return pz, pw


def run_closed_loop(world: PlantWorld, ctl, source, *, duration, monitor: TruthMonitor = None, sample_s=0.04, base_target=None,
                    print_every=2., label='', render=None, stop_on_violation=None, on_sample=None, control_period=None):
    """Returns (status, error, traces dict, wall time).  ``world.bind(ctl.xc)`` must have been called."""
    d, sm = world.d, world.sm
    xc = ctl.xc
    dt = world.dt
    Ts = control_period if control_period is not None else xc.Ts
    control_every = max(1, int(round(Ts / dt)))
    sample_every = max(1, int(round(sample_s / dt)))
    servo_every = max(1, int(round(.001 / dt)))
    read_u = world.reader(xc.ci.independent_ids)            # all independent coordinates (commanded and held)
    read_x = world.reader(xc.task.ids)
    free = xc.free
    total_steps = int(round(duration / dt))
    trace = {k: [] for k in SAMPLE_KEYS + CTRL_KEYS}
    contact_keys = ['contact_true', 'contact_model', 'contact_slip', 'c_contact_active', 'c_contact_slack', 'c_P_contact'] if xc.support is not None else []
    trace.update({k: [] for k in contact_keys})
    travels = world.drive_data is not None
    base_target = world.machine_pose().copy() if base_target is None else np.asarray(base_target, float)
    base_rate = np.zeros(3)
    track_cmd = np.zeros(2)
    status, error = 'INCOMPLETE', None
    next_print = 0.
    start = time.perf_counter()
    channels = None
    obs = dict(tracking_error=0., soil_speed=0., bucket_mass=0., base_pose=world.machine_pose(), site_mass=0., site_peak=0., site_rms=0.,
               safety_hold=False, speed=0., hold_pairs=[])
    pair_ids = list(xc.barriers.pair_ids)
    u_r_prev = None
    try:
        for step in range(total_steps + 1):
            t = float(d.time)
            ref = source.references(t)
            if isinstance(ref[0], tuple):
                (x_des, xd_des, xdd_des), base_target, base_rate = ref
            else:
                x_des, xd_des, xdd_des = ref
            if step % control_every == 0:
                u, ud = read_u()
                base = world.base_state()
                if world.bank is None:
                    # first call: the native state was placed without actuator efforts, so its acceleration is that of an
                    # unsupported machine; the hydraulic states start on the static holding efforts of the nominal controller
                    base['acceleration'] = np.zeros(6)
                    ctl.step(t, u, ud, base, np.zeros(xc.nports), (x_des, xd_des, xdd_des))
                    world.start_hydraulics(ctl.last['phi_r'])
                    ctl.reset()
                channels = ctl.step(t, u, ud, base, world.measured_port_efforts(), (x_des, xd_des, xdd_des))
                L = ctl.last
                hon = L['h']
                act = L['active']
                rows = L['rows']
                vals = dict(
                    c_time=t, c_storage=L['storage'], c_dissipation=L['dissipation'], c_P_cmd=L['P_cmd'], c_P_mis=L['P_mis'],
                    c_P_rows=L['P_rows'], c_P_bounds=L['P_bounds'], c_s=L['s'].copy(), c_u_r=L['u_r'].copy(), c_u_r0=L['u_r0'].copy(), c_ud=L['ud'].copy(),
                    c_e=L['e'].copy(), c_phi=L['phi'].copy(), c_phi_r=L['phi_r'].copy(), c_phi_meas=L['phi_meas'].copy(), c_lo=L['lo'].copy(),
                    c_hi=L['hi'].copy(), c_tau_ext=L['tau_ext'].copy(), c_n_active=int(np.sum(act)), c_feasible=bool(L['feasible']),
                    c_bound=bool(np.any(L['bound_active'])), c_hmin=float(np.min(hon)) if len(hon) else np.inf,
                    c_psi1_min=float(np.min(L['psi1'])) if len(hon) else np.inf,
                    c_psi2_min=float(np.min(L['psi2'] - L['margin_components']['load_error'])) if len(hon) else np.inf,
                    c_row_min=float(np.min(L['psi2_pred'][act] - L['margin'][act])) if np.any(act) else np.nan,
                    c_margin_max=float(np.max(L['margin'][act])) if np.any(act) else 0.,
                    c_slack_max=float(np.max(L['slack'])) if len(hon) else 0., c_velocity_active=int(np.sum(L['velocity_rows_active'])),
                    c_velocity_level=int(L['velocity_level']), c_effort_active=bool(L['effort_active']), c_speed_active=bool(L['speed_active']),
                    c_wall=L['wall_s'], c_load_comp=L['load_comp'].copy(),
                    c_pair_h=np.array([float(np.min(rows.h[idx])) for idx in xc.barriers.pair_rows.values()]),
                    c_pair_active=L['pair_active'], c_pair_velocity_active=L['pair_velocity_active'], c_x=L['x'].copy(),
                    c_force_model_err=L['force_model_error'].copy(), c_resid_max=L['row_residual_max'], c_psi2_after=L['psi2_after_active'],
                    c_sigma_max=float(np.max(L['sigma'][act])) if np.any(act) else 0., c_P_acc=L['P_acc'],
                    c_velocity_slack=float(L['velocity_slack']),
                    c_zs_min=float(np.min(L['zs'])) if len(hon) else np.inf,
                    c_uD_norm=float(np.sqrt(max(L['uD'] @ L['Mr'] @ L['uD'], 0.))), c_a_r=L['a_r'].copy(),
                    c_pair_zs=pair_slack_log(ctl, L, len(pair_ids))[0], c_pair_w=pair_slack_log(ctl, L, len(pair_ids))[1],
                    c_no_authority=bool(np.any(L['no_authority']) or np.any(L['velocity_no_authority'])),
                    # relative stationarity residual of the effort QP (rcx.qp: |H x + g + A^T mu| over the size of its terms)
                    c_qp_stationarity=float(L['kkt'].get('stationarity_relative', np.nan)) if L.get('kkt') else 0.)
                for k, val in vals.items():
                    trace[k].append(val)
                if contact_keys:
                    trace['c_contact_active'].append(int(np.sum(L['contact_active'])))
                    trace['c_contact_slack'].append(float(np.max(L['contact_slack'], initial=0.)))
                    trace['c_P_contact'].append(float(L['P_contact']))
            if travels and step % servo_every == 0:
                track_cmd = world.travel_command(base_target, base_rate, .001, update_outer=step % control_every == 0)
            solve = world.step(channels, track_cmd)
            if monitor is not None:
                monitor.update(t)
            if step % control_every == 0:
                # model residual with the native acceleration of this instant: the external load the reduced model must
                # carry (soil, payload, friction, base-motion filter error) and the error of its estimate
                L = ctl.last
                udd = read_u.acceleration()[free]
                tau_true = L['Mr'] @ udd + L['hr'] - L['L'].T @ L['f_meas']
                err = L['tau_ext'] - tau_true
                trace['c_tau_ext_true'].append(tau_true)
                trace['c_P_load'].append(-float(L['s'] @ (tau_true - L['load_comp'])))      # external load that is not compensated
                # Zero-order hold of the required velocity (as in the earlier single-machine release): the nodal ledger
                # integrates nodal values by the trapezoidal rule; the exact storage change of a held interval differs by
                #     Ts / 2 (du_r^T M_r u_dot - s^T M_r_dot du_r + du_r^T M_r_dot du_r / 2) - du_r^T M_r du_r / 2
                # per sample (c_P_zoh: this energy divided by Ts) and by the end term s^T M_r du_r / 2 (c_E_step).
                du_r = np.zeros_like(L['u_r']) if u_r_prev is None else L['u_r'] - u_r_prev
                u_r_prev = L['u_r'].copy()
                Md = L['Mr_dot']
                e_zoh = (0.5 * Ts * float(du_r @ L['Mr'] @ udd - L['s'] @ Md @ du_r + 0.5 * du_r @ Md @ du_r)
                         - 0.5 * float(du_r @ L['Mr'] @ du_r))
                trace['c_P_zoh'].append(e_zoh / Ts)
                trace['c_E_step'].append(0.5 * float(L['s'] @ L['Mr'] @ du_r))
                trace['c_load_err'].append(float(np.sqrt(max(err @ np.linalg.solve(L['Mr'], err), 0.))))
                trace['c_load_bound'].append(L['load_bound']); trace['c_load_norm'].append(L['load_norm']); trace['c_udd'].append(udd)
                trace['c_f_applied'].append(world.applied_port_efforts())
            sampled = step % sample_every == 0 or step == total_steps
            if sampled or t >= next_print:
                x_now, xd_now = read_x()
                tracking = float(np.max(np.abs(x_now - x_des))) if len(x_des) == len(x_now) else 0.
            if sampled:
                idx, hv = world.sync_terrain()
                if render is not None:
                    render.add(idx, hv, sm)
                mt = sm.metrics()
                wm = work_metrics(sm, world.boxes['work'])
                world.peak['bucket_mass_kg'] = max(world.peak['bucket_mass_kg'], mt['bucket_mass'])
                L = ctl.last
                vals = dict(time=t, qpos=d.qpos.copy(), qvel=d.qvel.copy(), phase=source.phase, u=x_now.copy(), u_des=np.array(x_des),
                            ud=xd_now.copy(), ud_des=np.array(xd_des), lip=d.site_xpos[world.lip_site].copy(), channels=channels.copy(),
                            phi_nominal=L['phi_r'].copy(), port_effort=world.applied_port_efforts(), f_meas=world.measured_port_efforts(),
                            pressure=world.bank.pressure.copy(), tracking_error=tracking, closure_gap=world.closure,
                            force_residual=solve['final_residual'], soil_force=world.soil_F.copy(), soil_torque=world.soil_T.copy(),
                            soil_mode=MODES.get(sm.last['mode'], -1), cut_depth=sm.last['depth'], bucket_volume=sm.content,
                            bucket_mass=mt['bucket_mass'], payload_mass=world.coupler.payload, base_pose=world.machine_pose(),
                            valve_saturated=world.item['valve_saturated'], pressure_limited=world.item['pressure_limit_active'],
                            pair_h=monitor.pair_values() if monitor is not None else np.zeros(0), tau_ext_hat=L['tau_ext'].copy(),
                            storage=L['storage'], n_active=int(np.sum(L['active'])), feasible=bool(L['feasible']),
                            intervening=bool(L['intervening']), energy=d.energy.copy(), site_rms=mt.get('site_rms', 0.),
                            pit_volume=mt.get('pit_volume', 0.), pit_depth=mt.get('pit_depth', 0.), work_mean=wm['mean'], work_rms=wm['rms'],
                            mass_error=mt['mass_error'], airborne_volume=mt['airborne_volume'])
                for k, val in vals.items():
                    trace[k].append(val)
                if contact_keys:                                # ground contacts: plant forces, model forces at the measured efforts, slip
                    trace['contact_true'].append(world.contact['force'].copy())
                    trace['contact_model'].append(L['contact_force'].copy())
                    trace['contact_slip'].append(world.contact['slip'].copy())
                obs = dict(tracking_error=tracking, soil_speed=mt['soil_speed'], bucket_mass=mt['bucket_mass'], base_pose=world.machine_pose(),
                           site_mass=mt.get('site_mass', 0.), site_peak=mt.get('site_peak', 0.), site_rms=mt.get('site_rms', 0.),
                           safety_hold=bool(np.any(L['active']) or np.any(L['velocity_rows_active'])),
                           hold_pairs=[pair_ids[i] for i in np.flatnonzero(L['pair_active'] | L['pair_velocity_active'])],
                           speed=float(np.max(np.abs(xd_now))), s_norm=float(np.max(np.abs(L['s']))))
                if on_sample is not None:
                    on_sample(t, obs)
                source.update(t, obs)
                if source.done:
                    status = 'TASK_FAILED' if source.failures else 'COMPLETED'
                    break
                if stop_on_violation is not None and monitor is not None and stop_on_violation(monitor):
                    status = 'STOPPED_ON_VIOLATION'
                    break
            if t >= next_print:
                L = ctl.last
                hm = float(np.min(L['h'])) if len(L['h']) else np.inf
                print(f"{label} t={t:6.2f}s [{source.phase:2d}] {source.names[source.phase][:30]:30s} err={tracking:.4f} soil={sm.content * sm.p.rho_loose:5.0f}kg "
                      f"F={np.linalg.norm(world.soil_F) / 1e3:5.1f}kN hmin={hm:6.3f} act={int(np.sum(L['active'])):2d} nu={L['storage']:8.2f} "
                      f"gap={world.closure * 1e6:4.1f}um wall={time.perf_counter() - start:4.0f}s", flush=True)
                next_print += print_every
            if step == total_steps:
                # a guarded mission that is still inside its phases when the time limit is reached has not completed
                unfinished = bool(getattr(source, 'guarded', False)) and not source.done
                status = 'TASK_FAILED' if source.failures else ('TIME_LIMIT' if unfinished else 'COMPLETED')
                break
            world.advance()
    except (Exception, KeyboardInterrupt):
        error = traceback.format_exc()
        status = 'ERROR'
        print(error, flush=True)
    # an exception inside a control sample leaves traces of different length: cut every group back to its common length
    for keys in (SAMPLE_KEYS + [k for k in contact_keys if not k.startswith('c_')], CTRL_KEYS + [k for k in contact_keys if k.startswith('c_')]):
        common = min(len(trace[k]) for k in keys)
        for k in keys:
            del trace[k][common:]
    out = {}
    for k, v in trace.items():
        out[k] = np.asarray(v) if len(v) else np.zeros(0)
    return status, error, out, time.perf_counter() - start
