"""Metrics of the excavator runs as they are reported in the paper (Tables II-V, Figs. 5-7 and the numbers of Section
VII-B), computed from the logged results of one run: ``report.json`` and ``trace.npz`` of results/<machine>/hazards/<name>
or results/<machine>/<mission name>.

Definitions (captions of Tables II-V):
  S_max            largest storage S = 1/2 |s|_M^2 over the control samples (J)
  onset / release  largest |s|_M = sqrt(2 S) over the control samples of the six approach phases ('cycle j: go') and of
                   the six return phases ('cycle j: back') (sqrt(J))
  energy into S    E_j^+ = Ts sum_l max(P_j(t_l), 0) over the control samples (positive parts), for the terms of the sampled
                   storage balance: port P_f = -s^T C (phi - phi_r) (trace c_P_cmd), feedforward <s, du_r/Ts - a_r>_M (c_P_acc),
                   force loops -s^T C (f - phi) (c_P_mis - c_P_cmd), other: uncompensated load (c_P_load) and zero-order hold
                   (c_P_zoh), each with its positive part
  port share       E_port^+ / (sum of the four positive energies) (%)
  clearance        smallest plant-side value of the barrier functions of the pairs of the scenario (truth monitor, every
                   physics step; negative: violated) (mm)
  effort           share of the control samples with a binding barrier constraint of the effort layer (%)
  RMS              largest RMS error (40 ms samples) of the task coordinates not driven into the limit (mrad)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tracefile import load_trace      # noqa: E402

# task coordinates driven into the limit by the scenario 'cycles' and the coordinates left out of the RMS column
CYCLE_PAIR = dict(komatsu=['p3', 'p4'], heap=['P_BOOM'])            # boom stroke limit (Komatsu: two boom cylinders)
CYCLE_DRIVEN = dict(komatsu=['boom'], heap=None)                    # HEAP: given by ARM_RMS below
SCENARIO_PAIRS = dict(komatsu=dict(efence=['efence_ceiling', 'efence_front', 'efence_side'], self=['self_track_L', 'self_track_R'],
                                   site_mission=['efence_floor', 'pipe', 'efence_ceiling', 'efence_front']),
                      heap=dict(efence=['efence_ceiling', 'efence_front', 'efence_side'], self=['self_wheel_LF', 'self_wheel_RF'],
                                site_mission=['efence_floor', 'pipe', 'efence_ceiling', 'efence_front']))


class LogTrace:
    """The control-sample signals of a run read back from its exported log (results/csv/runs/<machine>/<run>.csv, written
    by paper/tables.py), under the key names of trace.npz, so that the metrics can be recomputed without the raw traces.
    Pair signals are available for the pairs of the scenario that the log contains (``pair_ids``)."""

    def __init__(self, csv_path):
        import csv as _csv
        with open(csv_path) as f:
            rows = [r for r in _csv.reader(l for l in f if not l.startswith('#'))]
        head = rows[0]
        data = np.array([[float(x) for x in r] for r in rows[1:]], float)
        col = {h: data[:, i] for i, h in enumerate(head)}
        self._d = dict(c_time=col['t_s'], c_storage=col['storage_J'], c_P_cmd=col['P_port_W'], c_P_acc=col['P_feedforward_W'],
                       c_P_mis=col['P_force_loops_W'] + col['P_port_W'], c_P_load=col['P_load_W'], c_P_zoh=col['P_zoh_W'],
                       c_dissipation=col['dissipation_W'], c_n_active=col['effort_rows_binding'], c_feasible=col['qp_feasible'] > 0.5,
                       c_qp_stationarity=col['qp_stationarity_rel'], c_wall=1e-3 * col['controller_wall_ms'],
                       c_phase=col['phase'].astype(int))            # phase index as assigned from the unrounded sample times
        pairs = [h[2:] for h in head if h.startswith('h_')]
        self._d['pair_ids'] = np.array(pairs)
        for key, prefix in (('c_pair_h', 'h_'), ('c_pair_active', 'effort_active_'), ('c_pair_velocity_active', 'first_layer_active_'),
                            ('c_pair_zs', 'zs_'), ('c_pair_w', 'w_')):
            if all(prefix + q in col for q in pairs):
                self._d[key] = np.stack([col[prefix + q] for q in pairs], axis=1) if pairs else np.zeros((len(data), 0))
        self.files = list(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def close(self):
        pass


def load(folder):
    """(report, trace) of a run: the raw trace.npz of a run folder, or, for a folder of the exported logs
    (results/runs/<machine>/<run>, see paper/tables.py), the control samples of results/csv/runs/<machine>/<run>.csv."""
    folder = Path(folder)
    rep = json.loads((folder / 'report.json').read_text())
    if (folder / 'trace.npz').exists():
        return rep, load_trace(folder)
    csv_path = folder.parents[2] / 'csv' / 'runs' / folder.parent.name / f'{folder.name}.csv'
    return rep, LogTrace(csv_path)


def phase_index(rep, t):
    """Index of the phase (in report['phase_names']) of every time in t, as the timeline assigns it (bench.reference)."""
    starts = np.asarray(rep['phase_starts'], float)
    return np.clip(np.searchsorted(starts, np.asarray(t, float), side='right') - 1, 0, len(rep['phase_names']) - 1)


def phase_mask(rep, t, word, idx=None):
    names = rep['phase_names']
    idx = phase_index(rep, t) if idx is None else idx
    sel = np.array([word in n for n in names])
    return sel[idx]


def positive_energy(p, Ts):
    return float(Ts * np.sum(np.maximum(np.asarray(p, float), 0.)))


def energies(tr, Ts):
    """Positive energies of the terms of the sampled storage balance.  c_P_mis = -s^T C (f - phi_r) is the power of the
    delivered efforts against the nominal ones; the force-loop defect is c_P_mis - c_P_cmd = -s^T C (f - phi)."""
    P_cmd = np.asarray(tr['c_P_cmd'], float)
    E = dict(port=positive_energy(P_cmd, Ts), feedforward=positive_energy(tr['c_P_acc'], Ts),
             force_loops=positive_energy(np.asarray(tr['c_P_mis'], float) - P_cmd, Ts),
             other=positive_energy(tr['c_P_load'], Ts) + positive_energy(tr['c_P_zoh'], Ts))
    total = sum(E.values())
    E['port_share_percent'] = 100. * E['port'] / total if total > 0. else 0.
    return E


def snorm(tr):
    return np.sqrt(2. * np.maximum(np.asarray(tr['c_storage'], float), 0.))


def clearance_mm(rep, pairs):
    vals = [rep['truth']['pairs'][p]['h_min'] for p in pairs if p in rep['truth']['pairs']]
    return 1e3 * float(min(vals)) if vals else float('nan')


def rms_mrad(rep, exclude_labels=(), only_labels=None):
    labels = rep['task_labels']
    rms = np.asarray(rep['tracking']['rms'], float)
    keep = [i for i, l in enumerate(labels) if l not in exclude_labels and (only_labels is None or l in only_labels)]
    return 1e3 * float(np.max(rms[keep])) if keep else float('nan')


# HEAP: the arm coordinates of the RMS column of Tables II and III (the boom is driven into the limit, the telescope left out)
HEAP_ARM_RMS = None          # filled from the task labels of the report (see arm_labels)


def arm_labels(rep):
    """Task labels of the arm of HEAP without the boom and the telescope (Table II: 'HEAP: of the arm, except the
    telescope'); the chassis and leg coordinates are not part of the arm."""
    out = []
    for l in rep['task_labels']:
        low = l.lower()
        if any(w in low for w in ('chassis', 'leg', 'base', 'wheel', 'steer', 'hfe', 'haa', 'b_')):
            continue
        if 'boom' in low or 'tele' in low:
            continue
        out.append(l)
    return out


def cycles(folder, machine):
    """Row of Table II / III for one run of the scenario 'cycles'."""
    rep, tr = load(folder)
    Ts = float(rep['sample_time_s'])
    t = np.asarray(tr['c_time'], float)
    sn = snorm(tr)
    ph = np.asarray(tr['c_phase'], int) if 'c_phase' in tr else phase_index(rep, t)
    go, back = phase_mask(rep, t, ': go', ph), phase_mask(rep, t, ': back', ph)
    E = energies(tr, Ts)
    if machine == 'komatsu':
        rms = rms_mrad(rep, exclude_labels=CYCLE_DRIVEN['komatsu'])
    else:
        rms = rms_mrad(rep, only_labels=arm_labels(rep))
    act = np.asarray(tr['c_n_active']) > 0
    out = dict(status=rep['status'], S_max=float(np.max(tr['c_storage'])), onset=float(np.max(sn[go])), release=float(np.max(sn[back])),
               **{f'E_{k}': v for k, v in E.items() if k != 'port_share_percent'}, port_share_percent=E['port_share_percent'],
               clearance_mm=clearance_mm(rep, CYCLE_PAIR[machine]), effort_percent=100. * float(np.mean(act)), rms_mrad=rms,
               infeasible_samples=int(rep['filter']['infeasible_steps']), violated_pairs=rep['violated_pairs'],
               mean_wall_ms=1e3 * float(rep['controller'].get('mean_wall_s', np.nan)),
               onsets=[float(np.max(sn[ph == j])) for j, n in enumerate(rep['phase_names']) if n.endswith(': go')],
               releases=[float(np.max(sn[ph == j])) for j, n in enumerate(rep['phase_names']) if n.endswith(': back')])
    tr.close()
    return out


def scenario(folder, machine, name):
    """Row of Table IV for one run of a workspace scenario (efence, self)."""
    rep, tr = load(folder)
    Ts = float(rep['sample_time_s'])
    E = energies(tr, Ts)
    out = dict(status=rep['status'], S_max=float(np.max(tr['c_storage'])), E_port=E['port'],
               clearance_mm=clearance_mm(rep, SCENARIO_PAIRS[machine][name]), rms_mrad=rms_mrad(rep),
               infeasible_samples=int(rep['filter']['infeasible_steps']), violated_pairs=rep['violated_pairs'])
    tr.close()
    return out


def mission(folder):
    """Column of Table V for one mission run."""
    rep, tr = load(folder)
    Ts = float(rep['sample_time_s'])
    E = energies(tr, Ts)
    h = [r['h_min'] for r in rep['truth']['pairs'].values()]
    out = dict(status=rep['status'], duration_s=float(rep['duration_s']), S_max=float(np.max(tr['c_storage'])), E_port=E['port'],
               E_feedforward=E['feedforward'], E_force_loops=E['force_loops'], E_other=E['other'], port_share_percent=E['port_share_percent'],
               clearance_mm=1e3 * float(min(h)), violated_pairs=rep['violated_pairs'],
               excavated_m3=float(rep['soil']['pit_volume_end_m3']), infeasible_samples=int(rep['filter']['infeasible_steps']))
    tr.close()
    return out
