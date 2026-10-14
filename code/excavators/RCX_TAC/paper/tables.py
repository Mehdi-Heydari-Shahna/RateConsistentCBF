#!/usr/bin/env python3
"""Tables II-V, Figs. 5-7 and the numbers of Section VII-B of the paper, computed from the logged results of the runs of
``batch.py all paper``, with the values printed in the paper next to them.

    python paper/tables.py [--out <folder>] [--allow-other-sources]
    python paper/tables.py --from-logs <supplement>/results --out <folder>      (Tables II-V and numbers from the exported logs)

Writes into <out> (default: ../../results relative to the supplement's code folder, i.e. TAC_supplement/results):
    tables/table2.csv ... table5.csv, tables/tables_ii_v.md     regenerated values and the values printed in the paper
    numbers/section_vii_b.json, numbers/section_vii_b.md        numbers quoted in the text of Section VII-B
    figures/fig5.pdf, fig6.pdf, fig7.pdf                        figures
    csv/fig5_<machine>.csv, csv/fig6_<machine>.csv, csv/fig6_<machine>_clearance.csv, csv/fig7_gain.csv
    runs/<machine>/<run>/report.json and run.log, csv/runs/<machine>/<run>.csv    logged results of every run (control samples)

Every report must carry the hash of the present sources (bench.world.code_hash); otherwise the script stops, unless
--allow-other-sources is given (the tables then say so).  The hash includes the machine packages, which are supplied
separately (machine_models.py): with --from-logs and without a machine package, the logged reports of that machine are
used without this check (the tables then say so).  Definitions of the columns: paper/metrics.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import metrics as Mx      # noqa: E402

E = np.e
# ------------------------------------------------------------------------------------------------ values printed in the paper
PAPER_T2 = {          # as printed (strings keep the printed precision)
    ('komatsu', 'kinematic_only'): ('402', '28.4', '28.3', '0', '6763', '5769', '668', '0', '2.8', '0', '0.30'),
    ('komatsu', 'full'): ('401', '15.9', '28.3', '1835', '3788', '3813', '349', '18.8', '2.8', '10.2', '1.95'),
    ('komatsu', 'full_ffraw'): ('173', '12.7', '18.6', '1006', '280', '2686', '294', '23.6', '2.8', '8.8', '1.24'),
    ('komatsu', 'accel_b40'): ('36.8', '8.58', '8.40', '340', '280', '786', '144', '21.9', '2.8', '35.2', '0.81'),
    ('komatsu', 'consistent'): ('35.3', '4.78', '8.40', '0.0063', '99.3', '797', '149', '<0.1', '2.8', '34.5', '0.28'),
    ('heap', 'kinematic_only'): ('407', '28.5', '28.1', '125', '5844', '6204', '710', '1.0', '19.9', '0', '0.59'),
    ('heap', 'full'): ('405', '28.5', '27.5', '490', '5657', '5996', '873', '3.8', '20.1', '12.3', '0.70'),
    ('heap', 'accel_b40'): ('63.4', '11.3', '10.5', '131', '257', '1749', '504', '5.0', '20.0', '34.1', '0.63'),
    ('heap', 'consistent'): ('54.8', '7.87', '10.5', '4.73', '144', '1462', '398', '0.2', '20.0', '34.3', '0.59')}
T2_COLS = ('S_max', 'onset', 'release', 'E_port', 'E_feedforward', 'E_force_loops', 'E_other', 'port_share_percent', 'clearance_mm',
           'effort_percent', 'rms_mrad')
T2_HEAD = ('S_max (J)', 'onset (sqrt J)', 'release (sqrt J)', 'port (J)', 'feedfwd. (J)', 'force loops (J)', 'other (J)', 'port (%)',
           'clear. (mm)', 'effort (%)', 'RMS (mrad)')
T2_ROWS = (('komatsu', 'kinematic_only', 'first layer only'), ('komatsu', 'full', 'baseline'), ('komatsu', 'full_ffraw', 'raw feedforward'),
           ('komatsu', 'accel_b40', 'acceleration level, beta = 40'), ('komatsu', 'consistent', 'proposed'),
           ('heap', 'kinematic_only', 'first layer only'), ('heap', 'full', 'baseline'), ('heap', 'accel_b40', 'acceleration level, beta = 40'),
           ('heap', 'consistent', 'proposed'))
PAPER_T3 = {
    'full_k2': ('436', '20.5', '29.5', '1465', '3.7', '2.9', '5.66'), 'full': ('401', '15.9', '28.3', '1835', '2.8', '10.2', '1.95'),
    'full_k24': ('287', '13.8', '24.0', '2334', '2.8', '11.4', '1.35'), 'consistent_k2': ('51.5', '6.26', '10.1', '5.03', '2.8', '18.4', '1.23'),
    'consistent': ('35.3', '4.78', '8.40', '0.0063', '2.8', '34.5', '0.28'), 'consistent_k24': ('26.0', '3.91', '7.21', '<1e-3', '2.8', '34.9', '0.16'),
    'consistent_b8': ('35.3', '4.97', '8.40', '<1e-3', '2.8', '30.6', '0.17'), 'kinematic_only_k2': ('650', '36.1', '29.5', '0', '-0.4', '0', '0.87'),
    'kinematic_only': ('402', '28.4', '28.3', '0', '2.8', '0', '0.30'), 'full_bucket150': ('454', '17.7', '30.1', '2029', '3.0', '6.5', '2.80'),
    'consistent_bucket150': ('50.4', '6.59', '10.0', '0.711', '3.0', '31.4', '1.83'), 'kinematic_only_bucket150': ('454', '29.0', '30.1', '0', '3.0', '0', '1.84')}
T3_COLS = ('S_max', 'onset', 'release', 'E_port', 'clearance_mm', 'effort_percent', 'rms_mrad')
T3_HEAD = ('S_max (J)', 'onset (sqrt J)', 'release (sqrt J)', 'port (J)', 'clear. (mm)', 'effort (%)', 'RMS (mrad)')
T3_ROWS = (('full_k2', 'baseline', 2), ('full', 'baseline', 12), ('full_k24', 'baseline', 24), ('consistent_k2', 'proposed', 2),
           ('consistent', 'proposed', 12), ('consistent_k24', 'proposed', 24), ('consistent_b8', 'proposed, beta = 8', 12),
           ('kinematic_only_k2', 'first layer only', 2), ('kinematic_only', 'first layer only', 12),
           ('full_bucket150', 'baseline, bucket x1.5', 12), ('consistent_bucket150', 'proposed, bucket x1.5', 12),
           ('kinematic_only_bucket150', 'first layer only, bucket x1.5', 12))
# Tables IV and V and the numbers of Section VII-B as printed in the revised manuscript.  The first submission printed,
# from runs with the earlier QP routine (rcx.qp before the correction of its steps for nearly dependent rows, see
# pinocchio/num/qp_before_fix.py), in Table IV for the Komatsu e-fence port energies 344 (baseline) and 1.80 (proposed)
# and for the Komatsu self-collision port energy 83.6 (baseline); all other values are unchanged.
PAPER_T4 = {('komatsu', 'efence'): ('177', '22.4', '343', '1.82', '77.4', '76.7', '243.3', '241.9'),
            ('komatsu', 'self'): ('48.4', '22.5', '83.5', '5.89', '67.1', '67.0', '60.1', '58.2'),
            ('heap', 'efence'): ('337', '74.0', '31.0', '0.176', '100.9', '100.7', '269.2', '268.7'),
            ('heap', 'self'): ('36.8', '6.76', '31.8', '0.800', '93.2', '92.5', '107.2', '106.6')}
PAPER_T5 = {('komatsu', 'full'): ('397', '118.8', '556.9', '1679', '1606', '7593', '4.9', '35.3', 'none', '0.526'),
            ('komatsu', 'consistent'): ('397', '84.1', '95.0', '130', '819', '6148', '1.3', '35.3', 'none', '0.527'),
            ('heap', 'full'): ('406', '90.3', '190.8', '513', '1465', '10399', '1.5', '32.0', 'none', '0.501'),
            ('heap', 'consistent'): ('406', '57.4', '21.0', '130', '1251', '9965', '0.2', '32.0', 'none', '0.507')}
T5_COLS = ('duration_s', 'S_max', 'E_port', 'E_feedforward', 'E_force_loops', 'E_other', 'port_share_percent', 'clearance_mm', 'violated', 'excavated_m3')
T5_HEAD = ('Duration T (s)', 'Largest storage S_max (J)', 'Port energy (J)', 'Feedforward energy (J)', 'Force-loop energy (J)',
           'Other energy (J)', 'Port share (%)', 'Minimum clearance (mm)', 'Violated pairs', 'Excavated volume (m^3)')
GAINS = (2, 4, 6, 12, 24)


def printed_like(v, ref):
    """v with the decimals (or the significant digits of a mantissa) of the printed value ref (a string)."""
    from decimal import Decimal
    if 'e' in ref.lower():
        mant = ref.lower().split('e')[0]
        digits = len(mant.replace('-', '').replace('.', '').lstrip('0'))
        return f'{v:.{max(digits - 1, 0)}e}'
    return f'{v:.{max(-Decimal(ref).as_tuple().exponent, 0)}f}'


def fmt(x, ref=None):
    """Regenerated value as the paper would print it: with the precision of the printed value ``ref``."""
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    if isinstance(ref, str) and ref.startswith('<'):
        return ref if abs(x) < float(ref[1:]) else f'{x:.2g}'
    if isinstance(ref, str):
        try:
            return printed_like(float(x), ref)
        except Exception:                                   # noqa: BLE001  (non-numeric printed value)
            pass
    ax = abs(x)
    if x == 0:
        return '0'
    if ax >= 100:
        return f'{x:.0f}'
    return f'{x:.3g}' if ax >= 0.01 else f'{x:.2g}'


def cell(v, ref):
    """Markdown cell: the regenerated value as the paper prints it and, in parentheses, the printed value; bold where the
    two differ at the printed precision."""
    shown = fmt(v, ref)
    if isinstance(ref, str) and ref.startswith('<'):
        differs = shown != ref
    elif isinstance(ref, str) and ref not in ('none',):
        try:
            differs = float(shown) != float(ref)
        except ValueError:
            differs = shown != ref
    else:
        differs = str(shown) != str(ref)
    return f'**{shown}** ({ref})' if differs else f'{shown} ({ref})'


class Runs:
    def __init__(self, results, allow_other, from_logs=False):
        from bench.world import code_hash
        from machine_models import present
        self.results, self.allow_other, self.sha, self.mismatch, self.cache = results, allow_other, {}, [], {}
        self.from_logs = from_logs
        # machines whose package is not present: their source hash cannot be formed
        self.unchecked = [m for m in ('komatsu', 'heap') if not present(m)]
        for m in ('komatsu', 'heap'):
            self.sha[m] = code_hash(m)

    def folder(self, machine, name):
        if self.from_logs:
            return self.results / 'runs' / machine / name
        return self.results / machine / ('' if name.startswith('mission') else 'hazards') / name

    def check(self, machine, name):
        if machine in self.unchecked and not self.from_logs:
            from machine_models import require
            require(machine)
        rep = json.loads((self.folder(machine, name) / 'report.json').read_text())
        if machine not in self.unchecked and rep.get('code_sha256') != self.sha[machine]:
            self.mismatch.append(f'{machine}/{name}')
            if not self.allow_other:
                raise SystemExit(f'{machine}/{name}: report of other sources (run batch.py again or use --allow-other-sources)')
        if rep.get('status') != 'COMPLETED':
            raise SystemExit(f'{machine}/{name}: status {rep.get("status")}')

    def cycles(self, machine, variant):
        key = (machine, 'cycles', variant)
        if key not in self.cache:
            name = f'cycles__{variant}'
            self.check(machine, name)
            self.cache[key] = Mx.cycles(self.folder(machine, name), machine)
        return self.cache[key]

    def scenario(self, machine, sc, variant):
        key = (machine, sc, variant)
        if key not in self.cache:
            name = f'{sc}__{variant}'
            self.check(machine, name)
            self.cache[key] = Mx.scenario(self.folder(machine, name), machine, sc)
        return self.cache[key]

    def mission(self, machine, variant):
        key = (machine, 'mission', variant)
        if key not in self.cache:
            name = f'mission_{variant}'
            self.check(machine, name)
            self.cache[key] = Mx.mission(self.folder(machine, name))
        return self.cache[key]


def write_csv(path, header, rows, comment=None, digits=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        for line in (comment or '').splitlines():
            f.write(f'# {line}\n')
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f'{x:.{digits}g}' if isinstance(x, (float, np.floating)) else x for x in r])


def md_table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


# ------------------------------------------------------------------------------------------------ tables
def table2(R, out):
    rows_csv, rows_md = [], []
    for m, va, label in T2_ROWS:
        r = R.cycles(m, va)
        vals = [r[c] for c in T2_COLS]
        ref = PAPER_T2[(m, va)]
        rows_csv.append([m, label, va] + vals + list(ref))
        rows_md.append([m, label] + [cell(v, p) for v, p in zip(vals, ref)])
    write_csv(out / 'tables' / 'table2.csv', ['machine', 'variant', 'run_variant'] + list(T2_COLS) + [f'paper_{c}' for c in T2_COLS], rows_csv,
              'Table II: six approaches to the extended boom stroke end (k = 12); regenerated values and the values printed in the paper')
    return md_table(('machine', 'variant') + T2_HEAD, rows_md)


def table3(R, out):
    rows_csv, rows_md = [], []
    for va, label, k in T3_ROWS:
        r = R.cycles('komatsu', va)
        vals = [r[c] for c in T3_COLS]
        ref = PAPER_T3[va]
        rows_csv.append([label, va, k] + vals + list(ref))
        rows_md.append([label, k] + [cell(v, p) for v, p in zip(vals, ref)])
    write_csv(out / 'tables' / 'table3.csv', ['variant', 'run_variant', 'k'] + list(T3_COLS) + [f'paper_{c}' for c in T3_COLS], rows_csv,
              'Table III: Komatsu, the scenario of Table II: gain k, rate of the proposed layer, first layer alone, heavier bucket')
    return md_table(('variant', 'k') + T3_HEAD, rows_md)


def table4(R, out):
    rows_csv, rows_md = [], []
    for (m, sc), ref in PAPER_T4.items():
        b, p = R.scenario(m, sc, 'full'), R.scenario(m, sc, 'consistent')
        vals = [b['S_max'], p['S_max'], b['E_port'], p['E_port'], b['clearance_mm'], p['clearance_mm'], b['rms_mrad'], p['rms_mrad']]
        rows_csv.append([m, sc] + vals + list(ref) + [b['violated_pairs'] or 'none', p['violated_pairs'] or 'none'])
        rows_md.append([m, {'efence': 'e-fence', 'self': 'self-coll.'}[sc]] + [cell(v, q) for v, q in zip(vals, ref)])
    head = ('S_max base (J)', 'S_max prop. (J)', 'port base (J)', 'port prop. (J)', 'clear. base (mm)', 'clear. prop. (mm)', 'RMS base (mrad)',
            'RMS prop. (mrad)')
    write_csv(out / 'tables' / 'table4.csv', ['machine', 'scenario', 'S_max_base', 'S_max_prop', 'E_port_base', 'E_port_prop', 'clearance_mm_base',
                                              'clearance_mm_prop', 'rms_mrad_base', 'rms_mrad_prop'] + [f'paper_{c}' for c in range(8)]
              + ['violated_base', 'violated_prop'], rows_csv, 'Table IV: further scenarios whose references violate the limits')
    return md_table(('machine', 'scenario') + head, rows_md)


def table5(R, out):
    cols = [(m, va) for m in ('komatsu', 'heap') for va in ('full', 'consistent')]
    data = {c: R.mission(*c) for c in cols}
    for c in cols:
        data[c]['violated'] = ', '.join(data[c]['violated_pairs']) or 'none'
    rows_csv = [[h] + [data[c][k] for c in cols] + [PAPER_T5[c][i] for c in cols] for i, (k, h) in enumerate(zip(T5_COLS, T5_HEAD))]
    write_csv(out / 'tables' / 'table5.csv', ['metric'] + [f'{m}_{va}' for m, va in cols] + [f'paper_{m}_{va}' for m, va in cols], rows_csv,
              'Table V: dig-and-level missions (soil model, digging, loading, levelling); full = baseline, consistent = proposed')
    rows_md = [[h] + [cell(data[c][k], PAPER_T5[c][i]) for c in cols] for i, (k, h) in enumerate(zip(T5_COLS, T5_HEAD))]
    return md_table(('metric', 'Komatsu baseline', 'Komatsu proposed', 'HEAP baseline', 'HEAP proposed'), rows_md)


# ------------------------------------------------------------------------------------------------ numbers of the text
def onset_ratios(folder, machine):
    """Corollary 1 at the six activations of the stroke row in the first layer (baseline): (|J s(t2)| + |J s(t_a)|/e) / (zs(t_a)/e),
    t2 = first control sample of a cycle at which the first-layer row of the boom pair is active (the flag chatters at
    round-off level while the row is held, so only the first activation of every cycle counts), t_a = t2 - 1/alpha2
    (linear interpolation between the samples)."""
    rep, tr = Mx.load(folder)
    ids = [str(x) for x in tr['pair_ids']]
    j = min(ids.index(p) for p in Mx.CYCLE_PAIR[machine] if p in ids)
    t = np.asarray(tr['c_time'], float)
    act = np.asarray(tr['c_pair_velocity_active'])[:, j] > 0
    zs, w = np.asarray(tr['c_pair_zs'])[:, j], np.asarray(tr['c_pair_w'])[:, j]
    alpha2 = 20.                                              # stroke rows: (alpha1, alpha2) = (8, 20)
    names = rep['phase_names']
    ph = np.asarray(tr['c_phase'], int) if 'c_phase' in tr else Mx.phase_index(rep, t)
    out = []
    for k, n in enumerate(names):
        if not n.endswith(': go'):
            continue
        cyc = n.split(':')[0]
        idx = np.flatnonzero(act & np.isin(ph, [i for i, m in enumerate(names) if m.startswith(cyc + ':')]))
        if not len(idx):
            continue
        i2 = int(idx[0])
        ta = t[i2] - 1. / alpha2
        z_a, w_a = float(np.interp(ta, t, zs)), float(np.interp(ta, t, w))
        out.append(dict(cycle=cyc, t2=float(t[i2]), zs_ta=z_a, w_t2=float(w[i2]), w_ta=w_a, ratio=(abs(float(w[i2])) + abs(w_a) / E) / (z_a / E)))
    tr.close()
    return out


def numbers(R, out):
    N = {}
    c = lambda m, va: R.cycles(m, va)
    for m in ('komatsu', 'heap'):
        rat = onset_ratios(R.folder(m, 'cycles__full'), m)
        N[f'{m}_onset_ratio_range'] = [min(r['ratio'] for r in rat), max(r['ratio'] for r in rat)]
        N[f'{m}_onset_activations'] = len(rat)
    N['first_layer_only_onset'] = [c('komatsu', 'kinematic_only')['onset'], c('heap', 'kinematic_only')['onset']]
    N['baseline_onset'] = [c('komatsu', 'full')['onset'], c('heap', 'full')['onset']]
    N['onset_reduction_factor_consistent'] = [c(m, 'full')['onset'] / c(m, 'consistent')['onset'] for m in ('komatsu', 'heap')]
    N['port_energy_baseline_to_consistent'] = [[c(m, 'full')['E_port'], c(m, 'consistent')['E_port']] for m in ('komatsu', 'heap')]
    N['stroke_clearance_consistent_mm'] = [c(m, 'consistent')['clearance_mm'] for m in ('komatsu', 'heap')]
    N['stroke_clearance_baseline_mm'] = [c(m, 'full')['clearance_mm'] for m in ('komatsu', 'heap')]
    N['storage_peaks_beta40_consistent_baseline'] = [[c(m, va)['S_max'] for m in ('komatsu', 'heap')] for va in ('accel_b40', 'consistent', 'full')]
    N['onset_factor_beta40_to_16'] = [c(m, 'accel_b40')['onset'] / c(m, 'consistent')['onset'] for m in ('komatsu', 'heap')]
    N['port_factor_beta40_to_16'] = [c(m, 'accel_b40')['E_port'] / max(c(m, 'consistent')['E_port'], 1e-300) for m in ('komatsu', 'heap')]
    N['raw_feedforward_komatsu'] = dict(S_max=c('komatsu', 'full_ffraw')['S_max'], onset=c('komatsu', 'full_ffraw')['onset'],
                                        port=c('komatsu', 'full_ffraw')['E_port'])
    N['conditioned_storage_peaks'] = [c(m, 'conditioned')['S_max'] for m in ('komatsu', 'heap')]
    N['conditioned_energies'] = {m: {k: c(m, 'conditioned')[k] for k in ('E_port', 'E_feedforward', 'E_force_loops', 'E_other')} for m in ('komatsu', 'heap')}
    gain = {k: (c('komatsu', 'full' if k == 12 else f'full_k{k}'), c('komatsu', 'consistent' if k == 12 else f'consistent_k{k}')) for k in GAINS}
    N['baseline_onset_change_k2_to_k24_percent'] = 100. * (gain[24][0]['onset'] / gain[2][0]['onset'] - 1.)
    N['baseline_port_change_k2_to_k24_percent'] = 100. * (gain[24][0]['E_port'] / gain[2][0]['E_port'] - 1.)
    N['consistent_onset_reduction_factor_over_sweep'] = [min(b['onset'] / p['onset'] for b, p in gain.values()),
                                                         max(b['onset'] / p['onset'] for b, p in gain.values())]
    N['consistent_port_energy_by_gain'] = {str(k): v[1]['E_port'] for k, v in gain.items()}
    N['baseline_port_energy_by_gain'] = {str(k): v[0]['E_port'] for k, v in gain.items()}
    N['payload_onset_consistent_vs_baseline'] = [c('komatsu', 'consistent_bucket150')['onset'], c('komatsu', 'full_bucket150')['onset']]
    N['payload_port_consistent_vs_baseline'] = [c('komatsu', 'consistent_bucket150')['E_port'], c('komatsu', 'full_bucket150')['E_port']]
    N['effort_layer_alone'] = {m: dict(S_max=c(m, 'force_only')['S_max'], max_s=max(c(m, 'force_only')['onset'], c(m, 'force_only')['release']),
                                      onset=c(m, 'force_only')['onset'], release=c(m, 'force_only')['release'],
                                      clearance_mm=c(m, 'force_only')['clearance_mm'], violated=c(m, 'force_only')['violated_pairs'])
                               for m in ('komatsu', 'heap')}
    N['first_layer_alone_k2_clearance_mm'] = c('komatsu', 'kinematic_only_k2')['clearance_mm']
    N['two_layer_clearance_min_mm_over_gain_and_payload'] = min(
        c('komatsu', v)['clearance_mm'] for v in ('full', 'consistent', 'full_k2', 'full_k4', 'full_k6', 'full_k24', 'consistent_k2', 'consistent_k4',
                                                  'consistent_k6', 'consistent_k24', 'consistent_b8', 'full_bucket150', 'consistent_bucket150'))
    N['binding_percent_consistent'] = [c(m, 'consistent')['effort_percent'] for m in ('komatsu', 'heap')]
    N['binding_percent_baseline'] = [c(m, 'full')['effort_percent'] for m in ('komatsu', 'heap')]
    N['mean_computation_ms_baseline_consistent'] = {m: [c(m, 'full')['mean_wall_ms'], c(m, 'consistent')['mean_wall_ms']] for m in ('komatsu', 'heap')}
    pf, sf, dc, rm = [], [], [], []
    for m in ('komatsu', 'heap'):
        for sc in ('efence', 'self'):
            b, p = R.scenario(m, sc, 'full'), R.scenario(m, sc, 'consistent')
            pf.append(b['E_port'] / p['E_port'])
            sf.append(b['S_max'] / p['S_max'])
            dc.append(abs(b['clearance_mm'] - p['clearance_mm']))
            rm.append(p['rms_mrad'] <= b['rms_mrad'] + 1e-9)
    N['workspace_port_reduction_factors'] = [min(pf), max(pf)]
    N['workspace_storage_reduction_factors'] = [min(sf), max(sf)]
    N['workspace_clearance_difference_max_mm'] = max(dc)
    N['workspace_rms_proposed_not_larger'] = all(rm)
    mi = {(m, va): R.mission(m, va) for m in ('komatsu', 'heap') for va in ('full', 'consistent')}
    N['mission_durations_s'] = {f'{m}_{va}': v['duration_s'] for (m, va), v in mi.items()}
    N['mission_min_clearance_mm'] = {f'{m}_{va}': v['clearance_mm'] for (m, va), v in mi.items()}
    N['mission_storage_reduction_factors'] = [mi[(m, 'full')]['S_max'] / mi[(m, 'consistent')]['S_max'] for m in ('komatsu', 'heap')]
    N['mission_port_reduction_factors'] = [mi[(m, 'full')]['E_port'] / mi[(m, 'consistent')]['E_port'] for m in ('komatsu', 'heap')]
    N['mission_excavated_volume_difference_percent'] = [100. * abs(mi[(m, 'consistent')]['excavated_m3'] / mi[(m, 'full')]['excavated_m3'] - 1.)
                                                        for m in ('komatsu', 'heap')]
    N['mission_status'] = {f'{m}_{va}': v['status'] for (m, va), v in mi.items()}
    retained = ([('komatsu', 'cycles', va) for va in ('full', 'consistent')] + [('heap', 'cycles', va) for va in ('full', 'consistent')]
                + [(m, sc, va) for m in ('komatsu', 'heap') for sc in ('efence', 'self') for va in ('full', 'consistent')])
    inf = sum(R.cycles(m, va)['infeasible_samples'] if sc == 'cycles' else R.scenario(m, sc, va)['infeasible_samples'] for m, sc, va in retained)
    inf += sum(v['infeasible_samples'] for v in mi.values())
    N['qp_infeasible_samples_in_retained_runs'] = int(inf)
    paper = {
        'komatsu_onset_ratio_range': [1.061, 1.064], 'heap_onset_ratio_range': [1.051, 1.145], 'first_layer_only_onset': [28.4, 28.5],
        'baseline_onset': [15.9, 28.5], 'onset_reduction_factor_consistent': [3.3, 3.6], 'port_energy_baseline_to_consistent': [[1835, 0.0063], [490, 4.73]],
        'stroke_clearance_consistent_mm': [2.8, 20.0], 'storage_peaks_beta40_consistent_baseline': [[36.8, 63.4], [35.3, 54.8], [401, 405]],
        'onset_factor_beta40_to_16': [1.8, 1.4], 'port_factor_beta40_to_16': [5.4e4, 28], 'raw_feedforward_komatsu': dict(S_max=173, onset=12.7, port=1006),
        'conditioned_storage_peaks': [401, 447], 'baseline_onset_change_k2_to_k24_percent': -33, 'baseline_port_change_k2_to_k24_percent': 59,
        'consistent_onset_reduction_factor_over_sweep': [3.2, 3.5], 'consistent_port_energy_by_gain': {'2': 5.03, '4': '<0.01', '6': '<0.01', '12': '<0.01', '24': '<0.01'},
        'baseline_port_energy_by_gain': {'2': 1465}, 'payload_onset_consistent_vs_baseline': [6.59, 17.7], 'payload_port_consistent_vs_baseline': [0.711, 2029],
        'effort_layer_alone': 'peaks 138 and 196 sqrt(J); limits maintained', 'first_layer_alone_k2_clearance_mm': -0.4,
        'binding_percent_consistent': [34.3, 34.5], 'binding_percent_baseline': [10.2, 12.3],
        'mean_computation_ms_baseline_consistent': dict(komatsu=[11.8, 11.7], heap=[21.8, 22.0]),
        'workspace_port_reduction_factors': [14, 190], 'workspace_storage_reduction_factors': [2.1, 7.9], 'workspace_clearance_difference_max_mm': 0.7,
        'workspace_rms_proposed_not_larger': True, 'mission_durations_s': dict(komatsu=397, heap=406), 'mission_min_clearance_mm': dict(komatsu=35.3, heap=32.0),
        'mission_storage_reduction_factors': [1.4, 1.6], 'mission_port_reduction_factors': [5.9, 9.1],
        'mission_excavated_volume_difference_percent': [0.25, 1.2], 'qp_infeasible_samples_in_retained_runs': 0}
    (out / 'numbers').mkdir(parents=True, exist_ok=True)
    json.dump(dict(regenerated=N, paper=paper), open(out / 'numbers' / 'section_vii_b.json', 'w'), indent=1, default=float)
    lines = ['# Numbers of Section VII-B: regenerated and printed in the paper', '',
             'Regenerated values with four significant digits (full precision in section_vii_b.json).  Computation times are wall-clock '
             'times of the Python controller on the machine that ran the batch; they depend on the hardware.', '',
             '| quantity | regenerated | paper |', '|---|---|---|']

    def show(v):
        if isinstance(v, dict):
            return '; '.join(f'{k}: {show(x)}' for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return '[' + ', '.join(show(x) for x in v) + ']'
        if isinstance(v, bool) or not isinstance(v, (float, int, np.floating, np.integer)):
            return str(v)
        v = float(v)
        if v == 0. or not np.isfinite(v):
            return f'{v:g}'
        return f'{v:.0f}' if abs(v) >= 1e4 else f'{v:.4g}'
    for k, v in N.items():
        lines.append(f'| {k} | {show(v)} | {show(paper.get(k, ""))} |')
    (out / 'numbers' / 'section_vii_b.md').write_text('\n'.join(lines) + '\n')
    return N


# ------------------------------------------------------------------------------------------------ figures
def figures(R, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'lines.linewidth': 1.1, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
    COL = dict(kinematic_only='#1b9e77', full='#e66101', accel_b40='#7b3294', consistent='#2166ac')
    LAB = dict(kinematic_only='First layer only', full='Baseline', accel_b40=r'$\bar\beta=40$', consistent='Consistent')
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    # Fig. 5: |s|_M over the six approaches, baseline and consistent
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.9), gridspec_kw=dict(hspace=0.65))
    for ax, m, title in zip(axs, ('komatsu', 'heap'), ('(a) Komatsu', '(b) HEAP')):
        series = {}
        for va in ('full', 'consistent'):
            rep, tr = Mx.load(R.folder(m, f'cycles__{va}'))
            series[va] = (np.asarray(tr['c_time'], float), Mx.snorm(tr))
            tr.close()
            ax.plot(*series[va], color=COL[va], label=LAB[va])
        t = series['full'][0]
        assert np.allclose(series['consistent'][0][:len(t)], t[:len(series['consistent'][0])])
        n = min(len(series['full'][0]), len(series['consistent'][0]))
        write_csv(out / 'csv' / f'fig5_{m}.csv', ['t_s', 'baseline_s_M', 'consistent_s_M'],
                  zip(t[:n], series['full'][1][:n], series['consistent'][1][:n]),
                  f'Fig. 5 ({m}): kinetic velocity error |s|_M = sqrt(2 S) (sqrt(J)) at the 10 ms control samples, scenario cycles')
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\|s\|_M$ ($\sqrt{\mathrm{J}}$)')
        ax.set_title(title, loc='left')
        if m == 'komatsu':
            ax.legend(ncol=2, loc='lower right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
    fig.savefig(out / 'figures' / 'fig5.pdf')
    plt.close(fig)
    # Fig. 6: third cycle, four variants: |s|_M, safety-port power, stroke clearance
    fig, axs = plt.subplots(3, 2, figsize=(7.16, 3.6), sharex='col', gridspec_kw=dict(hspace=0.15, wspace=0.18, height_ratios=[1.1, 0.9, 1.0]))
    for col, m, title in ((0, 'komatsu', '(a) Komatsu'), (1, 'heap', '(b) HEAP')):
        rows_ctl, rows_clr, base_active = {}, {}, None
        for va in ('kinematic_only', 'full', 'accel_b40', 'consistent'):
            rep, tr = Mx.load(R.folder(m, f'cycles__{va}'))
            names = rep['phase_names']
            starts = np.asarray(rep['phase_starts'], float)
            j0, j1 = names.index('cycle 3: go'), names.index('cycle 3: back')
            t0, t1 = starts[j0], starts[j1 + 1]
            ids = [str(x) for x in tr['pair_ids']]
            cols = [ids.index(p) for p in Mx.CYCLE_PAIR[m] if p in ids]
            t = np.asarray(tr['c_time'], float)
            sel = (t >= t0) & (t <= t1)
            ts = t[sel] - t0
            sn, pf = Mx.snorm(tr)[sel], np.asarray(tr['c_P_cmd'], float)[sel]
            bind = np.any(np.asarray(tr['c_pair_active'])[sel][:, cols] > 0, axis=1)
            vel = np.any(np.asarray(tr['c_pair_velocity_active'])[sel][:, cols] > 0, axis=1)
            tp = np.asarray(tr['time'], float)
            sp = (tp >= t0) & (tp <= t1)
            clr = 1e3 * np.min(np.asarray(tr['pair_h'])[sp][:, cols], axis=1)
            rows_ctl[va] = (ts, sn, pf, bind, vel)
            rows_clr[va] = (tp[sp] - t0, clr)
            if va == 'full':
                base_active = (ts, vel)
            tr.close()
            axs[0, col].plot(ts, sn, color=COL[va], ls='--' if va == 'accel_b40' else '-', label=LAB[va])
            axs[1, col].plot(ts, 1e-3 * pf, color=COL[va], ls='--' if va == 'accel_b40' else '-')
            axs[2, col].plot(tp[sp] - t0, clr, color=COL[va], ls='--' if va == 'accel_b40' else '-')
        ts, vel = base_active
        for ax in axs[:, col]:
            edges = np.flatnonzero(np.diff(np.r_[0, vel.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(ts[a], ts[min(b, len(ts) - 1)], color='#e8e8e8', zorder=0)
        ylo = 70.
        for k, va in enumerate(('full', 'consistent')):
            ts_, _, _, bind, _ = rows_ctl[va]
            edges = np.flatnonzero(np.diff(np.r_[0, bind.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                axs[2, col].plot([ts_[a], ts_[min(b, len(ts_) - 1)]], [-0.06 * ylo * (k + 1)] * 2, color=COL[va], lw=2.2, solid_capstyle='butt')
        axs[2, col].axhline(0., color='#555555', lw=0.7)
        axs[2, col].set_ylim(-11., 65.)
        axs[2, col].set_xlim(0.3, rows_ctl['full'][0][-1] - 0.5)
        axs[0, col].set_title(title, loc='left')
        axs[2, col].set_xlabel('Time from approach start (s)')
        # data
        va0 = 'full'
        n = min(len(rows_ctl[v][0]) for v in rows_ctl)
        header = ['t_from_approach_start_s'] + [f'{v}_{q}' for v in rows_ctl for q in ('s_M', 'P_f_W', 'effort_row_binding', 'first_layer_row_active')]
        data = [[rows_ctl[va0][0][i]] + [x for v in rows_ctl for x in (rows_ctl[v][1][i], rows_ctl[v][2][i], int(rows_ctl[v][3][i]), int(rows_ctl[v][4][i]))]
                for i in range(n)]
        write_csv(out / 'csv' / f'fig6_{m}.csv', header, data,
                  f'Fig. 6 ({m}): third cycle of the scenario cycles, control samples (10 ms): |s|_M (sqrt(J)), safety-port power P_f (W), '
                  'binding effort-barrier row and active first-layer row of the boom stroke pair (0/1)')
        nc = min(len(rows_clr[v][0]) for v in rows_clr)
        write_csv(out / 'csv' / f'fig6_{m}_clearance.csv', ['t_from_approach_start_s'] + [f'{v}_clearance_mm' for v in rows_clr],
                  [[rows_clr[va0][0][i]] + [rows_clr[v][1][i] for v in rows_clr] for i in range(nc)],
                  f'Fig. 6 ({m}): plant-side clearance of the boom stroke limit (truth monitor, 40 ms samples), mm')
    axs[0, 0].set_ylabel(r'$\|s\|_M$ ($\sqrt{\mathrm{J}}$)')
    axs[1, 0].set_ylabel(r'$P_f$ (kW)')
    axs[2, 0].set_ylabel('Stroke clearance\n(mm)')
    axs[0, 0].legend(ncol=4, loc='upper center', bbox_to_anchor=(1.05, 1.42))
    fig.savefig(out / 'figures' / 'fig6.pdf')
    plt.close(fig)
    # Fig. 7: Komatsu gain sweep
    on_b = [R.cycles('komatsu', 'full' if k == 12 else f'full_k{k}')['onset'] for k in GAINS]
    on_c = [R.cycles('komatsu', 'consistent' if k == 12 else f'consistent_k{k}')['onset'] for k in GAINS]
    pe_b = [R.cycles('komatsu', 'full' if k == 12 else f'full_k{k}')['E_port'] for k in GAINS]
    pe_c = [R.cycles('komatsu', 'consistent' if k == 12 else f'consistent_k{k}')['E_port'] for k in GAINS]
    write_csv(out / 'csv' / 'fig7_gain.csv', ['k_per_s', 'baseline_onset', 'consistent_onset', 'baseline_port_J', 'consistent_port_J'],
              zip(GAINS, on_b, on_c, pe_b, pe_c), 'Fig. 7: Komatsu gain sweep (scenario cycles): onset |s|_M (sqrt(J)) and positive port energy (J)')
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7), gridspec_kw=dict(wspace=0.55))
    axs[0].plot(GAINS, on_b, color=COL['full'], marker='o', ms=3.2, label='Baseline')
    axs[0].plot(GAINS, on_c, color=COL['consistent'], marker='o', ms=3.2, label='Consistent')
    axs[0].set_ylim(bottom=0)
    axs[0].set_ylabel(r'Onset $\|s\|_M$ ($\sqrt{\mathrm{J}}$)')
    axs[1].semilogy(GAINS, pe_b, color=COL['full'], marker='o', ms=3.2)
    axs[1].semilogy(GAINS, np.maximum(pe_c, 1e-12), color=COL['consistent'], marker='o', ms=3.2)
    axs[1].set_ylabel('Positive port energy (J)')
    for ax in axs:
        ax.set_xticks([2, 6, 12, 24])
        ax.set_xlabel(r'$k$ (s$^{-1}$)')
    axs[0].legend(ncol=2, loc='upper center', bbox_to_anchor=(1.3, 1.32))
    fig.savefig(out / 'figures' / 'fig7.pdf')
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ logged results of every run
def export_runs(R, out):
    """report.json, the run log and a CSV of the control samples of every run of the paper (sanitised: no local paths)."""
    import batch
    rows = []
    for m in ('komatsu', 'heap'):
        for g in batch.PAPER_GROUPS:
            for name, report, argv, _ in batch.jobs_of(m, g):
                folder = Path(report).parent
                dst = out / 'runs' / m / name
                dst.mkdir(parents=True, exist_ok=True)
                rep = json.loads((folder / 'report.json').read_text())
                (dst / 'report.json').write_text(json.dumps(rep, indent=1))
                log = R.results / m / 'logs' / f'{name}.log'
                if log.exists():
                    text = log.read_text()
                    text = text.replace(str(ROOT), '<RCX_TAC>').replace(str(ROOT.parents[2]), '<supplement>')
                    (dst / 'run.log').write_text(text)
                rep_, tr = Mx.load(folder)
                t = np.asarray(tr['c_time'], float)
                ph = Mx.phase_index(rep_, t) if rep_.get('phase_starts') else np.zeros(len(t), int)
                ids = [str(x) for x in tr['pair_ids']]
                cols = {'t_s': t, 'phase': ph, 'storage_J': tr['c_storage'], 's_M': Mx.snorm(tr), 'P_port_W': tr['c_P_cmd'],
                        'P_feedforward_W': tr['c_P_acc'], 'P_force_loops_W': np.asarray(tr['c_P_mis']) - np.asarray(tr['c_P_cmd']),
                        'P_load_W': tr['c_P_load'], 'P_zoh_W': tr['c_P_zoh'], 'dissipation_W': tr['c_dissipation'],
                        'effort_rows_binding': tr['c_n_active'], 'qp_feasible': np.asarray(tr['c_feasible']).astype(int),
                        'qp_stationarity_rel': tr['c_qp_stationarity'] if 'c_qp_stationarity' in tr else np.full(len(t), np.nan),
                        'controller_wall_ms': 1e3 * np.asarray(tr['c_wall'], float)}
                focus = rep_.get('scenario')
                pairs = (Mx.CYCLE_PAIR[m] if focus == 'cycles' else Mx.SCENARIO_PAIRS[m].get(focus, [])) if focus else []
                for p in pairs:
                    if p in ids:
                        k = ids.index(p)
                        cols[f'h_{p}'] = np.asarray(tr['c_pair_h'])[:, k]
                        cols[f'effort_active_{p}'] = np.asarray(tr['c_pair_active'])[:, k].astype(int)
                        cols[f'first_layer_active_{p}'] = np.asarray(tr['c_pair_velocity_active'])[:, k].astype(int)
                        if focus == 'cycles':                   # first-layer slack and J s: onset quantity of Section VII-B
                            cols[f'zs_{p}'] = np.asarray(tr['c_pair_zs'])[:, k]
                            cols[f'w_{p}'] = np.asarray(tr['c_pair_w'])[:, k]
                tr.close()
                n = min(len(v) for v in cols.values())
                write_csv(out / 'csv' / 'runs' / m / f'{name}.csv', list(cols), zip(*[np.asarray(v)[:n].tolist() for v in cols.values()]),
                          f'{m} {name}: control samples (Ts = {rep["sample_time_s"]} s); phase = index into report.json phase_names; '
                          'h_<pair>: controller-side barrier value of the pair (m or rad); zs_<pair>, w_<pair>: slack of its first-order '
                          'constraint at the required velocity and J s (for the row of the pair with the smallest slack)', digits=7)
                rows.append([m, name, rep['status'], rep['scenario'] if 'scenario' in rep else 'mission', rep['variant'], rep['code_sha256'],
                             f"{rep['duration_s']:.2f}", len(rep['violated_pairs']), rep['filter']['infeasible_steps']])
    write_csv(out / 'runs' / 'index.csv', ['machine', 'run', 'status', 'scenario', 'variant', 'code_sha256', 'duration_s', 'violated_pairs',
                                           'infeasible_qp_samples'], rows, 'Runs of the paper (batch.py all paper)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(ROOT.parents[2] / 'results'))
    p.add_argument('--results', default=str(ROOT / 'results'), help='results folder of the runs (batch.py)')
    p.add_argument('--allow-other-sources', action='store_true')
    p.add_argument('--no-export', action='store_true', help='do not copy the logged results of the runs')
    p.add_argument('--from-logs', default=None, metavar='RESULTS',
                   help='recompute Tables II-V and the numbers of Section VII-B from the exported logs in RESULTS (the results folder of '
                        'the supplement: runs/ and csv/runs/) instead of the raw run folders; writes into --out (figures need the raw traces)')
    a = p.parse_args()
    out = Path(a.out)
    if a.from_logs:
        R = Runs(Path(a.from_logs), a.allow_other_sources, from_logs=True)
        if out.resolve() == Path(a.from_logs).resolve():
            raise SystemExit('--from-logs: choose an --out folder other than the logs folder')
    else:
        R = Runs(Path(a.results), a.allow_other_sources)
    parts = ['# Tables II-V: regenerated values (values printed in the paper in parentheses; bold: differs at the printed precision)', '',
             'The reference values are those of the revised manuscript; it differs from the first submission in three port energies of '
             'Table IV (see the comment above PAPER_T4 in paper/tables.py).', '']
    for title, fn in (('Table II', table2), ('Table III', table3), ('Table IV', table4), ('Table V', table5)):
        parts += [f'## {title}', '', fn(R, out), '']
    if R.mismatch:
        parts.insert(1, f'WARNING: reports of other sources: {sorted(set(R.mismatch))}\n')
    if R.unchecked:
        parts.insert(1, f'NOTE: machine packages not present ({", ".join(R.unchecked)}): their logged reports were used '
                        f'without checking the hash of the sources they ran with.\n')
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'tables' / 'tables_ii_v.md').write_text('\n'.join(parts))
    numbers(R, out)
    if not a.from_logs:
        figures(R, out)
        if not a.no_export:
            export_runs(R, out)
    print('wrote', out / 'tables' / 'tables_ii_v.md', out / 'numbers' / 'section_vii_b.md', '' if a.from_logs else 'and figures 5-7')


if __name__ == '__main__':
    main()
