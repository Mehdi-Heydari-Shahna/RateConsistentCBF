"""Table I and the numbers of Section VII-A, Remarks 4-5 and Section VIII of the paper, collected from the results of the
continuous-time scripts, with the values printed in the paper next to them.

    python paper_numbers.py [--out <folder>]          (default: ../../results)

Reads (all written by the scripts of reproduce.sh):
    num/t1_results.json, num/t2_komatsu_results.json, num/t2_heap_results.json, num/t7_results.json, num/t9_results.json   Table I
    num/t14_results.json, num/t15_results.json, onedof_sweep/sweep_alpha2.json                                             Section VII-A
    checks/arm_onset_corollary.json, checks/rho_bar_arm_narrow.json, checks/rho_bar_arm_model.json, checks/ex_metric.json
    num/t13_results.json                                                                                                   QP solver check
Writes  <out>/tables/table1.csv, table1.md  and  <out>/numbers/section_vii_a.json, section_vii_a.md.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
NUM, SWEEP, CHECKS = HERE / 'num', HERE / 'onedof_sweep', HERE / 'checks'


def load(path):
    return json.loads(Path(path).read_text())


def g(x):
    return f'{x:.3g}' if isinstance(x, float) else str(x)


def table1(out):
    t1, t7, t9 = load(NUM / 't1_results.json'), load(NUM / 't7_results.json'), load(NUM / 't9_results.json')
    tk, th = load(NUM / 't2_komatsu_results.json'), load(NUM / 't2_heap_results.json')
    p1 = t9['part1']
    rows = [('(2), 6 and 4 states', f"{tk['states']} and {th['states']} states", tk['largest_relative_errors']['a'], th['largest_relative_errors']['a'], '3.36e-11', '1.24e-10'),
            ('Storage balance', '', tk['largest_relative_errors']['b'], th['largest_relative_errors']['b'], '6.68e-10', '2.67e-9'),
            ('Constraint identity, 544 and 715 rows', f"{tk['rows']} and {th['rows']} rows", tk['largest_relative_errors']['c'], th['largest_relative_errors']['c'],
             '6.44e-11', '1.75e-10')]
    ineq = [('Port (a): 29 467 instances, six rates', f"{t1['counted_instances']} instances, {t1['rates_per_instance']} rates", t1['port_a_largest_relative_excess'], '4.1e-15'),
            ('Port (b): 18 917 dependent-gradient instances', f"{t7['counts']['inst']} instances", t7['port_b_largest_relative_excess'], '3.5e-12'),
            ('Storage: 20 000 instances, three rates', f"{p1['counts']['inst']} instances, {p1['counts']['evals']} evaluations", p1['worst']['M'], '6.8e-13')]
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    with open(out / 'tables' / 'table1.csv', 'w', newline='') as f:
        f.write('# Table I: independent numerical checks of the analytical identities and inequalities; regenerated and printed values\n')
        w = csv.writer(f)
        w.writerow(['check', 'regenerated_counts', 'komatsu_or_max_excess', 'heap', 'paper_komatsu_or_max_excess', 'paper_heap'])
        for r in rows:
            w.writerow([r[0], r[1], f'{r[2]:.3g}', f'{r[3]:.3g}', r[4], r[5]])
        for r in ineq:
            w.writerow([r[0], r[1], f'{r[2]:.3g}', '', r[3], ''])
    md = ['| identity | counts | Komatsu | HEAP |', '|---|---|---|---|']
    md += [f'| {r[0]} | {r[1]} | {r[2]:.3g} ({r[4]}) | {r[3]:.3g} ({r[5]}) |' for r in rows]
    md += ['', '| inequality | counts | max. excess |', '|---|---|---|']
    md += [f'| {r[0]} | {r[1]} | {r[2]:.3g} ({r[3]}) |' for r in ineq]
    (out / 'tables' / 'table1.md').write_text('# Table I: regenerated values (values printed in the paper in parentheses)\n\n' + '\n'.join(md) + '\n')


def section_vii_a(out):
    t14, t15, sw = load(NUM / 't14_results.json'), load(NUM / 't15_results.json'), load(SWEEP / 'sweep_alpha2.json')
    cor = load(CHECKS / 'arm_onset_corollary.json')
    rn, rm = load(CHECKS / 'rho_bar_arm_narrow.json'), load(CHECKS / 'rho_bar_arm_model.json')
    ex, t13 = load(CHECKS / 'ex_metric.json'), load(NUM / 't13_results.json')
    fo = {int(r['k']): r for r in t14['first_order']}
    co = t14['consistent']
    plane = {k: [o for o in r['onsets'] if o['row'] == 'plane'] for k, r in fo.items()}
    hi3 = {k: [o for o in r['onsets'] if o['row'] == 'hi3'] for k, r in fo.items()}
    cplane = [c for c in cor if c['row'] == 'plane']
    N = {}
    P = {}
    N['arm_first_order_peak_s_M'] = [max(fo[12]['sup_per_period']), max(fo[30]['sup_per_period'])]
    P['arm_first_order_peak_s_M'] = [0.2350, 0.2345]
    N['arm_floor_effort_binding_time_s'] = sorted({round(o['binding_time'], 4) for k in (12, 30) for o in plane[k]})
    P['arm_floor_effort_binding_time_s'] = 0.061
    N['arm_corollary1_lhs_floor_k12_k30'] = [max(c['lhs'] for c in cplane if c['k'] == 12.), max(c['lhs'] for c in cplane if c['k'] == 30.)]
    N['arm_corollary1_bound_floor'] = sorted({round(c['bound'], 4) for c in cplane})
    N['arm_corollary1_ratio_floor'] = [min(c['ratio'] for c in cplane), max(c['ratio'] for c in cplane)]
    P['arm_corollary1_lhs_floor_k12_k30'] = [0.1598, 0.1595]
    P['arm_corollary1_bound_floor'] = 0.1595
    P['arm_corollary1_ratio_floor'] = 'within 0.3 %'
    N['arm_elbow_integral_bound_ratio'] = [min(o['psi1_t2'] / o['lower'] for k in (12, 30) for o in hi3[k]),
                                           max(o['psi1_t2'] / o['lower'] for k in (12, 30) for o in hi3[k])]
    P['arm_elbow_integral_bound_ratio'] = 'attained within 0.8 %'
    N['arm_consistent_sup_s_M'] = [max(co['a_k12']['sup_per_period']), max(co['a_k30']['sup_per_period'])]
    P['arm_consistent_sup_s_M'] = 0.
    N['arm_decay_fitted_rate'] = [co['b_k12_beta16']['fitted_rate'], co['b_k30_beta16']['fitted_rate']]
    P['arm_decay_fitted_rate'] = [12.25, 29.9]
    N['arm_decay_theorem_rate_min'] = [co['b_k12_beta16']['rate_min'], co['b_k30_beta16']['rate_min']]
    P['arm_decay_theorem_rate_min'] = [11.99, 15.76]
    N['arm_curvature_rate_max'] = max(co['b_k12_beta16']['rho_max'], co['b_k30_beta16']['rho_max'])
    P['arm_curvature_rate_max'] = 'at most 0.24'
    N['arm_decay_initial_s_M'] = [co['b_k12_beta16']['s0'], co['b_k30_beta16']['s0']]
    P['arm_decay_initial_s_M'] = [8.226, 5.236]
    N['arm_mismatch_period_peaks'] = co['c_k12']['sup_per_period']
    P['arm_mismatch_period_peaks'] = 'between 0.049 and 0.055'
    N['arm_events_located_consistent_runs'] = {k: v.get('events') for k, v in co.items()}
    N['arm_task_error_mean_last_period'] = dict(baseline_k12=fo[12]['e_mean_last_period'], consistent_k12=co['a_k12']['e_mean_last_period'])
    P['arm_task_error_mean_last_period'] = dict(baseline_k12=0.0434, consistent_k12=0.0435)
    s16 = {int(r['k']): r for r in t15['first_order'] if r['alpha2'] == 16.}
    N['scalar_first_order_period_peaks_3_8'] = [min(min(r['sup_per_period'][2:]) for r in s16.values()), max(max(r['sup_per_period'][2:]) for r in s16.values())]
    P['scalar_first_order_period_peaks_3_8'] = [0.0868, 0.0869]
    N['scalar_corollary1_ratio_periods_3_8'] = [min(r['corollary_ratio_periods_3_8'][0] for r in s16.values()),
                                                max(r['corollary_ratio_periods_3_8'][1] for r in s16.values())]
    P['scalar_corollary1_ratio_periods_3_8'] = 'attained within 0.5 %'
    N['scalar_alpha2_peaks'] = {f"{r['alpha2']:g}": r['peak'] for r in sw}
    P['scalar_alpha2_peaks'] = {'8': 0.163, '16': 0.0868, '32': 0.0446, '64': 0.0226}
    N['scalar_alpha2_times_peak'] = [min(r['alpha2_times_peak'] for r in sw), max(r['alpha2_times_peak'] for r in sw)]
    P['scalar_alpha2_times_peak'] = [1.31, 1.45]
    so = {(int(r['k']), r['beta']): r for r in t15['second_order']}
    N['scalar_consistent_beta16_sup_s'] = [max(so[(k, 16.)]['sup_per_period']) for k in (12, 30, 100)]
    P['scalar_consistent_beta16_sup_s'] = 0.
    N['scalar_beta40_recurring_error'] = max(so[(12, 40.)]['sup_per_period'][2:])
    P['scalar_beta40_recurring_error'] = 0.054
    N['remark4_S_dot_over_s_M2'] = dict(euclidean=ex['remark4']['euclidean']['S_dot_over_s_M2'], kinetic=ex['remark4']['kinetic']['S_dot_over_s_M2'])
    P['remark4_S_dot_over_s_M2'] = dict(euclidean=12.2, kinetic=-7.21)
    r5 = ex['remark5']
    N['remark5'] = dict(S_dot_over_s_M2_K=r5['K_nonuniform']['S_dot_over_s_M2'], S_dot_over_s_M2_12I=r5['K_12I']['S_dot_over_s_M2'],
                        value_at_phi_r=r5['K_nonuniform']['constraint_value_at_phi_r'], P_f=r5['K_nonuniform']['P_f'], sKs=r5['K_nonuniform']['sKs'],
                        s2=r5['K_nonuniform']['s_M2'], value_12I=r5['K_12I']['constraint_value_at_phi_r'], P_f_12I=r5['K_12I']['P_f'], sKs_12I=r5['K_12I']['sKs'])
    P['remark5'] = dict(S_dot_over_s_M2_K=0.23, S_dot_over_s_M2_12I=-9.43, value_at_phi_r=-4.22, P_f=4.22, sKs=3.33, s2=3.89, value_12I=-10, P_f_12I=10, sKs_12I=46.68)
    N['rho_bar_lower_estimate'] = dict(narrow=rn['rho_bar_lower_bound'], model=rm['rho_bar_lower_bound'])
    P['rho_bar_lower_estimate'] = dict(narrow=80, model=9.1e3)
    N['qp_check_t13'] = {e: dict(instances=r['instances'], corrected_off=r['corrected']['off_1e7'], corrected_worst=r['corrected']['worst_distance'],
                                 before_fix_off=r['before_fix']['off_1e7'], before_fix_worst=r['before_fix']['worst_distance'],
                                 rated_empty=[r['corrected']['reported_infeasible'], r['before_fix']['reported_infeasible']],
                                 rated_empty_minimiser_distance=r['corrected']['reported_infeasible_minimiser_distance'])
                         for e, r in t13['nearly_dependent_rows'].items()}
    N['qp_check_t13_small_violations'] = {
        f: dict(feasible=r['feasible_instances'], empty=r['empty_instances'],
                **{f'{k}_{q}': r[k][q] for k in ('corrected', 'before_fix')
                   for q in ('backward_error_over_1e8', 'worst_backward_error_stationarity', 'worst_backward_error_rows', 'reported_infeasible',
                             'reported_feasible_though_empty', 'iteration_limit_reached', 'stationarity_over_1e8_relative_to_correction')})
        for f, r in t13['small_violations'].items()}
    N['qp_check_t13_adversarial'] = {
        k: dict(instances=sum(r['instances'] for r in t13['adversarial'].values()),
                backward_error_over_1e8=sum(r[k]['backward_error_over_1e8'] for r in t13['adversarial'].values()),
                of_which_below_stopping_tolerance=sum(r[k]['of_which_below_stopping_tolerance'] for r in t13['adversarial'].values()),
                worst_backward_error_stationarity=max(r[k]['worst_backward_error_stationarity'] for r in t13['adversarial'].values()),
                stationarity_over_1e8_relative_to_correction=sum(r[k]['stationarity_over_1e8_relative_to_correction'] for r in t13['adversarial'].values()),
                feasible_set_rated_empty=sum(r[k]['rated_empty_though_feasible'] for r in t13['adversarial'].values()),
                empty_set_rated_feasible=sum(r[k]['rated_feasible_though_empty'] for r in t13['adversarial'].values()),
                iteration_limit_reached=sum(r[k]['iteration_limit_reached'] for r in t13['adversarial'].values()))
        for k in ('corrected', 'before_fix')}
    N['qp_check_t13_corrected_worse_than_before_fix'] = sum(r['corrected_worse_than_before_fix'] for part in t13.values() for r in part.values())
    (out / 'numbers').mkdir(parents=True, exist_ok=True)
    json.dump(dict(regenerated=N, paper=P), open(out / 'numbers' / 'section_vii_a.json', 'w'), indent=1, default=float)

    def show(v):
        if isinstance(v, dict):
            return '; '.join(f'{k}: {show(x)}' for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return '[' + ', '.join(show(x) for x in v) + ']'
        return f'{v:.4g}' if isinstance(v, float) else str(v)
    lines = ['# Numbers of Section VII-A, Remarks 4-5 and Section VIII: regenerated and printed in the paper', '',
             '| quantity | regenerated | paper |', '|---|---|---|']
    lines += [f'| {k} | {show(v)} | {show(P.get(k, ""))} |' for k, v in N.items()]
    (out / 'numbers' / 'section_vii_a.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(HERE.parents[1] / 'results'))
    a = p.parse_args()
    out = Path(a.out)
    table1(out)
    section_vii_a(out)
    print('wrote', out / 'tables' / 'table1.md', out / 'numbers' / 'section_vii_a.md')
