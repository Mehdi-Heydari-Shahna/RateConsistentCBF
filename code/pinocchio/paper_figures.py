"""Figures 2 and 3 of the paper (continuous-time illustrations) and their data as CSV files.

Reads the results of
    num/t14_consistent.py         six-joint arm: first-order first layer (k = 12, 30), consistent layer (exact model, mismatch)
    num/t15_onedof.py             scalar example (Example 1): first-order layer (k = 12, 30, 100), consistent and beta = 40 layers
    onedof_sweep/sweep_alpha2.py  scalar example: second rate alpha2 = 8, 16, 32, 64
and writes  <out>/figures/fig2.pdf, fig3.pdf  and  <out>/csv/fig2a_arm.csv, fig2b_scalar.csv, fig2b_onset_bounds.csv,
fig3a_alpha2.csv, fig3b_gain.csv.

    python paper_figures.py [--out <folder>]          (default: ../../results)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np      # noqa: E402
import matplotlib      # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import matplotlib.ticker      # noqa: E402

HERE = Path(__file__).resolve().parent
NUM, SWEEP = HERE / 'num', HERE / 'onedof_sweep'
ORANGE, GREEN, PURPLE, BLUE, GREY = '#e66101', '#1b9e77', '#7b3294', '#2166ac', '#555555'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'lines.linewidth': 1.3, 'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def write_csv(path, header, rows, comment=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        if comment:
            for line in comment.splitlines():
                f.write(f'# {line}\n')
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f'{x:.9g}' if isinstance(x, (float, np.floating)) else x for x in r])


def fig2(out):
    A12, A30 = np.load(NUM / 't14_first_order_k12.npz'), np.load(NUM / 't14_first_order_k30.npz')
    Cm, C0 = np.load(NUM / 't14_consistent_mismatch_k12.npz'), np.load(NUM / 't14_consistent_k12.npz')
    t = A12['t']
    for X in (A30, Cm, C0):
        assert np.allclose(X['t'], t)
    write_csv(out / 'csv' / 'fig2a_arm.csv', ['t_s', 'first_order_k12', 'first_order_k30', 'consistent_mismatch_k12', 'consistent_exact_k12'],
              zip(t, A12['snorm'], A30['snorm'], Cm['snorm'], C0['snorm']),
              'Fig. 2(a): six-joint arm, kinetic velocity error |s|_M (sqrt(J)) on the 1 ms grid (t14_consistent.py)')
    T = json.load(open(NUM / 't15_results.json'))
    B12, B100 = np.load(NUM / 't15_first_order_a16_k12.npz'), np.load(NUM / 't15_first_order_a16_k100.npz')
    B40, B16 = np.load(NUM / 't15_second_order_b40_k12.npz'), np.load(NUM / 't15_second_order_b16_k12.npz')
    tb = B12['t']
    for X in (B100, B40, B16):
        assert np.allclose(X['t'], tb)
    write_csv(out / 'csv' / 'fig2b_scalar.csv', ['t_s', 'first_order_k12', 'first_order_k100', 'acceleration_level_beta40_k12', 'consistent_beta16_k12'],
              zip(tb, B12['s'], B100['s'], B40['s'], B16['s']),
              'Fig. 2(b): scalar example, velocity error s = u_r - u (m/s), every 5th point of the 0.2 ms grid (t15_onedof.py)')
    # onset bound of Corollary 1 at every activation of the first-order layer (k = 12):  |s(t2)| >= (zs(t_a) - |s(t_a)|) / e
    run = next(r for r in T['first_order'] if r['alpha2'] == 16. and r['k'] == 12.)
    ons = [o for o in run['onsets']]
    write_csv(out / 'csv' / 'fig2b_onset_bounds.csv', ['t_a_s', 't2_s', 'abs_s_t2', 'zs_ta', 'abs_s_ta', 'bound_corrected'],
              [(o['t_a'], o['t_2'], o['s_t2'], o['zs_ta'], o['s_ta'], (o['zs_ta'] - o['s_ta']) / np.e) for o in ons],
              'Fig. 2(b): Corollary 1 at every activation (first-order layer, k = 12, alpha2 = 16); the dotted segments are drawn at -bound_corrected')
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.6), gridspec_kw=dict(hspace=0.55))
    ax = axs[0]
    sel = t <= 6.
    ax.plot(t[sel], A12['snorm'][sel], color=ORANGE, label='First order, $k=12$')
    ax.plot(t[sel], A30['snorm'][sel], color=GREEN, ls='--', label='First order, $k=30$')
    ax.plot(t[sel], Cm['snorm'][sel], color=PURPLE, label=r'Consistent, $\|e_m\|\leq 0.3$')
    ax.plot(t[sel], C0['snorm'][sel], color=BLUE, label='Consistent, exact model')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\|s\|_M$ ($\sqrt{\mathrm{J}}$)')
    ax.set_title('(a) Six-joint arm', loc='left')
    ax.legend(ncol=2, loc='upper center', handlelength=1.8, columnspacing=1.0)
    ax = axs[1]
    sel = (tb >= 4.) & (tb <= 10.)
    ax.plot(tb[sel], B12['s'][sel], color=ORANGE, label='First order, $k=12$')
    ax.plot(tb[sel], B100['s'][sel], color=GREEN, ls='--', label='First order, $k=100$')
    ax.plot(tb[sel], B40['s'][sel], color=PURPLE, ls=':', label=r'Acceleration level, $\bar\beta=40$')
    ax.plot(tb[sel], B16['s'][sel], color=BLUE, label=r'Consistent, $\bar\beta=16$')
    for o in ons:
        if 4. <= o['t_2'] <= 10.:
            lvl = -(o['zs_ta'] - o['s_ta']) / np.e
            ax.plot([o['t_2'] - 0.25, o['t_2'] + 0.15], [lvl, lvl], color=GREY, ls=':', lw=1.2)
    ax.set_xlim(4, 10)
    ax.set_ylim(-0.11, 0.075)
    ax.set_yticks([-0.1, -0.05, 0.])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$s$ (m/s)')
    ax.set_title('(b) Scalar example', loc='left')
    ax.legend(ncol=2, loc='upper center', handlelength=1.8, columnspacing=1.0)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'figures' / 'fig2.pdf')
    plt.close(fig)


def fig3(out):
    S = json.load(open(SWEEP / 'sweep_alpha2.json'))
    T = json.load(open(NUM / 't15_results.json'))
    a2 = [r['alpha2'] for r in S]
    peak = [r['peak'] for r in S]
    bound = [r['bound_corrected_min'] for r in S]
    write_csv(out / 'csv' / 'fig3a_alpha2.csv', ['alpha2_per_s', 'peak_abs_s_periods_3_8', 'onset_bound_min', 'alpha2_times_peak'],
              [(x, p, b, x * p) for x, p, b in zip(a2, peak, bound)],
              'Fig. 3(a): first-order layer, k = 12, alpha1 = 6; peak |s| over periods 3-8 and the smallest Corollary 1 bound '
              '(zs(t_a) - |s(t_a)|)/e over the onsets of periods 3-8 (sweep_alpha2.py)')
    ks = (12., 30., 100.)
    fo = [max(next(r for r in T['first_order'] if r['alpha2'] == 16. and r['k'] == k)['sup_per_period'][2:]) for k in ks]
    co = [max(next(r for r in T['second_order'] if r['beta'] == 16. and r['k'] == k)['sup_per_period'][2:]) for k in ks]
    write_csv(out / 'csv' / 'fig3b_gain.csv', ['k_per_s', 'first_order_peak', 'consistent_peak'], zip(ks, fo, co),
              'Fig. 3(b): alpha2 = 16, max |s| over periods 3-8 (t15_onedof.py)')
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    ax.loglog(a2, peak, color=ORANGE, marker='o', ms=3.5, label='Peak error')
    ax.loglog(a2, bound, color=GREY, ls=':', marker='x', ms=4, label='Onset bound')
    ax.set_xscale('log', base=2)
    ax.set_xticks(a2)
    ax.set_xticklabels([f'{x:g}' for x in a2])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r'$\alpha_2$ (s$^{-1}$)')
    ax.set_ylabel(r'max$|s|$ (m/s)')
    ax.set_title('(a) Barrier rate', loc='left')
    ax.legend(loc='lower left', handlelength=1.5, fontsize=6.5)
    ax = axs[1]
    ax.plot(ks, fo, color=ORANGE, marker='o', ms=3.5, label='First order')
    ax.plot(ks, co, color=BLUE, marker='o', ms=3.5, label='Consistent')
    ax.set_xticks(ks)
    ax.set_xticklabels([f'{k:g}' for k in ks])
    ax.set_xlim(5., 105.)
    ax.set_ylim(-0.005, 0.12)
    ax.set_xlabel(r'$k$ (s$^{-1}$)')
    ax.set_ylabel(r'max$|s|$ (m/s)')
    ax.set_title(r'(b) Gain, $\alpha_2=16$', loc='left')
    ax.legend(loc='upper center', fontsize=6.5)
    fig.savefig(out / 'figures' / 'fig3.pdf')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(HERE.parents[1] / 'results'))
    a = p.parse_args()
    out = Path(a.out)
    fig2(out)
    fig3(out)
    print('wrote', out / 'figures' / 'fig2.pdf', out / 'figures' / 'fig3.pdf')
