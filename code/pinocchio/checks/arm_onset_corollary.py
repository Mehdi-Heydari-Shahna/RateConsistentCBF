"""Corollary 1 (size of the error at an activation) on the six-joint arm of Section VII-A: first-order first layer,
exact feedforward, periodic reference of period 3 s, k = 12 and 30 (the loop of t14_consistent.py, part 1).

At every activation t2 of a constraint of the first layer (first step at which it is active) and t_a = t2 - 1/alpha2
(linear interpolation between the 1 ms grid points), the script evaluates
    lhs   = |J_i s(t2)| + |J_i s(t_a)| / e,      bound = zs_i(t_a) / e        (Corollary 1: lhs >= bound),
and the norm |s(t2)|_M.  Writes arm_onset_corollary.json next to this script.
usage:  python arm_onset_corollary.py <folder with toy.py, run_toy.py, t14_consistent.py, paths.py>
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
NUM = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(NUM))
import numpy as np                       # noqa: E402
from toy import Arm, Controller, simulate          # noqa: E402
from run_toy import Q_A, barriers                   # noqa: E402
from t14_consistent import periodic_reference, T_P  # noqa: E402


def interp(t, x, ti):
    j = int(np.searchsorted(t, ti)) - 1
    j = min(max(j, 0), len(t) - 2)
    lam = (ti - t[j]) / (t[j + 1] - t[j])
    return (1. - lam) * x[j] + lam * x[j + 1]


def main(periods=4):
    arm = Arm()
    res = []
    for k in (12., 30.):
        bar = barriers(arm)
        ctl = Controller(arm, bar, periodic_reference, lam=12., k=k)
        L = simulate(ctl, Q_A, np.zeros(6), periods * T_P, dt=1e-3)
        t = np.asarray(L['t'])
        act_k = np.asarray(L['act_k'], bool)
        zs, w, sn = np.asarray(L['zs']), np.asarray(L['w']), np.asarray(L['snorm'])
        for row in range(bar.R):
            starts = np.flatnonzero(act_k[1:, row] & ~act_k[:-1, row]) + 1
            for i2 in starts:
                a2 = float(bar.a2[row])
                ta = t[i2] - 1. / a2
                if ta < t[0]:
                    continue
                zta, wta = float(interp(t, zs[:, row], ta)), float(interp(t, w[:, row], ta))
                lhs = abs(float(w[i2, row])) + abs(wta) / np.e
                bound = zta / np.e
                res.append(dict(k=k, row=bar.names[row], alpha2=a2, t2=float(t[i2]), t_a=float(ta), zs_ta=zta, w_t2=float(w[i2, row]),
                                w_ta=wta, lhs=lhs, bound=bound, ratio=lhs / bound if bound > 0 else float('nan'), s_t2=float(sn[i2])))
                print(f"k = {k:4.0f}  {bar.names[row]:6s} t2 = {t[i2]:7.3f}  zs(t2 - 1/a2)/e = {bound:.5f}  |w(t2)| + |w(ta)|/e = {lhs:.5f}  "
                      f"ratio {lhs / bound:.4f}  |s(t2)|_M = {sn[i2]:.4f}")
    out = Path(__file__).with_suffix('.json')
    json.dump(res, open(out, 'w'), indent=1)
    r = np.array([x['ratio'] for x in res])
    print(f"{len(res)} activations; ratio lhs/bound between {r.min():.4f} and {r.max():.4f}")


if __name__ == '__main__':
    main()
