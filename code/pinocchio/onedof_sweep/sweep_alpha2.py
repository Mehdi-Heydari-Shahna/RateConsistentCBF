"""Example 1 of the paper (Fig. 3(a)): size of the velocity error at the onsets against the second rate alpha2
(first-order first layer, exact feedforward, k = 12, alpha1 = 6), and the bound of Corollary 1 at every onset of periods
3-8 (t_a = t2 - 1/alpha2):

    |s(t2)|  >=  (zs(t_a) - |s(t_a)|) / e          (onset bound, with the correction for the preceding error),
    zs(t_a) / e                                     (the same without the correction).

Uses the loop and the event-located integrator of ../num/t15_onedof.py (RK4, 0.2 ms).

    python sweep_alpha2.py          # writes sweep_alpha2.json next to this script and prints a table
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'num'))
import numpy as np      # noqa: E402
import t15_onedof as T      # noqa: E402

rows = []
for a2 in (8., 16., 32., 64.):
    a1, k, periods = 6., 12., 8
    L = T.integrate(lambda t, x: T.first_order(t, x, k, a1, a2), [0.3, 0.], periods * T.T_P)
    t, s, zs = L['t'], np.abs(L['s']), L['zs']
    held = L['held']
    starts = np.flatnonzero(held[1:] & ~held[:-1]) + 1
    on = []
    for i2 in starts:
        if t[i2] < 2 * T.T_P:                  # from the third period on (as in the paper)
            continue
        ia = int(np.argmin(np.abs(t - (t[i2] - 1. / a2))))
        on.append(dict(t2=float(t[i2]), t_a=float(t[ia]), s_t2=float(s[i2]), s_ta=float(s[ia]), zs_ta=float(zs[ia]),
                       bound=float(zs[ia] / np.e), bound_corrected=float((zs[ia] - s[ia]) / np.e),
                       ratio=float((s[i2] + s[ia] / np.e) / (zs[ia] / np.e))))
    per = [float(s[(t >= j * T.T_P) & (t < (j + 1) * T.T_P)].max()) for j in range(2, periods)]
    r = dict(alpha1=a1, alpha2=a2, k=k, sup_s_periods_3_8=[min(per), max(per)], peak=max(per), alpha2_times_peak=a2 * max(per), onsets=on,
             bound_min=min(o['bound'] for o in on), bound_corrected_min=min(o['bound_corrected'] for o in on),
             s_t2_min=min(o['s_t2'] for o in on), s_t2_max=max(o['s_t2'] for o in on),
             corollary_ratio=[min(o['ratio'] for o in on), max(o['ratio'] for o in on)])
    rows.append(r)
    print(f"alpha2 = {a2:5.1f}: max|s| per period (3-8) {min(per):.4f} to {max(per):.4f} m/s;  alpha2 * max|s| = {a2 * max(per):.3f};  "
          f"at the onsets |s(t2)| {r['s_t2_min']:.4f}-{r['s_t2_max']:.4f}, bound (zs(t_a) - |s(t_a)|)/e >= {r['bound_corrected_min']:.4f} m/s, "
          f"zs(t_a)/e >= {r['bound_min']:.4f} m/s (|s(t_a)| <= {max(o['s_ta'] for o in on):.1e}); Corollary 1 ratio {r['corollary_ratio'][0]:.4f}-{r['corollary_ratio'][1]:.4f}")
json.dump(rows, open(HERE / 'sweep_alpha2.json', 'w'), indent=1)
