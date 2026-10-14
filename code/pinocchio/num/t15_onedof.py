"""One degree of freedom: the example of the section on the lower bound and the consistency condition.

Plant            q' = u,  u' = f          (M = 1, N = 0, g = 0, C = H = 1, ideal effort source)
Barrier          h = q >= 0, rates alpha1, alpha2, no standoff:   psi1 = u + alpha1 q,   psi2 = u' + (alpha1 + alpha2) u + alpha1 alpha2 q
Nominal velocity u_n(t) = v0 sin(2 pi t / T_p) - mbar   (a function of time: mean -mbar < 0, maximum v_plus = v0 - mbar > 0)

Part 1  First-order first layer   u_r = max(u_n, -alpha1 q),   a_r = derivative of u_r along the motion,   zs = u_r + alpha1 q
        VDC law  f_r = a_r + k (u_r - u),   effort layer  f = max(f_r, -(alpha1 + alpha2) u - alpha1 alpha2 q).
        For every gain k the velocity error s = u_r - u does not converge.  Evaluated at every onset of a hold (time t2):
            |s(t2)| = psi1(t2) >= exp(-alpha2 (t2 - ta)) psi1(ta)                            (lower bound, any ta <= t2)
            max(|s(t2)|, |s(t2 - 1/alpha2)|) >= zs(t2 - 1/alpha2) / (e + 1)
        and the value v / (e alpha2) for a slack that closes at the rate v while the motion tracks until the effort row binds.
        The proved bound  limsup |s| >= min(mbar, v_plus / (exp(alpha2 T_p) + 1))  is printed for comparison.

Part 2  Required velocity as a state:  u_r' = a_r = max(a_n, -(alpha1 u + beta zs)),  a_n = u_n' + kappa (u_n - u_r),
        i.e. the row  zs' + beta zs >= 0.  With beta <= alpha2 and u_r(0) = u(0): s = 0 for all times and the effort layer never
        binds.  With beta > alpha2 the required motion is not consistent with the effort row and a velocity error appears.

Part 3  Required velocity conditioned by the effort layer:  u_r' = a_r + (f - f_r).  With U = max(u_n, -alpha1 q), the output
        of the first-order layer:
          'add'   u_r = U + u_D,  u_D' = -kappa u_D + (f - f_r),  a_r = U' - kappa u_D     (conditioning state; kappa = k)
          'kin0'  a_r = kappa (U - u_r)                                                      (reference model, no derivative)
          'none'  a_r = u_n' + kappa (u_n - u_r)                                             (no first layer)
        Then s' = -k s for every correction: s = 0 if u_r(0) = u(0).  The motions of the loops are compared: for M = 1 the
        form 'add' with kappa = k is the loop of part 1 with -s in the role of u_D.
"""
import json
import sys

sys.dont_write_bytecode = True
import numpy as np      # noqa: E402

import paths            # noqa: E402

OUT = {}
T_P, V0, MBAR = 2.0, 0.5, 0.1
TIE = 1e-11                 # a row counts as binding if the nominal effort violates it by more than round-off
OM = 2. * np.pi / T_P


def u_n(t):
    return V0 * np.sin(OM * t) - MBAR, V0 * OM * np.cos(OM * t)


# ------------------------------------------------------------------------------------------------------------------ loops
def first_order(t, x, k, a1, a2):
    """Right-hand side, mode (held, binding) and the logged quantities for the first-order first layer."""
    q, u = x
    un, und = u_n(t)
    held = un + a1 * q <= 0.
    ur, ar = (-a1 * q, -a1 * u) if held else (un, und)
    s = ur - u
    f_r = ar + k * s
    low = -(a1 + a2) * u - a1 * a2 * q
    bind = f_r < low - TIE * (1. + abs(low))
    f = low if bind else f_r
    zs = ur + a1 * q
    return np.array([u, f]), (bool(held), bool(bind)), dict(s=s, psi1=u + a1 * q, zs=zs, zs_dot=ar + a1 * u, Pf=-s * (f - f_r), q=q, u=u, corr=f - f_r)


def second_order(t, x, k, a1, a2, beta, kappa):
    """The same loop with the required velocity as a state (x = (q, u, u_r))."""
    q, u, ur = x
    un, und = u_n(t)
    zs = ur + a1 * q
    a_n = und + kappa * (un - ur)
    low_r = -(a1 * u + beta * zs)                 # row  zs' + beta zs >= 0
    held = a_n < low_r                            # here: the row of the first layer binds
    ar = low_r if held else a_n
    s = ur - u
    f_r = ar + k * s
    low = -(a1 + a2) * u - a1 * a2 * q
    bind = f_r < low - TIE * (1. + abs(low))      # for beta = alpha2 and s = 0 the nominal effort meets the row with equality
    f = low if bind else f_r
    return np.array([u, f, ar]), (bool(held), bool(bind)), dict(s=s, psi1=u + a1 * q, zs=zs, zs_dot=ar + a1 * u, Pf=-s * (f - f_r), q=q, u=u,
                                                                cons=ar + a1 * u + beta * zs, corr=f - f_r)


def conditioned(t, x, k, a1, a2, kappa, first):
    """Required velocity conditioned by the correction of the effort layer (x = (q, u, u_r), or (q, u, u_D) for 'add')."""
    q, u, xc = x
    un, und = u_n(t)
    held = first != 'none' and un + a1 * q <= 0.     # the first-order row of the nominal velocity is held
    U, Ud = (-a1 * q, -a1 * u) if held else (un, und)
    if first == 'add':
        ur, ar = U + xc, Ud - kappa * xc
    elif first == 'none':
        ur, ar = xc, und + kappa * (un - xc)
    else:
        ur, ar = xc, kappa * (U - xc)
    s = ur - u
    f_r = ar + k * s
    low = -(a1 + a2) * u - a1 * a2 * q
    bind = f_r < low - TIE * (1. + abs(low))
    f = low if bind else f_r
    xc_dot = -kappa * xc + (f - f_r) if first == 'add' else ar + (f - f_r)
    return np.array([u, f, xc_dot]), (bool(held), bool(bind)), dict(s=s, psi1=u + a1 * q, Pf=-s * (f - f_r), q=q, u=u, corr=f - f_r, dev=ur - un,
                                                                    dev_U=ur - U, vel_U=U - u)


def integrate(rhs, x0, T, dt=2e-4):
    """RK4 with location of the changes of the mode (bisection; the step ends 1e-8 s before the change, which is crossed by
    one Euler step), as in toy.simulate."""
    x = np.array(x0, float)
    n = int(round(T / dt))
    keys = None
    log = {}

    def rk4(t, x, h):
        k1 = rhs(t, x)[0]
        k2 = rhs(t + h / 2, x + h / 2 * k1)[0]
        k3 = rhs(t + h / 2, x + h / 2 * k2)[0]
        k4 = rhs(t + h, x + h * k3)[0]
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for i in range(n + 1):
        t = i * dt
        _, mode, o = rhs(t, x)
        if keys is None:
            keys = list(o)
            log = {kk: [] for kk in keys + ['t', 'held', 'bind']}
        for kk in keys:
            log[kk].append(o[kk])
        log['t'].append(t); log['held'].append(mode[0]); log['bind'].append(mode[1])
        if i == n:
            break
        tt, t_end, ev = t, t + dt, 0
        while t_end - tt > 1e-13:
            h = t_end - tt
            m0 = rhs(tt, x)[1]
            new = rk4(tt, x, h)
            if ev >= 8 or rhs(tt + h, new)[1] == m0:
                x, tt = new, t_end
                break
            lo, hi = 0., h
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if rhs(tt + mid, rk4(tt, x, mid))[1] == m0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-13:
                    break
            h1 = max(lo - 1e-8, 0.)
            if h1 > 0.:
                x = rk4(tt, x, h1)
            d = min(hi - h1 + 1e-10, t_end - tt - h1)
            x = x + d * rhs(tt + h1, x)[0]
            tt += h1 + d
            ev += 1
    return {kk: np.array(v) for kk, v in log.items()}


# ------------------------------------------------------------------------------------------------------------------ part 1
def part1(periods=8):
    res = []
    print('Part 1: first-order first layer, exact feedforward;  u_n = %.2f sin(2 pi t / %.1f) - %.2f' % (V0, T_P, MBAR))
    for a1, a2 in ((6., 16.), (2., 4.)):
        proved = min(MBAR, (V0 - MBAR) / (np.exp(a2 * T_P) + 1.))
        print(f'  rates ({a1:g}, {a2:g}):  proved  limsup |s| >= min(mbar, v_plus / (exp(alpha2 T_p) + 1)) = {proved:.2e}')
        for k in (12., 30., 100.):
            L = integrate(lambda t, x: first_order(t, x, k, a1, a2), [0.3, 0.], periods * T_P)
            t, s = L['t'], np.abs(L['s'])
            dt = t[1] - t[0]
            per = [float(s[(t >= j * T_P) & (t < (j + 1) * T_P)].max()) for j in range(periods)]
            l2 = [float(np.sum(s[(t >= j * T_P) & (t < (j + 1) * T_P)] ** 2) * dt) for j in range(periods)]
            held, bind = L['held'], L['bind']
            starts = np.flatnonzero(held[1:] & ~held[:-1]) + 1
            ons = []
            for i2 in starts:
                ib = i2 - 1
                if not bind[ib]:
                    continue
                while ib > 0 and bind[ib - 1]:
                    ib -= 1
                ia = int(np.searchsorted(t, t[i2] - 1. / a2))
                v = -float(L['zs_dot'][ib])
                ons.append(dict(t_b=float(t[ib]), t_2=float(t[i2]), s_t2=float(s[i2]), psi1_t2=float(L['psi1'][i2]), psi1_tb=float(L['psi1'][ib]),
                                lower=float(np.exp(-a2 * (t[i2] - t[ib])) * L['psi1'][ib]), s_tb=float(s[ib]), zs_tb=float(L['zs'][ib]),
                                closing_rate=v, leading_order=v / (np.e * a2), binding_time=float(t[i2] - t[ib]),
                                two_times_lhs=float(max(s[i2], s[ia])), two_times_rhs=float(L['zs'][ia] / (np.e + 1.)),
                                # Corollary 1 with t_a = t2 - 1/alpha2:  |s(t2)| + |s(t_a)| / e  >=  zs(t_a) / e
                                t_a=float(t[ia]), zs_ta=float(L['zs'][ia]), s_ta=float(s[ia]), corollary_lhs=float(s[i2] + s[ia] / np.e),
                                corollary_bound=float(L['zs'][ia] / np.e)))
            one = dict(alpha1=a1, alpha2=a2, k=k, sup_per_period=per, l2_per_period=l2, onsets=ons, q_min=float(L['q'].min()), psi1_min=float(L['psi1'].min()),
                       proved_limsup=float(proved))
            res.append(one)
            o = ons[-1]
            print(f"    k = {k:5g}: sup |s| per period {np.round(per[2:], 4).tolist()};  integral of s^2 per period {['%.2e' % x for x in l2[2:]]};  min q {one['q_min']:+.1e}")
            print(f"              last onset t2 = {o['t_2']:.4f} s (effort row binding for {o['binding_time']:.4f} s, 1/alpha2 = {1 / a2:.4f}):  |s(t2)| = {o['s_t2']:.5f} = psi1(t2) = {o['psi1_t2']:.5f}"
                  f" >= e^(-alpha2 (t2-tb)) psi1(tb) = {o['lower']:.5f};  |s(tb)| = {o['s_tb']:.1e};  v = {o['closing_rate']:.4f}, v/(e alpha2) = {o['leading_order']:.5f};"
                  f"  max(|s(t2)|, |s(t2 - 1/alpha2)|) = {o['two_times_lhs']:.5f} >= zs(t2 - 1/alpha2)/(e+1) = {o['two_times_rhs']:.5f}")
            late = [x for x in ons if x['t_2'] >= 2 * T_P]
            one['corollary_ratio_periods_3_8'] = [float(min(x['corollary_lhs'] / x['corollary_bound'] for x in late)),
                                                  float(max(x['corollary_lhs'] / x['corollary_bound'] for x in late))]
            print(f"              Corollary 1 at the onsets of periods 3-8: (|s(t2)| + |s(t_a)|/e) / (zs(t_a)/e) between "
                  f"{one['corollary_ratio_periods_3_8'][0]:.4f} and {one['corollary_ratio_periods_3_8'][1]:.4f}")
            bad = [x for x in ons if x['psi1_t2'] < x['lower'] - 1e-6 or x['two_times_lhs'] < x['two_times_rhs'] - 1e-6 or abs(x['s_t2'] - x['psi1_t2']) > 1e-6
                   or x['corollary_lhs'] < x['corollary_bound'] - 1e-6]
            assert not bad, bad
            np.savez_compressed(paths.NUM / f't15_first_order_a{a2:g}_k{k:g}.npz', t=t[::5], s=L['s'][::5], q=L['q'][::5])
    OUT['first_order'] = res


# ------------------------------------------------------------------------------------------------------------------ part 2
def part2(periods=8, kappa=20.):
    res = []
    a1, a2 = 6., 16.
    print('Part 2: required velocity as a state with the row zs\' + beta zs >= 0, rates (6, 16), u_r(0) = u(0)')
    for k, beta in ((12., 16.), (30., 16.), (100., 16.), (12., 8.), (12., 40.), (30., 40.)):
        q0, u0 = 0.3, 0.
        L = integrate(lambda t, x: second_order(t, x, k, a1, a2, beta, kappa), [q0, u0, u0], periods * T_P)
        t, s = L['t'], np.abs(L['s'])
        per = [float(s[(t >= j * T_P) & (t < (j + 1) * T_P)].max()) for j in range(periods)]
        one = dict(k=k, beta=beta, consistent=bool(beta <= a2), sup_per_period=per, effort_binding_steps=int(L['bind'].sum()), first_layer_binding_steps=int(L['held'].sum()),
                   steps=int(len(t)), q_min=float(L['q'].min()), zs_min=float(L['zs'].min()), cons_min=float(L['cons'].min()), psi1_min=float(L['psi1'].min()),
                   largest_correction=float(np.abs(L['corr']).max()))
        res.append(one)
        print(f"    k = {k:5g}, beta = {beta:4g} ({'<=' if beta <= a2 else '> '} alpha2): sup |s| per period {['%.1e' % x for x in per[2:]]};  effort row binds in {one['effort_binding_steps']} steps, "
              f"row of the first layer in {one['first_layer_binding_steps']} of {one['steps']};  largest correction of the effort layer {one['largest_correction']:.1e};  min q {one['q_min']:+.2e}, min zs {one['zs_min']:+.1e}, min (zs' + beta zs) {one['cons_min']:+.1e}")
        if beta <= a2:
            assert max(per) < 1e-9 and one['effort_binding_steps'] == 0, one
        else:
            assert min(per[2:]) > 1e-3 and one['effort_binding_steps'] > 0, one
        np.savez_compressed(paths.NUM / f't15_second_order_b{beta:g}_k{k:g}.npz', t=t[::5], s=L['s'][::5], q=L['q'][::5])
    OUT['second_order'] = res


# ------------------------------------------------------------------------------------------------------------------ part 3
def part3(periods=8):
    res = []
    a1, a2 = 6., 16.
    print('Part 3: conditioned required velocity, rates (6, 16), and comparison of the motions (periods 3 to 8)')
    for k, kappa in ((12., 20.), (30., 20.), (100., 200.), (400., 2000.)):
        A = integrate(lambda t, x: first_order(t, x, k, a1, a2), [0.3, 0.], periods * T_P)
        B = integrate(lambda t, x: second_order(t, x, k, a1, a2, a2, kappa), [0.3, 0., 0.], periods * T_P)
        t = A['t']
        late = t >= 2 * T_P
        one = dict(k=k, kappa=kappa, sup_s_first_order=float(np.abs(A['s'][late]).max()), sup_s_second_order=float(np.abs(B['s'][late]).max()),
                   dq_first_second=float(np.abs(A['q'] - B['q'])[late].max()), q_range=[float(A['q'][late].min()), float(A['q'][late].max())])
        line = (f"    k = {k:g}, kappa = {kappa:g}: sup |s|: first-order layer {one['sup_s_first_order']:.4f}, second-order layer {one['sup_s_second_order']:.1e};  "
                f"sup |q_first - q_second| = {one['dq_first_second']:.2e} m (range of q {one['q_range'][0]:.4f} to {one['q_range'][1]:.4f} m)")
        for first in ('add', 'kin0', 'none'):
            kap = k if first == 'add' else kappa
            # u_r(0) = u(0) = 0:  for 'add' the state is u_D = u_r - U, and U(0) = u_n(0)
            x0 = [0.3, 0., -u_n(0.)[0]] if first == 'add' else [0.3, 0., 0.]
            Cn = integrate(lambda t, x: conditioned(t, x, k, a1, a2, kap, first), x0, periods * T_P)
            one[f'sup_s_conditioned_{first}'] = float(np.abs(Cn['s']).max())
            one[f'sup_dev_U_conditioned_{first}'] = float(np.abs(Cn['dev_U'][late]).max())
            one[f'sup_vel_U_conditioned_{first}'] = float(np.abs(Cn['vel_U'][late]).max())
            one[f'dq_first_conditioned_{first}'] = float(np.abs(A['q'] - Cn['q'])[late].max())
            one[f'binding_steps_conditioned_{first}'] = int(Cn['bind'].sum())
            one[f'q_min_conditioned_{first}'] = float(Cn['q'].min())
            one[f'sup_dev_conditioned_{first}'] = float(np.abs(Cn['dev'][late]).max())
            line += (f";\n          conditioned '{first}' (kappa = {kap:g}): sup |s| = {one[f'sup_s_conditioned_{first}']:.1e}, sup |u_r - U| = {one[f'sup_dev_U_conditioned_{first}']:.4f}, "
                     f"sup |U - u| = {one[f'sup_vel_U_conditioned_{first}']:.4f}, effort row binds in {one[f'binding_steps_conditioned_{first}']} steps, "
                     f"sup |q_first - q| = {one[f'dq_first_conditioned_{first}']:.2e} m, min q {one[f'q_min_conditioned_{first}']:+.1e}")
            assert one[f'sup_s_conditioned_{first}'] < 1e-9 and one[f'q_min_conditioned_{first}'] > -1e-9, one
            if (k, kappa) == (12., 20.):
                np.savez_compressed(paths.NUM / f't15_conditioned_{first}_k{k:g}.npz', t=t[::5], s=Cn['s'][::5], q=Cn['q'][::5], dev=Cn['dev'][::5])
        # with an initial velocity error the error decays at the rate k whatever the effort layer does
        Ce = integrate(lambda t, x: conditioned(t, x, k, a1, a2, kappa, 'kin0'), [0.3, 0., -0.5], 1.0)
        sel = np.abs(Ce['s']) > 1e-9 * abs(Ce['s'][0])
        ratio = np.abs(Ce['s'][sel]) * np.exp(k * Ce['t'][sel]) / abs(Ce['s'][0])
        one['decay_ratio'] = [float(ratio.min()), float(ratio.max())]
        one['decay_binding_steps'] = int(Ce['bind'][sel].sum())
        line += f";\n          'kin0', start with s(0) = -0.5: |s(t)| e^(kt) / |s(0)| in [{ratio.min():.8f}, {ratio.max():.8f}], effort row binding in {one['decay_binding_steps']} of these steps"
        print(line)
        res.append(one)
    OUT['conditioned'] = res


if __name__ == '__main__':
    part1()
    part2()
    part3()
    json.dump(OUT, open(paths.NUM / 't15_results.json', 'w'), indent=1, default=float)
