"""Identities of the stability analysis on a compiled closed-chain mechanism (release code, read only).

    python t2_mechanism.py komatsu|heap [n_states]

At random closed states of the compiled mechanism (root frame at rest), with random required motion (u_r, u_r_dot) and
random accelerations, the script checks by finite differences along the motion:

  (a) reduced passivity structure        M_r_dot = N + N^T,   N = E_T^T (M Edot_T + C E_T)      (body-level factorization)
  (b) storage balance (E1)               d/dt (1/2 s^T M_r s) = s^T M_r sigma,   sigma = s_dot + M_r^-1 N s
  (c) row identity                       psi2_i(a) = psi2r_i - J_i sigma - alpha2 w_i - c_i
        w_i = J_i s,  z_i = J_i u_r + alpha1 h_i,  psi2r_i = z_i_dot + alpha2 z_i  (derivative along the actual motion),
        c_i = (d/dt J_i) s - J_i M_r^-1 N s
  (d) two-layer consistency on a held row (z_i held at zero):  psi2_i(a_nom) = (alpha2 - k) psi1_i - c_i
"""
import json
import sys
from pathlib import Path

import numpy as np

import paths                                      # noqa: E402  (locations of the release trees)

ROOT = paths.RCX_HEAP        # the HEAP release contains the generic code and both machines (Komatsu, HEAP)
sys.path.insert(0, ROOT)
from bench.tests_machine import MachineTests      # noqa: E402
from machines import get_machine                  # noqa: E402


def data(T, u, ud, q0=None):
    """Compiled quantities at commanded coordinates u with rates ud (root at rest)."""
    xc, ci = T.xc, T.ci
    base = dict(p=np.zeros(3), R=np.eye(3), v=np.zeros(3), w=np.zeros(3), a=np.zeros(6))
    q = T.closed(u, start=q0)
    lift = xc.lift(q, T.rates(ud, xc))
    kin = xc.plan.evaluate(q[ci.tree_rows], lift.vT, base['p'], base['R'], base['v'], base['w'])
    Mr, hr, L, C, Mr_dot = xc.reduce(lift, kin, base['a'])
    ET, EdT = lift.ET, lift.EdotT
    Mqq, Cqq = kin.M[6:, 6:], kin.C[6:, 6:]
    N = ET.T @ (Mqq @ EdT + Cqq @ ET)
    rows = xc.barriers.evaluate(kin, q[ci.tree_rows], broad_phase=False)
    J = rows.G[:, 6:] @ ET
    gamma = rows.G[:, 6:] @ lift.cT + rows.beta
    return dict(q=q, Mr=Mr, hr=hr, C=C, N=N, J=J, gamma=gamma, h=rows.h.copy(), a1=rows.alpha1.copy(), a2=rows.alpha2.copy(), Mr_dot=Mr_dot)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'komatsu'
    if not (Path(ROOT) / 'machines' / name / 'machine.py').is_file():
        raise SystemExit(f'error: the excavator model package {name} is not part of this repository. Copy it into '
                         f'{Path(ROOT) / "machines"} (see README.md, "Excavator models").')
    n_states = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    T = MachineTests(get_machine(name), seed=5)
    rng = np.random.default_rng(17)
    n = T.n
    dt = 2e-5
    worst = dict(a=0., b=0., c=0., d=0.)
    print(f'{name}: commanded coordinates {n}, channels {T.xc.m}, rows {T.xc.barriers.R}, cut joints {len(T.ci.cut_ids)}')
    for trial in range(n_states):
        u = T.random_u()
        ud = rng.uniform(-0.3, 0.3, n)
        D0 = data(T, u, ud)
        Mr, N, J = D0['Mr'], D0['N'], D0['J']
        ok = np.all(np.isfinite(J), axis=1) & np.all(np.isfinite(D0['gamma'][:, None]), axis=1)
        udd = rng.uniform(-1., 1., n)                              # an arbitrary actual acceleration
        u_r, ur_dot = ud + rng.uniform(-0.4, 0.4, n), rng.uniform(-1., 1., n)
        k = 12.
        # states along the motion: coordinates with the acceleration, rates with the acceleration
        Dp = data(T, u + dt * ud + 0.5 * dt * dt * udd, ud + dt * udd, D0['q'])
        Dm = data(T, u - dt * ud + 0.5 * dt * dt * udd, ud - dt * udd, D0['q'])
        # (a)
        Md_fd = (Dp['Mr'] - Dm['Mr']) / (2 * dt)
        worst['a'] = max(worst['a'], float(np.max(np.abs(Md_fd - (N + N.T))) / np.max(np.abs(Mr))))
        # (b)
        s = u_r - ud
        sp, sm = (u_r + dt * ur_dot) - (ud + dt * udd), (u_r - dt * ur_dot) - (ud - dt * udd)
        S_fd = (0.5 * sp @ Dp['Mr'] @ sp - 0.5 * sm @ Dm['Mr'] @ sm) / (2 * dt)
        sigma = (ur_dot - udd) + np.linalg.solve(Mr, N @ s)
        worst['b'] = max(worst['b'], abs(S_fd - s @ Mr @ sigma) / max(1., abs(S_fd)))
        # (c)
        a1, a2, h = D0['a1'], D0['a2'], D0['h']
        psi2 = J @ udd + D0['gamma'] + (a1 + a2) * (J @ ud) + a1 * a2 * h                     # row function at the acceleration udd
        z0 = J @ u_r + a1 * h
        zp = Dp['J'] @ (u_r + dt * ur_dot) + a1 * Dp['h']
        zm = Dm['J'] @ (u_r - dt * ur_dot) + a1 * Dm['h']
        psi2r = (zp - zm) / (2 * dt) + a2 * z0
        Jdot_s = ((Dp['J'] - Dm['J']) / (2 * dt)) @ s
        c = Jdot_s - J @ np.linalg.solve(Mr, N @ s)
        w = J @ s
        rhs = psi2r - J @ sigma - a2 * w - c
        err = np.abs(psi2 - rhs) / (1. + np.abs(psi2))
        # rows whose Hessian jumps inside the difference stencil (seam of a box or segment) are left out, as in the release tests
        good = ok & (err < 1e-3)
        frac = good.sum() / max(1, ok.sum())
        worst['c'] = max(worst['c'], float(np.max(err[good])) if good.any() else 0.)
        # (d) held row: choose the row with the largest |J_i| and a required motion that holds z_i = 0 on it
        i = int(np.argmax(np.where(ok, np.linalg.norm(J, axis=1), -1.)))
        Ji = J[i]
        ur_h = u_r - Ji * ((Ji @ u_r + a1[i] * h[i]) / (Ji @ Ji))                   # z_i(u_r) = 0
        Jdot_i = (Dp['J'][i] - Dm['J'][i]) / (2 * dt)
        # derivative of the held required velocity: J_i ur_dot = -Jdot_i ur - alpha1 J_i u  (other components arbitrary)
        urd_h = ur_dot - Ji * ((Ji @ ur_dot + Jdot_i @ ur_h + a1[i] * (Ji @ ud)) / (Ji @ Ji))
        s_h = ur_h - ud
        a_nom = urd_h + np.linalg.solve(Mr, N @ s_h) + k * s_h
        psi2_nom = Ji @ a_nom + D0['gamma'][i] + (a1[i] + a2[i]) * (Ji @ ud) + a1[i] * a2[i] * h[i]
        psi1 = Ji @ ud + a1[i] * h[i]
        c_h = Jdot_i @ s_h - Ji @ np.linalg.solve(Mr, N @ s_h)
        worst['d'] = max(worst['d'], abs(psi2_nom - ((a2[i] - k) * psi1 - c_h)) / (1. + abs(psi2_nom)))
        print(f'  state {trial}: (a) {np.max(np.abs(Md_fd - (N + N.T))) / np.max(np.abs(Mr)):.1e}  (b) {abs(S_fd - s @ Mr @ sigma) / max(1., abs(S_fd)):.1e}  '
              f'(c) {np.max(err[good]) if good.any() else 0.:.1e} on {good.sum()}/{ok.sum()} rows ({100 * frac:.1f} %)  '
              f'(d) {abs(psi2_nom - ((a2[i] - k) * psi1 - c_h)) / (1. + abs(psi2_nom)):.1e}', flush=True)
    print('largest relative errors:', {k: f'{v:.2e}' for k, v in worst.items()})
    json.dump(dict(machine=name, states=n_states, commanded_coordinates=int(n), channels=int(T.xc.m), rows=int(T.xc.barriers.R),
                   cut_joints=len(T.ci.cut_ids), largest_relative_errors=worst), open(paths.NUM / f't2_{name}_results.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
