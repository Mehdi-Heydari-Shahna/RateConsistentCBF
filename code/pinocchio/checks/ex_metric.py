"""Remarks 4 and 5 of the paper and the numbers of their proofs (two degrees of freedom, one active constraint, no defect).

Remark 4 (the kinetic metric is needed): d = m = 2, C = I, M = diag(1, 0.01), k = 12, e_a = e_m = 0, one active constraint with
z~ = z~' = 0 (no curvature), alpha2 = 4, J = (-1, 0.01), s = (1, -10).  The Euclidean metric H = I gives S' = +12.2 |s|_M^2,
the kinetic metric S' = -7.21 |s|_M^2.

Remark 5 (the gain K = k M is needed): d = m = 2, M = C = H = I, K = [[12, 3.4], [3.4, 1]], one active constraint with
J = (1, 0), alpha2 = 2, no defect, s = (-1, 1.7): S' = +0.23 |s|_M^2, while K = 12 I gives S' = -9.43 |s|_M^2.
Proof numbers: psi~1 = -J s = 1, value of the constraint at phi_r (identity (19) with beta = 0) alpha2 psi~1 + J K s = -4.22,
Delta = 4.22 J^T, P_f = 4.22, s^T K s = 3.33, |s|^2 = 3.89; for K = 12 I: value -10, P_f = 10, s^T K s = 46.68.

    python ex_metric.py            # prints the numbers and writes ex_metric.json next to this script
"""
import json
from pathlib import Path

import numpy as np


def held_row_effort_correction(M, C, H, J, Pi):
    """Smallest H-norm correction Delta with  J M^-1 C Delta >= -Pi  (one constraint of the effort layer)."""
    G = J @ np.linalg.solve(M, C)
    lam = max(0., -Pi) / float(G @ np.linalg.solve(H, G))
    return lam * np.linalg.solve(H, G)


out = {}
# Remark 4
M = np.diag([1., 0.01]); C = np.eye(2); k = 12.; a2 = 4.
J = np.array([-1., 0.01]); s = np.array([1., -10.])
w = J @ s
Pi = (a2 - k) * (-w)                   # held row, exact feedforward, no curvature: value of the constraint at phi_r, psi1 = -w
r4 = dict(w=float(w), constraint_value_at_phi_r=float(Pi))
for name, H in (('euclidean', np.eye(2)), ('kinetic', C.T @ np.linalg.solve(M, C))):
    D = held_row_effort_correction(M, C, H, J, Pi)
    Pf = -float(s @ C @ D)
    S2 = float(s @ M @ s)
    Sdot = -k * S2 + Pf
    r4[name] = dict(Delta=D.tolist(), P_f=Pf, s_M2=S2, S_dot=Sdot, S_dot_over_s_M2=Sdot / S2)
    print(f'Remark 4, {name:9s} metric: Delta {np.round(D, 4)}, P_f = {Pf:.4f}, |s|_M^2 = {S2:.4f}, S_dot = {Sdot:.4f} = {Sdot / S2:+.3f} |s|_M^2')
out['remark4'] = r4
# Remark 5
M = C = H = np.eye(2); a2 = 2.
J = np.array([1., 0.]); s = np.array([-1., 1.7])
r5 = {}
for name, K in (('K_nonuniform', np.array([[12., 3.4], [3.4, 1.]])), ('K_12I', 12. * np.eye(2))):
    psi1 = -float(J @ s)                                    # z~ = 0
    Pi = a2 * psi1 + float(J @ K @ s)                       # identity (19), beta = 0, no defect and no curvature
    D = held_row_effort_correction(M, C, H, J, Pi)
    Pf = -float(s @ C @ D)
    sKs = float(s @ K @ s)
    S2 = float(s @ s)
    Sdot = -sKs + Pf
    r5[name] = dict(psi1=psi1, constraint_value_at_phi_r=Pi, Delta=D.tolist(), P_f=Pf, sKs=sKs, s_M2=S2, S_dot=Sdot, S_dot_over_s_M2=Sdot / S2)
    print(f'Remark 5, {name:12s}: psi1 = {psi1:.2f}, value at phi_r = {Pi:.2f}, Delta = {np.round(D, 2)}, P_f = {Pf:.2f}, s^T K s = {sKs:.2f}, '
          f'|s|^2 = {S2:.2f}, S_dot = {Sdot:.3f} = {Sdot / S2:+.3f} |s|_M^2')
ev = np.linalg.eigvalsh(np.array([[12., 3.4], [3.4, 1.]]))
r5['K_nonuniform']['eigenvalues'] = ev.tolist()
out['remark5'] = r5
json.dump(out, open(Path(__file__).with_suffix('.json'), 'w'), indent=1)
