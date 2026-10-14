"""Sampled lower bound of the worst-case curvature rate  rho_bar  of Theorem 5 (two-sided limits) for the six-joint
sample manipulator of Pinocchio with limits on all joints.

    rho_bar = sup_{(q,u) in C} || Gamma M^{-1/2} ||_2 / sqrt(lambda_min(L M^{-1} L^T)),   Gamma = Ldot - L M^{-1} N,
with l = q (L = I, Ldot = 0):  rho_bar = sup || M^{-1} N(q,u) M^{-1/2} ||_2 * sqrt(lambda_max(M(q))).

Safe set of the box with rates (alpha1, alpha2) and no standoff:  lo_j <= q_j <= hi_j  and
-alpha1 (q_j - lo_j) <= u_j <= alpha1 (hi_j - q_j).  N is linear in u, so for fixed q the norm is convex in u and its
maximum over the u-box is attained at a vertex; q is sampled uniformly.  N is Pinocchio's Coriolis matrix, which
satisfies Mdot = N + N^T.  The result is a lower bound of the supremum (sampling).
usage: python rho_bar_arm.py [--limits model|narrow] [--samples 4000] [--seed 0]     (writes rho_bar_arm_<limits>.json)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limits', default='model', choices=['model', 'narrow'])
    p.add_argument('--samples', type=int, default=4000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--alpha1', type=float, default=8.)
    a = p.parse_args()
    m = pin.buildSampleModelManipulator()
    d = m.createData()
    if a.limits == 'model':
        lo, hi = m.lowerPositionLimit.copy(), m.upperPositionLimit.copy()
    else:                                             # +-0.5 rad about the start pose Q_A of the paper's arm study
        QA = np.array([0., 0.5, 0., 1.0, 0., 0.3])
        lo, hi = QA - 0.5, QA + 0.5
    rng = np.random.default_rng(a.seed)
    best, arg = 0., None
    signs = np.array(list(itertools.product([0, 1], repeat=m.nv)))
    for _ in range(a.samples):
        q = lo + rng.uniform(0., 1., m.nq) * (hi - lo)
        ulo, uhi = -a.alpha1 * (q - lo), a.alpha1 * (hi - q)
        M = pin.crba(m, d, q)
        M = np.triu(M) + np.triu(M, 1).T
        w, V = np.linalg.eigh(M)
        Mmh = V @ np.diag(w ** -0.5) @ V.T
        lmax = float(w.max())
        Minv = np.linalg.inv(M)
        for sv in signs:
            u = np.where(sv == 1, uhi, ulo)
            N = pin.computeCoriolisMatrix(m, d, q, u).copy()
            r = float(np.linalg.norm(Minv @ N @ Mmh, 2)) * np.sqrt(lmax)
            if r > best:
                best, arg = r, (q.copy(), u.copy())
    # check of the passivity structure at the maximizer (finite differences of M along u)
    q, u = arg
    eps = 1e-6
    Mp = pin.crba(m, d, q + eps * u); Mp = np.triu(Mp) + np.triu(Mp, 1).T
    Mm = pin.crba(m, d, q - eps * u); Mm = np.triu(Mm) + np.triu(Mm, 1).T
    N = pin.computeCoriolisMatrix(m, d, q, u).copy()
    skew = float(np.max(np.abs((Mp - Mm) / (2 * eps) - N - N.T)) / max(1., float(np.max(np.abs(N)))))
    out = dict(limits=a.limits, lo=lo.tolist(), hi=hi.tolist(), alpha1=a.alpha1, samples=a.samples, seed=a.seed,
               rho_bar_lower_bound=best, q_at_max=q.tolist(), u_at_max=u.tolist(), speed_at_max=float(np.max(np.abs(u))),
               skew_check_rel=skew)
    print(json.dumps(out, indent=1))
    json.dump(out, open(Path(__file__).with_name(f'rho_bar_arm_{a.limits}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
