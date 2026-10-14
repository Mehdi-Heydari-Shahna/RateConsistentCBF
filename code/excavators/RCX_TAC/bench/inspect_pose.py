"""Inspection of task-coordinate vectors on the compiled graph (scenario design aid): bucket pose, smallest value of
every safety function, windowed coordinates.

    from bench.inspect_pose import PoseInspector
    insp = PoseInspector(machine, physical, xc, planner);  insp.show('name', u)
"""
from __future__ import annotations

import numpy as np

from rcx.lift import extended_lift


class PoseInspector:
    def __init__(self, machine, physical, xc, planner):
        self.machine, self.physical, self.xc, self.planner = machine, physical, xc, planner
        self.lip = machine.bucket_geometry().lip

    def rows(self, u):
        """Barrier values at the closed configuration of task coordinates u (root frame at the origin): {pair: h_min}."""
        ci = self.planner.ci
        q = self.planner.closed(u)
        kin = self.xc.plan.evaluate(q[ci.tree_rows], np.zeros(ci.nt), dynamics=False)
        # the planner's graph and the compiled interfaces share records and partition: same tree order
        h = self.xc.barriers.values(kin.R, kin.p, q[ci.tree_rows])
        return {pid: float(np.min(h[idx])) for pid, idx in self.xc.barriers.pair_rows.items()}, q

    def describe(self, u):
        pairs, q = self.rows(u)
        p, phi = self.planner.chain.pose(u, self.lip)
        ids = self.planner.ci.coordinate_ids
        cyl = {jid: float(q[ids.index(jid)]) for jid in self.physical['machine']['cylinders'] if jid in ids}
        return dict(lip=p, pitch_deg=float(np.degrees(phi)), pairs=pairs, cylinders=cyl)

    def show(self, name, u, worst=5):
        d = self.describe(u)
        order = sorted(d['pairs'], key=d['pairs'].get)[:worst]
        print(f"{name:22s} u={np.round(u, 3).tolist()}  lip=({d['lip'][0]:.2f}, {d['lip'][1]:.2f}, {d['lip'][2]:.2f}) pitch={d['pitch_deg']:.1f} deg")
        print('    smallest h: ' + ', '.join(f'{k} {d["pairs"][k]:+.3f}' for k in order))
        return d
