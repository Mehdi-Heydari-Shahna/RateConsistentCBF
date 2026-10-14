"""Computed facts of a machine that its documentation and its report quote.

    python run.py <machine> facts

Compiles the controller graph and the plant graph of the machine and records what follows from the records alone, so
that no such number has to be typed into a text: size of the graphs, masses, the windows of the task coordinates that
follow from the declared coordinate windows, the bucket (section, struck volume, retained volume against pitch), the
static support of the base (centre of mass and edge margins of the centre of pressure at poses given by the adapter)
and machine-specific findings of the adapter (``machine.facts``), e.g. the closest approach of the bucket to the
machine.  Output: results/<machine>/facts/report.json with the hash of the sources.
"""
from __future__ import annotations

import time

import numpy as np

from core.compiler import compile_graph
from env.soil import SoilModel, SoilParams
from rcx.compiler import compile_extended

from .reference import Planner
from .stability import G, Stability
from .world import code_hash, save_json


def graph_facts(physical, comp):
    s = comp.structure
    moving = [j for j in physical['joints'] if j['type'] != 'fixed']
    return dict(bodies=len(physical['bodies']), joints=len(physical['joints']), moving_joints=len(moving),
                revolute=sum(j['type'] == 'revolute' for j in moving), prismatic=sum(j['type'] == 'prismatic' for j in moving),
                tree_coordinates=int(s['tree_coordinates']), cut_joints=[c['id'] for c in s['cuts']], mobility=len(s['independent']),
                mass_kg=float(sum(b['mass_kg'] for b in physical['bodies'])))


def soil_frame_rotation(pitch):
    """Rotation of the soil frame of a bucket at the given pitch (heading +x, no lateral tilt): columns x_s, y_s, z_s."""
    c, s = np.cos(pitch), np.sin(pitch)
    xs, ys = np.array([c, 0., s]), np.array([0., 1., 0.])
    return np.column_stack((xs, ys, np.cross(xs, ys)))


def bucket_facts(machine):
    b = machine.bucket_geometry()
    p = SoilParams()
    n = 8
    sm = SoilModel.__new__(SoilModel)                             # only the content model is used (no terrain)
    sm.b, sm.p = b, p
    table = []
    for deg in range(-180, -89, 5):
        table.append(dict(pitch_deg=deg, volume_m3=float(SoilModel.capacity(sm, soil_frame_rotation(np.radians(deg))))))
    plate = np.arctan2(b.plate[1], b.plate[0])
    mouth = np.arctan2(b.mouth_normal[1], b.mouth_normal[0])
    return dict(width_m=float(b.width), section_area_m2=float(b.area), struck_volume_m3=float(b.area * b.width),
                loose_density_kg_m3=float(p.rho_loose), struck_mass_kg=float(b.area * b.width * p.rho_loose),
                lip_from_pivot_m=float(np.linalg.norm(b.lip)),
                pitch_floor_plate_horizontal_deg=float(np.degrees(-plate)),
                pitch_mouth_up_deg=float(np.degrees(((np.pi / 2 - mouth) + np.pi) % (2 * np.pi) - np.pi)),
                retained_volume_against_pitch=table)


def support_facts(machine, physical, xc, planner):
    poly = machine.support_polygon(physical)
    if poly is None:
        return None
    st = Stability(physical, xc)
    lip = machine.bucket_geometry().lip
    rows = []
    for label, u, payload in machine.support_poses(planner):
        p, _ = planner.chain.pose(u, lip)
        empty = st.margins(u, poly)
        loaded = st.margins(u, poly, loads=[((0., 0., -payload * G), p)])
        rows.append(dict(pose=label, lip_radius_m=float(np.hypot(p[0], p[1])), lip_azimuth_deg=float(np.degrees(np.arctan2(p[1], p[0]))),
                         lip_height_m=float(p[2]), com_xy_m=empty['com'][:2], margin_empty_m=empty['min_margin'], payload_kg=payload,
                         margin_with_payload_m=loaded['min_margin']))
    return dict(polygon_m=[list(map(float, v)) for v in poly], mass_kg=st.total, poses=rows)


def run(machine):
    sha = code_hash(machine.name)
    out = machine.results_dir() / 'facts'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    t0 = time.perf_counter()
    phys = machine.physical()
    plant = machine.physical(plant=True)
    xc = compile_extended(phys, machine.spec(phys))
    comp_plant = compile_graph(plant, 'joint_space')
    planner = Planner(phys, xc, machine)
    lo, hi, exact = planner.task_box()
    info = phys['machine']
    report = dict(
        status='RUNNING', machine=machine.name, title=machine.title, code_sha256=sha,
        controller_graph=dict(graph_facts(phys, xc.ci.comp), task_ids=list(xc.task.ids), task_labels=list(info.get('task_labels', xc.task.ids)),
                              ports=list(xc.ci.port_ids), channels=[dict(id=c.id, type=c.type, ports=len(c.port_index)) for c in xc.channels],
                              barrier_rows=int(xc.barriers.R), barrier_classes=xc.manifest['barriers']['classes'],
                              seed_closure_residual=float(np.max(np.abs(xc.ci.graph.residual(xc.ci.seed)[0])))),
        plant_graph=graph_facts(plant, comp_plant),
        provenance=phys.get('provenance', {}),
        cylinders={k: {kk: v[kk] for kk in ('kind', 'lower_m', 'upper_m', 'area_A_m2', 'area_B_m2', 'joint', 'connected', 'stroke_margin_m',
                                             'reference_margin_m') if kk in v} for k, v in info.get('cylinders', {}).items()},
        rotary={k: dict(v) for k, v in info.get('rotary', {}).items()},
        reference_windows={k: [None if not np.isfinite(a) else float(a), None if not np.isfinite(b) else float(b)]
                           for k, a, b in zip(planner.task_ids, lo, hi)},
        bucket=bucket_facts(machine),
        support=support_facts(machine, phys, xc, planner))
    report['machine_specific'] = machine.facts(phys, xc, planner)
    report['wall_s'] = time.perf_counter() - t0
    report['status'] = 'COMPLETED'
    save_json(out / 'report.json', report)
    c = report['controller_graph']
    print(f"facts of {machine.name}: controller graph {c['bodies']} bodies, {c['moving_joints']} moving joints, mobility {c['mobility']}, "
          f"{len(c['cut_joints'])} cuts, {c['barrier_rows']} barrier rows; mass {c['mass_kg']:.0f} kg; {report['wall_s']:.0f} s")
    return report
