"""MJCF export of a compiled plant graph with native loop closures on the construction-site environment.

Machine independent (generalised from the MJCF export of the earlier release): the body tree is the compiled spanning tree of the plant
records, placed at the compiled closed seed (joint ``ref`` = seed), so that MuJoCo's ``connect`` equalities (two point
pairs per cut revolute joint: joint origin and a point 0.1 m along the axis) are anchored on the closed configuration.
No body, inertia or joint is invented beyond the physical records; geometry (render meshes, support surrogates) comes
from the machine adapter and is massless.

Root: the body that the records fix to the graph root frame carries the free joint (floating machine) and is placed so
that the graph root frame has the requested site pose; with ``floating=False`` it is welded to the site instead.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from core.loop_graph import TreeWorld
from env import scene as env_scene

CUT_LEVER = 0.1


def fmt(x):
    return ' '.join(format(float(v), '.17g') for v in np.ravel(x))


def quat_of(R):
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat()          # x y z w
    return fmt([q[3], q[0], q[1], q[2]])


def _inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def site_pose(x=0., y=0., z=0., yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def export_plant(path, machine, physical, comp, *, dt=0.001, base=(0., 0., 0., 0.), floating=True, asset_dir=None, grid=None, layout=None,
                 environment=True, cone='elliptic', solver_tolerance=1e-14, gravity=None):
    """Writes the scene and returns its description: root body, pose of the root body in the graph root frame (T0),
    cut closure gaps at the seed, names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = comp.structure
    cids = list(s['coordinate_ids'])
    seed = np.asarray(s['seed_closed'], float)
    bodies = {b['id']: b for b in physical['bodies']}
    jrec = {j['id']: j for j in physical['joints']}
    passive = physical.get('machine', {}).get('passive_joints', {})
    edges_by_parent = {}
    for e in s['edges']:
        edges_by_parent.setdefault(e['parent'], []).append(e)

    root = ET.Element('mujoco', model=f'{machine.name}_compiled_plant')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false', meshdir=str(machine.mesh_dir()))
    opt = ET.SubElement(root, 'option', timestep=str(dt), gravity=fmt(physical['gravity_m_s2'] if gravity is None else gravity),
                        integrator='implicitfast', solver='Newton', jacobian='sparse', cone=cone, iterations='100', ls_iterations='30',
                        tolerance=str(solver_tolerance))
    for key, value in machine.plant_option.items():           # declared by the machine (bench.machine.Machine.plant_option)
        opt.set(key, str(value))
    ET.SubElement(opt, 'flag', contact='enable', energy='enable', autoreset='disable', warmstart='enable')
    ET.SubElement(root, 'size', memory='512M')
    vis = ET.SubElement(root, 'visual')
    ET.SubElement(vis, 'global', offwidth='1440', offheight='900')
    ET.SubElement(root, 'statistic', center='3 3 1', extent='22')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', solref='.008 1', solimp='.99 .99 .001 .5 2', friction='.65 .003 .008', condim='6', margin='0', gap='0')
    ET.SubElement(default, 'equality', solref='.004 1', solimp='.9999 .9999 .001 .5 2')
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    machine.add_assets(asset)
    body_elems = {}

    def _full(I):
        I = np.asarray(I, float)
        return [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]

    def add_body(parent_elem, bid, T):
        b = bodies[bid]
        elem = ET.SubElement(parent_elem, 'body', name=bid, pos=fmt(T[:3, 3]), quat=quat_of(T[:3, :3]))
        ET.SubElement(elem, 'inertial', pos=fmt(b['com_m']), mass=format(float(b['mass_kg']), '.17g'), fullinertia=fmt(_full(b['inertia_com_kg_m2'])))
        body_elems[bid] = elem
        return elem

    def edge_transform(e):
        k = e['coord']
        q = seed[k] if k is not None else 0.
        Q = np.eye(4)
        if e['kind'] == 'revolute':
            Q[:3, :3] = Rotation.from_rotvec(np.asarray(e['axis'], float) * q).as_matrix()
        elif e['kind'] == 'prismatic':
            Q[:3, 3] = np.asarray(e['axis'], float) * q
        return np.asarray(e['enter'], float) @ Q @ np.asarray(e['leave'], float), q

    def add_joint(celem, e, q):
        linv = _inv(np.asarray(e['leave'], float))
        anchor = linv[:3, 3]
        axis_c = linv[:3, :3] @ np.asarray(e['axis'], float)
        jid = cids[e['coord']]
        extra = {k: format(float(v), '.17g') for k, v in passive.get(jid, {}).items()}
        attrs = dict(name=jid, type='hinge' if e['kind'] == 'revolute' else 'slide', pos=fmt(anchor), axis=fmt(axis_c),
                     ref=format(float(q), '.17g'), damping='0', armature='0', limited='false')
        attrs.update(extra)
        ET.SubElement(celem, 'joint', **attrs)

    def recurse(bid, elem):
        for e in edges_by_parent.get(bid, []):
            T, q = edge_transform(e)
            celem = add_body(elem, e['child'], T)
            if e['coord'] is not None:
                add_joint(celem, e, q)
            machine.add_geoms(e['child'], celem, physical)
            recurse(e['child'], celem)

    T_site = site_pose(*base)
    root_body, T0 = None, np.eye(4)
    for e in edges_by_parent.get(s['root'], []):
        T, q = edge_transform(e)
        celem = add_body(world, e['child'], T_site @ T)
        if e['kind'] == 'fixed':
            if root_body is not None:
                raise ValueError('more than one body is fixed to the graph root frame')
            root_body, T0 = e['child'], T
            if floating:
                ET.SubElement(celem, 'freejoint', name='floating_root')
        else:
            if not np.allclose(T_site, np.eye(4)):
                raise ValueError('a graph whose root joints move must be exported at the identity site pose')
            add_joint(celem, e, q)
        machine.add_geoms(e['child'], celem, physical)
        recurse(e['child'], celem)

    # ---------------------------------------------------------------- tool sites and carried-soil mesh
    tool = physical['machine']['tool_body']
    R_bs, p_bs = machine.soil_frame()
    bucket = machine.bucket_geometry()
    te = body_elems[tool]
    for name, xz, rgba in (('lip', bucket.lip, '1 0 0 1'), ('heel', bucket.heel, '0 0 1 1')):
        ET.SubElement(te, 'site', name=name, pos=fmt(p_bs + R_bs @ np.array([xz[0], 0., xz[1]])), size='.03', rgba=rgba, group='5')
    ET.SubElement(te, 'site', name='soil_frame', pos=fmt(p_bs), quat=quat_of(R_bs), size='.03', rgba='0 1 0 1', group='5')

    # ---------------------------------------------------------------- loop closures (cuts)
    equality = ET.SubElement(root, 'equality')
    poses = TreeWorld(comp).poses(seed)
    cut_meta = []
    for cut in s['cuts']:
        rec = jrec[cut['id']]
        TA, TB = np.asarray(rec['T_AJ'], float), np.asarray(rec['T_BJ'], float)
        axis = np.asarray(rec['axis'], float)
        gap = []
        for k, dist in enumerate((0., CUT_LEVER)):
            pa = TA[:3, 3] + TA[:3, :3] @ (dist * axis)
            pb = TB[:3, 3] + TB[:3, :3] @ (dist * axis)
            wa = poses[rec['body_a']][:3, :3] @ pa + poses[rec['body_a']][:3, 3]
            wb = poses[rec['body_b']][:3, :3] @ pb + poses[rec['body_b']][:3, 3]
            gap.append(float(np.linalg.norm(wa - wb)))
            ET.SubElement(equality, 'connect', name=f'cut_{cut["id"]}_{k}', body1=rec['body_a'], body2=rec['body_b'], anchor=fmt(pa))
        cut_meta.append(dict(id=cut['id'], body_a=rec['body_a'], body_b=rec['body_b'], seed_gap_m=gap))

    # ---------------------------------------------------------------- actuators (effort ports)
    act = ET.SubElement(root, 'actuator')
    for a in physical['actuators']:
        ET.SubElement(act, 'motor', name='effort_' + a['joint_id'], joint=a['joint_id'], gear='1', ctrllimited='false', forcelimited='false')

    if environment:
        env_scene.ensure_assets(asset_dir, grid, layout)
        env_scene.add_environment(root, asset_dir, grid, layout, statistic=machine.statistic(), support_friction=machine.support_friction)
        env_scene.add_content_mesh(root, te, bucket, R_bs, p_bs)
    else:
        ET.SubElement(world, 'light', directional='true', pos='6 -8 14', dir='-.3 .4 -.8', diffuse='.85 .85 .85')
        ET.SubElement(world, 'geom', name='pad', type='plane', pos='0 0 0', size='40 40 .1', contype='4', conaffinity='11', condim='3',
                      friction='1.0 .003 .008', rgba='.5 .5 .45 1')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return dict(model=f'{machine.name}_compiled_plant', dt=dt, base=list(base), root_body=root_body, floating=bool(floating and root_body is not None),
                T0=T0.tolist(), cuts=cut_meta, tool_body=tool)
