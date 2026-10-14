"""Control / safety / actuator specification records of the extended compiler (schema rcx.control-safety-spec/1.0;
1.1 adds the optional section ``support``).

The specification is the second input of the extended compiler, next to the physical mechanism graph.  Like the
physical records it contains declarative data only:

  control      task coordinates (physical joint ids), task rate Lambda, body velocity-feedback rate k (K_B = k M_B)
  geometry     collision spheres attached to graph bodies; obstacle primitives owned by the inertial frame or a body
  safety       clearance pairs (sphere groups x obstacle, minimum clearance, class-K rates alpha1, alpha2) and
               coordinate limits (strokes, joint limits) with their margin and rates
  actuators    sample time and one record per actuator channel (ports, type, capabilities, force-tracking error bound)
  uncertainty  tool-wrench estimation error bound, control-effectiveness uncertainty, intersample jerk bound
  support      (schema 1.1) ground contacts of a mechanism that stands on its own legs: per contact the three prismatic
               joints of the graph whose coordinates are the position of the contact point (held coordinates: the
               ground holds them, the controller does not command them), friction coefficient and smallest normal
               force; the static support limit (centre of mass inside the support polygon, with a margin and
               class-K rates); weight of the tangential contact forces in the distribution of the nominal efforts

Not accepted (the compiler generates them): Jacobians, distance functions, constraint rows, coordinate partitions,
closed-chain kinematics, force maps.  Every record is closed: ``validate`` rejects any key that the schema does not
list (so generated structure has no place to enter and a misspelt key cannot silently become a default), checks the
records against the physical graph (bodies, joints, actuated ports and actuator groups) and their value ranges.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SCHEMA = 'rcx.control-safety-spec/1.0'
SCHEMA_SUPPORT = 'rcx.control-safety-spec/1.1'             # 1.0 plus the optional section ``support``
FORBIDDEN = {'jacobian', 'jacobians', 'rows', 'constraint_rows', 'distance_function', 'force_map', 'transmission_matrix',
             'independent_ids', 'dependent_ids', 'tree', 'cuts', 'closure', 'barrier_function', 'gradient'}
# Every record is closed: a key that is not listed here is an error (a misspelt key must not silently become a default,
# and generated structure such as Jacobians, partitions or force maps has no place to enter).
KEYS = dict(
    top={'schema', 'name', 'control', 'geometry', 'safety', 'actuators', 'uncertainty', 'provenance', 'support'},
    support={'contacts', 'centroid', 'tangential_weight', 'force_reserve_N'},
    contact={'id', 'anchor', 'friction', 'normal_force_min_N'},
    centroid={'id', 'margin', 'alpha', 'class'},
    control={'task_ids', 'lambda', 'k', 'nominal_effort_fraction', 'correction_filter_s', 'speed_headroom_gain'},
    geometry={'spheres', 'obstacles'},
    sphere={'id', 'body', 'center', 'radius', 'group'},
    obstacle=dict(plane={'id', 'frame', 'type', 'normal', 'offset'}, box={'id', 'frame', 'type', 'center', 'half', 'rpy'},
                  capsule={'id', 'frame', 'type', 'a', 'b', 'radius'}, sphere={'id', 'frame', 'type', 'center', 'radius'}),
    safety={'clearance', 'coordinate_limits', 'motion_bounds'},
    clearance={'id', 'groups', 'obstacle', 'd_min', 'alpha', 'class', 'activation_distance'},
    limit={'id', 'joint', 'lower', 'upper', 'margin', 'alpha', 'class'},
    motion_bounds={'point_speed_m_s', 'point_acceleration_m_s2', 'row_margin_m_s2'},
    actuators={'sample_time_s', 'channels'},
    channel=dict(hydraulic_cylinder={'id', 'ports', 'type', 'derate', 'speed_fraction', 'force_error', 'response_time', 'bore_m', 'rod_m',
                                     'supply_pressure_Pa', 'valve_flow_m3_s', 'min_pressure_Pa', 'back_pressure_factor', 'back_pressure_cap'},
                 rotary_torque={'id', 'ports', 'type', 'derate', 'speed_fraction', 'force_error', 'response_time', 'torque_limit_Nm',
                                'speed_limit_rad_s'}),
    uncertainty={'tool_body', 'tool_point', 'wrench_force_N', 'wrench_moment_Nm', 'load_relative', 'effectiveness', 'jerk_m_s3'})
REQUIRED = dict(
    top={'schema', 'control', 'actuators'}, control={'task_ids'}, sphere={'id', 'body', 'center', 'radius', 'group'},
    obstacle=dict(plane={'id', 'type', 'normal', 'offset'}, box={'id', 'type', 'center', 'half'}, capsule={'id', 'type', 'a', 'b', 'radius'},
                  sphere={'id', 'type', 'center', 'radius'}),
    clearance={'id', 'groups', 'obstacle', 'alpha'}, limit={'id', 'joint', 'alpha'}, actuators={'sample_time_s', 'channels'},
    support={'contacts'}, contact={'id', 'anchor', 'friction'}, centroid={'id', 'alpha'},
    channel=dict(hydraulic_cylinder={'id', 'ports', 'type', 'bore_m', 'rod_m', 'supply_pressure_Pa', 'valve_flow_m3_s'},
                 rotary_torque={'id', 'ports', 'type', 'torque_limit_Nm', 'speed_limit_rad_s'}))


class SpecError(ValueError):
    pass


def _walk(node, path=''):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path + '/' + str(k), k
            yield from _walk(v, path + '/' + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f'{path}[{i}]')


def _closed(record, allowed, required, where):
    if not isinstance(record, dict):
        raise SpecError(f'{where}: a record (mapping) is expected')
    unknown = sorted(set(record) - set(allowed))
    if unknown:
        raise SpecError(f'{where}: unknown keys {unknown} (allowed: {sorted(allowed)})')
    missing = sorted(set(required) - set(record))
    if missing:
        raise SpecError(f'{where}: missing keys {missing}')


def _unique(ids, what):
    seen, dup = set(), set()
    for i in ids:
        (dup if i in seen else seen).add(i)
    if dup:
        raise SpecError(f'{what} ids must be unique: {sorted(dup)}')


def _vector(value, n, where):
    try:
        v = np.asarray(value, float)
    except (TypeError, ValueError):
        raise SpecError(f'{where}: numeric vector of length {n} expected') from None
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise SpecError(f'{where}: finite vector of length {n} expected')
    return v


def _rigid_clusters(physical):
    """Bodies joined by fixed joints, directly or through the root frame of the records: their relative pose never
    changes."""
    parent = {b['id']: b['id'] for b in physical['bodies']}
    for j in physical['joints']:                             # frames that are not bodies (the root of the records)
        for key in ('body_a', 'body_b'):
            if j.get(key) is not None:
                parent.setdefault(j[key], j[key])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for j in physical['joints']:
        a, b = j.get('body_a'), j.get('body_b')
        if j['type'] == 'fixed' and a in parent and b in parent:
            parent[find(a)] = find(b)
    return {b: find(b) for b in parent}


def validate(spec, physical):
    """Checks the declarative specification against the physical records; raises ``SpecError`` with the offending record."""
    if not isinstance(spec, dict) or spec.get('schema') not in (SCHEMA, SCHEMA_SUPPORT):
        raise SpecError(f'schema must be {SCHEMA} or {SCHEMA_SUPPORT}')
    if 'support' in spec and spec['schema'] != SCHEMA_SUPPORT:
        raise SpecError(f'the section support needs schema {SCHEMA_SUPPORT}')
    bad = sorted({p for p, k in _walk(spec) if str(k).lower() in FORBIDDEN})
    if bad:
        raise SpecError(f'Specification must not pre-author generated structure: {bad}')
    _closed(spec, KEYS['top'], REQUIRED['top'], 'specification')
    bodies = {b['id'] for b in physical['bodies']}
    joints = {j['id']: j for j in physical['joints']}
    moving = {k for k, j in joints.items() if j['type'] != 'fixed'}
    cluster = _rigid_clusters(physical)
    # control objective
    ctrl = spec['control']
    _closed(ctrl, KEYS['control'], REQUIRED['control'], 'control')
    if any(t not in moving for t in ctrl['task_ids']) or len(set(ctrl['task_ids'])) != len(ctrl['task_ids']):
        raise SpecError('task_ids must be distinct moving physical joints')
    if np.any(np.asarray(ctrl.get('lambda', 12.), float) <= 0.):
        raise SpecError('control.lambda must be positive')
    k = ctrl.get('k', 12.)
    if isinstance(k, dict):
        if set(k) - bodies - {'default'}:
            raise SpecError(f"control.k: unknown bodies {sorted(set(k) - bodies - {'default'})}")
        k = list(k.values())
    if np.any(np.asarray(k, float) < 0.):
        raise SpecError('control.k must be non-negative')
    if not 0. < float(ctrl.get('nominal_effort_fraction', 1.)) <= 1.:
        raise SpecError('control.nominal_effort_fraction must be in (0, 1]')
    if float(ctrl.get('correction_filter_s', 0.03)) < 0. or float(ctrl.get('speed_headroom_gain', 0.)) < 0.:
        raise SpecError('control.correction_filter_s and control.speed_headroom_gain must be non-negative')
    # collision geometry
    geo = spec.get('geometry', {})
    _closed(geo, KEYS['geometry'], (), 'geometry')
    sph = geo.get('spheres', [])
    for s in sph:
        _closed(s, KEYS['sphere'], REQUIRED['sphere'], f"sphere {s.get('id') if isinstance(s, dict) else s}")
        _vector(s['center'], 3, f"sphere {s['id']}: center")
        if s['body'] not in bodies or not float(s['radius']) >= 0.:
            raise SpecError(f"sphere {s['id']}: unknown body or negative radius")
    _unique([s['id'] for s in sph], 'sphere')
    obs = {}
    for o in geo.get('obstacles', []):
        kind = o.get('type') if isinstance(o, dict) else None
        if kind not in KEYS['obstacle']:
            raise SpecError(f"obstacle {o.get('id') if isinstance(o, dict) else o}: unsupported type {kind}")
        _closed(o, KEYS['obstacle'][kind], REQUIRED['obstacle'][kind], f"obstacle {o.get('id')}")
        if o.get('frame', 'world') != 'world' and o['frame'] not in bodies:
            raise SpecError(f"obstacle {o['id']}: unknown owner frame")
        where = f"obstacle {o['id']}"
        if kind == 'plane':
            if not np.linalg.norm(_vector(o['normal'], 3, where + ': normal')) > 0. or not np.isfinite(float(o['offset'])):
                raise SpecError(f'{where}: the normal must be non-zero and the offset finite')
        elif kind == 'box':
            _vector(o['center'], 3, where + ': center')
            _vector(o.get('rpy', (0., 0., 0.)), 3, where + ': rpy')
            if not np.all(_vector(o['half'], 3, where + ': half') > 0.):
                raise SpecError(f'{where}: half sizes must be positive')
        elif kind == 'capsule':
            _vector(o['a'], 3, where + ': a')
            _vector(o['b'], 3, where + ': b')
            if not float(o['radius']) >= 0.:
                raise SpecError(f'{where}: radius must be non-negative')
        else:
            _vector(o['center'], 3, where + ': center')
            if not float(o['radius']) >= 0.:
                raise SpecError(f'{where}: radius must be non-negative')
        obs[o['id']] = o
    _unique([o['id'] for o in geo.get('obstacles', [])], 'obstacle')
    groups = {s['group'] for s in sph}
    # safety limits
    saf = spec.get('safety', {})
    _closed(saf, KEYS['safety'], (), 'safety')
    for c in saf.get('clearance', []):
        where = f"clearance {c.get('id') if isinstance(c, dict) else c}"
        _closed(c, KEYS['clearance'], REQUIRED['clearance'], where)
        if c['obstacle'] not in obs or not c['groups'] or not set(c['groups']) <= groups:
            raise SpecError(f'{where}: unknown obstacle or sphere group')
        if len(c['alpha']) != 2 or not (float(c['alpha'][0]) > 0 and float(c['alpha'][1]) > 0) or float(c.get('d_min', 0.)) < 0:
            raise SpecError(f'{where}: two positive rates alpha and a non-negative d_min are required')
        if 'activation_distance' in c and not float(c['activation_distance']) > 0.:
            raise SpecError(f'{where}: activation_distance must be positive')
        owner = obs[c['obstacle']].get('frame', 'world')
        if owner != 'world':
            fixed = sorted(s['id'] for s in sph if s['group'] in c['groups'] and cluster[s['body']] == cluster[owner])
            if fixed:
                raise SpecError(f'{where}: spheres {fixed} are rigidly attached to the owner body of the obstacle (constant distance)')
    for l in saf.get('coordinate_limits', []):
        where = f"limit {l.get('id') if isinstance(l, dict) else l}"
        _closed(l, KEYS['limit'], REQUIRED['limit'], where)
        if l['joint'] not in moving:
            raise SpecError(f'{where}: unknown joint')
        lo, hi, margin = l.get('lower'), l.get('upper'), float(l.get('margin', 0.))
        if lo is None and hi is None:
            raise SpecError(f'{where}: at least one of lower, upper is required')
        if margin < 0.:
            raise SpecError(f'{where}: margin must be non-negative')
        if lo is not None and hi is not None and not float(hi) - float(lo) > 2 * margin:
            raise SpecError(f'{where}: empty admissible interval')
        if len(l['alpha']) != 2 or not (float(l['alpha'][0]) > 0 and float(l['alpha'][1]) > 0):
            raise SpecError(f'{where}: two positive rates alpha are required')
    _unique([c['id'] for c in saf.get('clearance', [])] + [l['id'] for l in saf.get('coordinate_limits', [])], 'clearance / limit')
    mb = saf.get('motion_bounds', {})
    _closed(mb, KEYS['motion_bounds'], (), 'safety.motion_bounds')
    if any(not float(v) >= 0. for v in mb.values()):
        raise SpecError('safety.motion_bounds must be non-negative')
    # actuator capabilities
    act = spec['actuators']
    _closed(act, KEYS['actuators'], REQUIRED['actuators'], 'actuators')
    if not float(act['sample_time_s']) > 0:
        raise SpecError('sample time must be positive')
    declared = {a['joint_id'] for a in physical['actuators']}
    graph_groups = {g: [p for p in ports if p in declared] for g, ports in physical.get('actuator_groups', {}).items()}
    used = []
    for ch in act['channels']:
        kind = ch.get('type') if isinstance(ch, dict) else None
        where = f"channel {ch.get('id') if isinstance(ch, dict) else ch}"
        if kind not in KEYS['channel']:
            raise SpecError(f'{where}: unsupported type {kind}')
        _closed(ch, KEYS['channel'][kind], REQUIRED['channel'][kind], where)
        if not ch['ports'] or not set(ch['ports']) <= declared:
            raise SpecError(f'{where}: ports must be actuated joints of the graph')
        if ch['id'] in graph_groups and sorted(ch['ports']) != sorted(graph_groups[ch['id']]):
            raise SpecError(f"{where}: ports {sorted(ch['ports'])} differ from the actuator group of the graph {sorted(graph_groups[ch['id']])}")
        if not 0. < float(ch.get('derate', 1.)) <= 1. or not 0. < float(ch.get('speed_fraction', 1.)) <= 1.:
            raise SpecError(f'{where}: derate and speed_fraction must be in (0, 1]')
        if float(ch.get('force_error', 0.)) < 0. or float(ch.get('response_time', 0.)) < 0.:
            raise SpecError(f'{where}: force_error and response_time must be non-negative')
        if kind == 'hydraulic_cylinder':
            if not (float(ch['bore_m']) > float(ch['rod_m']) >= 0. and float(ch['supply_pressure_Pa']) > 0. and float(ch['valve_flow_m3_s']) > 0.):
                raise SpecError(f'{where}: bore > rod >= 0, supply pressure > 0 and valve flow > 0 are required')
            if float(ch.get('min_pressure_Pa', 0.)) < 0. or float(ch.get('back_pressure_factor', 1.)) < 0. or not 0. < float(ch.get('back_pressure_cap', 1.)) <= 1.:
                raise SpecError(f'{where}: min_pressure_Pa, back_pressure_factor >= 0 and back_pressure_cap in (0, 1] are required')
        elif not (float(ch['torque_limit_Nm']) > 0. and float(ch['speed_limit_rad_s']) > 0.):
            raise SpecError(f'{where}: torque and speed limits must be positive')
        used += list(ch['ports'])
    _unique([ch['id'] for ch in act['channels']], 'channel')
    if len(set(used)) != len(used):
        raise SpecError('an actuator port belongs to more than one channel')
    # declared uncertainty
    unc = spec.get('uncertainty', {})
    _closed(unc, KEYS['uncertainty'], (), 'uncertainty')
    if not 0. <= float(unc.get('effectiveness', 0.)) < 1.:
        raise SpecError('effectiveness uncertainty must be in [0, 1)')
    if unc.get('tool_body') is not None and unc['tool_body'] not in bodies:
        raise SpecError('uncertainty.tool_body must be a graph body')
    if 'tool_point' in unc:
        _vector(unc['tool_point'], 3, 'uncertainty.tool_point')
    if any(float(unc.get(key, 0.)) < 0. for key in ('wrench_force_N', 'wrench_moment_Nm', 'load_relative', 'jerk_m_s3')):
        raise SpecError('uncertainty bounds must be non-negative')
    if unc.get('tool_body') is None and (float(unc.get('wrench_force_N', 0.)) > 0. or float(unc.get('wrench_moment_Nm', 0.)) > 0.):
        raise SpecError('uncertainty: a tool wrench bound needs tool_body (the body on which the unknown wrench acts)')
    # ground support
    if 'support' in spec:
        sup = spec['support']
        _closed(sup, KEYS['support'], REQUIRED['support'], 'support')
        if not isinstance(sup['contacts'], (list, tuple)) or not sup['contacts']:
            raise SpecError('support: at least one contact is required')
        root = 'world'                                           # root frame of the records (generated by the compiler)
        held = []
        for c in sup['contacts']:
            where = f"contact {c.get('id') if isinstance(c, dict) else c}"
            _closed(c, KEYS['contact'], REQUIRED['contact'], where)
            anchor = list(c['anchor'])
            if len(anchor) != 3 or any(a not in moving or joints[a]['type'] != 'prismatic' for a in anchor):
                raise SpecError(f'{where}: anchor must name three prismatic joints of the graph')
            if joints[anchor[0]]['body_a'] != root or joints[anchor[1]]['body_a'] != joints[anchor[0]]['body_b'] \
                    or joints[anchor[2]]['body_a'] != joints[anchor[1]]['body_b']:
                raise SpecError(f'{where}: the anchor joints must form a chain that starts at the root frame of the records')
            # x, y and z translation of the root frame with all attachment frames at the identity: the anchor coordinates are
            # then the position of the contact point (origin of the last anchor body) in the root frame
            axes = np.array([np.asarray(joints[a]['axis'], float) for a in anchor])
            if any(not np.allclose(np.asarray(joints[a][T], float), np.eye(4), atol=1e-12) for a in anchor for T in ('T_AJ', 'T_BJ')) \
                    or not np.allclose(axes, np.eye(3), atol=1e-12):
                raise SpecError(f'{where}: the anchor joints must be the x, y and z translation of the root frame, in this order, '
                                'with attachment frames at the identity')
            # point contact: the last anchor body is joined to the mechanism by three revolute joints in series whose axes pass
            # through the contact point; such a chain transmits a force and no moment, so that the generalized forces of the
            # anchor are the contact force
            body, prev = joints[anchor[2]]['body_b'], anchor[2]
            for k in range(3):
                nxt = [j for j in physical['joints'] if j['id'] != prev and body in (j.get('body_a'), j.get('body_b'))]
                if len(nxt) != 1 or nxt[0]['type'] != 'revolute':
                    raise SpecError(f'{where}: the anchor must be joined to the mechanism by three revolute joints in series (point contact)')
                j = nxt[0]
                here, there, other = ('T_AJ', 'T_BJ', 'body_b') if j['body_a'] == body else ('T_BJ', 'T_AJ', 'body_a')
                if np.linalg.norm(np.asarray(j[here], float)[:3, 3]) > 1e-12 or (k < 2 and np.linalg.norm(np.asarray(j[there], float)[:3, 3]) > 1e-12):
                    raise SpecError(f'{where}: the axes of the three revolute joints behind the anchor must pass through the contact point')
                body, prev = j[other], j['id']
            if not float(c['friction']) > 0. or float(c.get('normal_force_min_N', 0.)) < 0.:
                raise SpecError(f'{where}: friction must be positive and normal_force_min_N non-negative')
            held += anchor
        _unique([c['id'] for c in sup['contacts']], 'contact')
        if len(set(held)) != len(held) or set(held) & set(ctrl['task_ids']):
            raise SpecError('support: an anchor joint belongs to one contact only and is not a task coordinate')
        if 'centroid' in sup:
            cen = sup['centroid']
            _closed(cen, KEYS['centroid'], REQUIRED['centroid'], 'support.centroid')
            if len(sup['contacts']) < 3:
                raise SpecError('support.centroid needs at least three contacts (a support polygon)')
            if len(cen['alpha']) != 2 or not (float(cen['alpha'][0]) > 0 and float(cen['alpha'][1]) > 0) or float(cen.get('margin', 0.)) < 0:
                raise SpecError('support.centroid: two positive rates alpha and a non-negative margin are required')
            if cen['id'] in {c['id'] for c in saf.get('clearance', [])} | {l['id'] for l in saf.get('coordinate_limits', [])}:
                raise SpecError('support.centroid: id is already used by a clearance pair or a coordinate limit')
        if float(sup.get('tangential_weight', 1.)) < 0. or float(sup.get('force_reserve_N', 0.)) < 0.:
            raise SpecError('support: tangential_weight and force_reserve_N must be non-negative')
    return True


def held_ids(spec):
    """Anchor joints of the declared contacts (coordinates that the ground holds), contact by contact."""
    return [a for c in spec.get('support', {}).get('contacts', []) for a in c['anchor']]


def digest(obj):
    def plain(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        raise TypeError(type(x).__name__)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=plain).encode()).hexdigest()
