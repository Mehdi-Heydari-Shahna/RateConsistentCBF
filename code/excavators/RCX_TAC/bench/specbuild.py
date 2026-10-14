"""Assembly of a control / safety / actuator specification (schema rcx.control-safety-spec/1.0) from machine data.

Declarative records only (rcx.spec): the machine adapter supplies collision spheres, machine-fixed obstacles and the
self-collision pairs; the site (e-fence planes, buried pipe) is declared by every run; the actuator capabilities are
taken from the cylinder and drive records that travel with the physical records (``physical['machine']``).
"""
from __future__ import annotations

import numpy as np

SCHEMA = 'rcx.control-safety-spec/1.0'
SCHEMA_SUPPORT = 'rcx.control-safety-spec/1.1'


def shell_stations(shell_xz, spacing=0.30):
    """Points on a section polyline at about ``spacing`` arc length (both ends included)."""
    P = np.asarray(shell_xz, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0., np.cumsum(seg)]
    n = max(1, int(np.ceil(s[-1] / spacing)))
    t = np.linspace(0., s[-1], n + 1)
    return np.c_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])]


def bucket_spheres(body, shell_xz, width, R_bs, p_bs, spacing=0.30, radius=0.04):
    """Spheres on the bucket shell: side edges (group bucket_edge) and three inner lines (group bucket_inner)."""
    out = []
    half = 0.5 * width
    R_bs, p_bs = np.asarray(R_bs, float), np.asarray(p_bs, float)
    for k, (x, z) in enumerate(shell_stations(shell_xz, spacing)):
        for name, y, group in (('L', half, 'bucket_edge'), ('R', -half, 'bucket_edge'), ('l', 0.5 * half, 'bucket_inner'), ('c', 0., 'bucket_inner'),
                               ('r', -0.5 * half, 'bucket_inner')):
            c = p_bs + R_bs @ np.array([x, y, z])
            out.append(dict(id=f'bucket_s{k}_{name}', body=body, center=[float(v) for v in c], radius=radius, group=group))
    return out


def channel_records(physical, *, derate=0.95, speed_fraction=0.9, force_error_fraction=0.01, back_pressure_factor=1.3, back_pressure_cap=0.35,
                    skip_prefix=('track',)):
    """Actuator channel records of the specification from the cylinder / drive records of the machine.  The bound of the
    force-loop model error of a channel is ``force_error_fraction`` of its capacity, or the fraction that the record of
    its actuator declares (key ``force_error_fraction``)."""
    info = physical['machine']
    hyd = info['hydraulics']
    declared = {a['joint_id'] for a in physical['actuators']}
    out = []
    for gid, joints in physical['actuator_groups'].items():
        ports = [j for j in joints if j in declared]
        if not ports or gid.startswith(tuple(skip_prefix)):
            continue
        if ports[0] in info['rotary']:
            r = info['rotary'][ports[0]]
            out.append(dict(id=gid, ports=ports, type='rotary_torque', torque_limit_Nm=float(r['limit_Nm']), speed_limit_rad_s=float(r['speed_rad_s']),
                            force_error=float(r.get('force_error_fraction', force_error_fraction)) * float(r['limit_Nm']),
                            response_time=float(r['response_s']), derate=derate,
                            speed_fraction=speed_fraction))
        else:
            c = info['cylinders'][ports[0]]
            if any(info['cylinders'][p]['area_A_m2'] != c['area_A_m2'] or info['cylinders'][p]['area_B_m2'] != c['area_B_m2'] for p in ports):
                raise ValueError(f'channel {gid}: the cylinders of one channel must be equal')
            A, B = float(c['area_A_m2']), float(c['area_B_m2'])
            bore, rod = float(np.sqrt(4. * A / np.pi)), float(np.sqrt(4. * (A - B) / np.pi))
            ps = float(hyd['supply_pressure_Pa'])
            out.append(dict(id=gid, ports=ports, type='hydraulic_cylinder', bore_m=bore, rod_m=rod, supply_pressure_Pa=ps,
                            valve_flow_m3_s=float(c['valve_flow_m3_s']), min_pressure_Pa=float(hyd['minimum_command_pressure_Pa']),
                            back_pressure_factor=back_pressure_factor, back_pressure_cap=back_pressure_cap,
                            force_error=float(c.get('force_error_fraction', force_error_fraction)) * A * ps,
                            response_time=float(hyd['pressure_response_s']), derate=derate,
                            speed_fraction=speed_fraction))
    return out


def stroke_limits(physical, margin=None, alpha=(8., 20.), margin_fraction=0.08, default_margin=0.03):
    """Stroke end-stops of every cylinder of the machine (connected or passive) as coordinate limits.  Margin of a
    cylinder: ``margin`` if given, else the ``stroke_margin_m`` of its record, else ``default_margin``; at most
    ``margin_fraction`` of the stroke."""
    out = []
    moving = {j['id'] for j in physical['joints'] if j['type'] != 'fixed'}
    for jid, c in physical['machine']['cylinders'].items():
        if jid not in moving:
            continue
        m = min(c.get('stroke_margin_m', default_margin) if margin is None else margin, margin_fraction * (c['upper_m'] - c['lower_m']))
        out.append(dict(id=jid, joint=jid, lower=float(c['lower_m']), upper=float(c['upper_m']), margin=float(m), alpha=list(alpha),
                        **{'class': 'stroke'}))
    return out


def site_records(groups, *, floor_z=None, ceiling_z=None, wall_front_x=None, wall_side_y=None, pipe=None, d_fence=0., d_pipe=0.12,
                 alpha_floor=(6., 16.), alpha_pipe=(6., 16.), alpha_ceiling=(3., 16.), alpha_wall=(1.5, 20.)):
    """Obstacles and clearance pairs of a site: planes  z >= floor_z,  z <= ceiling_z,  x <= wall_front_x,  y >= -wall_side_y
    (inertial frame) and ``pipe`` = dict(a=[x, y, z], b=[x, y, z], radius=r).  ``groups``: sphere groups per site element."""
    obstacles, clearance = [], []
    if floor_z is not None:
        obstacles.append(dict(id='floor', frame='world', type='plane', normal=[0., 0., 1.], offset=float(floor_z)))
        clearance.append(dict(id='efence_floor', groups=list(groups['floor']), obstacle='floor', d_min=d_fence, alpha=list(alpha_floor),
                              **{'class': 'efence'}))
    if ceiling_z is not None:
        obstacles.append(dict(id='ceiling', frame='world', type='plane', normal=[0., 0., -1.], offset=-float(ceiling_z)))
        clearance.append(dict(id='efence_ceiling', groups=list(groups['ceiling']), obstacle='ceiling', d_min=d_fence, alpha=list(alpha_ceiling),
                              **{'class': 'efence'}))
    if wall_front_x is not None:
        obstacles.append(dict(id='wall_front', frame='world', type='plane', normal=[-1., 0., 0.], offset=-float(wall_front_x)))
        clearance.append(dict(id='efence_front', groups=list(groups['wall']), obstacle='wall_front', d_min=d_fence, alpha=list(alpha_wall),
                              **{'class': 'efence'}))
    if wall_side_y is not None:
        obstacles.append(dict(id='wall_side', frame='world', type='plane', normal=[0., 1., 0.], offset=-float(wall_side_y)))
        clearance.append(dict(id='efence_side', groups=list(groups['wall']), obstacle='wall_side', d_min=d_fence, alpha=list(alpha_wall),
                              **{'class': 'efence'}))
    if pipe is not None:
        obstacles.append(dict(id='pipe', frame='world', type='capsule', a=[float(v) for v in pipe['a']], b=[float(v) for v in pipe['b']],
                              radius=float(pipe['radius'])))
        clearance.append(dict(id='pipe', groups=list(groups['pipe']), obstacle='pipe', d_min=d_pipe, alpha=list(alpha_pipe),
                              **{'class': 'utility'}))
    return obstacles, clearance


def assemble(physical, *, name, spheres, obstacles, clearance, limits, tool_point, sample_time=0.01, Lambda=12., k=12., wrench_force=0.75e3,
             wrench_moment=0.25e3, load_relative=0.1, effectiveness=0.2, jerk=2.0, point_speed=12., point_acceleration=15., row_margin=40.,
             nominal_effort_fraction=1.0, correction_filter=0.03, speed_headroom=3.0, channel_kw=None, provenance='', support=None):
    info = physical['machine']
    return dict(
        schema=SCHEMA if support is None else SCHEMA_SUPPORT, name=name, **({} if support is None else dict(support=support)),
        control=dict(task_ids=list(info['task_ids']), **{'lambda': Lambda}, k=k, nominal_effort_fraction=nominal_effort_fraction,
                     correction_filter_s=correction_filter, speed_headroom_gain=speed_headroom),
        geometry=dict(spheres=spheres, obstacles=obstacles),
        safety=dict(clearance=clearance, coordinate_limits=limits,
                    motion_bounds=dict(point_speed_m_s=point_speed, point_acceleration_m_s2=point_acceleration, row_margin_m_s2=row_margin)),
        actuators=dict(sample_time_s=sample_time, channels=channel_records(physical, **(channel_kw or {}))),
        uncertainty=dict(tool_body=info['tool_body'], tool_point=[float(v) for v in tool_point], wrench_force_N=wrench_force,
                         wrench_moment_Nm=wrench_moment, load_relative=load_relative, effectiveness=effectiveness, jerk_m_s3=jerk),
        provenance=provenance)
