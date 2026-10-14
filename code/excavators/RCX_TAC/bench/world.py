"""Native MuJoCo plant of a machine on the continuum-soil site, wrapped for the extended-compiler runs.

Machine-independent version of the site world of the earlier release (``SiteWorld``, ``TruthMonitor``): compiled plant graph with native
loop closures, floating root body on its support, synthetic hydraulics at the physics rate, continuum soil coupled
explicitly every step, residual-gated forward solves.  The machine adapter supplies the records, the bucket section
and the support/drive of the base; the generated interfaces (``xc``) supply the actuator ports and the channel map.

``TruthMonitor`` evaluates every generated safety function from the *native* body poses at every physics step,
independently of the controller's shadow configuration (rcx.monitor.RowMonitor).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from core.compiler import compile_graph
from core.native import accurate_forward, advance_after_forward, validate_model
from core.reduced import CompiledInterface
from env import scene as env_scene
from env.terrain import ELEV, Z_BASE
from rcx.monitor import RowMonitor

from .export import export_plant
from .hydraulics import HydraulicBank, TrackDrive, TravelController
from .reference import closed_configuration

ROOT = Path(__file__).resolve().parents[1]
MODES = {'none': 0, 'scoop': 1, 'doze': 2}
CONTACT_SETTLE_S = 0.5                 # a machine placed on its ground contacts settles on them: contact statistics start after this time
CODE_FILES = ('rcx/*.py', 'core/*.py', 'env/*.py', 'bench/*.py', 'machines/__init__.py')


def plain(x):
    """Plain JSON value: numpy types converted, non-finite numbers written as null (strict JSON has no Infinity / NaN)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if x is None or isinstance(x, str):
        return x
    raise TypeError(type(x).__name__)


def save_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=1, allow_nan=False))


def code_files(machine_name):
    """Files that determine a result of one machine: shared compiler, extension, environment and bench, and the machine's
    own package with its source records (meshes excluded: render only)."""
    files = []
    for pattern in CODE_FILES + (f'machines/{machine_name}/*.py', f'machines/{machine_name}/source/*.json',
                                 f'machines/{machine_name}/source/*.xml', f'machines/{machine_name}/source/*.urdf'):
        files += sorted(ROOT.glob(pattern))
    return files


def code_hash(machine_name):
    """SHA-256 over ``code_files``.  Every runner takes it when it starts and writes it into its report; the report
    generator compares the hashes with each other and with the present files and fails on a mismatch."""
    digest = hashlib.sha256()
    for f in code_files(machine_name):
        digest.update(str(f.relative_to(ROOT)).encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------- soil coupling
class BucketCoupler:
    """Exact tool kinematics of the current state (scratch MjData, the plant data is not touched), the soil wrench as a
    Cartesian force/torque at the centre of mass of the tool body, and the payload (content) mass properties of that body."""

    def __init__(self, m, body, R_bs, p_bs):
        self.m = m
        self.dk = mujoco.MjData(m)
        self.bid = m.body(body).id
        self.R_bs, self.p_bs = np.asarray(R_bs, float), np.asarray(p_bs, float)
        self.base = dict(mass=float(m.body_mass[self.bid]), ipos=m.body_ipos[self.bid].copy(),
                         iquat=m.body_iquat[self.bid].copy(), inertia=m.body_inertia[self.bid].copy())
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, self.base['iquat'])
        Rb = Rb.reshape(3, 3)
        self.I_body = Rb @ np.diag(self.base['inertia']) @ Rb.T     # tool inertia about its centre of mass, body frame
        chain = []
        b = self.bid
        while True:
            chain.append(b)
            if b == 0:
                break
            b = int(m.body_parentid[b])
        self.ancestors = np.array(chain)
        self.subtree0 = m.body_subtreemass[self.ancestors].copy()
        self.payload = 0.0
        self.energy_injected = 0.0
        self._res = np.zeros(6)

    def kinematics(self, d):
        """Soil frame of the bucket: rotation, origin, velocity of the origin, angular velocity; and the body centre of mass."""
        dk = self.dk
        dk.qpos[:] = d.qpos
        dk.qvel[:] = d.qvel
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        mujoco.mj_comVel(self.m, dk)
        mujoco.mj_objectVelocity(self.m, dk, mujoco.mjtObj.mjOBJ_BODY, self.bid, self._res, 0)
        Rb = dk.xmat[self.bid].reshape(3, 3)
        omega, vb = self._res[:3].copy(), self._res[3:].copy()
        r = Rb @ self.p_bs
        return Rb @ self.R_bs, dk.xpos[self.bid] + r, vb + np.cross(omega, r), omega, dk.xipos[self.bid].copy()

    @staticmethod
    def wrench_at_com(F, T_origin, pos, com):
        """xfrc_applied row: force and torque about the body centre of mass (MuJoCo applies xfrc_applied at xipos)."""
        return np.r_[F, T_origin + np.cross(pos - com, F)]

    def _energy(self, d):
        dk = self.dk
        dk.qpos[:] = d.qpos
        dk.qvel[:] = d.qvel
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        mujoco.mj_crb(self.m, dk)
        mujoco.mj_energyPos(self.m, dk)
        mujoco.mj_energyVel(self.m, dk)
        return float(dk.energy.sum())

    def set_payload(self, d, mass, centroid_s, inertia_s):
        """Tool body = tool + content (rigid payload); centroid and inertia of the content are given in the soil frame.
        Returns the mechanical energy change of this model update at the current state (energy bookkeeping)."""
        m = self.m
        if abs(mass - self.payload) < 1e-9 and mass == 0.0:
            return 0.0
        e0 = self._energy(d)
        centroid = self.p_bs + self.R_bs @ np.asarray(centroid_s, float)
        inertia_c = self.R_bs @ np.asarray(inertia_s, float) @ self.R_bs.T
        mb, cb, Ib = self.base['mass'], self.base['ipos'], self.I_body
        M = mb + mass
        c = (mb * cb + mass * centroid) / M

        def shift(mm, r):
            return mm * ((r @ r) * np.eye(3) - np.outer(r, r))
        I = Ib + shift(mb, cb - c) + (inertia_c + shift(mass, centroid - c) if mass > 0 else 0.)
        I = 0.5 * (I + I.T)
        w, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] = -V[:, 0]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, V.ravel())
        m.body_mass[self.bid] = M
        m.body_ipos[self.bid] = c
        m.body_iquat[self.bid] = q
        m.body_inertia[self.bid] = np.maximum(w, 1e-6)
        m.body_subtreemass[self.ancestors] = self.subtree0 + mass
        self.payload = mass
        de = self._energy(d) - e0
        self.energy_injected += de
        return de


# ---------------------------------------------------------------------------------------------------- coordinates
class CoordinateReader:
    """Values, rates and accelerations of coordinates of the controller graph on the plant state.  A coordinate is a joint
    of the plant or, for a machine whose controller graph has virtual joints (chassis pose, contact anchors), a value
    that the machine adapter evaluates from the plant state (``machine.read_virtual``)."""

    def __init__(self, world, ids):
        self.world = world
        self.ids = list(ids)
        virtual = set(world.machine.virtual_ids(world.ctrl_phys))
        self.vidx = [(k, i) for k, i in enumerate(self.ids) if i in virtual]
        real = [k for k, i in enumerate(self.ids) if i not in virtual]
        self.ridx = np.array(real, int)
        j = np.array([world.m.joint(self.ids[k]).id for k in real], int)
        self.qi, self.vi = world.m.jnt_qposadr[j], world.m.jnt_dofadr[j]
        self.n = len(self.ids)

    def __call__(self):
        d = self.world.d
        if not self.vidx:
            return d.qpos[self.qi].copy(), d.qvel[self.vi].copy()
        q, v = np.zeros(self.n), np.zeros(self.n)
        q[self.ridx], v[self.ridx] = d.qpos[self.qi], d.qvel[self.vi]
        virt = self.world.machine.read_virtual(self.world)
        for k, i in self.vidx:
            q[k], v[k] = virt[i][0], virt[i][1]
        return q, v

    def acceleration(self):
        d = self.world.d
        if not self.vidx:
            return d.qacc[self.vi].copy()
        a = np.zeros(self.n)
        a[self.ridx] = d.qacc[self.vi]
        virt = self.world.machine.read_virtual(self.world, acceleration=True)
        for k, i in self.vidx:
            a[k] = virt[i][2]
        return a


# ---------------------------------------------------------------------------------------------------- plant
class PlantWorld:
    def __init__(self, machine, out_dir, *, dt=0.001, base=(0., 0., 0., 0.), payload_scale=1., capacity_scale=1., backpressure_factor=1.3,
                 soil_params=None, soil_enabled=True, variant=None):
        self.machine = machine
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dt = float(dt)
        self.capacity_scale, self.backpressure_factor = capacity_scale, backpressure_factor
        variant = dict(variant or {})
        self.plant_phys = machine.physical(plant=True, **variant)
        self.ctrl_phys = machine.physical(plant=False, **variant)
        info = self.plant_phys['machine']
        self.tool = info['tool_body']
        if payload_scale != 1.:
            for b in self.plant_phys['bodies']:
                if b['id'] == self.tool:
                    b['mass_kg'] *= payload_scale
                    b['inertia_com_kg_m2'] = (np.asarray(b['inertia_com_kg_m2']) * payload_scale).tolist()
        self.payload_scale = payload_scale
        self.plant_comp = compile_graph(self.plant_phys, 'joint_space')
        self.ci_ctrl = CompiledInterface(self.ctrl_phys, 'joint_space')
        self.grid, self.layout = machine.grid, machine.layout
        self.meta = export_plant(self.out / 'scene.xml', machine, self.plant_phys, self.plant_comp, dt=dt, base=base,
                                 floating=machine.floating, asset_dir=machine.asset_dir(), grid=self.grid, layout=self.layout)
        self.scene = self.out / 'scene.xml'
        self.m = mujoco.MjModel.from_xml_path(str(self.scene))
        self.d = mujoco.MjData(self.m)
        validate_model(self.m)
        m = self.m
        self.soil_enabled = bool(soil_enabled)
        self.bucket = machine.bucket_geometry()
        self.R_bs, self.p_bs = machine.soil_frame()
        self.sm, self.boxes = env_scene.initial_soil(machine.asset_dir(), self.grid, self.layout, self.bucket, soil_params)
        env_scene.load_terrain_into_model(m, self.sm.h)
        self.hadr = int(m.hfield_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, 'soil')])
        self.ground = env_scene.ground_function(self.grid, self.sm.ground)
        self.coupler = BucketCoupler(m, self.tool, self.R_bs, self.p_bs)
        self.tool_id = self.coupler.bid
        self.lip_site = m.site('lip').id
        # root: body fixed to the graph root frame (free joint) and its pose T0 in that frame
        self.root_body = self.meta['root_body']
        self.T0 = np.asarray(self.meta['T0'], float)
        self._root_z = self.T0[:3, :3].T[:, 2].copy()               # z axis of the root frame in the frame of the root body
        self.floating = bool(self.meta['floating'])
        if self.root_body is not None:
            self.root_id = m.body(self.root_body).id
            if self.floating:
                bj = int(m.body_jntadr[self.root_id])
                self.base_qi, self.base_vi = int(m.jnt_qposadr[bj]), int(m.jnt_dofadr[bj])
        # base support / travel drive
        self.drive_data = machine.base_drive(self.plant_phys)
        if self.drive_data is not None:
            dd = self.drive_data
            self.ai_track = np.array([m.actuator('effort_' + k).id for k in dd['left'] + dd['right']])
            self.vi_left = m.jnt_dofadr[np.array([m.joint(k).id for k in dd['left']])]
            self.vi_right = m.jnt_dofadr[np.array([m.joint(k).id for k in dd['right']])]
            self.travel = TravelController(dd['radius'], dd['gauge'], kp=dd['kp'], ki=dd['ki'], forward_bound=dd.get('forward_bound', 0.55),
                                           yaw_rate_bound=dd.get('yaw_rate_bound', 0.20))
            self.drive = TrackDrive(dd['torque_limit'] * capacity_scale, len(dd['left']))
        self.bank = None
        self.xc = None
        self.warnings0 = np.array([w.number for w in self.d.warning])
        self.work = np.zeros(4)                       # actuator, equality constraint, contact, soil wrench
        self.abswork = np.zeros(4)
        self.initial_energy = None
        self.last_payload_R = None
        self.item = self.di = None
        self.soil_F, self.soil_T = np.zeros(3), np.zeros(3)
        self.peak = dict(force_residual=0., closure_m=0., bucket_mass_kg=0., soil_force_N=0., pressure_Pa=0., base_tilt_rad=0.,
                         support_force_N=0., fee_H_N=0.)
        self.steps = 0
        self._tmp_p, self._tmp_r = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.contact, self.contact_peak = None, None

    # ------------------------------------------------------------------ generated interfaces
    def bind(self, xc):
        """Actuator ports and channel map of the generated interfaces (ports of the controller graph)."""
        m, info = self.m, self.ctrl_phys['machine']
        self.xc = xc
        self.port_ids = list(xc.ci.port_ids)
        self.G = xc.G.copy()
        jid = np.array([m.joint(k).id for k in self.port_ids])
        self.port_qi, self.port_vi = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.port_act = np.array([m.actuator('effort_' + k).id for k in self.port_ids])
        self.cyl_idx = np.array([i for i, k in enumerate(self.port_ids) if k in info['cylinders']], int)
        self.rot_idx = np.array([i for i, k in enumerate(self.port_ids) if k in info['rotary']], int)
        if len(self.cyl_idx) + len(self.rot_idx) != len(self.port_ids):
            raise ValueError('every actuator port needs a cylinder or rotary-drive record of the machine')
        self.cyl_records = [info['cylinders'][self.port_ids[i]] for i in self.cyl_idx]
        self.rot_records = [info['rotary'][self.port_ids[i]] for i in self.rot_idx]
        return self

    def joint_addresses(self, ids):
        j = np.array([self.m.joint(k).id for k in ids])
        return self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]

    def reader(self, ids):
        """Reader of controller-graph coordinates on the plant (CoordinateReader)."""
        return CoordinateReader(self, ids)

    def full_coordinates(self, xc, u_task):
        """All independent coordinates of the controller graph for task coordinates: held coordinates (contact anchors) at
        their seed values."""
        return xc.compose(np.asarray(u_task, float))

    def place(self, ci, u0):
        """Put the plant on the closed configuration of the compiled graph for independent coordinates u0."""
        q_closed = closed_configuration(ci, u0)
        virtual = set(self.machine.virtual_ids(self.ctrl_phys))
        # every coordinate of the compiled graph that is a joint of the plant: the tree joints, and the joints that the
        # controller graph cuts while the tree of the plant keeps them (the two graphs need not be cut at the same joints)
        for k, jid in enumerate(ci.coordinate_ids):
            if jid in virtual:
                continue
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if j >= 0:
                self.d.qpos[self.m.jnt_qposadr[j]] = q_closed[k]
            elif jid in ci.tree_ids:
                raise ValueError(f'tree joint {jid} of the controller graph is no joint of the plant')
        if virtual:
            self.machine.place_virtual(self, {jid: q_closed[ci.coordinate_ids.index(jid)] for jid in virtual if jid in ci.coordinate_ids})
        mujoco.mj_forward(self.m, self.d)
        eq = self.d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        return float(np.max(np.abs(self.d.efc_pos[eq]), initial=0.))

    def start_hydraulics(self, channels):
        """Hydraulic states on the static holding efforts of the given channel efforts."""
        port = self.G @ np.asarray(channels, float)
        self.bank = HydraulicBank(self.cyl_records, self.rot_records, self.ctrl_phys['machine']['hydraulics'], port[self.cyl_idx],
                                  port[self.rot_idx], capacity_scale=self.capacity_scale, backpressure_factor=self.backpressure_factor)

    # ------------------------------------------------------------------ measurements
    def root_pose(self, d=None):
        """Rotation and origin of the graph root frame in the site frame."""
        d = self.d if d is None else d
        if self.root_body is None or self.machine.ground_root:
            return np.eye(3), np.zeros(3)
        Rb, pb = d.xmat[self.root_id].reshape(3, 3), d.xpos[self.root_id]
        R = Rb @ self.T0[:3, :3].T
        return R, pb - R @ self.T0[:3, 3]

    def base_state(self):
        """Pose, velocity and acceleration of the graph root frame (linear parts: of its origin, site frame)."""
        d = self.d
        if self.root_body is None or not self.floating or self.machine.ground_root:
            R, p = self.root_pose()
            return dict(position=p, rotation=R, velocity=np.zeros(6), acceleration=np.zeros(6))
        qi, vi = self.base_qi, self.base_vi
        quat = d.qpos[qi + 3:qi + 7]
        Rb = np.zeros(9)
        mujoco.mju_quat2Mat(Rb, quat / np.linalg.norm(quat))
        Rb = Rb.reshape(3, 3)
        v, a = d.qvel[vi:vi + 6], d.qacc[vi:vi + 6]
        w, al = Rb @ v[3:], Rb @ a[3:]                          # angular velocity and acceleration, site frame
        R = Rb @ self.T0[:3, :3].T
        r = -R @ self.T0[:3, 3]                                 # root-frame origin relative to the body origin
        vel = v[:3] + np.cross(w, r)
        acc = a[:3] + np.cross(al, r) + np.cross(w, np.cross(w, r))
        return dict(position=d.qpos[qi:qi + 3] + r, rotation=R, velocity=np.r_[vel, w], acceleration=np.r_[acc, al])

    def machine_pose(self):
        """(x, y, yaw) of the graph root frame on the site."""
        R, p = self.root_pose()
        return np.r_[p[:2], np.arctan2(R[1, 0], R[0, 0])]

    def measured_port_efforts(self):
        """Port efforts as a machine would measure them: cylinder pressure forces (A p_A - B p_B) and drive torque states;
        friction is not included (the plant applies pressure force minus friction)."""
        out = np.zeros(len(self.port_ids))
        out[self.cyl_idx] = self.bank.pressure_force()
        out[self.rot_idx] = self.bank.rotary
        return out

    def applied_port_efforts(self):
        out = np.zeros(len(self.port_ids))
        out[self.cyl_idx] = self.item['force']
        out[self.rot_idx] = self.item['torque']
        return out

    def travel_command(self, target, rate, dt, update_outer=True):
        d = self.d
        R, _ = self.root_pose()
        fwd = R[:, 0]
        fdot = np.cross(d.cvel[self.root_id, :3], fwd)
        yaw_rate = float((fwd[0] * fdot[1] - fwd[1] * fdot[0]) / max(1e-8, fwd[0] ** 2 + fwd[1] ** 2))
        speeds = np.array([d.qvel[self.vi_left].mean(), d.qvel[self.vi_right].mean()])
        return self.travel.evaluate(self.machine_pose(), yaw_rate, speeds, np.asarray(target, float), np.asarray(rate, float), dt,
                                    update_outer=update_outer)

    # ------------------------------------------------------------------ one physics step
    def step(self, channels, track_cmd=None):
        """Hydraulics, soil coupling and native forward solve at the current state.  Returns the diagnostics of the forward
        solve evaluated *before* the advance (state at time t)."""
        m, d, sm, dt = self.m, self.d, self.sm, self.dt
        port = self.G @ np.asarray(channels, float)
        item = self.bank.evaluate(d.qpos[self.port_qi[self.cyl_idx]], d.qvel[self.port_vi[self.cyl_idx]], port[self.cyl_idx],
                                  port[self.rot_idx], d.qvel[self.port_vi[self.rot_idx]], dt)
        d.ctrl[self.port_act[self.cyl_idx]] = item['force']
        d.ctrl[self.port_act[self.rot_idx]] = item['torque']
        self.item = item
        if self.drive_data is not None:
            di = self.drive.evaluate(np.zeros(2) if track_cmd is None else track_cmd, np.r_[d.qvel[self.vi_left], d.qvel[self.vi_right]], dt)
            d.ctrl[self.ai_track] = di['effort']
            self.di = di
        tool, coupler = self.tool_id, self.coupler
        R, pos, v_s, om, com = coupler.kinematics(d)
        if self.soil_enabled:
            F, T = sm.step(dt, R, pos, v_s, om)
            sm.spill(dt, R, pos, v_s, om)
            sm.advance_clods(dt)
            sm.relax()
            mass_now = sm.content * sm.p.rho_loose
            rot_change = 0. if self.last_payload_R is None else float(np.linalg.norm(R - self.last_payload_R))
            if abs(mass_now - coupler.payload) > 0.5 or (mass_now > 0 and rot_change > 0.035) or (mass_now == 0 and coupler.payload != 0):
                mass, cen, Ic = sm.content_mass_properties(R)
                coupler.set_payload(d, mass, cen, Ic)
                self.last_payload_R = R
                dk = coupler.dk                                 # kinematics of the present state (the plant data holds the last solve)
                com = dk.xpos[tool] + dk.xmat[tool].reshape(3, 3) @ m.body_ipos[tool]
            d.xfrc_applied[tool] = coupler.wrench_at_com(F, T, pos, com)
        else:
            F, T = np.zeros(3), np.zeros(3)
        self.soil_F, self.soil_T, self.bucket_R, self.bucket_pos = F, T, R, pos
        if np.any(d.qfrc_applied) or np.any(np.delete(d.xfrc_applied, tool, axis=0)):
            raise ValueError('Unexpected external force')
        solve = accurate_forward(m, d)
        if self.initial_energy is None:
            self.initial_energy = float(d.energy.sum())
        eqmask = d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        closure = float(np.max(np.abs(d.efc_pos[eqmask]), initial=0.))
        pk = self.peak
        pk['force_residual'] = max(pk['force_residual'], solve['final_residual'])
        pk['closure_m'] = max(pk['closure_m'], closure)
        pk['pressure_Pa'] = max(pk['pressure_Pa'], item['pressure_max'])
        pk['soil_force_N'] = max(pk['soil_force_N'], float(np.linalg.norm(F)))
        pk['support_force_N'] = max(pk['support_force_N'], float(sm.last['support']))
        pk['fee_H_N'] = max(pk['fee_H_N'], float(sm.last['H']))
        if self.root_body is not None and self.floating:
            # inclination of the graph root frame (angle of its z axis to the vertical): the base leaving its support
            zr = d.xmat[self.root_id].reshape(3, 3) @ self._root_z
            pk['base_tilt_rad'] = max(pk['base_tilt_rad'], float(np.arctan2(np.hypot(zr[0], zr[1]), zr[2])))
        self.contact = self.machine.contact_truth(self)          # ground contacts of a machine on legs (None otherwise)
        if self.contact is not None and d.time >= CONTACT_SETTLE_S:
            ct = self.contact
            if self.contact_peak is None:
                depth = np.asarray(ct.get('depth', np.zeros(len(ct['normal']))), float)
                self.contact_peak = dict(normal_min_N=ct['normal'].copy(), friction_ratio_max=ct['ratio'].copy(), slip_max_m=ct['slip'].copy(),
                                         lost_steps=np.zeros(len(ct['normal']), int), depth_min_m=depth.copy(), depth_max_m=depth.copy())
            cp = self.contact_peak
            if 'depth' in ct:
                np.minimum(cp['depth_min_m'], ct['depth'], out=cp['depth_min_m'])
                np.maximum(cp['depth_max_m'], ct['depth'], out=cp['depth_max_m'])
            np.minimum(cp['normal_min_N'], ct['normal'], out=cp['normal_min_N'])
            np.maximum(cp['friction_ratio_max'], ct['ratio'], out=cp['friction_ratio_max'])
            np.maximum(cp['slip_max_m'], ct['slip'], out=cp['slip_max_m'])
            cp['lost_steps'] += ct['normal'] <= 0.
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            raise FloatingPointError('Nonfinite native state')
        if np.any(np.array([w.number for w in d.warning]) != self.warnings0):
            raise RuntimeError('MuJoCo warning: ' + str([(i, w.number) for i, w in enumerate(d.warning) if w.number]))
        self.solve, self.closure = solve, closure
        return solve

    def advance(self):
        m, d, dt = self.m, self.d, self.dt
        eqmask = d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        conmask = d.efc_type >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        old_v = d.qvel.copy()
        eqf, cof, tau = d.efc_force[eqmask].copy(), d.efc_force[conmask].copy(), d.qfrc_actuator.copy()
        jacp, jacr = self._tmp_p, self._tmp_r
        mujoco.mj_jacBodyCom(m, d, jacp, jacr, self.tool_id)
        qf_soil = jacp.T @ d.xfrc_applied[self.tool_id, :3] + jacr.T @ d.xfrc_applied[self.tool_id, 3:]
        advance_after_forward(m, d)
        mid = .5 * (old_v + d.qvel)
        jmid = np.zeros(d.nefc)
        mujoco.mj_mulJacVec(m, d, jmid, mid)
        powers = np.array([tau @ mid, eqf @ jmid[eqmask], cof @ jmid[conmask], qf_soil @ mid])
        self.work += dt * powers
        self.abswork += dt * np.abs(powers)
        self.bank.advance(self.item, dt)
        if self.drive_data is not None:
            self.drive.advance(self.di, dt)
        self.steps += 1

    def sync_terrain(self):
        """Changed soil cells into the MuJoCo heightfield (render state); returns the change."""
        idx, hv = self.sm.take_dirty()
        if len(idx):
            self.m.hfield_data[self.hadr + idx] = np.clip((hv - Z_BASE) / ELEV, 0, 1)
        return idx, hv

    def energy_report(self):
        final = float(self.d.energy.sum())
        defect = final - (self.initial_energy or 0.) - float(self.work.sum()) - self.coupler.energy_injected
        return dict(mechanical_work_J=dict(actuators=self.work[0], equality=self.work[1], contact=self.work[2], soil=self.work[3]),
                    payload_energy_injection_J=self.coupler.energy_injected, mechanical_energy_defect_J=defect,
                    mechanical_energy_relative_defect=abs(defect) / max(1., float(self.abswork.sum())))


# ---------------------------------------------------------------------------------------------------- truth monitor
class TruthMonitor(RowMonitor):
    """Safety functions of the generated barrier set evaluated on the native plant state at the physics rate
    (rcx.monitor.RowMonitor fed with the native MuJoCo body poses)."""

    def __init__(self, world: PlantWorld, xc, window_s=None):
        super().__init__(xc, world.dt, window_s)
        self.world = world
        m = world.m
        plan = xc.plan
        self.is_root = np.array([name == plan.root for name in plan.frame_ids])
        # frames of the controller graph that are no bodies of the plant (virtual chains of a machine on ground contacts):
        # their poses come from the machine adapter (plant state), identity where the adapter gives none
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
        self.virtual = [(i, name) for i, name in enumerate(plan.frame_ids) if name != plan.root and name not in names]
        self.body = np.array([0 if (name == plan.root or name not in names) else m.body(name).id for name in plan.frame_ids])
        self.tree = world.reader(xc.ci.tree_ids)
        self.has_support = xc.barriers.support is not None
        # smallest value of the static-support functions for the centre of mass of the plant with its payload (the declared
        # functions use the bodies of the graph; the soil in the bucket is an unmodelled load of the plant)
        self.support_with_payload_min = np.inf

    def poses(self):
        d = self.world.d
        R, p = d.xmat[self.body].reshape(-1, 3, 3).copy(), d.xpos[self.body].copy()
        Rr, pr = self.world.root_pose()
        R[self.is_root], p[self.is_root] = Rr, pr
        if self.virtual:
            given = self.world.machine.virtual_frames(self.world)
            for i, name in self.virtual:
                R[i], p[i] = given.get(name, (np.eye(3), np.zeros(3)))
        return R, p

    def update(self, t):
        """Call after every native forward solve (poses of time t)."""
        R, p = self.poses()
        com = self.world.machine.plant_com(self.world) if self.has_support else None
        if self.has_support:
            full = self.xc.barriers.support_values(p, self.world.machine.plant_com(self.world, payload=True))
            self.support_with_payload_min = min(self.support_with_payload_min, float(np.min(full)))
        return self.update_values(t, R, p, self.tree()[0], com)

    def summary(self):
        out = super().summary()
        if self.has_support:
            out['support_with_payload_h_min'] = self.support_with_payload_min
        return out


def work_metrics(sm, box):
    rs, cs = sm.box_slices(box)
    dh = sm.h[rs, cs] - sm.ground[rs, cs]
    return dict(mean=float(dh.mean()), rms=float(dh.std()), peak=float(dh.max()), low=float(dh.min()), volume=float(dh.sum() * sm.A))
