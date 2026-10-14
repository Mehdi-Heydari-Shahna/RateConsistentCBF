#!/usr/bin/env python3
"""Videos of the runs with the generated safety layer: barrier rows off and complete filter side by side.

    python video.py heap dig_pipe                    # results/heap/videos/heap_dig_pipe.mp4
    python video.py heap mission --stride 4          # 4 x speed
    python video.py heap self --stills 6             # stills of both halves instead of a video

Nothing is simulated here.  Runs recorded with --render (python run.py <machine> hazard ... --render, or
python batch.py <machine> paper --render for the runs of the paper) store, besides
their traces, the soil state at the 40 ms samples (render.npz: terrain changes, bucket content, falling clods); the
video replays machine poses and soil state of those very runs and adds, from the specification of the run:
  the obstacle primitives of the scenario's pairs (amber; red while the pair is violated; declared planes as finite
  patches), their collision spheres up to a radius, and the position of a buried pipe on the present surface
  (yellow dots);
and a read-out, from the traces:
  the safety functions h of those pairs from the plant-side monitor (``pair_h``: evaluated from the native body poses,
  not from the controller model) at the 40 ms samples; a pair that was violated stays marked with its worst value;
  for the filter, whether a barrier row was binding at a control sample since the previous frame, in the
  required-motion problem ("motion shaped") or in the effort QP ("efforts corrected");
  for a machine on declared ground contacts, the smallest normal force of a contact and the largest ratio of tangential
  to normal force of the plant at the sample, and, if the static support limit is among the pairs shown, the polygon of
  the contact points, the declared limit inside it and the ground projection of the centre of mass of the machine.
Rendering uses MUJOCO_GL=egl unless MUJOCO_GL is set.  Cameras, labels and the pairs shown come from the machine
adapter (``video_config``).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SOIL_RGB = np.array([0.36, 0.28, 0.21])
AMBER, RED, CYAN, YELLOW = (1.0, 0.72, 0.10), (0.92, 0.12, 0.10), (0.25, 0.80, 0.95), (1.0, 0.92, 0.15)
TITLE = dict(strokes='Stroke ends and joint limits', self='Protective distance to the machine', efence='E-fence planes',
             dig_floor='Design-depth floor (digging)', dig_pipe='Buried pipe (digging)', mission='Complete site mission',
             support='Static support limit')
UNIT = dict(stroke=('mm', 1e3), joint_limit=('mrad', 1e3))          # read-out unit and factor per class of a pair (default: m)


def cam_from(pos, look):
    pos, look = np.asarray(pos, float), np.asarray(look, float)
    f = look - pos
    dist = float(np.linalg.norm(f))
    return dict(lookat=look, distance=dist, azimuth=float(np.degrees(np.arctan2(f[1], f[0]))), elevation=float(np.degrees(np.arcsin(f[2] / dist))))


class Director:
    """First-order camera follower (lookat, distance, azimuth, elevation)."""

    def __init__(self, tau=1.2):
        self.tau, self.state = tau, None

    def __call__(self, target, dt):
        v = np.r_[target['lookat'], target['distance'], target['azimuth'], target['elevation']]
        if self.state is None:
            self.state = v.copy()
        else:
            d = v - self.state
            d[4] = (d[4] + 180) % 360 - 180
            self.state += (1 - np.exp(-dt / self.tau)) * d
        s = self.state
        return dict(lookat=s[:3], distance=s[3], azimuth=s[4], elevation=s[5])


def relocated_scene(scene_xml):
    """Text of the scene file of a run with the workspace folder of that run replaced by the present one: the scene names
    its mesh folder and its textures by absolute paths, and the parts of a release are unpacked somewhere else."""
    import re
    text = Path(scene_xml).read_text()
    found = re.search(r'"([^"]*?)/(?:machines|results)/', text)
    if found and Path(found.group(1)).resolve() != ROOT:
        text = text.replace(found.group(1) + '/', str(ROOT) + '/')
    return text


class SiteRenderer:
    """Replay of a run scene: machine pose, soil heightfield with the fresh-soil texture, bucket content, clods."""

    def __init__(self, machine, scene_xml, h0, width, height, shadow=4096, samples=4):
        import mujoco
        from env import content_mesh, scene as env_scene
        from env.terrain import ELEV, Z_BASE
        self.mj, self.cm, self.Z_BASE, self.ELEV = mujoco, content_mesh, Z_BASE, ELEV
        m = mujoco.MjModel.from_xml_string(relocated_scene(scene_xml))
        m.vis.quality.shadowsize = shadow
        m.vis.quality.offsamples = samples
        m.vis.global_.offwidth = max(width, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(height, m.vis.global_.offheight)
        self.m, self.d = m, mujoco.MjData(m)
        self.grid = machine.grid
        self.h0 = np.array(h0, float)
        self.h = self.h0.copy()
        env_scene.load_terrain_into_model(m, self.h)
        self.r = mujoco.Renderer(m, height, width)
        self.ctx = self.r._mjr_context
        self.hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, 'soil')
        self.hadr = int(m.hfield_adr[self.hid])
        mujoco.mjr_uploadHField(m, self.ctx, self.hid)
        self.tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, 'soil_tex')
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, 'fresh_tex')
        tw, th, nc = int(m.tex_width[self.tid]), int(m.tex_height[self.tid]), int(m.tex_nchannel[self.tid])
        self.tshape = (th, tw, nc)
        a, b = int(m.tex_adr[self.tid]), int(m.tex_adr[fid])
        self.tex_base = m.tex_data[a:a + th * tw * nc].reshape(th, tw, nc).astype(np.float32)
        self.tex_fresh = m.tex_data[b:b + th * tw * nc].reshape(th, tw, nc).astype(np.float32)
        self.touch_w = np.zeros_like(self.h)             # fresh-soil weight: grows with the local height change, never decays
        self.tex_dirty = self.hf_dirty = False
        # soil content mesh (fixed topology) on the tool body; its vertices are written in the frame of that body
        self.mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, 'soil_content')
        self.gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'soil_content')
        self.vadr, self.nv = int(m.mesh_vertadr[self.mid]), int(m.mesh_vertnum[self.mid])
        self.nadr, self.nn = int(m.mesh_normaladr[self.mid]), int(m.mesh_normalnum[self.mid])
        self.bucket = machine.bucket_geometry()
        self.R_bs, self.p_bs = (np.asarray(x, float) for x in machine.soil_frame())
        self.bid = int(m.geom_bodyid[self.gid])
        if self.nv != len(content_mesh.initial_local(self.bucket)):
            raise RuntimeError('content mesh vertex count changed by the MuJoCo compiler')
        self.normals_per_vertex = self.nn == self.nv
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0                        # support surrogates (rollers, pad) are not drawn

    def apply_terrain(self, idx, hv):
        if len(idx) == 0:
            return
        self.h.ravel()[idx] = hv
        self.m.hfield_data[self.hadr + idx] = np.clip((hv - self.Z_BASE) / self.ELEV, 0, 1)
        tw = self.touch_w.ravel()
        tw[idx] = np.maximum(tw[idx], np.clip(np.abs(hv - self.h0.ravel()[idx]) / 0.02, 0, 1))
        self.hf_dirty = self.tex_dirty = True

    def upload_terrain(self, texture=True):
        from scipy.ndimage import gaussian_filter, zoom
        mj = self.mj
        if self.hf_dirty:
            mj.mjr_uploadHField(self.m, self.ctx, self.hid)
            self.hf_dirty = False
        if texture and self.tex_dirty:
            th, tw, nc = self.tshape
            w = np.clip(gaussian_filter(self.touch_w.astype(np.float32), 2.0) * 1.25, 0, 1)      # soft transition into undisturbed soil
            W = zoom(w, (th / w.shape[0], tw / w.shape[1]), order=1)[::-1, :, None]              # texture rows run from +y down
            tex = self.tex_base * (1 - W) + self.tex_fresh * W
            a = int(self.m.tex_adr[self.tid])
            self.m.tex_data[a:a + th * tw * nc] = np.clip(tex, 0, 255).astype(np.uint8).ravel()
            mj.mjr_uploadTexture(self.m, self.ctx, self.tid)
            self.tex_dirty = False

    def set_pose(self, qpos):
        mj = self.mj
        self.d.qpos[:] = qpos
        mj.mj_kinematics(self.m, self.d)
        mj.mj_comPos(self.m, self.d)
        mj.mj_camlight(self.m, self.d)                   # light directions (else the sun is dark)

    def set_content(self, volume):
        d, m = self.d, self.m
        Rb, pb = d.xmat[self.bid].reshape(3, 3), d.xpos[self.bid]
        Vw = self.cm.vertices_world(self.bucket, Rb @ self.R_bs, pb + Rb @ self.p_bs, volume)
        gR = d.geom_xmat[self.gid].reshape(3, 3)
        Vg = (Vw - d.geom_xpos[self.gid]) @ gR
        m.mesh_vert[self.vadr:self.vadr + self.nv] = Vg
        if self.normals_per_vertex:
            m.mesh_normal[self.nadr:self.nadr + self.nn] = self.cm.vertex_normals(Vg)
        self.mj.mjr_uploadMesh(m, self.ctx, self.mid)

    def add_clods(self, scn, pos, vol, ids, t):
        mj = self.mj
        for p, v, i in zip(pos, vol, ids):
            rng = np.random.default_rng(int(i) + 12345)
            req = (3 * float(v) / (4 * np.pi)) ** (1 / 3)
            scales = np.array([0.72, 0.58, 0.48])
            offs = rng.normal(0, 0.55 * req, (3, 3))
            spin = rng.uniform(2, 7) * t + rng.uniform(0, 6.28)
            for k in range(3):
                if scn.ngeom >= scn.maxgeom:
                    return
                size = req * scales[k] * rng.uniform(0.7, 1.3, 3)
                q = np.zeros(4)
                ax = rng.normal(size=3)
                mj.mju_axisAngle2Quat(q, ax / np.linalg.norm(ax), spin + k)
                mat = np.zeros(9)
                mj.mju_quat2Mat(mat, q)
                rgba = np.r_[SOIL_RGB * rng.uniform(0.8, 1.15), 1.0]
                mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_ELLIPSOID, size, p + offs[k], mat, rgba.astype(np.float32))
                scn.ngeom += 1


class Geometry:
    """Declared geometry of the specification as render primitives at the current pose of the render model."""

    def __init__(self, mj, m, spec, focus, cfg, key):
        self.mj, self.m, self.cfg = mj, m, cfg
        self.obstacles = {o['id']: o for o in spec['geometry']['obstacles']}
        self.pairs = [c for c in spec['safety']['clearance'] if c['id'] in focus]
        groups = {g for c in self.pairs for g in c['groups']}
        rmax = cfg.get('sphere_max_radius', {}).get(key, 0.13)
        self.spheres = [s for s in spec['geometry']['spheres'] if s['group'] in groups and s['radius'] <= rmax]
        self.body = {}
        for name in {o.get('frame', 'world') for o in self.obstacles.values()} | {s['body'] for s in self.spheres}:
            if name != 'world':
                self.body[name] = m.body(name).id
        self.pipe = self.obstacles.get('pipe')
        # static support limit among the pairs shown: polygon of the contact points (sites contact_<id> of the plant scene),
        # the declared limit inside it and the centre of mass of the machine
        sup = spec.get('support', {})
        self.support = None
        if sup.get('centroid') and sup['centroid']['id'] in focus:
            ids = [c['id'] for c in sup['contacts']]
            sites = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f'contact_{i}') for i in ids]
            root = cfg.get('support_body')
            if all(k >= 0 for k in sites) and root is not None:
                self.support = dict(pair=sup['centroid']['id'], sites=sites, margin=float(sup['centroid'].get('margin', 0.)), body=m.body(root).id)
        self.floor_xy = cfg['floor_xy']
        if self.pipe is not None:
            mid = 0.5 * (np.asarray(self.pipe['a'], float) + np.asarray(self.pipe['b'], float))
            self.floor_xy = (float(mid[0]), float(mid[1]))

    def frame(self, d, name):
        if name == 'world':
            return np.eye(3), np.zeros(3)
        b = self.body[name]
        return d.xmat[b].reshape(3, 3), d.xpos[b]

    def _geom(self, scn, kind, size, pos, mat, rgba):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        self.mj.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def add(self, scn, d, violated, surface):
        """Obstacles of the focus pairs (red while the pair is violated), their collision spheres and the surface mark of
        the pipe.  ``violated``: pair id -> bool at this sample; ``surface(x, y)``: current terrain height."""
        from rcx.geometry import rpy_matrix
        mj, cfg = self.mj, self.cfg
        G = mj.mjtGeom
        for c in self.pairs:
            o = self.obstacles[c['obstacle']]
            col = RED if violated.get(c['id'], False) else AMBER
            R, p = self.frame(d, o.get('frame', 'world'))
            if o['type'] == 'box':
                self._geom(scn, G.mjGEOM_BOX, o['half'], p + R @ np.asarray(o['center'], float), R @ rpy_matrix(o.get('rpy', (0., 0., 0.))), (*col, 0.30))
            elif o['type'] == 'capsule':
                a, b = p + R @ np.asarray(o['a'], float), p + R @ np.asarray(o['b'], float)
                solid = o.get('frame', 'world') == 'world'                 # the pipe is a real object, the envelopes are not
                g = self._geom(scn, G.mjGEOM_CAPSULE, [o['radius'], 0., 0.], a, np.eye(3), (*col, 1.0 if solid else 0.30))
                if g is not None:
                    mj.mjv_connector(g, G.mjGEOM_CAPSULE, float(o['radius']), a, b)
            elif o['type'] == 'sphere':
                self._geom(scn, G.mjGEOM_SPHERE, [o['radius'], 0., 0.], p + R @ np.asarray(o['center'], float), np.eye(3), (*col, 0.30))
            elif o['type'] == 'plane':
                n = np.asarray(o['normal'], float)
                off = float(o['offset']) / np.linalg.norm(n)
                n = n / np.linalg.norm(n)
                anchor = np.asarray(cfg['anchor'], float)
                centre = anchor - (anchor @ n - off) * n
                t1 = np.cross([0., 0., 1.], n)
                if np.linalg.norm(t1) < 1e-6:                              # horizontal plane: floor patch under the dig area, ceiling patch
                    t1 = np.array([1., 0., 0.])
                    half = [*cfg['floor_half'], 0.004] if n[2] > 0 else [*cfg['ceiling_half'], 0.01]
                    if n[2] > 0:
                        centre = np.array([self.floor_xy[0], self.floor_xy[1], off])
                else:
                    t1 = t1 / np.linalg.norm(t1)
                    half = [*cfg['wall_half'], 0.01]
                    centre = centre.copy()
                    centre[2] = cfg['wall_half'][1]
                t2 = np.cross(n, t1)
                self._geom(scn, G.mjGEOM_BOX, half, centre, np.c_[t1, t2, n], (*col, 0.34 if n[2] > 0 else 0.16))
        for s in self.spheres:
            R, p = self.frame(d, s['body'])
            self._geom(scn, G.mjGEOM_SPHERE, [s['radius'], 0., 0.], p + R @ np.asarray(s['center'], float), np.eye(3),
                       (*CYAN, 0.95 if s['radius'] <= 0.13 else 0.12))
        if self.support is not None:
            from rcx.barriers import hull_order
            S = self.support
            col = RED if violated.get(S['pair'], False) else AMBER
            P = np.array([d.site_xpos[k][:2] for k in S['sites']])
            P = P[hull_order(P)]
            centre = P.mean(axis=0)

            def ring(points, z, radius, rgba):
                for a, b in zip(points, np.roll(points, -1, axis=0)):
                    g = self._geom(scn, G.mjGEOM_CAPSULE, [radius, 0., 0.], [*a, z], np.eye(3), rgba)
                    if g is not None:
                        mj.mjv_connector(g, G.mjGEOM_CAPSULE, radius, np.r_[a, z], np.r_[b, z])
            ring(P, 0.03, 0.02, (0.9, 0.9, 0.9, 0.9))                    # polygon of the contact points
            # declared limit: every edge moved inwards by the margin (intersection of neighbouring offset edges)
            n = len(P)
            lines = []
            for a, b in zip(P, np.roll(P, -1, axis=0)):
                e = (b - a) / np.linalg.norm(b - a)
                nrm = np.array([-e[1], e[0]])
                if nrm @ (centre - a) < 0.:
                    nrm = -nrm
                lines.append((a + S['margin'] * nrm, e))
            inner = []
            for k in range(n):
                (p0, e0), (p1, e1) = lines[k - 1], lines[k]
                tpar = np.linalg.solve(np.c_[e0, -e1], p1 - p0)
                inner.append(p0 + tpar[0] * e0)
            ring(np.array(inner), 0.03, 0.03, (*col, 0.95))
            com = d.subtree_com[S['body']]
            self._geom(scn, G.mjGEOM_SPHERE, [0.09, 0., 0.], [com[0], com[1], 0.06], np.eye(3), (*CYAN, 1.0))
            g = self._geom(scn, G.mjGEOM_CAPSULE, [0.012, 0., 0.], com, np.eye(3), (*CYAN, 0.6))
            if g is not None:
                mj.mjv_connector(g, G.mjGEOM_CAPSULE, 0.012, np.array([com[0], com[1], 0.06]), np.asarray(com, float))
        if self.pipe is not None:                                           # position of the buried pipe, marked on the present surface
            a, b = np.asarray(self.pipe['a'], float), np.asarray(self.pipe['b'], float)
            for f in np.linspace(0., 1., 13):
                q = a + f * (b - a)
                z = float(surface(q[0], q[1]))
                if z > q[2] + self.pipe['radius'] + 0.03:                   # not where the pipe itself is exposed
                    self._geom(scn, G.mjGEOM_SPHERE, [0.04, 0., 0.], [q[0], q[1], z + 0.02], np.eye(3), (*YELLOW, 1.0))


def hud(img, lines, tag, tag_rgb, footer, state=None):
    """Overlay: tag of the variant, optional state tag (text, colour), read-out lines (text, colour) and a footer line."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, 'RGBA')
    try:
        bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 21)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:                                      # noqa: BLE001
        bold = font = small = ImageFont.load_default()
    W, H = im.size
    pad = 10
    tw = dr.textlength(tag, font=bold)
    dr.rectangle([pad, pad, pad + tw + 20, pad + 34], fill=(*tag_rgb, 235))
    dr.text((pad + 10, pad + 5), tag, fill=(255, 255, 255, 255), font=bold)
    if state is not None:
        x0 = pad + tw + 28
        sw = dr.textlength(state[0], font=font)
        dr.rectangle([x0, pad, x0 + sw + 16, pad + 34], fill=(*state[1], 235))
        dr.text((x0 + 8, pad + 8), state[0], fill=(255, 255, 255, 255), font=font)
    y = pad + 44
    if lines:
        wmax = max(dr.textlength(t, font=font) for t, _ in lines)
        dr.rectangle([pad, y - 4, pad + wmax + 20, y + 23 * len(lines) + 4], fill=(0, 0, 0, 175))
        for text, rgb in lines:
            dr.text((pad + 10, y), text, fill=(*rgb, 255), font=font)
            y += 23
    dr.rectangle([0, H - 26, W, H], fill=(0, 0, 0, 165))
    size = 14
    while dr.textlength(footer, font=small) > W - 2 * pad and size > 9:      # a long footer is set smaller (down to 9 px)
        size -= 1
        try:
            small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        except Exception:                                  # noqa: BLE001
            break
    dr.text((pad, H - 22 + (14 - size) // 2), footer, fill=(255, 255, 255, 235), font=small)
    return np.asarray(im.convert('RGB'))


def run_folder(machine, key, variant):
    return machine.results_dir() / (f'mission_{variant}' if key == 'mission' else Path('hazards') / f'{key}__{variant}')


def load_half(machine, key, variant):
    """Trace, render record and specification of one run."""
    folder = run_folder(machine, key, variant)
    for name in ('trace.npz', 'render.npz', 'spec.json', 'scene.xml', 'report.json'):
        if not (folder / name).exists():
            raise SystemExit(f'{folder / name} is missing: run  python batch.py {machine.name} paper --render  (or run.py with --render) first')
    from tracefile import load_trace
    with load_trace(folder) as z:
        tr = {k: z[k] for k in ('time', 'qpos', 'phase', 'pair_h', 'bucket_mass', 'pair_ids', 'phase_names', 'c_time', 'c_n_active', 'c_velocity_active')}
        if 'contact_true' in z.files:                                       # machine on ground contacts: forces of the plant
            tr['contact_true'] = z['contact_true']
    with np.load(folder / 'render.npz') as z:
        rd = {k: z[k] for k in z.files}
    h0 = rd.pop('h_initial')
    spec = json.loads((folder / 'spec.json').read_text())
    report = json.loads((folder / 'report.json').read_text())
    cls = {c['id']: c.get('class', 'clearance') for c in spec['safety']['clearance']}
    cls.update({c['id']: c.get('class', 'limit') for c in spec['safety']['coordinate_limits']})
    return SimpleNamespace(folder=folder, tr=tr, rd=rd, h0=h0, spec=spec, cls=cls, code_sha256=report.get('code_sha256'),
                           free_motion=bool(report.get('soil', {}).get('free_motion_scenario', False)),
                           pair_ids=[str(x) for x in tr['pair_ids']], names=[str(x) for x in tr['phase_names']])


def render_half(machine, args):
    """One half picture (one variant) as an intermediate video or as stills."""
    import imageio_ffmpeg
    import mujoco
    from PIL import Image
    key, cfg = args.key, machine.video_config()
    data = load_half(machine, key, args.variant)
    out = machine.results_dir() / 'videos' / f'{key}__{args.variant}'
    out.mkdir(parents=True, exist_ok=True)
    tr, rd = data.tr, data.rd
    W, H = args.width // 2, args.height
    sr = SiteRenderer(machine, data.folder / 'scene.xml', data.h0, W, H, shadow=args.shadow, samples=args.samples)
    grid = sr.grid
    focus = [p for p in cfg['focus'][key] if p in data.pair_ids]
    col = [data.pair_ids.index(p) for p in focus]
    geo = Geometry(mujoco, sr.m, data.spec, focus, cfg, key)
    label = cfg.get('labels', {})

    def surface(x, y):
        ix = int(np.clip(round((x - grid.xmin) / grid.dx), 0, grid.n - 1))
        iy = int(np.clip(round((y - grid.ymin) / grid.dx), 0, grid.n - 1))
        return sr.h[iy, ix]
    t = tr['time']
    n = min(len(t), len(rd['content']))
    frames = list(range(0, n, max(1, args.stride)))
    if args.stills:
        frames = [frames[int(round(f * (len(frames) - 1)))] for f in np.linspace(0., 1., args.stills)]
    toff, tidx, th, coff = rd['terrain_off'], rd['terrain_idx'], rd['terrain_h'], rd['clod_off']
    tag, tag_rgb = (('BARRIER ROWS OFF', (150, 30, 30)) if args.variant == 'off' else ('SAFETY FILTER ON', (16, 104, 130)))
    worst = {p: np.inf for p in focus}
    writer = None
    if not args.stills:
        writer = imageio_ffmpeg.write_frames(str(out / 'half.mp4'), (W, H), fps=args.fps, codec='libx264', quality=None,
                                             macro_block_size=8, output_params=['-crf', '13', '-preset', 'veryfast'])
        writer.send(None)
    director = Director(tau=1.2)
    ct = tr['c_time']
    shaped, corrected = tr['c_velocity_active'] > 0, tr['c_n_active'] > 0
    free_motion = data.free_motion
    title = dict(TITLE, **getattr(machine, 'SCENARIO_TITLES', {}))
    applied, last_t, last_k, last_frame_t = 0, None, -1, None
    wall = time.time()
    for count, k in enumerate(frames):
        while applied <= k:
            a, b = int(toff[applied]), int(toff[applied + 1])
            sr.apply_terrain(tidx[a:b], th[a:b])
            applied += 1
        sr.upload_terrain(texture=(count % 3 == 0) or bool(args.stills))
        qpos = tr['qpos'][k].astype(float)
        sr.set_pose(qpos)
        sr.set_content(float(rd['content'][k]))
        a, b = int(coff[k]), int(coff[k + 1])
        name = data.names[int(tr['phase'][k])]
        h_now = tr['pair_h'][k, col].astype(float)
        seg = tr['pair_h'][last_k + 1:k + 1, col].astype(float)        # every sample since the last frame (min over skipped samples)
        last_k = k
        for p, v in zip(focus, seg.min(axis=0) if len(seg) else h_now):
            worst[p] = min(worst[p], float(v))
        cam = cam_from(*machine.camera(key, name, qpos))
        cam = director(cam, 0. if last_t is None else t[k] - last_t) if not args.stills else cam
        last_t = t[k]
        c = sr.cam
        c.lookat[:] = cam['lookat']
        c.distance, c.azimuth, c.elevation = cam['distance'], cam['azimuth'], cam['elevation']
        sr.r.update_scene(sr.d, c, sr.opt)
        scn = sr.r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
        sr.add_clods(scn, rd['clod_pos'][a:b], rd['clod_vol'][a:b], rd['clod_id'][a:b], t[k])
        geo.add(scn, sr.d, {p: bool(v < 0.) for p, v in zip(focus, h_now)}, surface)
        img = sr.r.render().copy()
        lines = [('margin to the declared limit (measured on the plant)', (215, 215, 215))]
        for p, v in zip(focus, h_now):
            unit, scale = UNIT.get(data.cls.get(p), ('m', 1.))
            fmt = (lambda x: f'{scale * x:+.3f} {unit}') if unit == 'm' else (lambda x: f'{scale * x:+.0f} {unit}')
            text = f'{label.get(p, p)}:  {fmt(v)}'
            if v < 0.:
                lines.append((text + '   LIMIT VIOLATED', (255, 90, 80)))
            elif worst[p] < 0.:
                lines.append((text + f'   (was violated: {fmt(worst[p])})', (255, 150, 120)))
            else:
                lines.append((text, (150, 235, 150)))
        if not free_motion:
            lines.append((f'soil in the bucket:  {float(tr["bucket_mass"][k]):.0f} kg', (235, 235, 235)))
        if 'contact_true' in tr:
            F = tr['contact_true'][k].astype(float)
            ratio = np.linalg.norm(F[:, :2], axis=1) / np.maximum(F[:, 2], 1.)
            lines.append((f'ground contacts: normal force min {F[:, 2].min() / 1e3:.0f} kN, tangential / normal max {ratio.max():.2f}',
                          (235, 235, 235)))
        if geo.support is not None:
            lines.append(('white: contact polygon   amber: declared limit   cyan: centre of mass', (215, 215, 215)))
        if geo.pipe is not None:
            lines.append(('yellow dots: position of the buried pipe', (255, 235, 90)))
        speed = args.stride * args.fps * float(np.median(np.diff(t)))
        footer = f'{machine.short}: {title.get(key, key)}   t = {t[k]:6.1f} s   {name}' + ('' if abs(speed - 1.) < 0.05 else f'   ({speed:.0f} x speed)')
        state = None
        if args.variant != 'off':                              # control samples since the last frame
            i0, i1 = np.searchsorted(ct, [(-1. if last_frame_t is None else last_frame_t) + 1e-9, t[k] + 1e-9])
            sh, co = bool(shaped[i0:i1].any()), bool(corrected[i0:i1].any())
            state = (('ACTIVE: ' + ' + '.join(x for x, on in (('motion shaped', sh), ('efforts corrected', co)) if on), (200, 110, 0))
                     if sh or co else ('idle', (70, 70, 70)))
        last_frame_t = t[k]
        img = hud(img, lines, tag, tag_rgb, footer, state)
        if writer is not None:
            writer.send(np.ascontiguousarray(img))
        else:
            Image.fromarray(img).save(out / f'still_{count:02d}.png')
        if count % 100 == 0:
            print(f'{key}__{args.variant}: frame {count}/{len(frames)} t={t[k]:.1f}s {(time.time() - wall) / (count + 1):.2f} s/frame', flush=True)
    if writer is not None:
        writer.close()
    (out / 'half.json').write_text(json.dumps(dict(run=str(data.folder.relative_to(ROOT)), code_sha256=data.code_sha256, frames=len(frames),
                                                   stride=args.stride, fps=args.fps, worst_margin={k: float(v) for k, v in worst.items()})))
    print(f'{key}__{args.variant}: {len(frames)} frames in {time.time() - wall:.0f} s; worst margins {worst}')
    return 0


def video(machine, args):
    """Both halves (two processes), then one picture with the variant without barrier rows on the left."""
    import imageio_ffmpeg
    base = [sys.executable, str(Path(__file__).resolve()), machine.name, args.key, '--stride', str(args.stride), '--fps', str(args.fps), '--width',
            str(args.width), '--height', str(args.height), '--shadow', str(args.shadow), '--samples', str(args.samples)]
    if args.stills:
        base += ['--stills', str(args.stills)]
    procs = [subprocess.Popen(base + ['--half', v]) for v in ('off', 'full')]
    if any(p.wait() != 0 for p in procs):
        raise SystemExit('rendering of a half picture failed')
    if args.stills:
        return 0
    vid = machine.results_dir() / 'videos'
    halves = [vid / f'{args.key}__{v}' for v in ('off', 'full')]
    out = vid / f'{machine.name}_{args.key}.mp4'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(halves[0] / 'half.mp4'), '-i', str(halves[1] / 'half.mp4'),
           '-filter_complex', 'hstack=inputs=2:shortest=1', '-c:v', 'libx264', '-crf', str(args.crf), '-preset', 'slow', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(out)]
    subprocess.run(cmd, check=True)
    info = [json.loads((h / 'half.json').read_text()) for h in halves]
    (vid / f'{machine.name}_{args.key}.json').write_text(json.dumps(dict(video=out.name, left=info[0], right=info[1]), indent=1))
    print(f'written {out} ({out.stat().st_size / 2 ** 20:.1f} MiB)')
    return 0


def main():
    from machines import NAMES, get_machine
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('machine', choices=NAMES)
    p.add_argument('key', help='scenario name or "mission"')
    p.add_argument('--stride', type=int, default=1, help='every n-th 40 ms sample is drawn (n x speed at 25 frames per second)')
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--shadow', type=int, default=4096)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--crf', type=int, default=23)
    p.add_argument('--stills', type=int, default=0, help='write this many evenly spaced stills instead of a video')
    p.add_argument('--half', default=None, choices=['full', 'off'], help='render one half only (used by the video command)')
    args = p.parse_args()
    from machine_models import require
    require(args.machine)
    machine = get_machine(args.machine)
    if args.half:
        args.variant = args.half
        return render_half(machine, args)
    return video(machine, args)


if __name__ == '__main__':
    raise SystemExit(main())
