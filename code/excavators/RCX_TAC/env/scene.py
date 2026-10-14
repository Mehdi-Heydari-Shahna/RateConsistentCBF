"""Construction-site environment of a plant scene (machine independent): level working platform, deformable soil
heightfield, far terrain, soil textures, light, and the render geometry of the soil carried in the bucket.

Same scene as in the earlier single-machine release, split from the machine: the functions here
add the environment to an MJCF tree that already contains the machine (bench/export.py) and build the initial soil.

The bucket has no MuJoCo contact with the soil; the soil-bucket interaction is computed by ``env.soil`` and applied as a
wrench, so the heightfield is geometry for rendering and for the continuum model; the machine stands on the platform
(a hidden box whose top face is the level pad of the terrain).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from . import content_mesh
from .soil import BucketGeometry, SoilModel, SoilParams
from .terrain import ELEV, Z_BASE, GridSpec, SiteLayout, initial_terrain, outer_terrain, save_png, soil_texture

OUTER_HALF = 200.0


def aligned_surround(grid: GridSpec, approx=OUTER_HALF, max_cell=3.2):
    """(half size, vertex count) of a far-terrain mesh whose texture tiles line up with the soil grid and whose vertex
    lines fall on the soil-grid border (the far terrain has a hole exactly over the soil grid: no visible seam)."""
    tile = 2 * grid.half
    if abs(grid.xmin - grid.ymin) > 1e-9:
        raise ValueError('grid must be centred on the diagonal for a square aligned surround')
    S = float(round((approx + grid.xmin) / tile) * tile - grid.xmin)
    for k in range(1, 200):
        c = tile / k
        cells = 2 * S / c
        if c <= max_cell and abs(cells - round(cells)) < 1e-9:
            return S, int(round(cells)) + 1
    raise ValueError('no aligned surround resolution found')


def _write_atomic(path: Path, write):
    """``write(tmp)`` into a temporary file of the same folder, then an atomic rename to ``path``: runs started in parallel
    may build the same (deterministic) asset, and none of them reads a partly written file."""
    tmp = path.with_name(f'.{path.stem}.{os.getpid()}.tmp{path.suffix}')
    write(tmp)
    os.replace(tmp, path)


def ensure_assets(asset_dir: Path, grid: GridSpec, layout: SiteLayout, tex_size=2048):
    """Initial terrain (npz) and the two soil textures in ``asset_dir`` (built once, deterministic)."""
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    key = json.dumps(dict(center=list(grid.center), half=grid.half, dx=grid.dx, site=list(layout.site_center), pile=list(layout.pile_radii),
                          height=layout.pile_height, dig=[float(v) for v in layout.dig_center], pad=list(layout.pad), seed=layout.seed),
                     sort_keys=True)
    f = asset_dir / 'terrain_initial.npz'
    if f.exists():
        with np.load(f) as z:
            if str(z['key']) != key:
                f.unlink(missing_ok=True)
                (asset_dir / 'terrain_settled.npz').unlink(missing_ok=True)
    if not f.exists():
        h, loose, ground = initial_terrain(grid, layout)
        _write_atomic(f, lambda t: np.savez_compressed(t, h=h, loose=loose, ground=ground, key=key))
    if not (asset_dir / 'soil_base.png').exists():         # soil_fresh.png first: soil_base.png marks both as present
        _write_atomic(asset_dir / 'soil_fresh.png', lambda t: save_png(soil_texture(tex_size, fresh=True, meters=2 * grid.half), t))
        _write_atomic(asset_dir / 'soil_base.png', lambda t: save_png(soil_texture(tex_size, fresh=False, meters=2 * grid.half), t))
    return asset_dir


def initial_soil(asset_dir: Path, grid: GridSpec, layout: SiteLayout, bucket: BucketGeometry, params: SoilParams | None = None):
    """Soil model on the initial terrain; the loose pile is relaxed to its angle of repose once (cached)."""
    params = params or SoilParams()
    asset_dir = ensure_assets(asset_dir, grid, layout)
    with np.load(asset_dir / 'terrain_initial.npz') as z:
        h0, loose0, ground = z['h'], z['loose'], z['ground']
    boxes = layout.boxes()
    cache = asset_dir / 'terrain_settled.npz'
    if cache.exists():
        with np.load(cache) as s:
            h, loose = s['h'], s['loose']
    else:
        sm = SoilModel(grid, h0, loose0, ground, SoilParams(), bucket)
        sx, sy = layout.site_center
        ax, ay = layout.pile_radii
        r0, r1, c0, c1 = sm._window(sx - ax - 1.35, sx + ax + 1.35, sy - ay - 1.35, sy + ay + 1.35, pad=2)
        sweeps = sm.settle(box=[r0, r1, c0, c1], max_sweeps=5000, tol=1e-8)
        h, loose = sm.h, sm.loose
        _write_atomic(cache, lambda t: np.savez_compressed(t, h=h, loose=loose, sweeps=sweeps))
    sm = SoilModel(grid, h, loose, ground, params, bucket, site=boxes['site'], dig=boxes['dig'])
    sm.h_initial = np.array(h)
    sm.touched[:] = False
    sm.dirty[:] = False
    return sm, boxes


def ground_function(grid: GridSpec, ground):
    ground = np.asarray(ground, float)

    def fn(x, y):
        c = (np.asarray(x, float) - grid.xmin) / grid.dx
        r = (np.asarray(y, float) - grid.ymin) / grid.dx
        r0 = np.clip(np.floor(r).astype(int), 0, grid.n - 2)
        c0 = np.clip(np.floor(c).astype(int), 0, grid.n - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        g = ground
        return ((1 - fr) * (1 - fc) * g[r0, c0] + (1 - fr) * fc * g[r0, c0 + 1] + fr * (1 - fc) * g[r0 + 1, c0]
                + fr * fc * g[r0 + 1, c0 + 1])
    return fn


def _surround_mesh(grid: GridSpec, S, NS, outer):
    x = np.linspace(-S, S, NS)
    X, Y = np.meshgrid(x, x)
    tile = 2 * grid.half
    V = np.c_[X.ravel(), Y.ravel(), np.asarray(outer, float).ravel()]
    x0, x1 = grid.xmin, grid.xmin + tile
    y0, y1 = grid.ymin, grid.ymin + tile
    eps = 1e-6
    faces = []
    for i in range(NS - 1):
        for j in range(NS - 1):
            cx0, cx1, cy0, cy1 = x[j], x[j + 1], x[i], x[i + 1]
            if cx0 >= x0 - eps and cx1 <= x1 + eps and cy0 >= y0 - eps and cy1 <= y1 + eps:
                continue                                     # cell covered by the soil grid
            a, b, c, d = i * NS + j, i * NS + j + 1, (i + 1) * NS + j + 1, (i + 1) * NS + j
            faces += [(a, b, c), (a, c, d)]
    uv = np.c_[(X.ravel() - x0) / tile, (y1 - Y.ravel()) / tile]      # the heightfield maps texture row 0 to the +y edge
    return V, np.asarray(faces, np.int64), uv


def add_environment(root, asset_dir: Path, grid: GridSpec, layout: SiteLayout, *, support_contype=4, support_conaffinity=11,
                    support_friction='1.0 .003 .008', statistic=('3 3 1', '22')):
    """Adds platform, soil heightfield, far terrain, textures, light and render settings to the MJCF tree ``root``."""
    asset_dir = Path(asset_dir).resolve()
    asset, world = root.find('asset'), root.find('worldbody')
    ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='.42 .58 .80', rgb2='.86 .88 .90', width='512', height='512')
    ET.SubElement(asset, 'texture', name='soil_tex', type='2d', file=str(asset_dir / 'soil_base.png'))
    ET.SubElement(asset, 'material', name='soil_mat', texture='soil_tex', texrepeat='1 1', texuniform='false', specular='0.05',
                  shininess='0.05', reflectance='0')
    ET.SubElement(asset, 'texture', name='fresh_tex', type='2d', file=str(asset_dir / 'soil_fresh.png'))
    ET.SubElement(asset, 'material', name='m_soil_content', texture='fresh_tex', texrepeat='1 1', texuniform='false', specular='0.02',
                  shininess='0.02', reflectance='0', rgba='0.80 0.76 0.72 1')
    ET.SubElement(asset, 'texture', name='surround_tex', type='2d', file=str(asset_dir / 'soil_base.png'))
    ET.SubElement(asset, 'material', name='surround_mesh_mat', texture='surround_tex', texrepeat='1 1', texuniform='false',
                  specular='0.05', shininess='0.05', reflectance='0')
    n = grid.n
    ET.SubElement(asset, 'hfield', name='soil', nrow=str(n), ncol=str(n), size=f'{grid.half} {grid.half} {ELEV} 0.4')
    px0, px1, py = layout.pad
    ET.SubElement(world, 'geom', name='pad', type='box', pos=f'{(px0 + px1) / 2} 0 -0.5', size=f'{(px1 - px0) / 2} {py} 0.5',
                  contype=str(support_contype), conaffinity=str(support_conaffinity), condim='3', friction=support_friction, group='3',
                  rgba='.5 .5 .5 1')
    ET.SubElement(world, 'geom', name='soil', type='hfield', hfield='soil', pos=f'{grid.center[0]} {grid.center[1]} {Z_BASE}',
                  material='soil_mat', contype='0', conaffinity='0', group='0')
    S, NS = aligned_surround(grid)
    outer = outer_terrain(NS, S, inner_half=grid.half, inner_center=grid.center, hide_margin=1e-6)
    Vs, Fs, UVs = _surround_mesh(grid, S, NS, outer)
    fm = lambda a: ' '.join(format(float(v), '.5f') for v in np.ravel(a))
    ET.SubElement(asset, 'mesh', name='surround_mesh', inertia='shell', vertex=fm(Vs), texcoord=fm(UVs),
                  face=' '.join(str(int(i)) for i in Fs.ravel()))
    ET.SubElement(world, 'geom', name='surround', type='mesh', mesh='surround_mesh', material='surround_mesh_mat', contype='0',
                  conaffinity='0', mass='0', group='0')
    ET.SubElement(world, 'light', name='sun', directional='true', pos='-20 -30 40', dir='0.42 0.55 -0.72', diffuse='0.95 0.90 0.80',
                  specular='0.35 0.33 0.30', ambient='0.0 0.0 0.0', castshadow='true')
    ET.SubElement(world, 'light', name='fill', directional='true', pos='20 30 20', dir='-0.5 -0.6 -0.6', diffuse='0.20 0.23 0.28',
                  specular='0 0 0', castshadow='false')
    vis = root.find('visual')
    for tag in ('headlight', 'quality', 'map', 'global'):
        e = vis.find(tag)
        if e is not None:
            vis.remove(e)
    ET.SubElement(vis, 'global', offwidth='1920', offheight='1080', fovy='40')
    ET.SubElement(vis, 'quality', shadowsize='8192', offsamples='8', numslices='32', numstacks='16')
    ET.SubElement(vis, 'headlight', ambient='0.36 0.36 0.38', diffuse='0.12 0.12 0.12', specular='0.05 0.05 0.05')
    ET.SubElement(vis, 'map', znear='0.02', zfar='600', fogstart='3', fogend='12', haze='0.30', shadowclip='1.1', shadowscale='0.6')
    rgba = vis.find('rgba')
    if rgba is None:
        rgba = ET.SubElement(vis, 'rgba')
    rgba.set('fog', '0.80 0.84 0.88 1')
    rgba.set('haze', '0.86 0.88 0.90 1')
    stat = root.find('statistic')
    stat.set('center', statistic[0])
    stat.set('extent', statistic[1])


def add_content_mesh(root, tool_body_elem, bucket: BucketGeometry, R_bs, p_bs):
    """Render mesh of the soil carried in the bucket on the tool body (vertices are rewritten every frame by the
    renderer).  ``R_bs``, ``p_bs``: soil frame of the bucket in the tool-body frame."""
    V0 = content_mesh.initial_local(bucket) @ np.asarray(R_bs, float).T + np.asarray(p_bs, float)
    fmtv = lambda a: ' '.join(format(float(v), '.6f') for v in np.ravel(a))
    ET.SubElement(root.find('asset'), 'mesh', name='soil_content', vertex=fmtv(V0), texcoord=fmtv(content_mesh.texcoords()),
                  face=' '.join(str(int(i)) for i in content_mesh.FACES.ravel()))
    ET.SubElement(tool_body_elem, 'geom', name='soil_content', type='mesh', mesh='soil_content', material='m_soil_content', contype='0',
                  conaffinity='0', mass='0', group='1')
    return V0


def load_terrain_into_model(m, h):
    """Write surface heights (m) into the MuJoCo soil heightfield (normalised data)."""
    import mujoco
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, 'soil')
    adr, nr, nc = m.hfield_adr[sid], m.hfield_nrow[sid], m.hfield_ncol[sid]
    m.hfield_data[adr:adr + nr * nc] = np.clip((np.asarray(h) - Z_BASE) / ELEV, 0, 1).ravel()
    return sid


class RenderRecorder:
    """Per-sample data for offline rendering: terrain changes, airborne clods, bucket content volume."""

    def __init__(self):
        self.idx, self.h, self.idx_off = [], [], [0]
        self.clod_p, self.clod_v, self.clod_i, self.clod_off = [], [], [], [0]
        self.content = []

    def add(self, idx, hv, sm: SoilModel):
        self.idx.append(idx)
        self.h.append(hv)
        self.idx_off.append(self.idx_off[-1] + len(idx))
        self.clod_p.append(sm.clod_pos.astype(np.float32))
        self.clod_v.append(sm.clod_vol.astype(np.float32))
        self.clod_i.append(sm.clod_id.astype(np.int64))
        self.clod_off.append(self.clod_off[-1] + len(sm.clod_vol))
        self.content.append(sm.content)

    def arrays(self):
        cat = lambda lst, dt, shp: np.concatenate(lst).astype(dt) if len(lst) else np.zeros(shp, dt)
        return dict(terrain_idx=cat(self.idx, np.int32, (0,)), terrain_h=cat(self.h, np.float32, (0,)),
                    terrain_off=np.asarray(self.idx_off, np.int64), clod_pos=cat(self.clod_p, np.float32, (0, 3)).reshape(-1, 3),
                    clod_vol=cat(self.clod_v, np.float32, (0,)), clod_id=cat(self.clod_i, np.int64, (0,)),
                    clod_off=np.asarray(self.clod_off, np.int64), content=np.asarray(self.content, float))

    def __len__(self):
        return len(self.content)
