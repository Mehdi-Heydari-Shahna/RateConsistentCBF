#!/usr/bin/env python3
"""Side-by-side videos of the paper runs: baseline with the first-order first layer (left) and the proposed
acceleration-level first layer (right), with the velocity error and the energy of the safety port in the read-out.

    python paper/paper_video.py komatsu cycles                 # results/komatsu/videos/paper_komatsu_cycles.mp4
    python paper/paper_video.py heap mission --stride 4        # 4 x speed
    python paper/paper_video.py komatsu cycles --stills 4      # stills of both halves (results/<m>/videos/paper_<key>__<variant>/)

Nothing is simulated here: the video replays the machine poses and the soil records of the runs (hazard runs with
``--render``; missions with ``--render``) and reads the controller signals of the same runs (trace.npz: c_storage,
c_P_cmd).  The rendering is that of video.py (same scene, obstacle primitives, plant-side safety functions).
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import video as V  # noqa: E402

TAGS = dict(full=('BASELINE: FIRST-ORDER FIRST LAYER', (150, 70, 20)), consistent=('PROPOSED: ACCELERATION-LEVEL FIRST LAYER', (16, 84, 160)),
            off=('NO BARRIER CONSTRAINTS', (150, 30, 30)))
TITLE = dict(V.TITLE, cycles='Six approaches to the boom stroke end')


def render_half(machine, args):
    import imageio_ffmpeg
    import mujoco
    from PIL import Image
    key, cfg, variant = args.key, machine.video_config(), args.half
    data = V.load_half(machine, key, variant)
    with np.load(data.folder / 'trace.npz', allow_pickle=True) as z:
        c_t, c_S, c_P = z['c_time'], z['c_storage'], z['c_P_cmd']
    dtc = float(np.median(np.diff(c_t)))
    E_port = np.cumsum(np.maximum(c_P, 0.)) * dtc
    out = machine.results_dir() / 'videos' / f'paper_{key}__{variant}'
    out.mkdir(parents=True, exist_ok=True)
    tr, rd = data.tr, data.rd
    W, H = args.width // 2, args.height
    sr = V.SiteRenderer(machine, data.folder / 'scene.xml', data.h0, W, H, shadow=args.shadow, samples=args.samples)
    grid = sr.grid
    focus = [p for p in cfg['focus'][key] if p in data.pair_ids]
    col = [data.pair_ids.index(p) for p in focus]
    geo = V.Geometry(mujoco, sr.m, data.spec, focus, cfg, key)
    label = cfg.get('labels', {})

    def surface(x, y):
        ix = int(np.clip(round((x - grid.xmin) / grid.dx), 0, grid.n - 1))
        iy = int(np.clip(round((y - grid.ymin) / grid.dx), 0, grid.n - 1))
        return sr.h[iy, ix]
    t = tr['time']
    n = min(len(t), len(rd['content']))
    frames = list(range(0, n, max(1, args.stride)))
    if args.at:
        frames = [int(np.argmin(np.abs(t[:n] - x))) for x in args.at]
    elif args.stills:
        frames = [frames[int(round(f * (len(frames) - 1)))] for f in np.linspace(0., 1., args.stills)]
    toff, tidx, th, coff = rd['terrain_off'], rd['terrain_idx'], rd['terrain_h'], rd['clod_off']
    tag, tag_rgb = TAGS.get(variant, (variant.upper(), (60, 60, 60)))
    worst = {p: np.inf for p in focus}
    s_peak = 0.
    writer = None
    if not (args.stills or args.at):
        writer = imageio_ffmpeg.write_frames(str(out / 'half.mp4'), (W, H), fps=args.fps, codec='libx264', quality=None,
                                             macro_block_size=8, output_params=['-crf', '13', '-preset', 'veryfast'])
        writer.send(None)
    director = V.Director(tau=1.2)
    ct = tr['c_time']
    shaped, corrected = tr['c_velocity_active'] > 0, tr['c_n_active'] > 0
    cam_key = 'strokes' if key == 'cycles' else key
    applied, last_t, last_k, last_frame_t = 0, None, -1, None
    wall = time.time()
    for count, k in enumerate(frames):
        while applied <= k:
            a, b = int(toff[applied]), int(toff[applied + 1])
            sr.apply_terrain(tidx[a:b], th[a:b])
            applied += 1
        sr.upload_terrain(texture=(count % 3 == 0) or bool(args.stills) or bool(args.at))
        qpos = tr['qpos'][k].astype(float)
        sr.set_pose(qpos)
        sr.set_content(float(rd['content'][k]))
        a, b = int(coff[k]), int(coff[k + 1])
        name = data.names[int(tr['phase'][k])]
        h_now = tr['pair_h'][k, col].astype(float)
        seg = tr['pair_h'][last_k + 1:k + 1, col].astype(float)
        last_k = k
        for p, v in zip(focus, seg.min(axis=0) if len(seg) else h_now):
            worst[p] = min(worst[p], float(v))
        cam = V.cam_from(*machine.camera(cam_key, name, qpos))
        cam = director(cam, 0. if last_t is None else t[k] - last_t) if not (args.stills or args.at) else cam
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
        # controller signals up to this frame
        i1 = int(np.searchsorted(c_t, t[k] + 1e-9))
        i0 = int(np.searchsorted(c_t, (-1. if last_frame_t is None else last_frame_t) + 1e-9))
        s_now = float(np.sqrt(2. * max(c_S[max(i1 - 1, 0)], 0.)))
        if i1 > i0:
            s_peak = max(s_peak, float(np.sqrt(2. * max(np.max(c_S[i0:i1]), 0.))))
        e_now = float(E_port[max(i1 - 1, 0)])
        lines = [(f'velocity error |s|_M = {s_now:6.2f} sqrt(J)   (largest so far {s_peak:6.2f})', (235, 235, 235)),
                 (f'energy put into the storage by the safety port: {e_now:8.2f} J', (235, 235, 235)),
                 ('margin to the declared limit (measured on the plant)', (215, 215, 215))]
        for p, v in zip(focus, h_now):
            unit, scale = V.UNIT.get(data.cls.get(p), ('m', 1.))
            fmt = (lambda x: f'{scale * x:+.3f} {unit}') if unit == 'm' else (lambda x: f'{scale * x:+.0f} {unit}')
            text = f'{label.get(p, p)}:  {fmt(v)}'
            if v < 0.:
                lines.append((text + '   LIMIT VIOLATED', (255, 90, 80)))
            elif worst[p] < 0.:
                lines.append((text + f'   (was violated: {fmt(worst[p])})', (255, 150, 120)))
            else:
                lines.append((text, (150, 235, 150)))
        if not data.free_motion:
            lines.append((f'soil in the bucket:  {float(tr["bucket_mass"][k]):.0f} kg', (235, 235, 235)))
        speed = args.stride * args.fps * float(np.median(np.diff(t)))
        footer = f'{machine.short}: {TITLE.get(key, key)}   t = {t[k]:6.1f} s   {name}' + ('' if abs(speed - 1.) < 0.05 else f'   ({speed:.0f} x speed)')
        state = None
        if variant != 'off':
            j0, j1 = np.searchsorted(ct, [(-1. if last_frame_t is None else last_frame_t) + 1e-9, t[k] + 1e-9])
            sh, co = bool(shaped[j0:j1].any()), bool(corrected[j0:j1].any())
            state = (('ACTIVE: ' + ' + '.join(x for x, on in (('motion shaped', sh), ('efforts corrected', co)) if on), (200, 110, 0))
                     if sh or co else ('idle', (70, 70, 70)))
        last_frame_t = t[k]
        if state is not None:
            lines.insert(0, ('filter ' + state[0].lower(), (255, 190, 90) if state[0] != 'idle' else (190, 190, 190)))
        if not args.nohud:
            img = V.hud(img, lines, tag, tag_rgb, footer, None)
        if writer is not None:
            writer.send(np.ascontiguousarray(img))
        else:
            Image.fromarray(img).save(out / (f'still_t{t[k]:06.2f}.png' if args.at else f'still_{count:02d}.png'))
        if count % 200 == 0:
            print(f'{key}__{variant}: frame {count}/{len(frames)} t={t[k]:.1f}s {(time.time() - wall) / (count + 1):.2f} s/frame', flush=True)
    if writer is not None:
        writer.close()
    (out / 'half.json').write_text(json.dumps(dict(run=str(data.folder.relative_to(ROOT)), code_sha256=data.code_sha256, frames=len(frames),
                                                   stride=args.stride, fps=args.fps, worst_margin={k: float(v) for k, v in worst.items()},
                                                   largest_s_M=s_peak, port_energy_in_J=float(E_port[-1]))))
    print(f'{key}__{variant}: {len(frames)} frames in {time.time() - wall:.0f} s; worst margins {worst}')
    return 0


def video(machine, args):
    import imageio_ffmpeg
    base = [sys.executable, str(Path(__file__).resolve()), machine.name, args.key, '--stride', str(args.stride), '--fps', str(args.fps),
            '--width', str(args.width), '--height', str(args.height), '--shadow', str(args.shadow), '--samples', str(args.samples)]
    if args.stills:
        base += ['--stills', str(args.stills)]
    if args.at:
        base += ['--at'] + [str(x) for x in args.at]
    if args.nohud:
        base += ['--nohud']
    left, right = args.left, args.right
    if args.sequential:
        for v in (left, right):
            if subprocess.call(base + ['--half', v]) != 0:
                raise SystemExit('rendering of a half picture failed')
    else:
        procs = [subprocess.Popen(base + ['--half', v]) for v in (left, right)]
        if any(p.wait() != 0 for p in procs):
            raise SystemExit('rendering of a half picture failed')
    if args.stills or args.at:
        return 0
    vid = machine.results_dir() / 'videos'
    halves = [vid / f'paper_{args.key}__{v}' for v in (left, right)]
    out = vid / f'paper_{machine.name}_{args.key}.mp4'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(halves[0] / 'half.mp4'), '-i', str(halves[1] / 'half.mp4'),
           '-filter_complex', 'hstack=inputs=2:shortest=1', '-c:v', 'libx264', '-crf', str(args.crf), '-preset', 'slow', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(out)]
    subprocess.run(cmd, check=True)
    info = [json.loads((h / 'half.json').read_text()) for h in halves]
    (vid / f'paper_{machine.name}_{args.key}.json').write_text(json.dumps(dict(video=out.name, left=info[0], right=info[1]), indent=1))
    print(f'written {out} ({out.stat().st_size / 2 ** 20:.1f} MiB)')
    return 0


def main():
    from machines import NAMES, get_machine
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('machine', choices=NAMES)
    p.add_argument('key', help='scenario name or "mission"')
    p.add_argument('--left', default='full')
    p.add_argument('--right', default='consistent')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--shadow', type=int, default=4096)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--crf', type=int, default=23)
    p.add_argument('--stills', type=int, default=0)
    p.add_argument('--at', type=float, nargs='*', default=None, help='stills at these times (s)')
    p.add_argument('--nohud', action='store_true', help='no read-out (pictures for the paper)')
    p.add_argument('--sequential', action='store_true', help='render the two halves one after the other (one process)')
    p.add_argument('--half', default=None)
    args = p.parse_args()
    from machine_models import require
    require(args.machine)
    machine = get_machine(args.machine)
    if args.half:
        return render_half(machine, args)
    return video(machine, args)


if __name__ == '__main__':
    raise SystemExit(main())
