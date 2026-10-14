#!/usr/bin/env python3
"""Batches of runs, sequential by default and resumable.  The groups are the runs that the paper reports:

    python batch.py komatsu cycles        Tables II and III, Figs. 5-7: six approaches to the boom stroke limit with the
                                          baseline ('full'), the proposed layer ('consistent'), the first layer alone
                                          ('kinematic_only'), the effort layer alone ('force_only'), raw feedforward ('full_ffraw',
                                          Komatsu), the acceleration-level layer with beta = 40 ('accel_b40'), conditioning
                                          ('conditioned'); Komatsu also: gains k = 2, 4, 6, 24, the slower layer ('consistent_b8')
                                          and the heavier bucket (payload scale 1.5)
    python batch.py komatsu workspace     Table IV: e-fence and self-collision scenarios, baseline and proposed layer
    python batch.py komatsu mission       Table V: dig-and-level mission, baseline and proposed layer
    python batch.py komatsu paper         the three groups above
    python batch.py all paper --jobs 2    both machines (komatsu, heap), the longest runs first
    python batch.py heap tests            offline tests of the generated interfaces and the computed facts of the machine
    python batch.py heap regeneration     regeneration study (not reported in this paper)

Every run compares the controller variants of the paper on the same plant, references and nominal VDC law; the
variant names are those of bench/hazards.py (VARIANTS).  ``--render`` also records the soil state for the videos
(render.npz) of the runs that the comparison videos show (cycles and missions, baseline and proposed layer).

A run whose report exists with status COMPLETED and the hash of the present sources is skipped (use --force to repeat
it), so an interrupted batch can simply be started again.  Every run writes results/<machine>/logs/<name>.log; runs
that did not complete are listed in results/<machine>/logs/FAILED and make the exit status non-zero.

The sources must not change while a batch runs: every report carries the hash of the sources taken when its run
started, and paper/tables.py refuses reports that do not belong to the present sources.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PAPER_GROUPS = ('cycles', 'workspace', 'mission')
ALL_GROUPS = ('tests',) + PAPER_GROUPS + ('regeneration',)
MACHINES = ('komatsu', 'heap')

# variants of the scenario 'cycles' that the paper reports (Table II, Table III, Figs. 5-7 and Section VII-B)
CYCLE_VARIANTS = dict(
    komatsu=('full', 'consistent', 'kinematic_only', 'force_only', 'full_ffraw', 'accel_b40', 'conditioned', 'consistent_b8',
             'full_k2', 'full_k4', 'full_k6', 'full_k24', 'consistent_k2', 'consistent_k4', 'consistent_k6', 'consistent_k24',
             'kinematic_only_k2'),
    heap=('full', 'consistent', 'kinematic_only', 'force_only', 'accel_b40', 'conditioned'))
# heavier bucket (Table III): bucket mass and inertia of the plant 1.5 times those of the model
PAYLOAD_VARIANTS = dict(komatsu=('full', 'consistent', 'kinematic_only'), heap=())
WORKSPACE = ('efence', 'self')
COMPARED = ('full', 'consistent')                 # baseline (first-order first layer) and proposed (acceleration-level) layer
VIDEO = {('cycles', 'full'), ('cycles', 'consistent')}
# rough cost of a run relative to a run of the scenario 'cycles' on the Komatsu plant (order of the jobs only)
COST = dict(komatsu=dict(cycles=1., efence=1., self=0.8, mission=9.), heap=dict(cycles=2., efence=2., self=2., mission=16.))


def jobs_of(machine, group, render=False):
    """List of (name, report path, argument list of run.py, relative cost)."""
    H = ROOT / 'results' / machine / 'hazards'
    R = ROOT / 'results' / machine
    cost = COST.get(machine, {})

    def hazard(sc, va, name=None, *extra):
        name = name or f'{sc}__{va}'
        rend = ['--render'] if render and (sc, va) in VIDEO and not extra else []
        return (name, H / name / 'report.json', ['hazard', '--scenario', sc, '--variant', va] + (['--name', name] if name != f'{sc}__{va}' else [])
                + list(extra) + rend, cost.get(sc, 1.))
    if group == 'cycles':
        return ([hazard('cycles', va) for va in CYCLE_VARIANTS[machine]]
                + [hazard('cycles', va, f'cycles__{va}_bucket150', '--payload-scale', '1.5') for va in PAYLOAD_VARIANTS[machine]])
    if group == 'workspace':
        return [hazard(sc, va) for sc in WORKSPACE for va in COMPARED]
    if group == 'mission':
        return [(f'mission_{va}', R / f'mission_{va}' / 'report.json', ['mission', '--variant', va] + (['--render'] if render else []),
                 cost.get('mission', 9.)) for va in COMPARED]
    if group == 'regeneration':
        return [('regeneration', R / 'regeneration' / 'report.json', ['regeneration'], 3.)]
    if group == 'tests':
        return [('tests', R / 'tests' / 'report.json', ['tests'], 1.), ('facts', R / 'facts' / 'report.json', ['facts'], 1.)]
    raise ValueError(group)


def done(report, sha, argv=()):
    """The run is complete for the present sources; a run with the render record also needs that record (a later call of
    run.py without --render removes it)."""
    try:
        rep = json.loads(Path(report).read_text())
    except (OSError, ValueError):
        return False
    if '--render' in argv and not (Path(report).parent / 'render.npz').exists():
        return False
    return rep.get('status') == 'COMPLETED' and rep.get('code_sha256') == sha


def main():
    from bench.world import code_hash
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('machine', choices=MACHINES + ('all',))
    p.add_argument('groups', nargs='+', choices=list(ALL_GROUPS) + ['paper', 'all'])
    p.add_argument('--jobs', type=int, default=1, help='parallel jobs (runs that start together build the site assets of a machine '
                                                          'each into a temporary file that is renamed when complete)')
    p.add_argument('--force', action='store_true')
    p.add_argument('--render', action='store_true', help='also record the soil state for the comparison videos')
    p.add_argument('--list', action='store_true', help='only list the runs and whether they are up to date')
    args = p.parse_args()
    machines = MACHINES if args.machine == 'all' else (args.machine,)
    from machine_models import require
    require(*machines)
    groups = []
    for g in args.groups:
        groups += list(ALL_GROUPS) if g == 'all' else list(PAPER_GROUPS) if g == 'paper' else [g]
    groups = list(dict.fromkeys(groups))
    sha = {m: code_hash(m) for m in machines}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MUJOCO_GL=os.environ.get('MUJOCO_GL', 'egl'))
    jobs = [(m, g) + j for m in machines for g in groups for j in jobs_of(m, g, args.render)]
    if args.list:
        for m, g, name, report, argv, _ in jobs:
            print(f'{m:8s} {g:12s} {name:36s} {"up to date" if done(report, sha[m], argv) else "to run"}')
        return 0
    for m in machines:
        (ROOT / 'results' / m / 'logs').mkdir(parents=True, exist_ok=True)
    failed = {m: [] for m in machines}

    def one(job):
        m, g, name, report, argv, _ = job
        if not args.force and done(report, sha[m], argv):
            return job, 'up to date', 0.
        t0 = time.time()
        with open(ROOT / 'results' / m / 'logs' / f'{name}.log', 'w') as log:
            rc = subprocess.call([sys.executable, str(ROOT / 'run.py'), m] + argv, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        ok = rc == 0 and done(report, sha[m], argv)
        return job, 'completed' if ok else f'FAILED (exit status {rc})', time.time() - t0

    # the longest runs first, so that parallel workers finish together
    order = sorted(jobs, key=lambda j: -j[5])
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for (m, g, name, _, _, _), state, wall in pool.map(one, order):
            print(f'{time.strftime("%H:%M:%S")} {m:8s} {g:12s} {name:36s} {state}  {wall:6.0f} s', flush=True)
            if state.startswith('FAILED'):
                failed[m].append(name)
    status = 0
    for m in machines:
        logs = ROOT / 'results' / m / 'logs'
        every = {name: (report, argv) for g in ALL_GROUPS for name, report, argv, _ in jobs_of(m, g, args.render)}
        listed = (logs / 'FAILED').read_text().split() if (logs / 'FAILED').exists() else []
        failed_all = [n for n in dict.fromkeys(listed + failed[m]) if n not in every or not done(every[n][0], sha[m], every[n][1])]
        (logs / 'FAILED').write_text('\n'.join(failed_all) + ('\n' if failed_all else ''))
        if code_hash(m) != sha[m]:
            print(f'WARNING: the sources of {m} changed while the batch was running')
            status = 1
        if failed[m]:
            status = 1
    return status


if __name__ == '__main__':
    raise SystemExit(main())
