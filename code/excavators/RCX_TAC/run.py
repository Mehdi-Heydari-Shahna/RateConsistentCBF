#!/usr/bin/env python3
"""Runs of the extended RoboCompiler on one machine (machines: heap, komatsu).

    python run.py heap hazard --scenario cycles --variant consistent
    python run.py heap mission --variant consistent
    python run.py heap regeneration
    python run.py heap tests
    python run.py heap facts

Results are written to results/<machine>/...; the exit status is non-zero unless the run completed.  A hazard or
mission run replaces the result of the same name (report, trace and its further pieces).  The render record of the
earlier run is removed as well; it is written again only with ``--render`` (the videos need it; ``batch.py ... --render``
records it for the runs that the comparison videos show).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    from bench.hazards import VARIANTS
    from machines import NAMES, get_machine
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('machine', choices=NAMES)
    sub = p.add_subparsers(dest='command', required=True)
    h = sub.add_parser('hazard', help='one hazard scenario')
    h.add_argument('--scenario', required=True)
    h.add_argument('--variant', default='full', choices=sorted(VARIANTS))
    h.add_argument('--name', default=None)
    h.add_argument('--partition', default='joint_space')
    h.add_argument('--duration', type=float, default=None)
    h.add_argument('--observer-gain', type=float, default=100.)
    h.add_argument('--payload-scale', type=float, default=1.)
    h.add_argument('--soil-strength', type=float, default=1., help='scales bank cohesion, subgrade modulus and bearing pressure')
    h.add_argument('--render', action='store_true', help='also record the soil state for the video (render.npz)')
    m = sub.add_parser('mission', help='complete site mission')
    m.add_argument('--variant', default='full', choices=sorted(VARIANTS))
    m.add_argument('--name', default=None)
    m.add_argument('--duration', type=float, default=900.)
    m.add_argument('--render', action='store_true')
    m.add_argument('--export-spec', action='store_true', help='also write the specification of this run next to the machine records')
    r = sub.add_parser('regeneration', help='regeneration study')
    r.add_argument('--only', nargs='*', default=None)
    r.add_argument('--samples', type=int, default=5)
    r.add_argument('--seed', type=int, default=3)
    r.add_argument('--duration', type=float, default=None)
    r.add_argument('--no-closed-loop', action='store_true')
    sub.add_parser('tests', help='offline tests of the generated interfaces on this machine')
    sub.add_parser('facts', help='computed facts of the machine that its documentation quotes')
    args = p.parse_args()
    from machine_models import require
    require(args.machine)
    machine = get_machine(args.machine)
    if args.command in ('hazard', 'mission'):
        # further pieces of a packed trace of the same run (trace.part2.npz, ...) belong to the result
        # that this run replaces
        name = args.name or (f'{args.scenario}__{args.variant}' if args.command == 'hazard' else f'mission_{args.variant}')
        folder = machine.results_dir() / ('hazards' if args.command == 'hazard' else '') / name
        for piece in folder.glob('trace.part*.npz'):
            piece.unlink()
    if args.command == 'hazard':
        from bench import hazards
        rep = hazards.run(machine, args.scenario, args.variant, name=args.name, partition=args.partition, duration=args.duration,
                          observer_gain=args.observer_gain, payload_scale=args.payload_scale, soil_strength=args.soil_strength, render=args.render)
    elif args.command == 'mission':
        from bench import mission
        rep = mission.run(machine, args.variant, name=args.name, duration=args.duration, render=args.render, export_spec=args.export_spec)
    elif args.command == 'regeneration':
        from bench import regeneration
        rep = regeneration.run(machine, only=args.only, samples=args.samples, seed=args.seed, duration=args.duration,
                               closed_loop=not args.no_closed_loop)
    elif args.command == 'facts':
        from bench import facts
        rep = facts.run(machine)
    else:
        from bench import tests_machine
        rep = tests_machine.run(machine)
    return 0 if rep['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
