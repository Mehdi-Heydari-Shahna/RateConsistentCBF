"""The excavator machine packages, which are supplied separately from this repository.

A machine is a package ``machines/<name>/`` of this folder (``machine.py`` with ``MACHINE``, a subclass of
``bench.machine.Machine``; ``model.py``; ``control_safety_spec.json``; source records in ``source/``).  The packages
of the paper (``komatsu`` and ``heap``) hold the mechanism records, geometry, actuator and hydraulic parameters and
the site and safety specification of the two excavators and are not distributed: copy them into ``machines/``.
The source hash of a run (``bench.world.code_hash``) includes the machine package, so results can be compared with
the logged results of the paper only with the same packages.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MACHINES_DIR = ROOT / 'machines'


def present(name):
    """True when the machine package ``name`` is in ``machines/``."""
    return (MACHINES_DIR / name / 'machine.py').is_file()


def require(*names):
    """End the command with a message unless every named machine package is present."""
    missing = [n for n in names if not present(n)]
    if missing:
        what = ', '.join(missing)
        raise SystemExit(f'error: the excavator model package{"s" if len(missing) > 1 else ""} {what} '
                         f'{"are" if len(missing) > 1 else "is"} not part of this repository. Copy the folder'
                         f'{"s" if len(missing) > 1 else ""} {what} of the separately supplied model archive into '
                         f'{MACHINES_DIR} (see README.md, "Excavator models").')
