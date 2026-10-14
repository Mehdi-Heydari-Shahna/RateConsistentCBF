"""Machine adapters of the bench: ``get_machine(name)``."""
import importlib

NAMES = ('komatsu', 'heap')


def get_machine(name):
    if name not in NAMES:
        raise ValueError(f'unknown machine {name} (known: {NAMES})')
    return importlib.import_module(f'machines.{name}.machine').MACHINE()
