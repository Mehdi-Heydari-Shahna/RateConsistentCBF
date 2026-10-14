"""Local code locations for the numerical examples.

RCX_HEAP is the excavator code (code/excavators/RCX_TAC: both machines and the QP solver rcx.qp); the environment
variable RCX_HEAP_ROOT can override it.
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
NUM = Path(__file__).resolve().parent
EXCAVATORS = NUM.parents[1] / 'excavators'
RCX_HEAP = os.environ.get('RCX_HEAP_ROOT', str(EXCAVATORS / 'RCX_TAC'))
