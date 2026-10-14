"""RoboCompiler extension: control and safety interfaces generated from the same physical mechanism graph.

Inputs:  physical records (bodies, joints, attachment frames, inertias, actuator ports) and a control /
         safety / actuator specification (control objectives, collision geometry, safety limits, actuator
         capabilities).
Outputs: a generated geometric VDC evaluator (body modules with loop-consistent required twists), generated
         barrier and storage evaluators (second-order CBF rows in physical actuator efforts, VDC storage and
         port virtual power flow) and one coordinated actuator-force filter.

The modules build on the ported RoboCompiler pieces in ``core`` (graph compiler, generated loop evaluators,
unchanged PACDM core); nothing here is specific to one mechanism.
"""
__version__ = '0.1.0'
