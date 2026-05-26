# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    TEETH_IN = 16
    TEETH_OUT = 48
    DECLARED_RATIO = 3.0
    parts = [
        {'id': 'frame', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},
        {'id': 'pinion', 'role': 'gear_input', 'mass_kg': 0.02, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'teeth': TEETH_IN}},
        {'id': 'gear', 'role': 'gear_output', 'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'teeth': TEETH_OUT}},
    ]
    joints = [
        {'id': 'pinion_axis', 'type': 'revolute', 'parent': 'frame', 'child': 'pinion', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'gear_axis', 'type': 'revolute', 'parent': 'frame', 'child': 'gear', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (40.0, 0.0, 0.0)},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'pinion_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'gear_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'teeth_in': TEETH_IN,
            'teeth_out': TEETH_OUT,
            'declared_ratio': DECLARED_RATIO,
        },
    }
