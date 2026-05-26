# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    D_IN = 24.37
    D_OUT = 48.74
    RATIO = 2.0
    parts = [
        {'id': 'frame', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},
        {'id': 'drive', 'role': 'pulley_drive', 'mass_kg': 0.03, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'diameter_mm': D_IN}},
        {'id': 'driven', 'role': 'pulley_driven', 'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'diameter_mm': D_OUT}},
    ]
    joints = [
        {'id': 'drive_axis', 'type': 'revolute', 'parent': 'frame', 'child': 'drive', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'driven_axis', 'type': 'revolute', 'parent': 'frame', 'child': 'driven', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (120.0, 0.0, 0.0)},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'drive_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'driven_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'drive_diameter_mm': D_IN,
            'driven_diameter_mm': D_OUT,
            'declared_ratio': RATIO,
        },
    }
