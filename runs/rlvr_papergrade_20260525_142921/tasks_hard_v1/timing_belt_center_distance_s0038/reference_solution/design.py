# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'frame',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'pulley_in',
        'role': 'input',
        'mass_kg': 0.02,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'pulley_out',
        'role': 'output',
        'mass_kg': 0.05,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'input_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'pulley_in',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'output_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'pulley_out',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (280.0, 0.0, 0.0),
    },
]
    ports = {
    'input_port': {
        'id': 'input_port',
        'part': 'input_axis',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'output_port': {
        'id': 'output_port',
        'part': 'output_axis',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'pitch_mm': 5.0,
    'belt_tooth_count': 120,
    'declared_belt_length_mm': 600.0,
    'declared_center_distance_mm': 280.0,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
