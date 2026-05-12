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
        'role': 'pulley',
        'mass_kg': 0.03,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'pulley_out',
        'role': 'pulley',
        'mass_kg': 0.05,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'in_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'pulley_in',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'out_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'pulley_out',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (155.433, 0.0, 0.0),
    },
]
    ports = {
    'input_port': {
        'id': 'input_port',
        'part': 'in_axis',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'output_port': {
        'id': 'output_port',
        'part': 'out_axis',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'center_distance_mm': 155.433,
    'alignment_error_mm': 0.0392,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
