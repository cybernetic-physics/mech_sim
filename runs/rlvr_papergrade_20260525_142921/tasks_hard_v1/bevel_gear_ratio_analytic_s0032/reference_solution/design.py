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
        'id': 'bevel_in',
        'role': 'input',
        'mass_kg': 0.02,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'bevel_out',
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
        'child': 'bevel_in',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'output_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'bevel_out',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (50.0, 0.0, 0.0),
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
    'teeth_in': 12,
    'teeth_out': 24,
    'declared_ratio': 2.0,
    'axis_angle_deg': 60.0,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
