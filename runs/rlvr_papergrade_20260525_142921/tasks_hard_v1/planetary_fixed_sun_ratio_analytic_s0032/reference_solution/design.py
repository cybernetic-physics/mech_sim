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
        'id': 'carrier',
        'role': 'carrier',
        'mass_kg': 0.05,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'ring',
        'role': 'ring',
        'mass_kg': 0.06,
        'com_local_mm': (0.0, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'input_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'carrier',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'output_axis',
        'type': 'revolute',
        'parent': 'frame',
        'child': 'ring',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
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
    'carrier_port': {
        'id': 'carrier_port',
        'part': 'carrier',
        'kind': 'frame',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
}
    params = {
    'sun_teeth': 20,
    'ring_teeth': 52,
    'declared_ratio': 0.722222,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
