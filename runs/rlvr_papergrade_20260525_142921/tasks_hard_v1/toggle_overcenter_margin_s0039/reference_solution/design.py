# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'ground',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (53.14, 0.0, 0.0),
    },
    {
        'id': 'crank',
        'role': 'crank',
        'mass_kg': 0.02,
        'com_local_mm': (19.45, 0.0, 0.0),
    },
    {
        'id': 'coupler',
        'role': 'coupler',
        'mass_kg': 0.06,
        'com_local_mm': (47.265, 0.0, 0.0),
    },
    {
        'id': 'rocker',
        'role': 'rocker',
        'mass_kg': 0.05,
        'com_local_mm': (39.91, 0.0, 0.0),
    },
]
    joints = [
    {
        'id': 'joint_input',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'crank',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (0.0, 0.0, 0.0),
    },
    {
        'id': 'joint_bc',
        'type': 'revolute',
        'parent': 'crank',
        'child': 'coupler',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (38.9, 0.0, 0.0),
    },
    {
        'id': 'joint_cd',
        'type': 'revolute',
        'parent': 'coupler',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (91.62152641733454, 78.46248499905096, 0.0),
    },
    {
        'id': 'joint_output',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (106.28, 0.0, 0.0),
    },
]
    ports = {
    'input_port': {
        'id': 'input_port',
        'part': 'joint_input',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'output_port': {
        'id': 'output_port',
        'part': 'joint_output',
        'kind': 'revolute_joint',
        'pose_local_mm': (0.0, 0.0, 0.0),
    },
    'coupler_point': {
        'id': 'coupler_point',
        'part': 'coupler',
        'kind': 'frame',
        'pose_local_mm': (19.45, 0.0, 0.0),
    },
}
    params = {
    'link_lengths_mm': {
        'ground': 106.28,
        'crank': 38.9,
        'coupler': 94.53,
        'rocker': 79.82,
    },
    'overcenter_margin_mm': 1.678,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
