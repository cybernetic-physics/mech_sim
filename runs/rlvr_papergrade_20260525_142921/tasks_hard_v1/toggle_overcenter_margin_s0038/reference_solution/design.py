# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [
    {
        'id': 'ground',
        'role': 'ground',
        'mass_kg': 0.0,
        'fixed': True,
        'com_local_mm': (57.295, 0.0, 0.0),
    },
    {
        'id': 'crank',
        'role': 'crank',
        'mass_kg': 0.02,
        'com_local_mm': (12.715, 0.0, 0.0),
    },
    {
        'id': 'coupler',
        'role': 'coupler',
        'mass_kg': 0.06,
        'com_local_mm': (41.765, 0.0, 0.0),
    },
    {
        'id': 'rocker',
        'role': 'rocker',
        'mass_kg': 0.05,
        'com_local_mm': (44.705, 0.0, 0.0),
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
        'anchor_world_mm': (25.43, 0.0, 0.0),
    },
    {
        'id': 'joint_cd',
        'type': 'revolute',
        'parent': 'coupler',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (64.30740242261103, 73.93110631439487, 0.0),
    },
    {
        'id': 'joint_output',
        'type': 'revolute',
        'parent': 'ground',
        'child': 'rocker',
        'axis_world': (0.0, 0.0, 1.0),
        'anchor_world_mm': (114.59, 0.0, 0.0),
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
        'pose_local_mm': (12.715, 0.0, 0.0),
    },
}
    params = {
    'link_lengths_mm': {
        'ground': 114.59,
        'crank': 25.43,
        'coupler': 83.53,
        'rocker': 89.41,
    },
    'overcenter_margin_mm': 2.3,
}
    ir = {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
    return ir
