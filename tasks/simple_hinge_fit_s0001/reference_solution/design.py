# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [{'id': 'leaf_a', 'role': 'leaf', 'mass_kg': 0.06, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},{'id': 'leaf_b', 'role': 'leaf', 'mass_kg': 0.06, 'com_local_mm': (0.0, 0.0, 0.0)}]
    joints = [{'id': 'hinge', 'type': 'revolute', 'parent': 'leaf_a', 'child': 'leaf_b', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)}]
    ports = {'mount_a': {'id': 'mount_a', 'part': 'leaf_a', 'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)},'hinge_joint': {'id': 'hinge_joint', 'part': 'hinge', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},'tip_b': {'id': 'tip_b', 'part': 'leaf_b', 'kind': 'frame', 'pose_local_mm': (66.66, 0.0, 0.0)}}
    params = {'leaf_length_mm': 66.66, 'knuckle_diameter_mm': 4.91, 'declared_pin_clearance_mm': 0.233}
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
