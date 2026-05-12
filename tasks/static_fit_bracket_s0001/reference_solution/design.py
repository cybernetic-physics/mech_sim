# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [{'id': 'bracket', 'role': 'frame', 'mass_kg': 0.05, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'hole_diameter_mm': 5.075, 'pitch_mm': 36.949}}]
    joints = []
    ports = {'mount_a': {'id': 'mount_a', 'part': 'bracket', 'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)},'mount_b': {'id': 'mount_b', 'part': 'bracket', 'kind': 'frame', 'pose_local_mm': (36.949, 0.0, 0.0)}}
    params = {'declared_min_wall_mm': 2.646, 'hole_diameter_mm': 5.075, 'pitch_mm': 36.949}
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
