# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    parts = [{'id': 'collar', 'role': 'frame', 'mass_kg': 0.05, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},{'id': 'shaft', 'role': 'shaft', 'mass_kg': 0.02, 'com_local_mm': (0.0, 0.0, 0.0)}]
    joints = [{'id': 'shaft_clamp', 'type': 'fixed', 'parent': 'collar', 'child': 'shaft', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)}]
    ports = {'collar_face': {'id': 'collar_face', 'part': 'collar', 'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)},'shaft_origin': {'id': 'shaft_origin', 'part': 'shaft', 'kind': 'frame', 'pose_local_mm': (0.0, 0.0, 0.0)}}
    params = {'shaft_diameter_mm': 8.527, 'collar_inner_diameter_mm': 9.021, 'declared_clearance_mm': 0.494}
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': params,
    }
