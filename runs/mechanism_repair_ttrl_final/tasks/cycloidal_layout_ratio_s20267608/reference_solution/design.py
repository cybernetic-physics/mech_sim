# auto-generated; do not edit by hand. See mech_bench.generators.
from pathlib import Path


def build_design(out_dir: Path) -> dict:
    PINS = 14
    RATIO = 13.0
    ECC_MM = 1.273
    parts = [
        {'id': 'housing', 'role': 'ground', 'mass_kg': 0.0, 'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'ring_pin_count': PINS}},
        {'id': 'eccentric_input', 'role': 'eccentric_input', 'mass_kg': 0.04, 'com_local_mm': (0.0, 0.0, 0.0), 'params': {'eccentricity_mm': ECC_MM}},
        {'id': 'cycloidal_disc', 'role': 'cycloidal_disc', 'mass_kg': 0.08, 'com_local_mm': (ECC_MM, 0.0, 0.0), 'params': {'lobes': PINS - 1}},
        {'id': 'output_carrier', 'role': 'output_carrier', 'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0)},
    ]
    joints = [
        {'id': 'input_axis', 'type': 'revolute', 'parent': 'housing', 'child': 'eccentric_input', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
        {'id': 'eccentric_disc_axis', 'type': 'revolute', 'parent': 'eccentric_input', 'child': 'cycloidal_disc', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (ECC_MM, 0.0, 0.0)},
        {'id': 'output_axis', 'type': 'revolute', 'parent': 'housing', 'child': 'output_carrier', 'axis_world': (0.0, 0.0, 1.0), 'anchor_world_mm': (0.0, 0.0, 0.0)},
    ]
    ports = {
        'input_port': {'id': 'input_port', 'part': 'input_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
        'output_port': {'id': 'output_port', 'part': 'output_axis', 'kind': 'revolute_joint', 'pose_local_mm': (0.0, 0.0, 0.0)},
    }
    return {
        'schema_version': 'design_ir.v2',
        'parts': parts,
        'joints': joints,
        'ports': ports,
        'params': {
            'ring_pin_count': PINS,
            'disc_lobe_count': PINS - 1,
            'declared_ratio': RATIO,
            'eccentricity_mm': ECC_MM,
        },
    }
