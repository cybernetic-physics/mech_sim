"""Tier 2 (transmission-analytic) and Tier 3 (contact-dynamics stub)
generators.

The Tier-2 families all reduce to a declared-ratio check:

* ``spur_gear_ratio_analytic`` — declared ``ratio = teeth_out / teeth_in``.
* ``rack_pinion_conversion``   — declared ``linear_per_rev = 2π · pitch_radius``.
* ``belt_pulley_ratio``        — declared ``ratio = D_out / D_in``.

Reference solutions encode the correct relationship; negative
controls perturb the declared value (wrong_ratio, hard-gate passes,
dense score drops) or remove a required port (missing_port, hard-gate
fails).

The Tier-3 stubs (``contact_gear_pair_stub``, ``cycloidal_lowN_stub``)
declare contact-force or torque-load probes. When the contact adapter is
available but the reference lacks credible contact geometry, the evaluator
surfaces ``missing_contact``. If the adapter is unavailable in a deployment,
the same probes surface ``capability_unavailable``.
"""

from __future__ import annotations

import random
from typing import Any

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    common_metadata,
    make_task_id,
)
from mech_bench.generators.static_fit import _negative_overlay


_PUBLIC_HEAD = (
    "# auto-generated; do not edit by hand. See mech_bench.generators.\n"
)


# --------------------------------------------------------------------- #
# Spur gear ratio (analytic)                                            #
# --------------------------------------------------------------------- #


def _two_gear_reference_py(teeth_in: int, teeth_out: int,
                             declared_ratio: float) -> str:
    return (
        _PUBLIC_HEAD
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    TEETH_IN = {teeth_in}\n"
        + f"    TEETH_OUT = {teeth_out}\n"
        + f"    DECLARED_RATIO = {declared_ratio}\n"
        + "    parts = [\n"
        + "        {'id': 'frame', 'role': 'ground', 'mass_kg': 0.0, "
        "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'pinion', 'role': 'gear_input', 'mass_kg': 0.02, "
        "'com_local_mm': (0.0, 0.0, 0.0), "
        "'params': {'teeth': TEETH_IN}},\n"
        + "        {'id': 'gear', 'role': 'gear_output', 'mass_kg': 0.05, "
        "'com_local_mm': (0.0, 0.0, 0.0), "
        "'params': {'teeth': TEETH_OUT}},\n"
        + "    ]\n"
        + "    joints = [\n"
        + "        {'id': 'pinion_axis', 'type': 'revolute', "
        "'parent': 'frame', 'child': 'pinion', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'gear_axis', 'type': 'revolute', "
        "'parent': 'frame', 'child': 'gear', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (40.0, 0.0, 0.0)},\n"
        + "    ]\n"
        + "    ports = {\n"
        + "        'input_port': {'id': 'input_port', "
        "'part': 'pinion_axis', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        'output_port': {'id': 'output_port', "
        "'part': 'gear_axis', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "    }\n"
        + "    return {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': {\n"
        + "            'teeth_in': TEETH_IN,\n"
        + "            'teeth_out': TEETH_OUT,\n"
        + "            'declared_ratio': DECLARED_RATIO,\n"
        + "        },\n"
        + "    }\n"
    )


class SpurGearRatioAnalyticGenerator(TaskGenerator):
    family = "spur_gear_ratio_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 5050)
        teeth_in = rng.choice([12, 14, 15, 16, 18, 20, 24])
        teeth_out = teeth_in * rng.choice([2, 3, 4, 5])
        ratio = round(teeth_out / teeth_in, 6)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Spur gear ratio (analytic)\n\n"
            f"Design a two-gear reducer with pinion ({teeth_in} teeth) "
            f"and gear ({teeth_out} teeth).\n\n"
            f"* Declare `params.declared_ratio` = teeth_out / teeth_in "
            f"= {ratio}.\n"
            "* Required ports: `input_port` (revolute_joint), "
            "`output_port` (revolute_joint), both grounded.\n"
            "* Mobility = 2 (two free axes, ungeared in this analytic "
            "tier).\n"
        )

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": (
                    f"Two-gear analytic ratio = {ratio}."
                ),
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "ports", "ratio"],
                "hidden_split": ["ratio"],
            },
        }

        def _cfg(target_ratio: float, tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 2, "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_grounded": ["input_port", "output_port"],
                     "require_kinds": {"input_port": "revolute_joint",
                                        "output_port": "revolute_joint"},
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ratio", "type": "analytic_param_check",
                     "path": "params.declared_ratio",
                     "expected": float(target_ratio),
                     "comparator": "eq",
                     "tolerance_pct": float(tol_pct),
                     "failure_code": "wrong_ratio",
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "ports.ports_required",
                        "ratio.observed", "ratio.expected",
                    ],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = _two_gear_reference_py(teeth_in, teeth_out, ratio)
        negatives = {
            "wrong_ratio": _negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 1.5, 6)}"
            ),
            "missing_port": _negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = {
            "description": "Tier 2 spur_gear_ratio_analytic negatives.",
            "controls": [
                {
                    "id": "wrong_ratio",
                    "submission": "negative_solutions/wrong_ratio",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.5,
                },
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(ratio, tol_pct=2.0),
            eval_config_hidden_toml=_cfg(ratio, tol_pct=1.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     teeth_in=teeth_in,
                                     teeth_out=teeth_out,
                                     ratio=ratio),
        )


# --------------------------------------------------------------------- #
# Rack-and-pinion conversion (analytic)                                 #
# --------------------------------------------------------------------- #


def _rack_pinion_reference_py(pitch_radius_mm: float,
                                linear_per_rev_mm: float) -> str:
    return (
        _PUBLIC_HEAD
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    PITCH_R = {pitch_radius_mm}\n"
        + f"    LIN_PER_REV = {linear_per_rev_mm}\n"
        + "    parts = [\n"
        + "        {'id': 'frame', 'role': 'ground', 'mass_kg': 0.0, "
        "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'pinion', 'role': 'pinion', 'mass_kg': 0.02, "
        "'com_local_mm': (0.0, 0.0, 0.0), "
        "'params': {'pitch_radius_mm': PITCH_R}},\n"
        + "        {'id': 'rack', 'role': 'rack', 'mass_kg': 0.04, "
        "'com_local_mm': (0.0, 0.0, 0.0)},\n"
        + "    ]\n"
        + "    joints = [\n"
        + "        {'id': 'pinion_axis', 'type': 'revolute', "
        "'parent': 'frame', 'child': 'pinion', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'rack_slide', 'type': 'prismatic', "
        "'parent': 'frame', 'child': 'rack', "
        "'axis_world': (1.0, 0.0, 0.0), "
        "'anchor_world_mm': (0.0, -PITCH_R, 0.0)},\n"
        + "    ]\n"
        + "    ports = {\n"
        + "        'input_port': {'id': 'input_port', "
        "'part': 'pinion_axis', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        'output_port': {'id': 'output_port', "
        "'part': 'rack_slide', 'kind': 'prismatic_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "    }\n"
        + "    return {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': {\n"
        + "            'pitch_radius_mm': PITCH_R,\n"
        + "            'declared_linear_per_rev_mm': LIN_PER_REV,\n"
        + "        },\n"
        + "    }\n"
    )


class RackPinionConversionGenerator(TaskGenerator):
    family = "rack_pinion_conversion"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        import math
        rng = random.Random(seed + 8181)
        pitch_radius_mm = round(rng.uniform(8.0, 25.0), 3)
        linear_per_rev_mm = round(2.0 * math.pi * pitch_radius_mm, 4)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Rack-and-pinion conversion\n\n"
            f"Design a rack-and-pinion with pinion pitch radius "
            f"{pitch_radius_mm} mm.\n\n"
            f"* Declare `params.declared_linear_per_rev_mm` = "
            f"2π · pitch_radius = {linear_per_rev_mm} mm.\n"
            f"* The observed output/input velocity ratio must be "
            f"pitch_radius = {pitch_radius_mm} mm/rad.\n"
            "* Ports: `input_port` (revolute_joint), `output_port` "
            "(prismatic_joint).\n"
            "* Mobility = 2 (kinematic rack-pinion tier).\n"
        )

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": (
                    f"Rack-and-pinion: declare linear/rev = "
                    f"{linear_per_rev_mm} mm."
                ),
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "ports", "linear_per_rev"],
                "hidden_split": ["linear_per_rev"],
            },
        }

        def _cfg(
            target: float,
            velocity_ratio: float,
            tol_pct: float,
        ) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 2, "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_grounded": ["input_port"],
                     "require_kinds": {"input_port": "revolute_joint",
                                        "output_port": "prismatic_joint"},
                     "hard_gate": True, "severity": "critical"},
                    {"id": "linear_per_rev", "type": "analytic_param_check",
                     "path": "params.declared_linear_per_rev_mm",
                     "expected": float(target),
                     "comparator": "eq",
                     "tolerance_pct": float(tol_pct),
                     "failure_code": "wrong_ratio",
                     "weight": 1.0, "severity": "major"},
                    {"id": "linear_velocity", "type": "port_velocity_ratio",
                     "input_port": "input_port",
                     "output_port": "output_port",
                     "expected": float(velocity_ratio),
                     "tolerance_pct": float(tol_pct),
                     "min_abs_input_velocity": 1e-6,
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "linear_per_rev.observed",
                        "linear_per_rev.expected",
                        "linear_velocity.ratio_observed",
                        "linear_velocity.ratio_expected",
                    ],
                    "hidden_metrics": [
                        "linear_per_rev.error_pct",
                        "linear_velocity.ratio_error_pct",
                    ],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = _rack_pinion_reference_py(pitch_radius_mm,
                                              linear_per_rev_mm)
        negatives = {
            "wrong_ratio": _negative_overlay(
                f"    ir['params']['declared_linear_per_rev_mm'] = "
                f"{round(linear_per_rev_mm * 0.7, 4)}"
            ),
            "missing_port": _negative_overlay(
                "    del ir['ports']['input_port']"
            ),
            "wrong_pinion_geometry": _negative_overlay(
                "    for part in ir['parts']:\n"
                "        if part['id'] == 'pinion':\n"
                "            part['params']['pitch_radius_mm'] *= 0.7"
            ),
        }
        expected = {
            "description": "Tier 2 rack_pinion_conversion negatives.",
            "controls": [
                {
                    "id": "wrong_ratio",
                    "submission": "negative_solutions/wrong_ratio",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.51,
                },
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "wrong_pinion_geometry",
                    "submission": "negative_solutions/wrong_pinion_geometry",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.8,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(
                linear_per_rev_mm, pitch_radius_mm, tol_pct=1.5),
            eval_config_hidden_toml=_cfg(
                linear_per_rev_mm, pitch_radius_mm, tol_pct=0.8),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     pitch_radius_mm=pitch_radius_mm,
                                     target_linear_per_rev_mm=linear_per_rev_mm),
        )


# --------------------------------------------------------------------- #
# Belt-pulley ratio (analytic)                                          #
# --------------------------------------------------------------------- #


def _belt_pulley_reference_py(d_in: float, d_out: float,
                                  ratio: float) -> str:
    return (
        _PUBLIC_HEAD
        + "from pathlib import Path\n\n\n"
        + "def build_design(out_dir: Path) -> dict:\n"
        + f"    D_IN = {d_in}\n"
        + f"    D_OUT = {d_out}\n"
        + f"    RATIO = {ratio}\n"
        + "    parts = [\n"
        + "        {'id': 'frame', 'role': 'ground', 'mass_kg': 0.0, "
        "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'drive', 'role': 'pulley_drive', "
        "'mass_kg': 0.03, 'com_local_mm': (0.0, 0.0, 0.0), "
        "'params': {'diameter_mm': D_IN}},\n"
        + "        {'id': 'driven', 'role': 'pulley_driven', "
        "'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0), "
        "'params': {'diameter_mm': D_OUT}},\n"
        + "    ]\n"
        + "    joints = [\n"
        + "        {'id': 'drive_axis', 'type': 'revolute', "
        "'parent': 'frame', 'child': 'drive', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
        + "        {'id': 'driven_axis', 'type': 'revolute', "
        "'parent': 'frame', 'child': 'driven', "
        "'axis_world': (0.0, 0.0, 1.0), "
        "'anchor_world_mm': (120.0, 0.0, 0.0)},\n"
        + "    ]\n"
        + "    ports = {\n"
        + "        'input_port': {'id': 'input_port', "
        "'part': 'drive_axis', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "        'output_port': {'id': 'output_port', "
        "'part': 'driven_axis', 'kind': 'revolute_joint', "
        "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
        + "    }\n"
        + "    return {\n"
        + "        'schema_version': 'design_ir.v2',\n"
        + "        'parts': parts,\n"
        + "        'joints': joints,\n"
        + "        'ports': ports,\n"
        + "        'params': {\n"
        + "            'drive_diameter_mm': D_IN,\n"
        + "            'driven_diameter_mm': D_OUT,\n"
        + "            'declared_ratio': RATIO,\n"
        + "        },\n"
        + "    }\n"
    )


class BeltPulleyRatioGenerator(TaskGenerator):
    family = "belt_pulley_ratio"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 9192)
        d_in = round(rng.uniform(20.0, 40.0), 2)
        d_out = round(d_in * rng.choice([1.5, 2.0, 2.5, 3.0, 4.0]), 2)
        ratio = round(d_out / d_in, 6)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Belt-pulley ratio (analytic)\n\n"
            f"Design a two-pulley belt drive with drive Ø{d_in} mm and "
            f"driven Ø{d_out} mm.\n\n"
            f"* Declare `params.declared_ratio` = D_out / D_in = {ratio}.\n"
            f"* The observed output/input angular velocity ratio must be "
            f"D_in / D_out = {round(d_in / d_out, 6)}.\n"
            "* Ports: `input_port`, `output_port` (revolute_joint, "
            "both grounded).\n"
            "* Mobility = 2 (ungeared analytic tier).\n"
        )

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": f"Belt-pulley analytic ratio = {ratio}.",
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "ports", "ratio"],
                "hidden_split": ["ratio"],
            },
        }

        def _cfg(target_ratio: float, tol_pct: float) -> dict[str, Any]:
            speed_ratio = 1.0 / target_ratio
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 2, "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_grounded": ["input_port", "output_port"],
                     "require_kinds": {"input_port": "revolute_joint",
                                        "output_port": "revolute_joint"},
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ratio", "type": "analytic_param_check",
                     "path": "params.declared_ratio",
                     "expected": float(target_ratio),
                     "comparator": "eq",
                     "tolerance_pct": float(tol_pct),
                     "failure_code": "wrong_ratio",
                     "weight": 1.0, "severity": "major"},
                    {"id": "speed_ratio", "type": "port_velocity_ratio",
                     "input_port": "input_port",
                     "output_port": "output_port",
                     "expected": float(speed_ratio),
                     "tolerance_pct": float(tol_pct),
                     "min_abs_input_velocity": 1e-6,
                     "weight": 1.0, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "ports.ports_required",
                        "ratio.observed", "ratio.expected",
                        "speed_ratio.ratio_observed",
                        "speed_ratio.ratio_expected",
                    ],
                    "hidden_metrics": [
                        "ratio.error_pct",
                        "speed_ratio.ratio_error_pct",
                    ],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = _belt_pulley_reference_py(d_in, d_out, ratio)
        negatives = {
            "wrong_ratio": _negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 0.5, 6)}"
            ),
            "missing_port": _negative_overlay(
                "    del ir['ports']['output_port']"
            ),
            "wrong_pulley_geometry": _negative_overlay(
                "    for part in ir['parts']:\n"
                "        if part['id'] == 'driven':\n"
                "            part['params']['diameter_mm'] *= 0.5"
            ),
        }
        expected = {
            "description": "Tier 2 belt_pulley_ratio negatives.",
            "controls": [
                {
                    "id": "wrong_ratio",
                    "submission": "negative_solutions/wrong_ratio",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.51,
                },
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "wrong_pulley_geometry",
                    "submission": "negative_solutions/wrong_pulley_geometry",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.8,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(ratio, tol_pct=2.0),
            eval_config_hidden_toml=_cfg(ratio, tol_pct=1.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     drive_d_mm=d_in,
                                     driven_d_mm=d_out, ratio=ratio),
        )


# --------------------------------------------------------------------- #
# Cycloidal layout ratio (analytic)                                     #
# --------------------------------------------------------------------- #


class CycloidalLayoutRatioGenerator(TaskGenerator):
    family = "cycloidal_layout_ratio"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 6060)
        pins = rng.choice([8, 10, 12, 14])
        eccentricity_mm = round(rng.uniform(1.0, 2.5), 3)
        ratio = float(pins - 1)
        task_id = make_task_id(self.family, seed)

        prompt = (
            "# Cycloidal reducer layout ratio\n\n"
            f"Design a single-stage cycloidal reducer layout with {pins} "
            f"fixed ring pins and target reduction ratio {ratio:g}:1.\n\n"
            "* Required topology: fixed housing/ring, eccentric input, "
            "cycloidal disc, and output carrier.\n"
            "* Required ports: `input_port` and `output_port`, both grounded "
            "revolute_joint ports.\n"
            f"* Declare `params.ring_pin_count = {pins}` and "
            f"`params.declared_ratio = {ratio:g}`.\n"
            f"* Declare `params.eccentricity_mm = {eccentricity_mm}`.\n"
        )

        ref_py = (
            _PUBLIC_HEAD
            + "from pathlib import Path\n\n\n"
            + "def build_design(out_dir: Path) -> dict:\n"
            + f"    PINS = {pins}\n"
            + f"    RATIO = {ratio}\n"
            + f"    ECC_MM = {eccentricity_mm}\n"
            + "    parts = [\n"
            + "        {'id': 'housing', 'role': 'ground', 'mass_kg': 0.0, "
            "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0), "
            "'params': {'ring_pin_count': PINS}},\n"
            + "        {'id': 'eccentric_input', 'role': 'eccentric_input', "
            "'mass_kg': 0.04, 'com_local_mm': (0.0, 0.0, 0.0), "
            "'params': {'eccentricity_mm': ECC_MM}},\n"
            + "        {'id': 'cycloidal_disc', 'role': 'cycloidal_disc', "
            "'mass_kg': 0.08, 'com_local_mm': (ECC_MM, 0.0, 0.0), "
            "'params': {'lobes': PINS - 1}},\n"
            + "        {'id': 'output_carrier', 'role': 'output_carrier', "
            "'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
            + "    ]\n"
            + "    joints = [\n"
            + "        {'id': 'input_axis', 'type': 'revolute', "
            "'parent': 'housing', 'child': 'eccentric_input', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
            + "        {'id': 'eccentric_disc_axis', 'type': 'revolute', "
            "'parent': 'eccentric_input', 'child': 'cycloidal_disc', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (ECC_MM, 0.0, 0.0)},\n"
            + "        {'id': 'output_axis', 'type': 'revolute', "
            "'parent': 'housing', 'child': 'output_carrier', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
            + "    ]\n"
            + "    ports = {\n"
            + "        'input_port': {'id': 'input_port', "
            "'part': 'input_axis', 'kind': 'revolute_joint', "
            "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
            + "        'output_port': {'id': 'output_port', "
            "'part': 'output_axis', 'kind': 'revolute_joint', "
            "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
            + "    }\n"
            + "    return {\n"
            + "        'schema_version': 'design_ir.v2',\n"
            + "        'parts': parts,\n"
            + "        'joints': joints,\n"
            + "        'ports': ports,\n"
            + "        'params': {\n"
            + "            'ring_pin_count': PINS,\n"
            + "            'disc_lobe_count': PINS - 1,\n"
            + "            'declared_ratio': RATIO,\n"
            + "            'eccentricity_mm': ECC_MM,\n"
            + "        },\n"
            + "    }\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 3, "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_grounded": ["input_port", "output_port"],
                     "require_kinds": {
                         "input_port": "revolute_joint",
                         "output_port": "revolute_joint"},
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ratio", "type": "analytic_param_check",
                     "path": "params.declared_ratio",
                     "expected": float(ratio),
                     "comparator": "eq",
                     "tolerance_pct": float(tol_pct),
                     "failure_code": "wrong_ratio",
                     "weight": 0.7, "severity": "major"},
                    {"id": "pin_count", "type": "analytic_param_check",
                     "path": "params.ring_pin_count",
                     "expected": float(pins),
                     "comparator": "eq",
                     "failure_code": "wrong_topology",
                     "weight": 0.3, "severity": "major"},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed", "mobility.expected",
                        "ratio.observed", "ratio.expected",
                        "pin_count.observed", "pin_count.expected",
                    ],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        negatives = {
            "wrong_ratio": _negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(max(1.0, ratio - 2.0), 6)}"
            ),
            "missing_output_port": _negative_overlay(
                "    del ir['ports']['output_port']"
            ),
            "wrong_pin_count": _negative_overlay(
                "    ir['params']['ring_pin_count'] = "
                f"{pins + 2}"
            ),
            "wrong_mobility_extra_fixed": _negative_overlay(
                "    ir['joints'].append({\n"
                "        'id': 'disc_lock', 'type': 'fixed',\n"
                "        'parent': 'housing', 'child': 'cycloidal_disc',\n"
                "        'axis_world': (0.0, 0.0, 1.0),\n"
                "        'anchor_world_mm': (0.0, 0.0, 0.0)})"
            ),
        }
        expected = {
            "description": (
                "Tier 2 cycloidal_layout_ratio negative controls."),
            "controls": [
                {
                    "id": "wrong_ratio",
                    "submission": "negative_solutions/wrong_ratio",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.8,
                },
                {
                    "id": "missing_output_port",
                    "submission": "negative_solutions/missing_output_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "wrong_pin_count",
                    "submission": "negative_solutions/wrong_pin_count",
                    "expected_failure_codes": ["wrong_topology"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.9,
                },
                {
                    "id": "wrong_mobility_extra_fixed",
                    "submission":
                        "negative_solutions/wrong_mobility_extra_fixed",
                    "expected_failure_codes": ["wrong_mobility"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
            ],
        }
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 3,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": (
                    f"Cycloidal layout with {pins} ring pins and "
                    f"{ratio:g}:1 declared ratio."
                ),
                "allow_massless_links": False,
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     ring_pin_count=pins,
                                     target_ratio=ratio,
                                     eccentricity_mm=eccentricity_mm),
        )


# --------------------------------------------------------------------- #
# Tier 3 stubs — contact-dynamics smoke tasks.                          #
# --------------------------------------------------------------------- #


class ContactGearPairStubGenerator(TaskGenerator):
    family = "contact_gear_pair_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 22222)
        teeth_in = rng.choice([12, 14, 16, 18])
        teeth_out = teeth_in * rng.choice([2, 3, 4])

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Contact gear pair (stub)\n\n"
            f"Two-gear contact-loaded design with {teeth_in}/{teeth_out} "
            "teeth.\n\n"
            "This task requires a contact-force-capable adapter and "
            "credible contact geometry. In this stub, the reference lacks "
            "contact geometry, so an available contact adapter should "
            "surface `missing_contact`.\n"
        )

        ref_py = _two_gear_reference_py(teeth_in, teeth_out,
                                          teeth_out / teeth_in)

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": "Contact-loaded two-gear pair.",
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "ports"],
                "hidden_split": ["contact"],
            },
            "capability": {
                "requires_adapter": "contact_forces",
                "expect_capability_unavailable": False,
            },
        }

        def _cfg(min_force: float) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "mobility", "type": "dof_grubler",
                     "space": "planar", "expected": 2, "tolerance": 0,
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_grounded": ["input_port", "output_port"],
                     "hard_gate": True, "severity": "critical"},
                    {"id": "contact", "type": "contact_engagement",
                     "required_pairs": ["pinion:gear"],
                     "min_rms_force_N": float(min_force),
                     "min_engagement_fraction": 0.2,
                     "weight": 1.0, "severity": "critical",
                     "hard_gate": True},
                ],
                "feedback": {
                    "public_metrics": [
                        "mobility.observed",
                        "contact.contact.pinion:gear.rms_N",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports", "contact"]},
                "adapters": {
                    "chrono_contact": {"samples": 720},
                },
            }

        expected = {
            "description": (
                "Tier 3 contact_gear_pair_stub missing-contact regression."),
            "controls": [
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "no_contact_geometry",
                    "submission": "negative_solutions/no_contact_geometry",
                    "expected_failure_codes": ["missing_contact"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
            ],
        }
        negatives = {
            "missing_port": _negative_overlay(
                "    del ir['ports']['output_port']"
            ),
            "no_contact_geometry": _negative_overlay(
                "    ir['params']['declared_ratio'] = -1.0"
            ),
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(0.5),
            eval_config_hidden_toml=_cfg(1.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     teeth_in=teeth_in,
                                     teeth_out=teeth_out,
                                     requires_adapter="contact_forces"),
        )


class CycloidalLowNStubGenerator(TaskGenerator):
    family = "cycloidal_lowN_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 33333)
        N_pins = rng.choice([8, 10, 12])
        target_ratio = float(N_pins - 1)

        task_id = make_task_id(self.family, seed)
        prompt = (
            "# Low-N cycloidal stub\n\n"
            f"Single-stage cycloidal reducer with {N_pins} ring pins "
            f"(target ratio {target_ratio:g}).\n\n"
            "Requires torque-load and contact-force capabilities; "
            "evaluated by the Chrono contact adapter when available.\n"
        )

        ref_py = (
            _PUBLIC_HEAD
            + "from pathlib import Path\n\n\n"
            + "def build_design(out_dir: Path) -> dict:\n"
            + f"    N_PINS = {N_pins}\n"
            + f"    RATIO = {target_ratio}\n"
            + "    parts = [\n"
            + "        {'id': 'housing', 'role': 'ground', 'mass_kg': 0.0, "
            "'fixed': True, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
            + "        {'id': 'eccentric', 'role': 'eccentric', "
            "'mass_kg': 0.05, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
            + "        {'id': 'disc', 'role': 'cycloidal_disc', "
            "'mass_kg': 0.08, 'com_local_mm': (0.0, 0.0, 0.0), "
            "'params': {'pins': N_PINS}},\n"
            + "        {'id': 'carrier', 'role': 'carrier', "
            "'mass_kg': 0.04, 'com_local_mm': (0.0, 0.0, 0.0)},\n"
            + "    ]\n"
            + "    joints = [\n"
            + "        {'id': 'input_revolute', 'type': 'revolute', "
            "'parent': 'housing', 'child': 'eccentric', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
            + "        {'id': 'eccentric_disc', 'type': 'revolute', "
            "'parent': 'eccentric', 'child': 'disc', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (1.0, 0.0, 0.0)},\n"
            + "        {'id': 'output_revolute', 'type': 'revolute', "
            "'parent': 'housing', 'child': 'carrier', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
            + "        {'id': 'ring_contact', 'type': 'contact_pair', "
            "'parent': 'housing', 'child': 'disc', "
            "'axis_world': (0.0, 0.0, 1.0), "
            "'anchor_world_mm': (0.0, 0.0, 0.0)},\n"
            + "    ]\n"
            + "    ports = {\n"
            + "        'input_port': {'id': 'input_port', "
            "'part': 'input_revolute', 'kind': 'revolute_joint', "
            "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
            + "        'output_port': {'id': 'output_port', "
            "'part': 'output_revolute', 'kind': 'revolute_joint', "
            "'pose_local_mm': (0.0, 0.0, 0.0)},\n"
            + "    }\n"
            + "    return {\n"
            + "        'schema_version': 'design_ir.v2',\n"
            + "        'parts': parts,\n"
            + "        'joints': joints,\n"
            + "        'ports': ports,\n"
            + "        'params': {\n"
            + "            'pins': N_PINS,\n"
            + "            'declared_ratio': RATIO,\n"
            + "        },\n"
            + "    }\n"
        )

        task_toml: dict[str, Any] = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": (
                    f"Cycloidal reducer with {N_pins} ring pins; "
                    f"declared ratio {target_ratio:g}."
                ),
                "allow_massless_links": False,
                "ground_required": True,
            },
            "visibility": {
                "public_split": ["mobility", "ports"],
                "hidden_split": ["torque"],
            },
            "capability": {
                "requires_adapter": "rigid_body_dynamics+contact_forces",
                "expect_capability_unavailable": False,
            },
        }

        def _cfg(input_speed: float, load: float) -> dict[str, Any]:
            return {
                "probes": [
                    {"id": "ports", "type": "required_ports",
                     "ports": ["input_port", "output_port"],
                     "require_kinds": {"input_port": "revolute_joint",
                                        "output_port": "revolute_joint"},
                     "hard_gate": True, "severity": "critical"},
                    {"id": "ratio", "type": "analytic_param_check",
                     "path": "params.declared_ratio",
                     "expected": float(target_ratio),
                     "comparator": "eq",
                     "tolerance_pct": 1.0,
                     "failure_code": "wrong_ratio",
                     "weight": 0.3, "severity": "major"},
                    {"id": "torque", "type": "torque_load_trial",
                     "input_port": "input_port",
                     "output_port": "output_port",
                     "input_speed_rad_s": float(input_speed),
                     "output_load_Nm": float(load),
                     "min_output_speed_rad_s": 0.001,
                     "max_power_error_pct": 25.0,
                     "max_torque_ripple_pct": 30.0,
                     "weight": 0.7, "severity": "critical"},
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "ratio.expected",
                        "torque.output_speed_observed_rad_s",
                    ],
                    "hidden_metrics": ["torque.torque_ripple_pct"],
                },
                "hard_gate": {"require": ["ports"]},
                "adapters": {
                    "chrono_contact": {"samples": 360},
                },
            }

        negatives = {
            "wrong_ratio": _negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(target_ratio * 0.5, 6)}"
            ),
            "missing_port": _negative_overlay(
                "    del ir['ports']['input_port']"
            ),
        }
        expected = {
            "description": (
                "Tier 3 cycloidal_lowN_stub — Chrono contact regression."),
            "controls": [
                {
                    "id": "wrong_ratio",
                    "submission": "negative_solutions/wrong_ratio",
                    "expected_failure_codes":
                        ["wrong_ratio"],
                    "expected_score_below": 0.5,
                },
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(input_speed=10.0, load=0.05),
            eval_config_hidden_toml=_cfg(input_speed=20.0, load=0.10),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     pins=N_pins,
                                     target_ratio=target_ratio,
                                     requires_adapter="rigid_body_dynamics+contact_forces"),
        )
