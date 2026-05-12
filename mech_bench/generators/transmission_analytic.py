"""Tier 2 — analytic transmission / simple-machine tasks.

Ten families that reduce to declared-ratio / declared-relation checks.
Each family advertises an input + output revolute (or prismatic) port,
mobility 2 (uncoupled in the analytic tier), and one or more analytic
parameter checks against the closed-form relation.
"""

from __future__ import annotations

import math
import random
from typing import Any

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    common_metadata,
    make_task_id,
)
from mech_bench.generators.common_designs import (
    dof_probe,
    frame_port,
    make_basic_design_py,
    make_expected_failures,
    make_ground_part,
    make_negative_overlay,
    make_revolute_part,
    make_slider_part,
    param_check_probe,
    prismatic_joint,
    prismatic_joint_port,
    required_ports_probe,
    revolute_joint,
    revolute_joint_port,
)


def _basic_pair_design(part_a: str, part_b: str,
                       second_anchor_mm: float,
                       prismatic_b: bool = False,
                       extra_params: dict[str, Any] | None = None,
                       ) -> tuple[list[dict], list[dict], dict[str, dict]]:
    parts = [
        make_ground_part("frame"),
        make_revolute_part(part_a, "input", 0.02),
    ]
    if prismatic_b:
        parts.append(make_slider_part(part_b))
        joints = [
            revolute_joint("input_axis", "frame", part_a, (0.0, 0.0, 0.0)),
            prismatic_joint(
                "output_axis", "frame", part_b,
                anchor_world_mm=(0.0, 0.0, 0.0)),
        ]
        ports = {
            "input_port": revolute_joint_port(
                "input_port", "input_axis"),
            "output_port": prismatic_joint_port(
                "output_port", "output_axis"),
        }
    else:
        parts.append(make_revolute_part(part_b, "output", 0.05))
        joints = [
            revolute_joint("input_axis", "frame", part_a, (0.0, 0.0, 0.0)),
            revolute_joint(
                "output_axis", "frame", part_b,
                (second_anchor_mm, 0.0, 0.0)),
        ]
        ports = {
            "input_port": revolute_joint_port(
                "input_port", "input_axis"),
            "output_port": revolute_joint_port(
                "output_port", "output_axis"),
        }
    return parts, joints, ports


# --------------------------------------------------------------------- #
# 21. compound_gear_ratio_analytic                                      #
# --------------------------------------------------------------------- #


class CompoundGearRatioAnalyticGenerator(TaskGenerator):
    family = "compound_gear_ratio_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4121)
        p1 = rng.choice([12, 14, 16])
        g1 = p1 * rng.choice([2, 3, 4])
        p2 = rng.choice([12, 14, 16])
        g2 = p2 * rng.choice([2, 3])
        ratio = round((g1 / p1) * (g2 / p2), 6)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "pinion", "gear", 40.0)
        params = {
            "stage1": {"pinion_teeth": p1, "gear_teeth": g1},
            "stage2": {"pinion_teeth": p2, "gear_teeth": g2},
            "declared_ratio": ratio,
        }
        prompt = (
            "# Compound gear ratio (analytic)\n\n"
            f"Two-stage gear train: (g1/p1) × (g2/p2) = "
            f"({g1}/{p1}) × ({g2}/{p2}) = {ratio}.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "ratio", "params.declared_ratio", ratio,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "ratio.expected"],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_stage_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 1.6, 6)}"
            ),
            "missing_output_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_stage_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_output_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": f"Compound gear ratio = {ratio}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     teeth=[p1, g1, p2, g2],
                                     ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 22. idler_gear_direction_analytic                                     #
# --------------------------------------------------------------------- #


class IdlerGearDirectionAnalyticGenerator(TaskGenerator):
    family = "idler_gear_direction_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4222)
        idlers = rng.choice([0, 1, 2, 3])
        direction = 1 if idlers % 2 == 0 else -1
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "pinion", "gear", 40.0)
        params = {
            "idler_count": int(idlers),
            "output_direction": int(direction),
        }
        prompt = (
            "# Idler gear direction (analytic)\n\n"
            f"With {idlers} idler gear(s), the output direction is "
            f"{direction}.\n"
        )

        def _cfg() -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "direction", "params.output_direction",
                        float(direction), tolerance_abs=0.0,
                        failure_code="wrong_ratio", weight=0.7,
                    ),
                    param_check_probe(
                        "idler_count", "params.idler_count",
                        float(idlers), tolerance_abs=0.0,
                        failure_code="invalid_artifact", weight=0.3,
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "direction.observed", "direction.expected",
                        "idler_count.observed",
                    ],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_sign": make_negative_overlay(
                "    ir['params']['output_direction'] = "
                "-ir['params']['output_direction']"
            ),
            "wrong_idler_count": make_negative_overlay(
                f"    ir['params']['idler_count'] = {idlers + 2}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_sign",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "wrong_idler_count",
                 "expected_failure_codes": ["invalid_artifact"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.9},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": f"Idler direction = {direction}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(),
            eval_config_hidden_toml=_cfg(),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     idler_count=idlers,
                                     output_direction=direction),
        )


# --------------------------------------------------------------------- #
# 23. planetary_fixed_ring_ratio_analytic                               #
# --------------------------------------------------------------------- #


class PlanetaryFixedRingRatioGenerator(TaskGenerator):
    family = "planetary_fixed_ring_ratio_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4323)
        sun = rng.choice([14, 16, 18, 20])
        ring = sun + 2 * rng.choice([10, 14, 18])
        ratio = round(1.0 + ring / sun, 6)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "sun", "carrier", 60.0)
        params = {
            "sun_teeth": sun, "ring_teeth": ring,
            "declared_ratio": ratio,
        }
        prompt = (
            "# Planetary (ring fixed) ratio (analytic)\n\n"
            f"Ratio = 1 + ring/sun = 1 + {ring}/{sun} = {ratio}.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "ratio", "params.declared_ratio", ratio,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "ratio.expected"],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 0.5, 6)}"
            ),
            "swapped_sun_ring": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(1.0 + sun / ring, 6)}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "swapped_sun_ring",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": f"Planetary fixed-ring ratio = {ratio}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, sun=sun, ring=ring,
                                     ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 24. planetary_fixed_sun_ratio_analytic                                #
# --------------------------------------------------------------------- #


class PlanetaryFixedSunRatioGenerator(TaskGenerator):
    family = "planetary_fixed_sun_ratio_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4424)
        sun = rng.choice([16, 18, 20])
        ring = sun + 2 * rng.choice([12, 16, 20])
        ratio = round(ring / (ring + sun), 6)
        task_id = make_task_id(self.family, seed)

        parts = [
            make_ground_part("frame"),
            make_revolute_part("carrier", "carrier", 0.05),
            make_revolute_part("ring", "ring", 0.06),
        ]
        joints = [
            revolute_joint("input_axis", "frame", "carrier",
                           (0.0, 0.0, 0.0)),
            revolute_joint("output_axis", "frame", "ring",
                           (0.0, 0.0, 0.0)),
        ]
        ports = {
            "input_port": revolute_joint_port("input_port", "input_axis"),
            "output_port": revolute_joint_port("output_port", "output_axis"),
            "carrier_port": frame_port("carrier_port", "carrier"),
        }
        params = {
            "sun_teeth": sun, "ring_teeth": ring,
            "declared_ratio": ratio,
        }
        prompt = (
            "# Planetary (sun fixed) ratio (analytic)\n\n"
            f"Carrier→ring ratio = ring/(ring+sun) = {ratio}.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports",
                        ["input_port", "output_port", "carrier_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "ratio", "params.declared_ratio", ratio,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "ratio.expected"],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 0.4, 6)}"
            ),
            "missing_carrier_port": make_negative_overlay(
                "    del ir['ports']['carrier_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_carrier_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": [
                    "input_port", "output_port", "carrier_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": f"Planetary fixed-sun ratio = {ratio}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, sun=sun, ring=ring,
                                     ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 25. worm_gear_ratio_analytic                                          #
# --------------------------------------------------------------------- #


class WormGearRatioAnalyticGenerator(TaskGenerator):
    family = "worm_gear_ratio_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4525)
        worm_starts = rng.choice([1, 2, 3])
        wheel_teeth = rng.choice([30, 40, 50, 60])
        ratio = round(wheel_teeth / worm_starts, 6)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design("worm", "wheel", 40.0)
        params = {
            "worm_starts": worm_starts, "wheel_teeth": wheel_teeth,
            "declared_ratio": ratio,
        }
        prompt = (
            "# Worm gear ratio (analytic)\n\n"
            f"Ratio = wheel_teeth / worm_starts = "
            f"{wheel_teeth}/{worm_starts} = {ratio}.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "ratio", "params.declared_ratio", ratio,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "ratio.expected"],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 0.5, 6)}"
            ),
            "wrong_starts": make_negative_overlay(
                f"    ir['params']['worm_starts'] = {worm_starts + 1}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                # wrong_starts mutates the metadata but ratio still matches
                # — declared_ratio remains the truth on the IR, so we
                # mark this as the soft-fail control.
                {"id": "wrong_starts",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": f"Worm gear ratio = {ratio}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     worm_starts=worm_starts,
                                     wheel_teeth=wheel_teeth,
                                     ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 26. lead_screw_linear_travel                                          #
# --------------------------------------------------------------------- #


class LeadScrewLinearTravelGenerator(TaskGenerator):
    family = "lead_screw_linear_travel"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4626)
        lead = round(rng.uniform(2.0, 10.0), 3)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "screw", "nut", 0.0, prismatic_b=True)
        params = {
            "lead_mm": lead,
            "declared_travel_per_rev_mm": lead,
        }
        prompt = (
            "# Lead screw linear travel\n\n"
            f"Declare `params.declared_travel_per_rev_mm` = lead_mm = "
            f"{lead}.\n"
            "* Input revolute, output prismatic.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "prismatic_joint"},
                    ),
                    param_check_probe(
                        "travel", "params.declared_travel_per_rev_mm",
                        lead, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "travel.observed", "travel.expected"],
                    "hidden_metrics": ["travel.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_lead": make_negative_overlay(
                f"    ir['params']['declared_travel_per_rev_mm'] = "
                f"{round(lead * 0.5, 4)}"
            ),
            "wrong_output_kind": make_negative_overlay(
                "    ir['ports']['output_port']['kind'] = "
                "'revolute_joint'"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_lead",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "wrong_output_kind",
                 "expected_failure_codes": ["wrong_topology"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 80, 50],
            },
            "objective": {
                "description": f"Lead screw travel/rev = {lead} mm.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, lead_mm=lead),
        )


# --------------------------------------------------------------------- #
# 27. bevel_gear_ratio_analytic                                         #
# --------------------------------------------------------------------- #


class BevelGearRatioAnalyticGenerator(TaskGenerator):
    family = "bevel_gear_ratio_analytic"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4727)
        t_in = rng.choice([12, 14, 16, 18])
        t_out = t_in * rng.choice([1, 2, 3])
        ratio = round(t_out / t_in, 6)
        axis_angle_deg = round(rng.choice([45.0, 60.0, 90.0]), 2)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "bevel_in", "bevel_out", 50.0)
        params = {
            "teeth_in": t_in, "teeth_out": t_out,
            "declared_ratio": ratio,
            "axis_angle_deg": axis_angle_deg,
        }
        prompt = (
            "# Bevel gear ratio + axis angle (analytic)\n\n"
            f"Ratio = {ratio}, axis angle = {axis_angle_deg}°.\n"
        )

        def _cfg(tol_pct: float, axis_tol: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "ratio", "params.declared_ratio", ratio,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio", weight=0.7,
                    ),
                    param_check_probe(
                        "axis_angle", "params.axis_angle_deg",
                        axis_angle_deg, tolerance_abs=axis_tol,
                        failure_code="wrong_topology", weight=0.3,
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "axis_angle.observed"],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 2.0, 6)}"
            ),
            "wrong_axis_angle": make_negative_overlay(
                "    ir['params']['axis_angle_deg'] = "
                f"{axis_angle_deg + 30.0}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "wrong_axis_angle",
                 "expected_failure_codes": ["wrong_topology"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.9},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": (
                    f"Bevel pair ratio={ratio}, angle="
                    f"{axis_angle_deg}°."
                ),
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(2.0, 2.0),
            eval_config_hidden_toml=_cfg(1.0, 1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, ratio=ratio,
                                     axis_angle_deg=axis_angle_deg),
        )


# --------------------------------------------------------------------- #
# 28. chain_sprocket_ratio                                              #
# --------------------------------------------------------------------- #


class ChainSprocketRatioGenerator(TaskGenerator):
    family = "chain_sprocket_ratio"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4828)
        driver = rng.choice([12, 14, 16])
        driven = driver * rng.choice([2, 3, 4])
        ratio = round(driven / driver, 6)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "sprocket_in", "sprocket_out", 80.0)
        params = {
            "driver_teeth": driver, "driven_teeth": driven,
            "declared_ratio": ratio,
        }
        prompt = (
            "# Chain sprocket ratio\n\n"
            f"Ratio = driven/driver = {driven}/{driver} = {ratio}.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "ratio", "params.declared_ratio", ratio,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "ratio.observed", "ratio.expected"],
                    "hidden_metrics": ["ratio.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 0.3, 6)}"
            ),
            "missing_chain_param": make_negative_overlay(
                "    del ir['params']['driver_teeth']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_chain_param",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 200, 80],
            },
            "objective": {
                "description": f"Chain ratio = {ratio}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, driver=driver,
                                     driven=driven, ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 29. timing_belt_center_distance                                       #
# --------------------------------------------------------------------- #


class TimingBeltCenterDistanceGenerator(TaskGenerator):
    family = "timing_belt_center_distance"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 4929)
        pitch = round(rng.choice([3.0, 5.0, 8.0]), 2)
        n_belt = rng.choice([60, 80, 100, 120])
        belt_length = round(n_belt * pitch, 3)
        center_dist = round(belt_length / 2.0 - 20.0, 3)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "pulley_in", "pulley_out", center_dist)
        params = {
            "pitch_mm": pitch, "belt_tooth_count": n_belt,
            "declared_belt_length_mm": belt_length,
            "declared_center_distance_mm": center_dist,
        }
        prompt = (
            "# Timing belt center distance\n\n"
            f"Belt length = pitch × tooth_count = {pitch}×{n_belt} = "
            f"{belt_length} mm.\n"
            f"Center distance = {center_dist} mm.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                    ),
                    param_check_probe(
                        "belt_length",
                        "params.declared_belt_length_mm",
                        belt_length, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                    param_check_probe(
                        "center",
                        "params.declared_center_distance_mm",
                        center_dist, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "belt_length.observed", "center.observed"],
                    "hidden_metrics": ["belt_length.error_pct"],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_center_distance": make_negative_overlay(
                f"    ir['params']['declared_center_distance_mm'] = "
                f"{round(center_dist * 1.5, 3)}"
            ),
            "wrong_tooth_count": make_negative_overlay(
                f"    ir['params']['declared_belt_length_mm'] = "
                f"{round(belt_length * 0.6, 3)}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_center_distance",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.7},
                {"id": "wrong_tooth_count",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.7},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [400, 200, 80],
            },
            "objective": {
                "description": "Timing belt geometry.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(tol_pct=2.0),
            eval_config_hidden_toml=_cfg(tol_pct=1.0),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     belt_length_mm=belt_length,
                                     center_distance_mm=center_dist),
        )


# --------------------------------------------------------------------- #
# 30. rack_pinion_force_direction                                       #
# --------------------------------------------------------------------- #


class RackPinionForceDirectionGenerator(TaskGenerator):
    family = "rack_pinion_force_direction"
    tier = "transmission_analytic"

    def generate(self, seed: int, difficulty: int = 2) -> GeneratedTask:
        rng = random.Random(seed + 5030)
        direction = rng.choice([-1, 1])
        pitch_radius = round(rng.uniform(8.0, 25.0), 3)
        task_id = make_task_id(self.family, seed)

        parts, joints, ports = _basic_pair_design(
            "pinion", "rack", 0.0, prismatic_b=True)
        params = {
            "pitch_radius_mm": pitch_radius,
            "output_direction_sign": int(direction),
        }
        prompt = (
            "# Rack & pinion force-direction\n\n"
            f"Output direction sign = {direction}.\n"
        )

        def _cfg() -> dict[str, Any]:
            return {
                "probes": [
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "prismatic_joint"},
                    ),
                    param_check_probe(
                        "direction", "params.output_direction_sign",
                        float(direction), tolerance_abs=0.0,
                        failure_code="wrong_ratio",
                    ),
                ],
                "feedback": {
                    "public_metrics": [
                        "direction.observed", "direction.expected"],
                    "hidden_metrics": [],
                },
                "hard_gate": {"require": ["mobility", "ports"]},
            }

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_direction": make_negative_overlay(
                "    ir['params']['output_direction_sign'] = "
                "-ir['params']['output_direction_sign']"
            ),
            "wrong_port_kind": make_negative_overlay(
                "    ir['ports']['output_port']['kind'] = "
                "'revolute_joint'"
            ),
        }
        expected = make_expected_failures(
            f"Tier 2 {self.family} negatives.",
            [
                {"id": "wrong_direction",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "wrong_port_kind",
                 "expected_failure_codes": ["wrong_topology"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["input_port", "output_port"],
                "expected_mobility": 2,
                "max_envelope_mm": [200, 80, 50],
            },
            "objective": {
                "description": f"Rack-pinion direction {direction}.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(),
            eval_config_hidden_toml=_cfg(),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, direction=direction),
        )
