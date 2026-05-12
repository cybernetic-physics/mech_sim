"""Tier 0 — static / analytic fit task generators.

Ten families that exercise only ``required_ports``, ``dof_grubler``,
and ``analytic_param_check``. They are simulator-free and run in
milliseconds.

Each generator emits a public + hidden eval config, two negative
controls (one hard-gate failure, one dense-score failure when
possible), and a reference solution that passes both configs.
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
from mech_bench.generators.common_designs import (
    dof_probe,
    fixed_joint,
    frame_port,
    make_basic_design_py,
    make_expected_failures,
    make_ground_part,
    make_negative_overlay,
    make_revolute_part,
    param_check_probe,
    required_ports_probe,
    revolute_joint,
    revolute_joint_port,
)


# --------------------------------------------------------------------- #
# Shared helpers                                                        #
# --------------------------------------------------------------------- #


def _two_pad_eval(
    probes: list[dict[str, Any]],
    public_metrics: list[str],
    hidden_metrics: list[str],
    hard_gate: list[str],
) -> dict[str, Any]:
    return {
        "probes": probes,
        "feedback": {
            "public_metrics": public_metrics,
            "hidden_metrics": hidden_metrics,
        },
        "hard_gate": {"require": hard_gate},
    }


# --------------------------------------------------------------------- #
# 1. mounting_plate_hole_pitch                                          #
# --------------------------------------------------------------------- #


class MountingPlateHolePitchGenerator(TaskGenerator):
    family = "mounting_plate_hole_pitch"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1101)
        pitch_mm = round(rng.uniform(15.0, 50.0), 3)
        hole_d = round(rng.uniform(3.0, 8.0), 3)
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("plate")]
        joints: list[dict[str, Any]] = []
        ports = {
            "mount_a": frame_port("mount_a", "plate", (0.0, 0.0, 0.0)),
            "mount_b": frame_port(
                "mount_b", "plate", (pitch_mm, 0.0, 0.0)),
        }
        params = {
            "hole_diameter_mm": hole_d,
            "declared_pitch_mm": pitch_mm,
        }

        prompt = (
            "# Mounting plate hole pitch\n\n"
            f"Design a flat mounting plate with two through-holes "
            f"(Ø{hole_d} mm).\n\n"
            f"* Declare `params.declared_pitch_mm` = {pitch_mm} mm.\n"
            "* Expose frame ports `mount_a` and `mount_b` on the plate.\n"
            "* Mobility = 0.\n"
        )

        def _cfg(target: float, tol_pct: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["mount_a", "mount_b"],
                        require_grounded=["mount_a", "mount_b"],
                        require_kinds={"mount_a": "frame",
                                       "mount_b": "frame"},
                    ),
                    param_check_probe(
                        "pitch", "params.declared_pitch_mm",
                        target, comparator="eq",
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                public_metrics=[
                    "mobility.observed", "mobility.expected",
                    "ports.ports_required",
                    "pitch.observed", "pitch.expected",
                ],
                hidden_metrics=["pitch.error_pct"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_pitch": make_negative_overlay(
                f"    ir['params']['declared_pitch_mm'] = "
                f"{round(pitch_mm * 1.35, 4)}"
            ),
            "missing_mount_port": make_negative_overlay(
                "    del ir['ports']['mount_b']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "wrong_pitch",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_mount_port",
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
                "required_ports": ["mount_a", "mount_b"],
                "expected_mobility": 0,
                "max_envelope_mm": [200, 100, 20],
            },
            "objective": {
                "description": (
                    f"Mounting plate; declared pitch {pitch_mm} mm."
                ),
                "allow_massless_links": True,
                "ground_required": False,
            },
        }

        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(pitch_mm, tol_pct=2.0),
            eval_config_hidden_toml=_cfg(pitch_mm, tol_pct=1.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     pitch_mm=pitch_mm,
                                     hole_diameter_mm=hole_d),
        )


# --------------------------------------------------------------------- #
# 2. flange_bolt_circle                                                 #
# --------------------------------------------------------------------- #


class FlangeBoltCircleGenerator(TaskGenerator):
    family = "flange_bolt_circle"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1202)
        bcd_mm = round(rng.uniform(40.0, 120.0), 3)
        bolt_count = rng.choice([4, 6, 8])
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("flange")]
        ports = {
            "flange_axis": frame_port("flange_axis", "flange"),
            "bolt_ref": frame_port(
                "bolt_ref", "flange", (bcd_mm / 2.0, 0.0, 0.0)),
        }
        params = {
            "declared_bolt_circle_mm": bcd_mm,
            "declared_bolt_count": bolt_count,
        }
        prompt = (
            "# Flange bolt circle\n\n"
            f"Design a circular flange with {bolt_count} bolt holes on "
            f"a Ø{bcd_mm} mm bolt circle.\n\n"
            "* Declare `params.declared_bolt_circle_mm` and "
            "`params.declared_bolt_count`.\n"
            "* Mobility = 0.\n"
        )

        def _cfg(target_bcd: float, target_n: int,
                 bcd_tol: float, n_tol: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["flange_axis", "bolt_ref"],
                        require_grounded=["flange_axis"],
                    ),
                    param_check_probe(
                        "bcd", "params.declared_bolt_circle_mm",
                        target_bcd, tolerance_pct=bcd_tol,
                        failure_code="wrong_ratio", weight=0.7,
                    ),
                    param_check_probe(
                        "bolt_count", "params.declared_bolt_count",
                        float(target_n),
                        tolerance_abs=float(n_tol),
                        failure_code="invalid_artifact", weight=0.3,
                    ),
                ],
                public_metrics=[
                    "mobility.observed", "bcd.observed", "bcd.expected",
                    "bolt_count.observed", "bolt_count.expected",
                ],
                hidden_metrics=["bcd.error_pct"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, [], ports, params)
        negatives = {
            "wrong_bcd": make_negative_overlay(
                f"    ir['params']['declared_bolt_circle_mm'] = "
                f"{round(bcd_mm * 0.6, 4)}"
            ),
            "wrong_bolt_count": make_negative_overlay(
                f"    ir['params']['declared_bolt_count'] = "
                f"{bolt_count + 2}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "wrong_bcd",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "wrong_bolt_count",
                 "expected_failure_codes": ["invalid_artifact"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.8},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["flange_axis", "bolt_ref"],
                "expected_mobility": 0,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": f"Flange BCD = {bcd_mm} mm, "
                               f"{bolt_count} bolts.",
                "ground_required": False,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(bcd_mm, bolt_count, 2.0, 0.0),
            eval_config_hidden_toml=_cfg(bcd_mm, bolt_count, 1.0, 0.0),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, bcd_mm=bcd_mm,
                                     bolt_count=bolt_count),
        )


# --------------------------------------------------------------------- #
# 3. bearing_seat_clearance                                             #
# --------------------------------------------------------------------- #


class BearingSeatClearanceGenerator(TaskGenerator):
    family = "bearing_seat_clearance"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1303)
        bore_d = round(rng.uniform(20.0, 45.0), 3)
        bearing_od = round(bore_d - 0.05, 3)
        clearance_mm = round(bore_d - bearing_od, 4)
        target_clearance = 0.02
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("housing"),
                 make_revolute_part("bearing", "bearing", 0.04)]
        joints = [fixed_joint("press", "housing", "bearing")]
        ports = {
            "bore_face": frame_port("bore_face", "housing"),
            "bearing_seat": frame_port(
                "bearing_seat", "bearing"),
        }
        params = {
            "bore_diameter_mm": bore_d,
            "bearing_od_mm": bearing_od,
            "clearance_mm": clearance_mm,
        }

        prompt = (
            "# Bearing seat clearance\n\n"
            f"Design a fixed bearing seat with bore Ø{bore_d} mm and "
            f"bearing OD Ø{bearing_od} mm.\n\n"
            f"* Declare `params.clearance_mm` ≥ {target_clearance} mm.\n"
            "* Mobility = 0 (bearing fixed in housing for this analytic "
            "task).\n"
        )

        def _cfg(target: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["bore_face", "bearing_seat"],
                        require_grounded=["bore_face"],
                    ),
                    param_check_probe(
                        "clearance", "params.clearance_mm",
                        target, comparator="ge",
                        tolerance_abs=0.001,
                        failure_code="insufficient_clearance",
                    ),
                ],
                public_metrics=[
                    "mobility.observed", "clearance.observed",
                    "clearance.expected",
                ],
                hidden_metrics=["clearance.error_abs"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "too_tight": make_negative_overlay(
                "    ir['params']['clearance_mm'] = 0.0"
            ),
            "too_loose": make_negative_overlay(
                "    ir['params']['clearance_mm'] = 5.0\n"
                "    ir['params']['bearing_od_mm'] = "
                "ir['params']['bore_diameter_mm'] - 5.0"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "too_tight",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                # "too_loose" still passes the ge check; force a hidden
                # tightening to catch it via a stricter tolerance.
                {"id": "too_loose",
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
                "required_ports": ["bore_face", "bearing_seat"],
                "expected_mobility": 0,
                "max_envelope_mm": [120, 120, 50],
            },
            "objective": {
                "description": "Bearing-seat clearance ≥ "
                               f"{target_clearance} mm.",
                "ground_required": False,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(target_clearance),
            eval_config_hidden_toml=_cfg(target_clearance * 1.5),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     bore_d_mm=bore_d,
                                     bearing_od_mm=bearing_od,
                                     target_clearance_mm=target_clearance),
        )


# --------------------------------------------------------------------- #
# 4. press_fit_hub_interference                                         #
# --------------------------------------------------------------------- #


class PressFitHubInterferenceGenerator(TaskGenerator):
    family = "press_fit_hub_interference"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1404)
        nominal_d = round(rng.uniform(15.0, 30.0), 3)
        interference = round(rng.uniform(0.015, 0.045), 4)
        min_interference = 0.01
        max_interference = 0.06
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("hub"),
                 make_revolute_part("shaft", "shaft", 0.05)]
        joints = [fixed_joint("press", "hub", "shaft")]
        ports = {
            "hub_face": frame_port("hub_face", "hub"),
            "shaft_origin": frame_port("shaft_origin", "shaft"),
        }
        params = {
            "nominal_diameter_mm": nominal_d,
            "interference_mm": interference,
        }
        prompt = (
            "# Press-fit hub interference\n\n"
            f"Design a hub pressed onto a Ø{nominal_d} mm shaft.\n\n"
            f"* `params.interference_mm` must satisfy "
            f"{min_interference} ≤ interference ≤ {max_interference} mm.\n"
            "* Mobility = 0.\n"
        )

        def _cfg(min_i: float, max_i: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["hub_face", "shaft_origin"],
                        require_grounded=["hub_face"],
                    ),
                    param_check_probe(
                        "interference_lower", "params.interference_mm",
                        min_i, comparator="ge",
                        failure_code="insufficient_clearance",
                        weight=0.5,
                    ),
                    param_check_probe(
                        "interference_upper", "params.interference_mm",
                        max_i, comparator="le",
                        failure_code="insufficient_clearance",
                        weight=0.5,
                    ),
                ],
                public_metrics=[
                    "mobility.observed",
                    "interference_lower.observed",
                    "interference_upper.observed",
                ],
                hidden_metrics=[
                    "interference_lower.error_pct",
                    "interference_upper.error_pct",
                ],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "slip_fit": make_negative_overlay(
                "    ir['params']['interference_mm'] = -0.005"
            ),
            "excessive_interference": make_negative_overlay(
                "    ir['params']['interference_mm'] = 0.20"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "slip_fit",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "excessive_interference",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["hub_face", "shaft_origin"],
                "expected_mobility": 0,
                "max_envelope_mm": [80, 80, 50],
            },
            "objective": {
                "description": "Press-fit interference within "
                               f"[{min_interference}, "
                               f"{max_interference}] mm.",
                "ground_required": False,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(min_interference, max_interference),
            eval_config_hidden_toml=_cfg(
                min_interference * 1.2, max_interference * 0.95),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     nominal_diameter_mm=nominal_d,
                                     interference_mm=interference),
        )


# --------------------------------------------------------------------- #
# 5. keyed_shaft_hub_fit                                                #
# --------------------------------------------------------------------- #


class KeyedShaftHubFitGenerator(TaskGenerator):
    family = "keyed_shaft_hub_fit"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1505)
        shaft_d = round(rng.uniform(8.0, 25.0), 3)
        keyway_w = round(rng.uniform(2.0, 8.0), 3)
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("hub"),
                 make_revolute_part("shaft", "shaft", 0.05)]
        joints = [revolute_joint("shaft_axis", "hub", "shaft",
                                 (0.0, 0.0, 0.0))]
        ports = {
            "hub_face": frame_port("hub_face", "hub"),
            "output_port": revolute_joint_port(
                "output_port", "shaft_axis"),
        }
        params = {
            "shaft_diameter_mm": shaft_d,
            "keyway_width_mm": keyway_w,
        }
        prompt = (
            "# Keyed shaft–hub fit\n\n"
            f"Design a keyed shaft (Ø{shaft_d} mm bore) with keyway "
            f"width {keyway_w} mm.\n\n"
            "* Required ports: `hub_face` (frame, grounded), "
            "`output_port` (revolute_joint).\n"
            "* Mobility = 1 (one revolute joint).\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=1),
                    required_ports_probe(
                        "ports", ["hub_face", "output_port"],
                        require_grounded=["hub_face"],
                        require_kinds={"hub_face": "frame",
                                       "output_port": "revolute_joint"},
                    ),
                    param_check_probe(
                        "shaft_d", "params.shaft_diameter_mm", shaft_d,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                    param_check_probe(
                        "keyway", "params.keyway_width_mm", keyway_w,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio", weight=0.5,
                    ),
                ],
                public_metrics=[
                    "mobility.observed", "shaft_d.observed",
                    "keyway.observed",
                ],
                hidden_metrics=["shaft_d.error_pct",
                                "keyway.error_pct"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_keyway": make_negative_overlay(
                f"    ir['params']['keyway_width_mm'] = "
                f"{round(keyway_w * 0.5, 3)}"
            ),
            "missing_output_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "wrong_keyway",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
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
                "required_ports": ["hub_face", "output_port"],
                "expected_mobility": 1,
                "max_envelope_mm": [120, 120, 80],
            },
            "objective": {
                "description": "Keyed shaft + hub; mobility=1.",
                "ground_required": False,
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
                                     shaft_diameter_mm=shaft_d,
                                     keyway_width_mm=keyway_w),
        )


# --------------------------------------------------------------------- #
# 6. spacer_stack_height                                                #
# --------------------------------------------------------------------- #


class SpacerStackHeightGenerator(TaskGenerator):
    family = "spacer_stack_height"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1606)
        spacer_count = rng.choice([3, 4, 5, 6])
        spacer_h = round(rng.uniform(2.0, 8.0), 3)
        stack_h = round(spacer_count * spacer_h, 4)
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("base"),
                 make_revolute_part("stack", "stack", 0.04)]
        joints = [fixed_joint("stack_fix", "base", "stack")]
        ports = {
            "base_face": frame_port("base_face", "base"),
            "stack_top": frame_port(
                "stack_top", "stack", (0.0, 0.0, stack_h)),
        }
        params = {
            "spacer_count": int(spacer_count),
            "spacer_height_mm": spacer_h,
            "declared_stack_height_mm": stack_h,
        }
        prompt = (
            "# Spacer stack height\n\n"
            f"Stack {spacer_count} spacers of height {spacer_h} mm.\n\n"
            f"* Declare `params.declared_stack_height_mm` = {stack_h}.\n"
            "* Mobility = 0; ports `base_face` (grounded) and `stack_top`.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["base_face", "stack_top"],
                        require_grounded=["base_face"],
                    ),
                    param_check_probe(
                        "stack", "params.declared_stack_height_mm",
                        stack_h, tolerance_pct=tol_pct,
                        failure_code="wrong_ratio",
                    ),
                ],
                public_metrics=[
                    "mobility.observed", "stack.observed", "stack.expected",
                ],
                hidden_metrics=["stack.error_pct"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "short_stack": make_negative_overlay(
                f"    ir['params']['declared_stack_height_mm'] = "
                f"{round(stack_h * 0.5, 4)}"
            ),
            "tall_stack": make_negative_overlay(
                f"    ir['params']['declared_stack_height_mm'] = "
                f"{round(stack_h * 1.7, 4)}"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "short_stack",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "tall_stack",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["base_face", "stack_top"],
                "expected_mobility": 0,
                "max_envelope_mm": [100, 100, 80],
            },
            "objective": {
                "description": (
                    f"Stack height = {stack_h} mm "
                    f"({spacer_count} × {spacer_h} mm)."
                ),
                "ground_required": False,
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
                                     spacer_count=spacer_count,
                                     spacer_height_mm=spacer_h,
                                     declared_stack_height_mm=stack_h),
        )


# --------------------------------------------------------------------- #
# 7. standoff_pattern_square                                            #
# --------------------------------------------------------------------- #


class StandoffPatternSquareGenerator(TaskGenerator):
    family = "standoff_pattern_square"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1707)
        side = round(rng.uniform(40.0, 100.0), 3)
        n = 4
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("plate")]
        ports = {
            "so_1": frame_port("so_1", "plate", (0.0, 0.0, 0.0)),
            "so_2": frame_port("so_2", "plate", (side, 0.0, 0.0)),
            "so_3": frame_port("so_3", "plate", (side, side, 0.0)),
            "so_4": frame_port("so_4", "plate", (0.0, side, 0.0)),
        }
        params = {
            "side_length_mm": side,
            "standoff_count": n,
        }
        prompt = (
            "# Standoff pattern (square)\n\n"
            f"Plate with four standoffs at the corners of a "
            f"{side} mm square.\n\n"
            "* Declare `params.side_length_mm` and "
            "`params.standoff_count`.\n"
            "* Required ports: `so_1`..`so_4`.\n"
        )

        def _cfg(tol_pct: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports",
                        ["so_1", "so_2", "so_3", "so_4"],
                        require_grounded=[
                            "so_1", "so_2", "so_3", "so_4"],
                    ),
                    param_check_probe(
                        "side", "params.side_length_mm", side,
                        tolerance_pct=tol_pct,
                        failure_code="wrong_ratio", weight=0.7,
                    ),
                    param_check_probe(
                        "count", "params.standoff_count", float(n),
                        tolerance_abs=0.0,
                        failure_code="invalid_artifact", weight=0.3,
                    ),
                ],
                public_metrics=[
                    "side.observed", "side.expected",
                    "count.observed", "count.expected",
                ],
                hidden_metrics=["side.error_pct"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, [], ports, params)
        negatives = {
            "wrong_spacing": make_negative_overlay(
                f"    ir['params']['side_length_mm'] = "
                f"{round(side * 1.3, 3)}"
            ),
            "wrong_count": make_negative_overlay(
                "    ir['params']['standoff_count'] = 3\n"
                "    del ir['ports']['so_4']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "wrong_spacing",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "wrong_count",
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
                    "so_1", "so_2", "so_3", "so_4"],
                "expected_mobility": 0,
                "max_envelope_mm": [200, 200, 50],
            },
            "objective": {
                "description": f"Standoff square of side {side} mm.",
                "ground_required": False,
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
                                     side_length_mm=side, standoff_count=n),
        )


# --------------------------------------------------------------------- #
# 8. pulley_bore_alignment_static                                       #
# --------------------------------------------------------------------- #


class PulleyBoreAlignmentStaticGenerator(TaskGenerator):
    family = "pulley_bore_alignment_static"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1808)
        center_dist = round(rng.uniform(80.0, 180.0), 3)
        alignment_err = round(rng.uniform(0.0, 0.04), 4)
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("frame"),
                 make_revolute_part("pulley_in", "pulley", 0.03),
                 make_revolute_part("pulley_out", "pulley", 0.05)]
        joints = [
            revolute_joint("in_axis", "frame", "pulley_in",
                           (0.0, 0.0, 0.0)),
            revolute_joint("out_axis", "frame", "pulley_out",
                           (center_dist, 0.0, 0.0)),
        ]
        ports = {
            "input_port": revolute_joint_port("input_port", "in_axis"),
            "output_port": revolute_joint_port("output_port", "out_axis"),
        }
        params = {
            "center_distance_mm": center_dist,
            "alignment_error_mm": alignment_err,
        }
        prompt = (
            "# Pulley bore alignment (static)\n\n"
            f"Two pulleys on parallel axes, center distance "
            f"{center_dist} mm.\n\n"
            "* Declare `params.alignment_error_mm` ≤ 0.05 mm.\n"
            "* Mobility = 2 (two independent revolutes; analytic tier).\n"
        )

        def _cfg(max_align: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=2),
                    required_ports_probe(
                        "ports", ["input_port", "output_port"],
                        require_grounded=["input_port", "output_port"],
                        require_kinds={
                            "input_port": "revolute_joint",
                            "output_port": "revolute_joint"},
                    ),
                    param_check_probe(
                        "alignment", "params.alignment_error_mm",
                        max_align, comparator="le",
                        failure_code="insufficient_clearance",
                    ),
                ],
                public_metrics=[
                    "alignment.observed", "alignment.expected",
                ],
                hidden_metrics=["alignment.error_abs"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "misaligned": make_negative_overlay(
                "    ir['params']['alignment_error_mm'] = 0.30"
            ),
            "missing_axis_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "misaligned",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.5},
                {"id": "missing_axis_port",
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
                "max_envelope_mm": [220, 80, 80],
            },
            "objective": {
                "description": "Pulley bore alignment ≤ 0.05 mm.",
                "ground_required": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(0.05),
            eval_config_hidden_toml=_cfg(0.03),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     center_distance_mm=center_dist,
                                     alignment_error_mm=alignment_err),
        )


# --------------------------------------------------------------------- #
# 9. snap_tab_clearance_static                                          #
# --------------------------------------------------------------------- #


class SnapTabClearanceStaticGenerator(TaskGenerator):
    family = "snap_tab_clearance_static"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 1909)
        gap_mm = round(rng.uniform(0.4, 1.2), 3)
        wall_mm = round(rng.uniform(1.5, 3.0), 3)
        min_gap = 0.3
        min_wall = 1.2
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("body"),
                 make_revolute_part("tab", "tab", 0.02)]
        joints = [fixed_joint("snap", "body", "tab")]
        ports = {
            "body_face": frame_port("body_face", "body"),
            "tab_face": frame_port("tab_face", "tab"),
        }
        params = {
            "gap_mm": gap_mm,
            "min_wall_mm": wall_mm,
        }
        prompt = (
            "# Snap-tab clearance (static)\n\n"
            "Design a snap-tab feature.\n\n"
            f"* `params.gap_mm` ≥ {min_gap} mm.\n"
            f"* `params.min_wall_mm` ≥ {min_wall} mm.\n"
            "* Mobility = 0.\n"
        )

        def _cfg(min_g: float, min_w: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["body_face", "tab_face"],
                        require_grounded=["body_face"],
                    ),
                    param_check_probe(
                        "gap", "params.gap_mm", min_g,
                        comparator="ge",
                        failure_code="insufficient_clearance",
                        weight=0.5,
                    ),
                    param_check_probe(
                        "wall", "params.min_wall_mm", min_w,
                        comparator="ge",
                        failure_code="insufficient_clearance",
                        weight=0.5,
                    ),
                ],
                public_metrics=[
                    "gap.observed", "wall.observed",
                ],
                hidden_metrics=["gap.error_abs", "wall.error_abs"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "insufficient_gap": make_negative_overlay(
                "    ir['params']['gap_mm'] = 0.05"
            ),
            "thin_wall": make_negative_overlay(
                "    ir['params']['min_wall_mm'] = 0.5"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "insufficient_gap",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
                {"id": "thin_wall",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.85},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["body_face", "tab_face"],
                "expected_mobility": 0,
                "max_envelope_mm": [100, 60, 50],
            },
            "objective": {
                "description": "Snap-tab clearance / min wall.",
                "ground_required": False,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(min_gap, min_wall),
            eval_config_hidden_toml=_cfg(
                min_gap * 1.2, min_wall * 1.1),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, gap_mm=gap_mm,
                                     min_wall_mm=wall_mm),
        )


# --------------------------------------------------------------------- #
# 10. box_lid_register_fit                                              #
# --------------------------------------------------------------------- #


class BoxLidRegisterFitGenerator(TaskGenerator):
    family = "box_lid_register_fit"
    tier = "artifact_static"

    def generate(self, seed: int, difficulty: int = 1) -> GeneratedTask:
        rng = random.Random(seed + 2010)
        lid_clearance = round(rng.uniform(0.10, 0.30), 3)
        min_clearance = 0.05
        max_clearance = 0.5
        task_id = make_task_id(self.family, seed)

        parts = [make_ground_part("base"),
                 make_revolute_part("lid", "lid", 0.04)]
        joints = [fixed_joint("register", "base", "lid")]
        ports = {
            "base_frame": frame_port("base_frame", "base"),
            "lid_frame": frame_port("lid_frame", "lid"),
        }
        params = {
            "register_clearance_mm": lid_clearance,
        }
        prompt = (
            "# Box / lid register fit\n\n"
            "Design a lid registering on a box.\n\n"
            f"* `params.register_clearance_mm` ∈ "
            f"[{min_clearance}, {max_clearance}] mm.\n"
            "* Ports: `base_frame` (grounded), `lid_frame`.\n"
        )

        def _cfg(min_c: float, max_c: float) -> dict[str, Any]:
            return _two_pad_eval(
                probes=[
                    dof_probe(expected=0),
                    required_ports_probe(
                        "ports", ["base_frame", "lid_frame"],
                        require_grounded=["base_frame"],
                    ),
                    param_check_probe(
                        "clearance_lo", "params.register_clearance_mm",
                        min_c, comparator="ge",
                        failure_code="insufficient_clearance",
                        weight=0.5,
                    ),
                    param_check_probe(
                        "clearance_hi", "params.register_clearance_mm",
                        max_c, comparator="le",
                        failure_code="insufficient_clearance",
                        weight=0.5,
                    ),
                ],
                public_metrics=[
                    "clearance_lo.observed", "clearance_hi.observed",
                ],
                hidden_metrics=["clearance_lo.error_abs"],
                hard_gate=["mobility", "ports"],
            )

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "no_lid_port": make_negative_overlay(
                "    del ir['ports']['lid_frame']"
            ),
            "wrong_clearance": make_negative_overlay(
                "    ir['params']['register_clearance_mm'] = 0.0"
            ),
        }
        expected = make_expected_failures(
            f"Tier 0 {self.family} negatives.",
            [
                {"id": "no_lid_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.001},
                {"id": "wrong_clearance",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.6},
            ],
        )
        task_toml = {
            "task": {"id": task_id, "family": self.family,
                     "difficulty": int(difficulty), "units": "mm",
                     "tier": self.tier},
            "requirements": {
                "required_ports": ["base_frame", "lid_frame"],
                "expected_mobility": 0,
                "max_envelope_mm": [200, 150, 80],
            },
            "objective": {
                "description": "Box-lid register clearance.",
                "ground_required": False,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_cfg(min_clearance, max_clearance),
            eval_config_hidden_toml=_cfg(
                min_clearance * 1.5, max_clearance * 0.9),
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty,
                                     register_clearance_mm=lid_clearance),
        )
