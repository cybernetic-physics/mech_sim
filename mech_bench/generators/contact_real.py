"""Chrono-backed contact mechanism generators.

These generators are intentionally separate from ``contact_synth``: they use
the real ``chrono_contact`` adapter with explicit collision geometry and do
not opt into ``fake_contact_oracle``.
"""

from __future__ import annotations

import math
from typing import Any

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    common_metadata,
    make_task_id,
)
from mech_bench.generators.common_designs import (
    make_basic_design_py,
    param_check_probe,
    prismatic_joint,
    prismatic_joint_port,
    required_ports_probe,
    revolute_joint,
    revolute_joint_port,
)
from mech_bench.generators.static_fit import _negative_overlay


def _inertia(mass_kg: float) -> tuple[tuple[float, float, float], ...]:
    scale = max(float(mass_kg), 1.0e-6)
    return (
        (scale * 1.0e-5, 0.0, 0.0),
        (0.0, scale * 1.2e-5, 0.0),
        (0.0, 0.0, scale * 1.5e-5),
    )


def _steel_part(
    part_id: str,
    role: str,
    mass_kg: float,
    *,
    fixed: bool = False,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": part_id,
        "role": role,
        "mass_kg": float(mass_kg),
        "fixed": bool(fixed),
        "material": "steel_1045",
        "com_local_mm": (0.0, 0.0, 0.0),
        "inertia_kg_m2": _inertia(mass_kg),
    }
    if params:
        out["params"] = dict(params)
    return out


def _cylinder(
    *,
    radius_mm: float,
    height_mm: float = 8.0,
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    return {
        "shape": "cylinder",
        "radius_mm": float(radius_mm),
        "height_mm": float(height_mm),
        "center_mm": tuple(float(v) for v in center_mm),
        "axis": (0.0, 0.0, 1.0),
    }


def _chrono_cfg(
    *,
    samples: int,
    duration_s: float,
    input_speed_rad_s: float,
    max_penetration_mm: float,
    min_rms_force_N: float,
    min_engagement_fraction: float,
    contact_pair: str,
    output_port_kind: str,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = [
        required_ports_probe(
            "ports",
            ["input_port", "output_port"],
            require_grounded=["input_port", "output_port"],
            require_kinds={
                "input_port": "revolute_joint",
                "output_port": output_port_kind,
            },
            hard_gate=True,
        ),
        {
            "id": "contact",
            "type": "contact_engagement",
            "required_pairs": [contact_pair],
            "min_rms_force_N": float(min_rms_force_N),
            "min_engagement_fraction": float(min_engagement_fraction),
            "adapter": "chrono_contact",
            "hard_gate": True,
            "severity": "critical",
            "weight": 1.0,
        },
        {
            "id": "penetration_bound",
            "type": "swept_collision",
            "max_penetration_mm": float(max_penetration_mm),
            "adapter": "chrono_contact",
            "hard_gate": True,
            "severity": "critical",
            "weight": 0.4,
        },
        {
            "id": "lockup",
            "type": "lockup",
            "input_port": "input_port",
            "output_port": "output_port",
            "min_output_motion_rad": 0.002,
            "adapter": "chrono_contact",
            "hard_gate": True,
            "severity": "critical",
            "weight": 0.4,
        },
    ]
    return {
        "probes": probes,
        "feedback": {
            "public_metrics": [
                "ports.ports_required",
                f"contact.contact.{contact_pair}.rms_N",
                f"contact.contact.{contact_pair}.engagement_fraction",
                "penetration_bound.max_penetration_mm",
                "lockup.output_motion_rad",
                "lockup.lockup_detected",
            ],
            "hidden_metrics": [
                "contact.worst_pair_score",
                "lockup.output_velocity_max",
            ],
        },
        "hard_gate": {"require": ["ports", "contact", "penetration_bound", "lockup"]},
        "adapters": {
            "chrono_contact": {
                "contact_model": "nsc",
                "procedural_cycloidal_fallback": False,
                "collision_filter_named_pairs": True,
                "samples": int(samples),
                "duration_s": float(duration_s),
                "dt": 1.0e-4,
                "timestep": 1.0e-4,
                "input_speed_rad_s": float(input_speed_rad_s),
                "friction_mu": 0.05,
                "restitution": 0.0,
                "contact_margin_m": 1.0e-5,
                "contact_envelope_m": 5.0e-5,
                "solver_iterations": 300,
                "solver_max_iterations": 300,
                "solver_tolerance": 1.0e-8,
            },
        },
    }


_MATERIALS_POST_MUTATION = (
    "    ir['materials'] = {\n"
    "        'steel_1045': {\n"
    "            'name': 'AISI 1045 steel',\n"
    "            'density_kg_m3': 7850.0,\n"
    "            'elastic_modulus_pa': 205000000000.0,\n"
    "            'poisson_ratio': 0.29,\n"
    "            'yield_strength_pa': 530000000.0,\n"
    "            'process': 'machined contact reference',\n"
    "            'provenance': 'MechanismRepair-Physics generator table',\n"
    "        },\n"
    "    }\n"
)


class CamFollowerEccentricChronoGenerator(TaskGenerator):
    family = "cam_follower_eccentric_chrono"
    tier = "contact_dynamics"

    _VARIANTS = (
        # cam radius, eccentricity, follower roller radius, follower x, lift target
        (18.0, 8.0, 6.0, 22.0, 16.0),
        (16.0, 8.0, 8.0, 18.0, 16.0),
        (18.0, 10.0, 8.0, 24.0, 20.0),
    )

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        cam_radius, eccentricity, follower_radius, follower_x, lift_mm = (
            self._VARIANTS[(int(seed) - 20260610) % len(self._VARIANTS)]
        )
        input_speed = 10.0
        task_id = make_task_id(self.family, seed)
        pair = "cam:follower"

        parts = [
            _steel_part("frame", "ground", 0.0, fixed=True),
            _steel_part(
                "cam",
                "eccentric_cam",
                0.12,
                params={
                    "chrono_collision": _cylinder(
                        radius_mm=cam_radius,
                        center_mm=(eccentricity, 0.0, 0.0),
                    ),
                },
            ),
            _steel_part(
                "follower",
                "roller_follower",
                0.08,
                params={
                    "initial_pose_mm": (follower_x, 0.0, 0.0),
                    "chrono_collision": _cylinder(radius_mm=follower_radius),
                },
            ),
        ]
        joints = [
            revolute_joint("cam_axis", "frame", "cam", (0.0, 0.0, 0.0)),
            prismatic_joint(
                "follower_slide",
                "frame",
                "follower",
                axis=(1.0, 0.0, 0.0),
                anchor_world_mm=(follower_x, 0.0, 0.0),
            ),
            {
                "id": "cam_follower_contact",
                "type": "contact_pair",
                "parent": "cam",
                "child": "follower",
                "axis_world": (0.0, 0.0, 1.0),
                "anchor_world_mm": (0.0, 0.0, 0.0),
            },
        ]
        ports = {
            "input_port": revolute_joint_port("input_port", "cam_axis"),
            "output_port": prismatic_joint_port("output_port", "follower_slide"),
        }
        params = {
            "target_lift_mm": float(lift_mm),
            "cam_radius_mm": float(cam_radius),
            "cam_eccentricity_mm": float(eccentricity),
            "follower_radius_mm": float(follower_radius),
            "declared_pair": pair,
            "chrono": {
                "collision_filter_named_pairs": True,
                "procedural_cycloidal_fallback": False,
            },
        }
        ref_py = make_basic_design_py(
            parts,
            joints,
            ports,
            params,
            post_mutation=_MATERIALS_POST_MUTATION,
        )
        negatives = {
            "wrong_lift": _negative_overlay(
                f"    ir['params']['target_lift_mm'] = {round(lift_mm * 0.5, 6)}"
            ),
            "missing_port": _negative_overlay(
                "    del ir['ports']['output_port']"
            ),
            "missing_contact_geometry": _negative_overlay(
                "    for part in ir.get('parts', []) or []:\n"
                "        if part.get('id') in {'cam', 'follower'}:\n"
                "            params = part.get('params') or {}\n"
                "            params.pop('chrono_collision', None)\n"
                "            part['params'] = params"
            ),
        }
        cfg = _chrono_cfg(
            samples=301,
            duration_s=0.6,
            input_speed_rad_s=input_speed,
            max_penetration_mm=8.0,
            min_rms_force_N=0.02,
            min_engagement_fraction=0.002,
            contact_pair=pair,
            output_port_kind="prismatic_joint",
        )
        cfg["probes"].insert(
            1,
            param_check_probe(
                "lift",
                "params.target_lift_mm",
                float(lift_mm),
                tolerance_pct=2.0,
                failure_code="wrong_ratio",
                weight=0.2,
            ),
        )
        cfg["feedback"]["public_metrics"].append("lift.observed")
        hidden_cfg = _chrono_cfg(
            samples=401,
            duration_s=0.6,
            input_speed_rad_s=input_speed,
            max_penetration_mm=8.5,
            min_rms_force_N=0.02,
            min_engagement_fraction=0.002,
            contact_pair=pair,
            output_port_kind="prismatic_joint",
        )
        hidden_cfg["probes"].insert(
            1,
            param_check_probe(
                "lift",
                "params.target_lift_mm",
                float(lift_mm),
                tolerance_pct=1.0,
                failure_code="wrong_ratio",
                weight=0.2,
            ),
        )
        hidden_cfg["feedback"]["public_metrics"].append("lift.observed")

        prompt = (
            "# Eccentric cam follower with Chrono contact\n\n"
            "Design a rotating eccentric cam named `cam` that drives a "
            "sliding roller follower named `follower` through physical "
            "contact.\n\n"
            f"* The follower must be on a prismatic `output_port` and the "
            f"cam must expose a revolute `input_port`.\n"
            f"* Use a real `contact_pair` joint for `{pair}` with "
            "`chrono_collision` geometry on both bodies.\n"
            f"* Preserve `params.target_lift_mm = {lift_mm:g}` and avoid "
            "lockup in the Chrono contact trial.\n"
        )
        expected = {
            "description": "Chrono-backed eccentric cam follower negatives.",
            "controls": [
                {
                    "id": "wrong_lift",
                    "submission": "negative_solutions/wrong_lift",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_score_below": 0.95,
                },
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "missing_contact_geometry",
                    "submission": "negative_solutions/missing_contact_geometry",
                    "expected_failure_codes": ["missing_contact"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.6,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id,
            family=self.family,
            difficulty=int(difficulty),
            prompt_md=prompt,
            task_toml={
                "task": {
                    "id": task_id,
                    "family": self.family,
                    "difficulty": int(difficulty),
                    "units": "mm",
                    "tier": self.tier,
                },
                "requirements": {
                    "required_ports": ["input_port", "output_port"],
                    "expected_mobility": 2,
                    "max_envelope_mm": [120, 120, 80],
                },
                "objective": {
                    "description": (
                        f"Eccentric cam lift target {lift_mm:g} mm with "
                        "real Chrono contact."
                    ),
                    "allow_massless_links": False,
                    "ground_required": True,
                },
                "capability": {
                    "requires_adapter": "rigid_body_dynamics+contact_forces",
                    "synthetic_oracle": False,
                },
            },
            eval_config_toml=cfg,
            eval_config_hidden_toml=hidden_cfg,
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(
                self.family,
                self.tier,
                seed,
                difficulty,
                target_lift_mm=lift_mm,
                contact_pair=pair,
                requires_adapter="rigid_body_dynamics+contact_forces",
            ),
        )


class GenevaIndexingPinChronoGenerator(TaskGenerator):
    family = "geneva_indexing_pin_chrono"
    tier = "contact_dynamics"

    _VARIANTS = (
        (4, 16.0, 24.0, 5.0),
        (5, 16.0, 24.0, 5.0),
        (6, 16.0, 24.0, 4.0),
    )

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        slot_count, driver_pin_radius_mm, center_distance_mm, slot_radius_mm = (
            self._VARIANTS[(int(seed) - 20270610) % len(self._VARIANTS)]
        )
        input_speed = 8.0
        task_id = make_task_id(self.family, seed)
        pair = "driver:geneva"

        slot_children = []
        for index in range(slot_count):
            theta = math.pi + 2.0 * math.pi * index / slot_count
            slot_children.append(
                _cylinder(
                    radius_mm=4.0,
                    center_mm=(
                        slot_radius_mm * math.cos(theta),
                        slot_radius_mm * math.sin(theta),
                        0.0,
                    ),
                )
            )
        parts = [
            _steel_part("frame", "ground", 0.0, fixed=True),
            _steel_part(
                "driver",
                "driver_wheel_with_pin",
                0.12,
                params={
                    "chrono_collision": _cylinder(
                        radius_mm=3.0,
                        center_mm=(driver_pin_radius_mm, 0.0, 0.0),
                    ),
                },
            ),
            _steel_part(
                "geneva",
                "indexed_slot_wheel",
                0.18,
                params={
                    "initial_pose_mm": (center_distance_mm, 0.0, 0.0),
                    "chrono_collision": {
                        "shape": "compound",
                        "children": slot_children,
                    },
                },
            ),
        ]
        joints = [
            revolute_joint("driver_axis", "frame", "driver", (0.0, 0.0, 0.0)),
            revolute_joint(
                "geneva_axis",
                "frame",
                "geneva",
                (center_distance_mm, 0.0, 0.0),
            ),
            {
                "id": "driver_geneva_contact",
                "type": "contact_pair",
                "parent": "driver",
                "child": "geneva",
                "axis_world": (0.0, 0.0, 1.0),
                "anchor_world_mm": (0.0, 0.0, 0.0),
            },
        ]
        ports = {
            "input_port": revolute_joint_port("input_port", "driver_axis"),
            "output_port": revolute_joint_port("output_port", "geneva_axis"),
        }
        params = {
            "index_count": int(slot_count),
            "declared_ratio": float(slot_count),
            "driver_pin_radius_mm": float(driver_pin_radius_mm),
            "center_distance_mm": float(center_distance_mm),
            "slot_radius_mm": float(slot_radius_mm),
            "declared_pair": pair,
            "chrono": {
                "collision_filter_named_pairs": True,
                "procedural_cycloidal_fallback": False,
            },
        }
        ref_py = make_basic_design_py(
            parts,
            joints,
            ports,
            params,
            post_mutation=_MATERIALS_POST_MUTATION,
        )
        negatives = {
            "wrong_index_count": _negative_overlay(
                f"    ir['params']['index_count'] = {slot_count + 2}\n"
                f"    ir['params']['declared_ratio'] = {float(slot_count + 2)!r}"
            ),
            "missing_port": _negative_overlay(
                "    del ir['ports']['output_port']"
            ),
            "missing_contact_geometry": _negative_overlay(
                "    for part in ir.get('parts', []) or []:\n"
                "        if part.get('id') in {'driver', 'geneva'}:\n"
                "            params = part.get('params') or {}\n"
                "            params.pop('chrono_collision', None)\n"
                "            part['params'] = params"
            ),
        }
        cfg = _chrono_cfg(
            samples=401,
            duration_s=0.8,
            input_speed_rad_s=input_speed,
            max_penetration_mm=5.0,
            min_rms_force_N=0.005,
            min_engagement_fraction=0.002,
            contact_pair=pair,
            output_port_kind="revolute_joint",
        )
        cfg["probes"].insert(
            1,
            param_check_probe(
                "index_count",
                "params.index_count",
                float(slot_count),
                tolerance_abs=0.0,
                failure_code="wrong_ratio",
                weight=0.2,
            ),
        )
        cfg["feedback"]["public_metrics"].append("index_count.observed")
        hidden_cfg = _chrono_cfg(
            samples=501,
            duration_s=0.8,
            input_speed_rad_s=input_speed,
            max_penetration_mm=5.5,
            min_rms_force_N=0.005,
            min_engagement_fraction=0.002,
            contact_pair=pair,
            output_port_kind="revolute_joint",
        )
        hidden_cfg["probes"].insert(
            1,
            param_check_probe(
                "index_count",
                "params.index_count",
                float(slot_count),
                tolerance_abs=0.0,
                failure_code="wrong_ratio",
                weight=0.2,
            ),
        )
        hidden_cfg["feedback"]["public_metrics"].append("index_count.observed")

        prompt = (
            "# Geneva drive-pin indexer with Chrono contact\n\n"
            f"Design a {slot_count}-slot Geneva-style indexer with a "
            "rotating `driver` body whose offset drive pin contacts an "
            "indexed slot wheel named `geneva`.\n\n"
            "* Expose revolute `input_port` and `output_port` joints.\n"
            f"* Preserve `params.index_count = {slot_count}` and "
            f"`params.declared_ratio = {float(slot_count):g}`.\n"
            f"* Use a real `contact_pair` joint for `{pair}` with "
            "`chrono_collision` geometry on the driver pin and Geneva "
            "slot-wall bodies; fake contact outputs do not count.\n"
        )
        expected = {
            "description": "Chrono-backed Geneva drive-pin negatives.",
            "controls": [
                {
                    "id": "wrong_index_count",
                    "submission": "negative_solutions/wrong_index_count",
                    "expected_failure_codes": ["wrong_ratio"],
                    "expected_hard_gate_passed": True,
                    "expected_score_below": 0.95,
                },
                {
                    "id": "missing_port",
                    "submission": "negative_solutions/missing_port",
                    "expected_failure_codes": ["missing_port"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.001,
                },
                {
                    "id": "missing_contact_geometry",
                    "submission": "negative_solutions/missing_contact_geometry",
                    "expected_failure_codes": ["missing_contact"],
                    "expected_hard_gate_passed": False,
                    "expected_score_below": 0.6,
                },
            ],
        }
        return GeneratedTask(
            task_id=task_id,
            family=self.family,
            difficulty=int(difficulty),
            prompt_md=prompt,
            task_toml={
                "task": {
                    "id": task_id,
                    "family": self.family,
                    "difficulty": int(difficulty),
                    "units": "mm",
                    "tier": self.tier,
                },
                "requirements": {
                    "required_ports": ["input_port", "output_port"],
                    "expected_mobility": 2,
                    "max_envelope_mm": [140, 140, 80],
                },
                "objective": {
                    "description": (
                        f"{slot_count}-slot Geneva drive-pin contact indexer."
                    ),
                    "allow_massless_links": False,
                    "ground_required": True,
                },
                "capability": {
                    "requires_adapter": "rigid_body_dynamics+contact_forces",
                    "synthetic_oracle": False,
                },
            },
            eval_config_toml=cfg,
            eval_config_hidden_toml=hidden_cfg,
            fixtures={},
            reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(
                self.family,
                self.tier,
                seed,
                difficulty,
                index_count=slot_count,
                contact_pair=pair,
                requires_adapter="rigid_body_dynamics+contact_forces",
            ),
        )
