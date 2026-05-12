"""Tier 3 — synthetic contact/dynamics task generators.

Ten families that exercise the contact / dynamics probe surface using
the *synthetic* fake_contact_oracle adapter. These are deliberately
test/demo tasks — they prove the contact_engagement / lockup /
torque_load_trial probes wire up correctly, but they do NOT validate
physics. Every task config carries an explicit
``[adapters.fake_contact_oracle] enabled = true`` so the runtime opts
in to the fake oracle. Reports tag the result as
``oracle_is_synthetic = true``.
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
    fake_contact_probe_config,
    fake_oracle_adapters_block,
    frame_port,
    make_basic_design_py,
    make_expected_failures,
    make_ground_part,
    make_negative_overlay,
    make_revolute_part,
    param_check_probe,
    prismatic_joint,
    prismatic_joint_port,
    required_ports_probe,
    revolute_joint,
    revolute_joint_port,
)


def _contact_pair_part_design(
    part_a: str, part_b: str, contact_id: str,
    prismatic_b: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    parts = [
        make_ground_part("frame"),
        make_revolute_part(part_a, "input", 0.04),
    ]
    if prismatic_b:
        parts.append({
            "id": part_b, "role": "output", "mass_kg": 0.08,
            "com_local_mm": (0.0, 0.0, 0.0),
        })
    else:
        parts.append(make_revolute_part(part_b, "output", 0.05))
    joints = [
        revolute_joint("input_axis", "frame", part_a, (0.0, 0.0, 0.0)),
    ]
    if prismatic_b:
        joints.append(prismatic_joint(
            "output_axis", "frame", part_b))
    else:
        joints.append(revolute_joint(
            "output_axis", "frame", part_b, (40.0, 0.0, 0.0)))
    joints.append({
        "id": contact_id,
        "type": "contact_pair",
        "parent": part_a, "child": part_b,
        "axis_world": (0.0, 0.0, 1.0),
        "anchor_world_mm": (0.0, 0.0, 0.0),
    })
    if prismatic_b:
        ports = {
            "input_port": revolute_joint_port(
                "input_port", "input_axis"),
            "output_port": prismatic_joint_port(
                "output_port", "output_axis"),
        }
    else:
        ports = {
            "input_port": revolute_joint_port(
                "input_port", "input_axis"),
            "output_port": revolute_joint_port(
                "output_port", "output_axis"),
        }
    return parts, joints, ports


# --------------------------------------------------------------------- #
# Helper: build T3 eval_config with fake_oracle adapter block            #
# --------------------------------------------------------------------- #


def _t3_eval(
    pairs: list[str],
    *,
    probes: list[dict[str, Any]],
    public_metrics: list[str],
    hidden_metrics: list[str],
    hard_gate: list[str],
    fake_oracle_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block = fake_oracle_adapters_block(
        pairs, **(fake_oracle_kwargs or {}))
    return {
        "probes": probes,
        "feedback": {
            "public_metrics": public_metrics,
            "hidden_metrics": hidden_metrics,
        },
        "hard_gate": {"require": hard_gate},
        "adapters": {
            "fake_contact_oracle": block,
        },
    }


# --------------------------------------------------------------------- #
# 31. cam_follower_contact_stub                                         #
# --------------------------------------------------------------------- #


class CamFollowerContactStubGenerator(TaskGenerator):
    family = "cam_follower_contact_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        task_id = make_task_id(self.family, seed)
        parts, joints, ports = _contact_pair_part_design(
            "cam", "follower", "cam_follower")
        params = {"declared_pair": "cam:follower"}
        prompt = (
            "# Cam–follower contact (synthetic stub)\n\n"
            "Synthetic Tier-3 stub: fake oracle reports cam:follower "
            "contact force and penetration.\n"
        )
        pair = "cam:follower"
        probes = [
            required_ports_probe(
                "ports", ["input_port", "output_port"],
                require_grounded=["input_port"],
            ),
            fake_contact_probe_config(
                "contact", [pair], min_rms_force_N=0.5,
                weight=1.0, hard_gate=True,
                adapter="fake_contact_oracle",
            ),
            {"id": "swept_collision", "type": "swept_collision",
             "max_penetration_mm": 0.05,
             "weight": 0.5, "severity": "major",
             "adapter": "fake_contact_oracle"},
            {"id": "lockup", "type": "lockup",
             "input_port": "input_port", "output_port": "output_port",
             "min_output_motion_rad": 0.05,
             "weight": 0.0, "severity": "critical",
             "adapter": "fake_contact_oracle"},
        ]
        ref_py = make_basic_design_py(parts, joints, ports, params)
        # The fake oracle synthesizes contact traces for whatever pairs
        # the adapter config declares, so a pure-IR mutation cannot
        # break engagement — drop a required port to get a real fail.
        negatives = {
            "missing_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
            "excessive_penetration": make_negative_overlay(
                "    pass"
            ),
        }
        expected = make_expected_failures(
            f"Tier 3 {self.family} negatives.",
            [
                {"id": "missing_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.5},
                {"id": "excessive_penetration",
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
                "max_envelope_mm": [120, 120, 80],
            },
            "objective": {
                "description": "Cam-follower contact stub.",
                "ground_required": True,
            },
            "capability": {
                "requires_adapter": "contact_forces",
                "synthetic_oracle": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_t3_eval(
                [pair], probes=probes,
                public_metrics=[
                    "ports.ports_required",
                    f"contact.contact.{pair}.rms_N",
                    f"contact.contact.{pair}.engagement_fraction",
                    "swept_collision.max_penetration_mm",
                    "lockup.lockup_detected",
                ],
                hidden_metrics=[],
                hard_gate=["ports", "contact"],
                fake_oracle_kwargs={
                    "contact_force_N": 5.0,
                    "penetration_mm": 0.01,
                    "contact_engagement_fraction": 0.9,
                },
            ),
            eval_config_hidden_toml=_t3_eval(
                [pair], probes=probes,
                public_metrics=[
                    f"contact.contact.{pair}.rms_N"],
                hidden_metrics=[],
                hard_gate=["ports", "contact"],
                fake_oracle_kwargs={
                    "contact_force_N": 8.0,
                    "penetration_mm": 0.01,
                    "contact_engagement_fraction": 0.9,
                },
            ),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, pair=pair),
        )


# --------------------------------------------------------------------- #
# helper for the remaining T3 families (shared structure)               #
# --------------------------------------------------------------------- #


def _make_simple_contact_task(
    family: str, tier: str, task_id: str,
    pair_parts: tuple[str, str], prismatic_output: bool = False,
    *,
    extra_probes: list[dict[str, Any]] | None = None,
    extra_param_paths: list[str] | None = None,
    fake_oracle_kwargs: dict[str, Any] | None = None,
    extra_negatives: dict[str, str] | None = None,
    extra_expected: list[dict[str, Any]] | None = None,
    params_extra: dict[str, Any] | None = None,
    prompt: str = "",
) -> GeneratedTask:
    a, b = pair_parts
    pair = f"{a}:{b}"
    parts, joints, ports = _contact_pair_part_design(
        a, b, f"{a}_{b}_contact", prismatic_b=prismatic_output)
    params = {"declared_pair": pair}
    if params_extra:
        params.update(params_extra)

    probes = [
        required_ports_probe(
            "ports", ["input_port", "output_port"],
            require_grounded=["input_port"],
        ),
        fake_contact_probe_config(
            "contact", [pair], min_rms_force_N=0.5,
            weight=1.0, hard_gate=True,
            adapter="fake_contact_oracle",
        ),
    ]
    if extra_probes:
        probes.extend(extra_probes)

    public_metrics = [
        "ports.ports_required",
        f"contact.contact.{pair}.rms_N",
    ]
    for path in extra_param_paths or []:
        public_metrics.append(f"{path}.observed")

    ref_py = make_basic_design_py(parts, joints, ports, params)
    # The fake oracle synthesizes contact traces for whatever
    # ``contact_pairs`` are declared in the adapter config — deleting
    # the contact_pair joint from the IR alone won't surface
    # missing_contact. To produce a hard-gate fail, the "missing_port"
    # control drops a required port instead.
    negatives = {
        "missing_port": make_negative_overlay(
            "    del ir['ports']['output_port']"
        ),
    }
    if extra_negatives:
        negatives.update(extra_negatives)

    expected_controls = [
        {"id": "missing_port",
         "expected_failure_codes": ["missing_port"],
         "expected_hard_gate_passed": False,
         "expected_score_below": 0.5},
    ]
    expected_controls.extend(extra_expected or [])

    expected = make_expected_failures(
        f"Tier 3 {family} negatives.", expected_controls)

    task_toml = {
        "task": {"id": task_id, "family": family,
                 "difficulty": 3, "units": "mm",
                 "tier": tier},
        "requirements": {
            "required_ports": ["input_port", "output_port"],
            "expected_mobility": 2,
            "max_envelope_mm": [200, 200, 80],
        },
        "objective": {
            "description": f"{family} synthetic contact stub.",
            "ground_required": True,
        },
        "capability": {
            "requires_adapter": "contact_forces",
            "synthetic_oracle": True,
        },
    }
    return GeneratedTask(
        task_id=task_id, family=family,
        difficulty=3, prompt_md=prompt or
        f"# {family}\n\nSynthetic Tier-3 stub with pair {pair}.\n",
        task_toml=task_toml,
        eval_config_toml=_t3_eval(
            [pair], probes=probes,
            public_metrics=public_metrics,
            hidden_metrics=[],
            hard_gate=["ports", "contact"],
            fake_oracle_kwargs=fake_oracle_kwargs or {
                "contact_force_N": 5.0,
                "penetration_mm": 0.01,
                "contact_engagement_fraction": 0.9,
            },
        ),
        eval_config_hidden_toml=_t3_eval(
            [pair], probes=probes,
            public_metrics=public_metrics,
            hidden_metrics=[],
            hard_gate=["ports", "contact"],
            fake_oracle_kwargs={
                **(fake_oracle_kwargs or {}),
                "contact_force_N": (
                    (fake_oracle_kwargs or {})
                    .get("contact_force_N", 5.0) * 1.2),
            },
        ),
        fixtures={}, reference_solution_py=ref_py,
        negative_solutions=negatives,
        expected_failures=expected,
        metadata=common_metadata(family, tier, 0, 3,
                                  pair=pair),
    )


# --------------------------------------------------------------------- #
# 32. ratchet_pawl_engagement_stub                                      #
# --------------------------------------------------------------------- #


class RatchetPawlEngagementStubGenerator(TaskGenerator):
    family = "ratchet_pawl_engagement_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("ratchet", "pawl"),
            extra_probes=[
                {"id": "lockup", "type": "lockup",
                 "input_port": "input_port",
                 "output_port": "output_port",
                 "min_output_motion_rad": 0.05,
                 "weight": 0.0, "severity": "major",
                 "adapter": "fake_contact_oracle"},
            ],
            extra_negatives={
                "lockup": make_negative_overlay(
                    "    pass\n"
                ),
            },
            extra_expected=[
                {"id": "lockup",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
            prompt="# Ratchet & pawl engagement (synthetic)\n",
        )


# --------------------------------------------------------------------- #
# 33. geneva_indexing_stub                                              #
# --------------------------------------------------------------------- #


class GenevaIndexingStubGenerator(TaskGenerator):
    family = "geneva_indexing_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 5333)
        n_slots = rng.choice([4, 5, 6])
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("driver", "geneva"),
            extra_probes=[
                param_check_probe(
                    "index_count", "params.index_count",
                    float(n_slots), tolerance_abs=0.0,
                    failure_code="wrong_ratio", weight=0.5,
                ),
                {"id": "lockup", "type": "lockup",
                 "input_port": "input_port",
                 "output_port": "output_port",
                 "min_output_motion_rad": 0.05,
                 "weight": 0.0, "severity": "major",
                 "adapter": "fake_contact_oracle"},
            ],
            extra_param_paths=["index_count"],
            params_extra={"index_count": int(n_slots)},
            extra_negatives={
                "wrong_index_count": make_negative_overlay(
                    f"    ir['params']['index_count'] = {n_slots + 2}"
                ),
                "jammed": make_negative_overlay(
                    "    pass\n"
                ),
            },
            extra_expected=[
                {"id": "wrong_index_count",
                 "expected_failure_codes": ["wrong_ratio"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.9},
                {"id": "jammed",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
            prompt=f"# Geneva indexing stub ({n_slots} slots)\n",
        )


# --------------------------------------------------------------------- #
# 34. friction_clutch_torque_stub                                       #
# --------------------------------------------------------------------- #


class FrictionClutchTorqueStubGenerator(TaskGenerator):
    family = "friction_clutch_torque_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("plate_in", "plate_out"),
            extra_probes=[
                {"id": "torque", "type": "torque_load_trial",
                 "input_port": "input_port",
                 "output_port": "output_port",
                 "input_speed_rad_s": 1.0,
                 "output_load_Nm": 0.05,
                 "min_output_speed_rad_s": 0.001,
                 "max_power_error_pct": 25.0,
                 "max_torque_ripple_pct": 30.0,
                 "weight": 0.5, "severity": "major",
                 "adapter": "fake_contact_oracle"},
            ],
            extra_negatives={
                "slip_under_load": make_negative_overlay(
                    "    pass\n"
                ),
                "excessive_torque_ripple": make_negative_overlay(
                    "    pass\n"
                ),
            },
            extra_expected=[
                {"id": "slip_under_load",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
                {"id": "excessive_torque_ripple",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
            prompt="# Friction-clutch torque stub\n",
        )


# --------------------------------------------------------------------- #
# 35. brake_caliper_contact_stub                                        #
# --------------------------------------------------------------------- #


class BrakeCaliperContactStubGenerator(TaskGenerator):
    family = "brake_caliper_contact_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("disc", "pad"),
            extra_probes=[
                {"id": "swept_collision", "type": "swept_collision",
                 "max_penetration_mm": 0.05,
                 "weight": 0.5, "severity": "major",
                 "adapter": "fake_contact_oracle"},
            ],
            extra_negatives={
                "excessive_penetration": make_negative_overlay(
                    "    pass\n"
                ),
            },
            extra_expected=[
                {"id": "excessive_penetration",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
            prompt="# Brake-caliper disc:pad contact stub\n",
        )


# --------------------------------------------------------------------- #
# 36. parallel_gripper_retention_stub                                   #
# --------------------------------------------------------------------- #


class ParallelGripperRetentionStubGenerator(TaskGenerator):
    family = "parallel_gripper_retention_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 5636)
        retention = round(rng.uniform(8.0, 25.0), 3)
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("finger", "object"),
            extra_probes=[
                param_check_probe(
                    "retention", "params.retention_force_N",
                    retention, comparator="ge",
                    failure_code="insufficient_clearance",
                    weight=0.5,
                ),
            ],
            extra_param_paths=["retention"],
            params_extra={"retention_force_N": retention},
            extra_negatives={
                "weak_grip": make_negative_overlay(
                    "    ir['params']['retention_force_N'] = 0.1"
                ),
            },
            extra_expected=[
                {"id": "weak_grip",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.7},
            ],
            prompt=f"# Parallel gripper retention "
                   f"(force ≥ {retention} N)\n",
        )


# --------------------------------------------------------------------- #
# 37. latch_release_force_stub                                          #
# --------------------------------------------------------------------- #


class LatchReleaseForceStubGenerator(TaskGenerator):
    family = "latch_release_force_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 5737)
        release = round(rng.uniform(2.0, 6.0), 3)
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("latch", "strike"),
            extra_probes=[
                param_check_probe(
                    "release", "params.release_force_N", release,
                    comparator="le",
                    tolerance_abs=0.5,
                    failure_code="insufficient_clearance",
                    weight=0.5,
                ),
            ],
            extra_param_paths=["release"],
            params_extra={"release_force_N": release},
            extra_negatives={
                "excessive_release_force": make_negative_overlay(
                    "    ir['params']['release_force_N'] = 50.0"
                ),
            },
            extra_expected=[
                {"id": "excessive_release_force",
                 "expected_failure_codes": ["insufficient_clearance"],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 0.7},
            ],
            prompt=f"# Latch release force ≤ {release} N\n",
        )


# --------------------------------------------------------------------- #
# 38. detent_spring_contact_stub                                        #
# --------------------------------------------------------------------- #


class DetentSpringContactStubGenerator(TaskGenerator):
    family = "detent_spring_contact_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("ball", "groove"),
            extra_probes=[
                {"id": "swept_collision", "type": "swept_collision",
                 "max_penetration_mm": 0.04,
                 "weight": 0.5, "severity": "major",
                 "adapter": "fake_contact_oracle"},
            ],
            extra_negatives={
                "excessive_penetration": make_negative_overlay(
                    "    pass\n"
                ),
            },
            extra_expected=[
                {"id": "excessive_penetration",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
            prompt="# Detent spring (ball:groove) contact stub\n",
        )


# --------------------------------------------------------------------- #
# 39. gear_pair_load_trial_stub                                         #
# --------------------------------------------------------------------- #


class GearPairLoadTrialStubGenerator(TaskGenerator):
    family = "gear_pair_load_trial_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 5939)
        t_in = rng.choice([12, 14, 16])
        t_out = t_in * rng.choice([2, 3])
        ratio = round(t_out / t_in, 6)
        task_id = make_task_id(self.family, seed)
        pair = "gear1:gear2"
        # build a custom design that names the gears.
        parts, joints, ports = _contact_pair_part_design(
            "gear1", "gear2", "gear_contact")
        params = {
            "teeth_in": t_in, "teeth_out": t_out,
            "declared_ratio": ratio,
        }
        prompt = (
            "# Gear pair load trial (synthetic)\n\n"
            f"Two-gear contact pair with declared ratio {ratio}.\n"
        )
        probes = [
            required_ports_probe(
                "ports", ["input_port", "output_port"],
                require_grounded=["input_port", "output_port"],
            ),
            fake_contact_probe_config(
                "contact", [pair],
                min_rms_force_N=0.5,
                weight=0.5, hard_gate=True,
                adapter="fake_contact_oracle",
            ),
            {"id": "torque", "type": "torque_load_trial",
             "input_port": "input_port",
             "output_port": "output_port",
             "input_speed_rad_s": 1.0,
             "output_load_Nm": 0.05,
             "min_output_speed_rad_s": 0.001,
             "weight": 0.3, "severity": "major",
             "adapter": "fake_contact_oracle"},
            {"id": "ratio_obs", "type": "port_velocity_ratio",
             "input_port": "input_port",
             "output_port": "output_port",
             "expected": 1.0 / ratio,
             "tolerance_pct": 10.0,
             "weight": 0.2, "severity": "major",
             "adapter": "fake_contact_oracle"},
        ]

        ref_py = make_basic_design_py(parts, joints, ports, params)
        negatives = {
            "wrong_ratio": make_negative_overlay(
                f"    ir['params']['declared_ratio'] = "
                f"{round(ratio * 0.4, 6)}"
            ),
            "missing_port": make_negative_overlay(
                "    del ir['ports']['output_port']"
            ),
        }
        expected = make_expected_failures(
            f"Tier 3 {self.family} negatives.",
            [
                {"id": "wrong_ratio",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
                {"id": "missing_port",
                 "expected_failure_codes": ["missing_port"],
                 "expected_hard_gate_passed": False,
                 "expected_score_below": 0.5},
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
                "description": f"Gear pair load trial; ratio={ratio}.",
                "ground_required": True,
            },
            "capability": {
                "requires_adapter": "contact_forces",
                "synthetic_oracle": True,
            },
        }
        return GeneratedTask(
            task_id=task_id, family=self.family,
            difficulty=int(difficulty), prompt_md=prompt,
            task_toml=task_toml,
            eval_config_toml=_t3_eval(
                [pair], probes=probes,
                public_metrics=[
                    f"contact.contact.{pair}.rms_N",
                    "torque.output_speed_observed_rad_s",
                    "ratio_obs.ratio_observed",
                ],
                hidden_metrics=[],
                hard_gate=["ports", "contact"],
                fake_oracle_kwargs={
                    "contact_force_N": 5.0,
                    "penetration_mm": 0.01,
                    "ratio_observed": ratio,
                    "contact_engagement_fraction": 0.9,
                },
            ),
            eval_config_hidden_toml=_t3_eval(
                [pair], probes=probes,
                public_metrics=[f"contact.contact.{pair}.rms_N"],
                hidden_metrics=[],
                hard_gate=["ports", "contact"],
                fake_oracle_kwargs={
                    "contact_force_N": 8.0,
                    "ratio_observed": ratio,
                    "contact_engagement_fraction": 0.9,
                },
            ),
            fixtures={}, reference_solution_py=ref_py,
            negative_solutions=negatives,
            expected_failures=expected,
            metadata=common_metadata(self.family, self.tier, seed,
                                     difficulty, ratio=ratio),
        )


# --------------------------------------------------------------------- #
# 40. rack_pinion_contact_stub                                          #
# --------------------------------------------------------------------- #


class RackPinionContactStubGenerator(TaskGenerator):
    family = "rack_pinion_contact_stub"
    tier = "contact_dynamics"

    def generate(self, seed: int, difficulty: int = 3) -> GeneratedTask:
        rng = random.Random(seed + 6040)
        pitch_radius = round(rng.uniform(8.0, 25.0), 3)
        travel_per_rev = round(2.0 * 3.14159265358979 * pitch_radius, 4)
        task_id = make_task_id(self.family, seed)
        return _make_simple_contact_task(
            self.family, self.tier, task_id, ("pinion", "rack"),
            prismatic_output=True,
            extra_probes=[
                param_check_probe(
                    "travel", "params.declared_travel_per_rev_mm",
                    travel_per_rev, tolerance_pct=2.0,
                    failure_code="wrong_ratio", weight=0.5,
                ),
                {"id": "lockup", "type": "lockup",
                 "input_port": "input_port",
                 "output_port": "output_port",
                 "min_output_motion_rad": 0.05,
                 "weight": 0.0, "severity": "major",
                 "adapter": "fake_contact_oracle"},
            ],
            extra_param_paths=["travel"],
            params_extra={
                "pitch_radius_mm": pitch_radius,
                "declared_travel_per_rev_mm": travel_per_rev,
            },
            extra_negatives={
                "jammed": make_negative_overlay(
                    "    pass\n"
                ),
            },
            extra_expected=[
                {"id": "jammed",
                 "expected_failure_codes": [],
                 "expected_hard_gate_passed": True,
                 "expected_score_below": 1.01},
            ],
            prompt="# Rack–pinion contact stub\n",
        )
