"""Lockup probe.

Catches the failure mode where the input is driven through a full
cycle but the output never moves (or moves below a useful threshold).
This is distinct from ``port_velocity_ratio``: that probe asks "is
the ratio correct"; this one asks "does the mechanism move at all".

Inputs (from sim_outputs):
  joint_positions[input_port]   (N,)
  joint_positions[output_port]  (N,)
  joint_velocities[output_port] (N,)  — optional, used if configured.
  time_s                        (N,)  — optional.

Reports:
  lockup_detected         (0 or 1)
  output_motion_rad       max-min unwrapped output angle
  input_motion_rad        max-min unwrapped input angle
  output_velocity_max     absolute max speed observed

Failure: LOCKUP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _as_1d(x) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _range(arr: np.ndarray | None) -> float:
    if arr is None or arr.size < 2:
        return 0.0
    return float(np.ptp(np.unwrap(arr)))


@register_probe
class Lockup(Probe):
    type_name = "lockup"
    # The probe only needs joint position traces. Any kinematic or
    # dynamic adapter advertising PLANAR_KINEMATICS provides them; a
    # future chrono_contact adapter operating on a planar mechanism
    # would also advertise this cap as a side effect.
    capabilities_required = frozenset({Capability.PLANAR_KINEMATICS})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        input_port = str(config.get("input_port", "input_port"))
        output_port = str(config.get("output_port", "output_port"))
        min_out_motion = float(config.get("min_output_motion_rad", 0.05))
        min_out_vel = config.get("min_output_velocity_rad_s")
        if min_out_vel is not None:
            min_out_vel = float(min_out_vel)

        positions = sim_outputs.get("joint_positions") or {}
        velocities = sim_outputs.get("joint_velocities") or {}
        in_pos = _as_1d(positions.get(input_port))
        out_pos = _as_1d(positions.get(output_port))
        out_vel = _as_1d(velocities.get(output_port))

        if in_pos is None and out_pos is None:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics={},
                failures=[Failure(
                    code=FailureCode.SIMULATOR_DIVERGENCE,
                    severity=Severity.CRITICAL,
                    message=(
                        f"lockup: no joint positions for {input_port!r} "
                        f"or {output_port!r} in sim_outputs."
                    ),
                )],
            )

        in_motion = _range(in_pos)
        out_motion = _range(out_pos)
        out_v_max = (float(np.max(np.abs(out_vel)))
                     if out_vel is not None else 0.0)

        # Lockup only makes sense if the input *was* driven.
        if in_pos is None or in_motion < 1e-6:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=True,
                score=1.0,
                metrics={
                    "input_motion_rad": float(in_motion),
                    "output_motion_rad": float(out_motion),
                    "output_velocity_max": float(out_v_max),
                    "lockup_detected": 0.0,
                    "skipped_no_input_drive": 1.0,
                },
                failures=[],
                skipped_reason=(
                    "Input port was not driven (motion below 1e-6 rad). "
                    "Lockup is undefined; reporting pass."
                ),
            )

        motion_lockup = out_motion < min_out_motion
        velocity_lockup = (
            min_out_vel is not None and out_v_max < min_out_vel
        )
        lockup = motion_lockup or velocity_lockup
        metrics = {
            "input_motion_rad": float(in_motion),
            "output_motion_rad": float(out_motion),
            "output_velocity_max": float(out_v_max),
            "lockup_detected": 1.0 if lockup else 0.0,
        }
        failures: list[Failure] = []
        if lockup:
            failures.append(Failure(
                code=FailureCode.LOCKUP,
                severity=Severity.CRITICAL,
                message=(
                    f"Output port {output_port!r} did not move: "
                    f"|Δθ_out| = {out_motion:.4f} rad over "
                    f"|Δθ_in| = {in_motion:.4f} rad (limit "
                    f"{min_out_motion:.4f})."
                ),
                metric="output_motion_rad",
                observed=float(out_motion),
                target=float(min_out_motion),
                public_hint=(
                    "Either the mechanism is structurally locked "
                    "(Grübler mobility 0 may not have caught this), "
                    "or the contact / joint chain that should transmit "
                    "motion is not closing."
                ),
            ))
        # Dense score: how close output motion is to the threshold.
        if min_out_motion > 0.0:
            score = max(0.0, min(out_motion / min_out_motion, 1.0))
        else:
            score = 1.0 if not lockup else 0.0
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=not lockup,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
