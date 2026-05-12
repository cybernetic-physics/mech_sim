"""Torque-load-trial probe.

Drives the input port at a prescribed speed, applies a prescribed
output load torque, and asks three questions:

  1. Does the output keep moving? (no LOCKUP)
  2. Does power balance hold within tolerance? (no POWER_BALANCE_ERROR)
  3. Is torque ripple acceptable? (no EXCESSIVE_TORQUE_RIPPLE)

The probe is generic: it consumes whatever a contact-and-dynamics
adapter (chrono_contact, mujoco, …) writes to sim_outputs, scoped to
the input/output ports named in config.

Consumed sim_outputs:
  joint_positions[input_port], joint_positions[output_port]
  joint_velocities[input_port], joint_velocities[output_port]
  scalar_metrics or top-level keys:
      "input_torque_Nm_mean", "output_torque_Nm_mean"
      "input_power_W_mean",   "output_power_W_mean"
      "input_torque_ripple_pct"
  time_s

The probe also tolerates the simpler shape where the adapter
provides only joint positions/velocities and the probe derives means
internally.

Failure codes:
  LOCKUP, POWER_BALANCE_ERROR, EXCESSIVE_TORQUE_RIPPLE.
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


def _scalar_get(sim_outputs: dict[str, Any], key: str) -> float | None:
    scalars = sim_outputs.get("scalar_metrics") or {}
    if key in scalars:
        try:
            return float(scalars[key])
        except (TypeError, ValueError):
            return None
    if key in sim_outputs:
        try:
            return float(sim_outputs[key])
        except (TypeError, ValueError):
            return None
    return None


@register_probe
class TorqueLoadTrial(Probe):
    type_name = "torque_load_trial"
    capabilities_required = frozenset({
        Capability.RIGID_BODY_DYNAMICS,
        Capability.MOTOR_DRIVES,
        Capability.LOAD_TORQUES,
        Capability.POSE_TRACES,
    })

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        input_port = str(config.get("input_port", "input_port"))
        output_port = str(config.get("output_port", "output_port"))
        in_speed = float(config.get("input_speed_rad_s", 1.0))
        out_load = float(config.get("output_load_Nm", 0.0))
        min_out_speed = float(config.get("min_output_speed_rad_s", 1e-3))
        max_power_err_pct = config.get("max_power_error_pct")
        max_ripple_pct = config.get("max_torque_ripple_pct")

        positions = sim_outputs.get("joint_positions") or {}
        velocities = sim_outputs.get("joint_velocities") or {}
        in_vel = _as_1d(velocities.get(input_port))
        out_vel = _as_1d(velocities.get(output_port))
        out_pos = _as_1d(positions.get(output_port))

        metrics: dict[str, float] = {}
        failures: list[Failure] = []

        # ----- Motion under load (LOCKUP) -----
        if out_vel is None and out_pos is None:
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
                        f"torque_load_trial: no output position or "
                        f"velocity for {output_port!r}."
                    ),
                )],
            )

        out_speed = (float(np.mean(np.abs(out_vel)))
                     if out_vel is not None else 0.0)
        metrics["input_speed_observed_rad_s"] = (
            float(np.mean(np.abs(in_vel))) if in_vel is not None else 0.0
        )
        metrics["output_speed_observed_rad_s"] = float(out_speed)
        metrics["output_load_Nm"] = float(out_load)

        if out_speed < min_out_speed:
            failures.append(Failure(
                code=FailureCode.LOCKUP,
                severity=Severity.CRITICAL,
                message=(
                    f"Output speed {out_speed:.4f} rad/s below "
                    f"threshold {min_out_speed:.4f} rad/s under "
                    f"{out_load} Nm load."
                ),
                metric="output_speed_observed_rad_s",
                observed=float(out_speed),
                target=float(min_out_speed),
            ))

        # ----- Power balance -----
        p_in = _scalar_get(sim_outputs, "input_power_W_mean")
        p_out = _scalar_get(sim_outputs, "output_power_W_mean")
        if p_in is None:
            t_in = _scalar_get(sim_outputs, "input_torque_Nm_mean")
            if t_in is not None:
                p_in = abs(t_in * in_speed)
        if p_out is None:
            t_out = _scalar_get(sim_outputs, "output_torque_Nm_mean")
            t_eff = t_out if t_out is not None else out_load
            p_out = abs(t_eff * out_speed)

        if p_in is not None and p_out is not None:
            metrics["input_power_W"] = float(p_in)
            metrics["output_power_W"] = float(p_out)
            if p_in > 0.0:
                err = abs(p_in - p_out) / p_in * 100.0
            else:
                err = 0.0 if p_out == 0.0 else 100.0
            metrics["power_balance_error_pct"] = float(err)
            if max_power_err_pct is not None and err > float(max_power_err_pct):
                failures.append(Failure(
                    code=FailureCode.POWER_BALANCE_ERROR,
                    severity=Severity.MAJOR,
                    message=(
                        f"Power balance error {err:.2f}% exceeds "
                        f"{float(max_power_err_pct):.2f}%."
                    ),
                    metric="power_balance_error_pct",
                    observed=float(err),
                    target=float(max_power_err_pct),
                ))

        # ----- Torque ripple -----
        ripple = _scalar_get(sim_outputs, "input_torque_ripple_pct")
        if ripple is None:
            # If a torque trace is available, derive ripple from it.
            t_trace = _as_1d(
                (sim_outputs.get("input_torque_Nm")
                 or sim_outputs.get("output_torque_Nm"))
            )
            if t_trace is not None and t_trace.size > 1:
                mean = float(np.mean(np.abs(t_trace)))
                if mean > 1e-12:
                    ripple = float((t_trace.max() - t_trace.min())
                                   / (2.0 * mean) * 100.0)
        if ripple is not None:
            metrics["torque_ripple_pct"] = float(ripple)
            if (max_ripple_pct is not None
                    and ripple > float(max_ripple_pct)):
                failures.append(Failure(
                    code=FailureCode.EXCESSIVE_TORQUE_RIPPLE,
                    severity=Severity.MAJOR,
                    message=(
                        f"Torque ripple {ripple:.2f}% exceeds "
                        f"{float(max_ripple_pct):.2f}%."
                    ),
                    metric="torque_ripple_pct",
                    observed=float(ripple),
                    target=float(max_ripple_pct),
                ))

        passed = not failures
        # Dense score: weighted by which checks are configured.
        components: list[float] = []
        if min_out_speed > 0:
            components.append(max(0.0, min(out_speed / min_out_speed, 1.0)))
        if (max_power_err_pct is not None
                and "power_balance_error_pct" in metrics):
            err = metrics["power_balance_error_pct"]
            t = float(max_power_err_pct)
            components.append(max(0.0, 1.0 - err / t) if t > 0 else
                              (1.0 if err == 0 else 0.0))
        if (max_ripple_pct is not None
                and "torque_ripple_pct" in metrics):
            r = metrics["torque_ripple_pct"]
            t = float(max_ripple_pct)
            components.append(max(0.0, 1.0 - r / t) if t > 0 else
                              (1.0 if r == 0 else 0.0))
        score = (sum(components) / len(components)) if components else (
            1.0 if passed else 0.0)
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
