"""Port-velocity-ratio probe.

Compares the angular- (or linear-) velocity ratio between two named
ports against an expected target. The probe is intentionally
mechanism-agnostic: a single-stage gear, a four-bar, a planetary
carrier-vs-sun, a chain drive all reduce to "input velocity ÷ output
velocity ought to equal R."

The probe accepts traces in two equivalent forms:

1. ``sim_outputs["joint_velocities"][port_id]`` — preferred. An (N,)
   array of d(joint coord)/dt samples on a common time axis.
2. ``sim_outputs["joint_positions"][port_id]`` plus
   ``sim_outputs["time_s"]`` — fallback. The probe finite-differences
   to recover velocity, after unwrapping angular signals.

A median (or signed median) is used to be robust to a handful of
near-zero input samples that would otherwise dominate a per-sample
mean of ratios.

Failure code: ``WRONG_RATIO``.
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


def _unwrap(theta: np.ndarray) -> np.ndarray:
    if theta.size < 2:
        return theta
    return np.unwrap(theta)


def _diff_velocity(
    pos: np.ndarray, t: np.ndarray | None,
) -> np.ndarray:
    pos = _unwrap(np.asarray(pos, dtype=float).reshape(-1))
    if pos.size < 2:
        return np.zeros_like(pos)
    if t is None or t.size != pos.size:
        # Assume uniform unit spacing.
        return np.gradient(pos, edge_order=1)
    return np.gradient(pos, t, edge_order=1)


def _resolve_velocity(
    sim_outputs: dict[str, Any], port_id: str,
) -> np.ndarray | None:
    vels = sim_outputs.get("joint_velocities") or {}
    v = _as_1d(vels.get(port_id))
    if v is not None:
        return v
    poses = sim_outputs.get("joint_positions") or {}
    p = _as_1d(poses.get(port_id))
    if p is None:
        return None
    t = _as_1d(sim_outputs.get("time_s"))
    return _diff_velocity(p, t)


def _robust_ratio(
    v_in: np.ndarray,
    v_out: np.ndarray,
    *,
    min_abs_input: float,
    use_median: bool,
) -> tuple[float | None, float, float]:
    """Return (ratio, input_median, output_median).

    Pairs are aligned by index. Samples where |v_in| < min_abs_input
    are dropped to avoid divide-by-near-zero blow-ups.
    """
    n = min(v_in.size, v_out.size)
    if n == 0:
        return None, 0.0, 0.0
    vi = v_in[:n]
    vo = v_out[:n]
    mask = np.abs(vi) >= float(min_abs_input)
    if mask.sum() == 0:
        # The input never moved: no observable ratio.
        return None, float(np.median(vi)), float(np.median(vo))
    ratios = vo[mask] / vi[mask]
    if use_median:
        r = float(np.median(ratios))
    else:
        r = float(np.mean(ratios))
    return r, float(np.median(vi)), float(np.median(vo))


@register_probe
class PortVelocityRatio(Probe):
    type_name = "port_velocity_ratio"
    # PLANAR_KINEMATICS is sufficient because the probe accepts either
    # explicit joint_velocities or derives them from joint_positions +
    # time_s, which planar adapters already emit. A spatial adapter
    # that also advertises POSE_TRACES will satisfy this requirement
    # once such an adapter is registered.
    capabilities_required = frozenset({Capability.PLANAR_KINEMATICS})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        input_port = str(config.get("input_port", "input_port"))
        output_port = str(config.get("output_port", "output_port"))
        expected = float(config.get("expected", 1.0))
        tolerance_pct = float(config.get("tolerance_pct", 5.0))
        min_abs_input = float(config.get("min_abs_input_velocity", 1e-6))
        use_median = bool(config.get("use_median", True))

        v_in = _resolve_velocity(sim_outputs, input_port)
        v_out = _resolve_velocity(sim_outputs, output_port)
        if v_in is None or v_out is None:
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
                        f"port_velocity_ratio missing velocity trace "
                        f"for {input_port!r} or {output_port!r} in "
                        f"simulator output."
                    ),
                    public_hint=(
                        "The adapter did not produce joint_velocities "
                        "or joint_positions for the requested ports. "
                        "Check that the ports exist and that the "
                        "adapter capabilities cover this trace."
                    ),
                )],
            )

        ratio, vi_med, vo_med = _robust_ratio(
            v_in, v_out,
            min_abs_input=min_abs_input,
            use_median=use_median,
        )
        if ratio is None:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics={
                    "input_velocity_median": float(vi_med),
                    "output_velocity_median": float(vo_med),
                },
                failures=[Failure(
                    code=FailureCode.WRONG_RATIO,
                    severity=Severity.MAJOR,
                    message=(
                        f"Input velocity for {input_port!r} stayed "
                        f"below {min_abs_input:g} rad/s over the entire "
                        f"trace; cannot infer ratio."
                    ),
                    metric="input_velocity_median",
                    observed=float(vi_med),
                    target=float(min_abs_input),
                )],
            )

        if abs(expected) < 1e-12:
            err_pct = abs(ratio - expected) * 100.0
        else:
            err_pct = abs(ratio - expected) / abs(expected) * 100.0

        passed = err_pct <= tolerance_pct
        if tolerance_pct > 0.0:
            score = max(0.0, 1.0 - err_pct / tolerance_pct)
        else:
            score = 1.0 if passed else 0.0
        metrics = {
            "ratio_observed": float(ratio),
            "ratio_expected": float(expected),
            "ratio_error_pct": float(err_pct),
            "input_velocity_median": float(vi_med),
            "output_velocity_median": float(vo_med),
        }
        failures: list[Failure] = []
        if not passed:
            failures.append(Failure(
                code=FailureCode.WRONG_RATIO,
                severity=Severity.MAJOR,
                message=(
                    f"Port velocity ratio {ratio:+.4f} differs from "
                    f"expected {expected:+.4f} by {err_pct:.2f}% "
                    f"(tolerance {tolerance_pct:.2f}%)."
                ),
                metric="ratio_observed",
                observed=float(ratio),
                target=float(expected),
                public_hint=(
                    "Check link/gear ratios. A negative ratio is "
                    "typical for reversers; a wrong sign means the "
                    "kinematic chain is flipped."
                ),
            ))
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
