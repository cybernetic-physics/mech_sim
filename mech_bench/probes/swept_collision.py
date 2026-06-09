"""Swept-collision probe.

Reports the worst penetration between part pairs while a mechanism
is swept through its motion. The probe is data-driven: it consumes
``sim_outputs["penetration"]`` — a ``dict[pair_key, ndarray]`` — and
optionally a shared ``time_s`` axis. The pair_key uses
``"partA:partB"`` (order-insensitive normalization is applied).

If the simulator did not emit penetration traces at all, the probe
either reports ``CAPABILITY_UNAVAILABLE`` (so the dispatcher knows
the underlying capability is missing) or ``SIMULATOR_DIVERGENCE``
(when penetration is present but malformed). The evaluator's missing-
capability handling will already short-circuit cases where no
adapter advertises the necessary capabilities; this probe only sees
sim_outputs from an adapter that *claimed* support.

Failure codes:
  * ``EXCESSIVE_PENETRATION`` — at least one pair exceeded the
    configured threshold.
  * ``COLLISION`` — falls back to this when allowed_pairs is non-empty
    and the offending pair is not in the allowlist.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _normalize_pair(p: str) -> str:
    a, _, b = p.partition(":")
    if not b:
        return p
    return ":".join(sorted([a, b]))


def _to_pair_set(items: Iterable[str]) -> set[str]:
    return {_normalize_pair(p) for p in items}


def _coerce_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


@register_probe
class SweptCollision(Probe):
    type_name = "swept_collision"
    capabilities_required = frozenset({
        Capability.MESH_OVERLAP,
    })

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        max_pen_mm = float(config.get("max_penetration_mm", 0.05))
        allowed = _to_pair_set(config.get("allowed_pairs", []) or [])
        ignored = _to_pair_set(config.get("ignored_pairs", []) or [])

        penetration = sim_outputs.get("penetration")
        if not isinstance(penetration, dict) or not penetration:
            # Adapter did not provide collision output. Distinguish
            # "explicitly unavailable" (a sentinel key) from "missing
            # but adapter said it ran."
            unavailable = sim_outputs.get(
                "__capability_unavailable__", False
            ) or sim_outputs.get("collision_unavailable", False)
            code = (FailureCode.CAPABILITY_UNAVAILABLE
                    if unavailable else
                    FailureCode.SIMULATOR_DIVERGENCE)
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics={},
                failures=[Failure(
                    code=code,
                    severity=Severity.CRITICAL,
                    message=(
                        "swept_collision: no penetration traces in "
                        "sim_outputs; the active adapter does not "
                        "produce collision data."
                    ),
                    public_hint=(
                        "This probe needs a simulator that emits "
                        "penetration depth per part pair. Either run a "
                        "mesh-overlap-capable adapter or relax the "
                        "task's collision requirement."
                    ),
                )],
            )

        time_s = _coerce_array(sim_outputs.get("time_s"))

        worst_pair = ""
        worst_pen = 0.0
        worst_idx = -1
        pair_worst: dict[str, float] = {}
        for raw_pair, trace in penetration.items():
            pair_key = _normalize_pair(str(raw_pair))
            if pair_key in ignored:
                continue
            arr = _coerce_array(trace)
            if arr is None:
                continue
            # Take the magnitude; penetration is usually positive but
            # different adapters use different sign conventions.
            mag = np.abs(arr)
            idx = int(np.argmax(mag))
            pen = float(mag[idx])
            pair_worst[pair_key] = max(pair_worst.get(pair_key, 0.0), pen)
            if pen > worst_pen:
                worst_pen = pen
                worst_pair = pair_key
                worst_idx = idx

        worst_time_s = 0.0
        if (time_s is not None and worst_idx >= 0
                and worst_idx < time_s.size):
            worst_time_s = float(time_s[worst_idx])

        metrics: dict[str, float] = {
            "max_penetration_mm": float(worst_pen),
            "n_pairs_seen": float(len(pair_worst)),
            "worst_time_s": worst_time_s,
        }
        for k, v in pair_worst.items():
            metrics[f"pair.{k}.max_pen_mm"] = float(v)

        failures: list[Failure] = []
        passed = True
        offender_in_allowlist = (
            allowed and worst_pair and worst_pair in allowed
        )
        if worst_pen > max_pen_mm and not offender_in_allowlist:
            passed = False
            code = (FailureCode.COLLISION
                    if allowed and worst_pair not in allowed
                    else FailureCode.EXCESSIVE_PENETRATION)
            failures.append(Failure(
                code=code,
                severity=Severity.CRITICAL,
                message=(
                    f"Pair {worst_pair!r} penetrated by "
                    f"{worst_pen:.4f} mm (limit {max_pen_mm:.4f} mm)."
                ),
                metric="max_penetration_mm",
                observed=float(worst_pen),
                target=float(max_pen_mm),
                public_hint=(
                    "Increase clearance for this pair, change the "
                    "joint geometry, or move it onto the allowed_pairs "
                    "list if the contact is intentional."
                ),
            ))
        if max_pen_mm > 0.0:
            score = max(0.0, 1.0 - worst_pen / max_pen_mm)
        else:
            score = 1.0 if passed else 0.0
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
