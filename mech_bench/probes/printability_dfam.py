"""Printability / DFAM probe.

v0 is intentionally thin: it does not analyze geometry itself. It
consumes precomputed mesh metrics from ``sim_outputs["mesh_metrics"]``
keyed by part id (or a flat per-mesh dict). When metrics are absent
the probe surfaces ``CAPABILITY_UNAVAILABLE`` so the evaluator can
distinguish "task wants printability" from "we forgot to run a
geometry adapter."

Expected per-part metric keys:
  ``min_wall_mm``                — thinnest wall thickness in part.
  ``max_overhang_deg``           — worst overhang from the build axis.
  ``support_volume_fraction``    — optional; v_support / v_part.

Failure: ``UNPRINTABLE``.
"""

from __future__ import annotations

from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _iter_parts(metrics: Any) -> list[tuple[str, dict]]:
    """Normalize either ``{part_id: {...}}`` or a flat metric dict into
    a list of (part_id, dict) tuples. A flat dict is treated as a
    single anonymous part for backwards compatibility.
    """
    if not isinstance(metrics, dict):
        return []
    if not metrics:
        return []
    sample = next(iter(metrics.values()))
    if isinstance(sample, dict):
        return [(str(k), v) for k, v in metrics.items()
                if isinstance(v, dict)]
    return [("__all__", metrics)]


def _f(d: dict, k: str) -> float | None:
    v = d.get(k)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@register_probe
class PrintabilityDFAM(Probe):
    type_name = "printability_dfam"
    capabilities_required = frozenset({Capability.MESH})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        min_wall = float(config.get("min_wall_mm", 1.2))
        max_overhang = float(config.get("max_overhang_deg", 50.0))
        max_support_frac = config.get("max_support_volume_fraction")

        mesh_metrics = sim_outputs.get("mesh_metrics")
        if mesh_metrics is None:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics={},
                failures=[Failure(
                    code=FailureCode.CAPABILITY_UNAVAILABLE,
                    severity=Severity.CRITICAL,
                    message=(
                        "printability_dfam: no mesh_metrics in "
                        "sim_outputs. A mesh-capable adapter must "
                        "compute min_wall / overhang before this probe "
                        "can run."
                    ),
                    public_hint=(
                        "Either ship a precomputed mesh-metric adapter "
                        "or remove printability from this task."
                    ),
                )],
            )

        per_part = _iter_parts(mesh_metrics)
        if not per_part:
            return ProbeResult(
                probe_id="",
                probe_type=self.type_name,
                passed=False,
                score=0.0,
                metrics={},
                failures=[Failure(
                    code=FailureCode.CAPABILITY_UNAVAILABLE,
                    severity=Severity.CRITICAL,
                    message=(
                        "printability_dfam: mesh_metrics is empty."
                    ),
                )],
            )

        worst_wall = float("inf")
        worst_overhang = 0.0
        worst_support = 0.0
        failures: list[Failure] = []
        metrics: dict[str, float] = {"n_parts": float(len(per_part))}
        for part_id, m in per_part:
            wall = _f(m, "min_wall_mm")
            ovh = _f(m, "max_overhang_deg")
            sup = _f(m, "support_volume_fraction")
            if wall is not None:
                metrics[f"part.{part_id}.min_wall_mm"] = float(wall)
                worst_wall = min(worst_wall, wall)
                if wall < min_wall:
                    failures.append(Failure(
                        code=FailureCode.UNPRINTABLE,
                        severity=Severity.MAJOR,
                        message=(
                            f"Part {part_id!r} has min wall "
                            f"{wall:.3f} mm < {min_wall:.3f} mm."
                        ),
                        metric="min_wall_mm",
                        observed=float(wall),
                        target=float(min_wall),
                        where=f"parts.{part_id}",
                    ))
            if ovh is not None:
                metrics[f"part.{part_id}.max_overhang_deg"] = float(ovh)
                worst_overhang = max(worst_overhang, ovh)
                if ovh > max_overhang:
                    failures.append(Failure(
                        code=FailureCode.UNPRINTABLE,
                        severity=Severity.MAJOR,
                        message=(
                            f"Part {part_id!r} has overhang "
                            f"{ovh:.1f}° > {max_overhang:.1f}°."
                        ),
                        metric="max_overhang_deg",
                        observed=float(ovh),
                        target=float(max_overhang),
                        where=f"parts.{part_id}",
                    ))
            if sup is not None:
                metrics[f"part.{part_id}.support_volume_fraction"] = float(sup)
                worst_support = max(worst_support, sup)
                if (max_support_frac is not None
                        and sup > float(max_support_frac)):
                    failures.append(Failure(
                        code=FailureCode.UNPRINTABLE,
                        severity=Severity.MAJOR,
                        message=(
                            f"Part {part_id!r} requires "
                            f"{sup*100:.1f}% support volume "
                            f"(max {float(max_support_frac)*100:.1f}%)."
                        ),
                        metric="support_volume_fraction",
                        observed=float(sup),
                        target=float(max_support_frac),
                        where=f"parts.{part_id}",
                    ))

        if worst_wall != float("inf"):
            metrics["worst_min_wall_mm"] = float(worst_wall)
        metrics["worst_overhang_deg"] = float(worst_overhang)
        if worst_support > 0.0:
            metrics["worst_support_fraction"] = float(worst_support)

        passed = not failures
        # Dense score uses the wall margin (most common failure mode);
        # falls back to a binary score if wall not measured.
        if worst_wall != float("inf") and min_wall > 0:
            wall_score = max(0.0, min(worst_wall / min_wall, 1.0))
        else:
            wall_score = 1.0 if passed else 0.0
        if max_overhang > 0 and worst_overhang > 0:
            ovh_score = max(0.0,
                            min(1.0 - (worst_overhang - max_overhang)
                                / max_overhang, 1.0))
        else:
            ovh_score = 1.0
        score = min(wall_score, ovh_score)
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(score),
            metrics=metrics,
            failures=failures,
        )
