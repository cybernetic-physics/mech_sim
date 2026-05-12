"""Safety-factor probe.

Consumes ``sim_outputs["safety_factors"]`` — either a flat dict of
``check_name -> fos`` or a nested ``part_id -> {check_name: fos}``.
The probe reports the minimum factor of safety across requested
checks and fails when it dips below ``min_fos``.

The probe does not run any structural analysis itself; it is a data
consumer for FEA or analytic adapters that have already computed
factors of safety.

Failure: ``INSUFFICIENT_SAFETY_FACTOR``.
"""

from __future__ import annotations

from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, register_probe
from mech_bench.schema import DesignIR, ProbeResult


def _flatten(safety: Any) -> list[tuple[str, float]]:
    """Return a list of (check_label, fos) tuples.

    ``safety`` may be:
      - a flat ``{check_name: float}`` dict, or
      - a nested ``{part_id: {check_name: float}}`` dict.
    """
    out: list[tuple[str, float]] = []
    if not isinstance(safety, dict):
        return out
    for k, v in safety.items():
        if isinstance(v, dict):
            for ck, cv in v.items():
                try:
                    out.append((f"{k}.{ck}", float(cv)))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out.append((str(k), float(v)))
            except (TypeError, ValueError):
                continue
    return out


@register_probe
class SafetyFactor(Probe):
    type_name = "safety_factor"
    capabilities_required = frozenset({
        Capability.SAFETY_FACTOR,
        Capability.FEA_STATIC,
    })

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        min_fos = float(config.get("min_fos", 1.5))
        wanted = config.get("checks") or []
        wanted_set = set(wanted)

        safety = sim_outputs.get("safety_factors")
        if safety is None:
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
                        "safety_factor: no safety_factors in "
                        "sim_outputs. A FEA or analytic adapter must "
                        "compute these before this probe can run."
                    ),
                )],
            )

        entries = _flatten(safety)
        if wanted_set:
            entries = [(k, v) for k, v in entries
                       if k in wanted_set
                       or any(k.endswith("." + w) for w in wanted_set)]
        if not entries:
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
                        "safety_factor: no matching checks in "
                        "safety_factors. Requested "
                        f"{sorted(wanted_set)!r}."
                    ),
                )],
            )

        metrics: dict[str, float] = {}
        min_observed = float("inf")
        worst_label = ""
        for label, fos in entries:
            metrics[f"fos.{label}"] = float(fos)
            if fos < min_observed:
                min_observed = float(fos)
                worst_label = label
        metrics["min_fos"] = float(min_observed)
        passed = min_observed >= min_fos
        failures: list[Failure] = []
        if not passed:
            failures.append(Failure(
                code=FailureCode.INSUFFICIENT_SAFETY_FACTOR,
                severity=Severity.MAJOR,
                message=(
                    f"Minimum safety factor {min_observed:.3f} "
                    f"(check {worst_label!r}) below required "
                    f"{min_fos:.3f}."
                ),
                metric="min_fos",
                observed=float(min_observed),
                target=float(min_fos),
                where=worst_label,
            ))
        if min_fos > 0.0:
            score = max(0.0, min(min_observed / min_fos, 1.0))
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
