"""Contact-engagement probe.

Verifies that named contact pairs in a mechanism actually carry
load. The motivating failure mode: an agent declares a cycloidal
disc/ring-pin contact in the IR (so the topology probe is satisfied)
but the geometry is such that no contact ever forms in simulation.
The mobility probe would still pass — only a load-carrying check can
distinguish the two.

Config:
  required_pairs: list of "partA:partB" strings.
  min_rms_force_N: RMS normal force threshold per pair.
  min_engagement_fraction: fraction of timesteps with non-trivial
                           force, in [0, 1].

Consumes:
  sim_outputs["contact_forces"][pair_key]  (N,) or (N, k) — normal
                                            force magnitudes.
  sim_outputs["time_s"] (N,)               — optional, only used for
                                            metadata.

Reports:
  contact.<pair>.rms_N
  contact.<pair>.engagement_fraction
  contact.<pair>.peak_N
  worst_pair                 (lexicographic id of weakest pair)

Failure: MISSING_CONTACT.
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


def _coerce(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _rms_and_engagement(
    arr: np.ndarray, threshold: float,
) -> tuple[float, float, float]:
    """Return (rms_magnitude, engagement_fraction, peak)."""
    # Flatten last dim so (N,k) → per-timestep magnitudes.
    if arr.ndim == 1:
        mag = np.abs(arr)
    else:
        mag = np.linalg.norm(arr, axis=tuple(range(1, arr.ndim)))
    rms = float(np.sqrt(np.mean(mag * mag)))
    peak = float(np.max(mag))
    engaged = float(np.mean(mag >= threshold))
    return rms, engaged, peak


@register_probe
class ContactEngagement(Probe):
    type_name = "contact_engagement"
    capabilities_required = frozenset({Capability.CONTACT_FORCES})

    def run(
        self,
        ir: DesignIR,
        sim_outputs: dict[str, Any],
        config: dict[str, Any],
    ) -> ProbeResult:
        required: Iterable[str] = config.get("required_pairs", []) or []
        min_rms = float(config.get("min_rms_force_N", 0.5))
        min_engagement = float(config.get("min_engagement_fraction", 0.05))
        # Force samples below this are treated as numerical noise for
        # engagement counting. Default chosen to be much smaller than
        # min_rms but non-zero.
        noise_floor = float(
            config.get("engagement_noise_floor_N", max(1e-6, min_rms * 0.05))
        )

        contact_forces = sim_outputs.get("contact_forces")
        if not isinstance(contact_forces, dict):
            unavailable = sim_outputs.get(
                "__capability_unavailable__", False
            )
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
                        "contact_engagement: sim_outputs.contact_forces "
                        "is absent or not a dict."
                    ),
                    public_hint=(
                        "This probe needs a contact-force-capable "
                        "simulator. Configure an adapter that "
                        "advertises CONTACT_FORCES."
                    ),
                )],
            )

        # Lookup map: normalized pair_key → original key (so reports
        # round-trip the original spelling).
        norm_map: dict[str, str] = {}
        for raw in contact_forces:
            norm_map[_normalize_pair(str(raw))] = str(raw)

        metrics: dict[str, float] = {
            "n_required_pairs": float(len(list(required))),
        }
        failures: list[Failure] = []
        worst_pair = ""
        worst_score = 1.0
        passed = True
        for raw_required in required:
            pair_key = _normalize_pair(str(raw_required))
            original = norm_map.get(pair_key)
            if original is None:
                failures.append(Failure(
                    code=FailureCode.MISSING_CONTACT,
                    severity=Severity.CRITICAL,
                    message=(f"Required contact pair {raw_required!r} "
                             f"has no entries in sim_outputs."),
                    metric="rms_N",
                    observed=0.0,
                    target=min_rms,
                    where=f"contact.{pair_key}",
                ))
                metrics[f"contact.{pair_key}.rms_N"] = 0.0
                metrics[f"contact.{pair_key}.engagement_fraction"] = 0.0
                metrics[f"contact.{pair_key}.peak_N"] = 0.0
                worst_pair = pair_key
                worst_score = 0.0
                passed = False
                continue
            arr = _coerce(contact_forces[original])
            if arr is None:
                failures.append(Failure(
                    code=FailureCode.MISSING_CONTACT,
                    severity=Severity.CRITICAL,
                    message=(f"Contact pair {raw_required!r} has empty "
                             f"force trace."),
                    where=f"contact.{pair_key}",
                ))
                metrics[f"contact.{pair_key}.rms_N"] = 0.0
                metrics[f"contact.{pair_key}.engagement_fraction"] = 0.0
                metrics[f"contact.{pair_key}.peak_N"] = 0.0
                worst_pair = pair_key
                worst_score = 0.0
                passed = False
                continue
            rms, engaged, peak = _rms_and_engagement(arr, noise_floor)
            metrics[f"contact.{pair_key}.rms_N"] = float(rms)
            metrics[f"contact.{pair_key}.engagement_fraction"] = float(engaged)
            metrics[f"contact.{pair_key}.peak_N"] = float(peak)
            pair_score = 1.0
            if rms < min_rms or engaged < min_engagement:
                passed = False
                failures.append(Failure(
                    code=FailureCode.MISSING_CONTACT,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Pair {raw_required!r} is under-engaged: "
                        f"RMS {rms:.3f} N (min {min_rms:.3f}), "
                        f"engaged {engaged*100:.1f}% (min "
                        f"{min_engagement*100:.1f}%)."
                    ),
                    metric="rms_N",
                    observed=float(rms),
                    target=float(min_rms),
                    where=f"contact.{pair_key}",
                    public_hint=(
                        "Adjust contact geometry (clearance, eccentricity) "
                        "so this pair actually carries load over the "
                        "cycle."
                    ),
                ))
                # Dense score: how close are we to both thresholds.
                rms_part = (rms / min_rms) if min_rms > 0 else 1.0
                eng_part = (engaged / min_engagement
                            if min_engagement > 0 else 1.0)
                pair_score = max(0.0, min(rms_part, eng_part, 1.0))
            if pair_score < worst_score:
                worst_score = pair_score
                worst_pair = pair_key

        metrics["worst_pair_score"] = float(worst_score)
        return ProbeResult(
            probe_id="",
            probe_type=self.type_name,
            passed=passed,
            score=float(worst_score if passed else min(worst_score, 0.5)),
            metrics=metrics,
            failures=failures,
        )
