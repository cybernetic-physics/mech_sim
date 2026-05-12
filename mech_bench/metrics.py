"""Tier and task-class metric aggregation.

The runtime computes three concentric scoring views on every report:

1. **General metrics** — high-level scalars an RLVR loop watches
   (``verified_score``, ``hard_gate_pass_rate``, ``runtime_s``).
2. **Tier metrics** — per-capability buckets (``artifact``, ``geometry``,
   ``kinematics``, ``collision``, ``contact``, ``dynamics``,
   ``structural``, ``manufacturability``, ``robustness``).
3. **Task-class metrics** — semantic channels tied to mechanism
   families (``linkage_path_score``, ``gearbox_ratio_score``,
   ``contact_health_score``, ``load_trial_score``,
   ``printability_score``, ``safety_factor_score``).

Tier and class assignment is driven by per-probe configuration: probes
set ``tier`` and ``class_metric`` on their ``ProbeSpec``. When a probe
doesn't declare either, the runtime falls back to a default derivation
from the probe's capabilities + type.
"""

from __future__ import annotations

from typing import Iterable

from mech_bench.probes import Capability
from mech_bench.schema import ProbeResult, ProbeSpec


# Canonical channel names — keep stable; dashboards rely on the keys.
TIER_CHANNELS: tuple[str, ...] = (
    "artifact",
    "geometry",
    "kinematics",
    "collision",
    "contact",
    "dynamics",
    "structural",
    "manufacturability",
    "robustness",
)

CLASS_CHANNELS: tuple[str, ...] = (
    "linkage_path_score",
    "gearbox_ratio_score",
    "contact_health_score",
    "load_trial_score",
    "printability_score",
    "safety_factor_score",
)


# Default tier per probe type. Probes can override via ProbeSpec.tier.
_DEFAULT_TIER_BY_TYPE: dict[str, str] = {
    "dof_grubler": "kinematics",
    "required_ports": "artifact",
    "path_trace_chamfer": "kinematics",
    "port_velocity_ratio": "kinematics",
    "swept_collision": "collision",
    "contact_engagement": "contact",
    "lockup": "dynamics",
    "torque_load_trial": "dynamics",
    "printability_dfam": "manufacturability",
    "safety_factor": "structural",
    "analytic_param_check": "artifact",
}


# Default class metric per probe type.
_DEFAULT_CLASS_BY_TYPE: dict[str, str] = {
    "path_trace_chamfer": "linkage_path_score",
    "port_velocity_ratio": "gearbox_ratio_score",
    "contact_engagement": "contact_health_score",
    "torque_load_trial": "load_trial_score",
    "printability_dfam": "printability_score",
    "safety_factor": "safety_factor_score",
}


def derive_tier(spec: ProbeSpec, caps: Iterable[Capability]) -> str:
    """Pick the channel for *spec*.

    Order: explicit ``spec.tier`` → default by ``spec.type`` → derived
    from capabilities → ``"artifact"`` as a conservative fallback.
    """
    if spec.tier:
        return str(spec.tier)
    if spec.type in _DEFAULT_TIER_BY_TYPE:
        return _DEFAULT_TIER_BY_TYPE[spec.type]
    pruned = {c for c in caps if c != Capability.NONE}
    if not pruned:
        return "artifact"
    if Capability.CONTACT_FORCES in pruned:
        return "contact"
    if Capability.MESH_OVERLAP in pruned:
        return "collision"
    if {Capability.RIGID_BODY_DYNAMICS, Capability.MOTOR_DRIVES,
            Capability.LOAD_TORQUES} & pruned:
        return "dynamics"
    if {Capability.PLANAR_KINEMATICS, Capability.SPATIAL_KINEMATICS,
            Capability.PATH_TRACE} & pruned:
        return "kinematics"
    if Capability.SAFETY_FACTOR in pruned or Capability.FEA_STATIC in pruned:
        return "structural"
    if Capability.MESH in pruned:
        return "geometry"
    return "artifact"


def derive_class_metric(spec: ProbeSpec) -> str | None:
    """Pick the task-class channel for *spec*, or ``None`` to skip."""
    if spec.class_metric:
        return str(spec.class_metric)
    return _DEFAULT_CLASS_BY_TYPE.get(spec.type)


def compute_tier_metrics(
    probe_results: list[ProbeResult],
    specs_by_id: dict[str, ProbeSpec],
    caps_by_id: dict[str, Iterable[Capability]],
) -> dict[str, dict[str, float]]:
    """Aggregate scores per tier.

    Each bucket is ``{"score": weighted_mean, "passed": all_passed,
    "n": count}``. Buckets for tiers with no probes are omitted, but
    the caller can fill in defaults.
    """
    buckets: dict[str, dict[str, float]] = {}
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        if spec is None:
            continue
        tier = derive_tier(spec, caps_by_id.get(r.probe_id, []))
        bucket = buckets.setdefault(tier, {
            "score_sum": 0.0,
            "weight_sum": 0.0,
            "n": 0.0,
            "n_passed": 0.0,
        })
        # Use weight when set; else uniform 1.0 so the bucket isn't empty.
        w = float(spec.weight) if spec.weight > 0 else 1.0
        bucket["score_sum"] += w * float(r.score)
        bucket["weight_sum"] += w
        bucket["n"] += 1
        if r.passed:
            bucket["n_passed"] += 1
    out: dict[str, dict[str, float]] = {}
    for tier, b in buckets.items():
        n = b["n"]
        out[tier] = {
            "score": (b["score_sum"] / b["weight_sum"]
                       if b["weight_sum"] > 0 else 0.0),
            "passed": bool(b["n_passed"] >= n),
            "n": int(n),
            "n_passed": int(b["n_passed"]),
        }
    return out


def compute_class_metrics(
    probe_results: list[ProbeResult],
    specs_by_id: dict[str, ProbeSpec],
) -> dict[str, float]:
    """Aggregate scores per task-class channel.

    Returns one float per channel. Channels are 0.0 by default when no
    probe contributes — the dashboard can render the full set.
    """
    out: dict[str, list[float]] = {c: [] for c in CLASS_CHANNELS}
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        if spec is None:
            continue
        ch = derive_class_metric(spec)
        if ch is None or ch not in out:
            continue
        out[ch].append(float(r.score))
    return {
        ch: (sum(vals) / len(vals)) if vals else 0.0
        for ch, vals in out.items()
    }


def compute_general_metrics(
    probe_results: list[ProbeResult],
    *,
    hard_gate_passed: bool,
    score: float,
    runtime_s: float,
    oracle_passed: bool | None = None,
) -> dict[str, float]:
    """Top-line scalars for an RLVR loop."""
    n = len(probe_results)
    pass_at_1 = (1.0 if hard_gate_passed and score > 0 else 0.0)
    return {
        "verified_score": float(score) if hard_gate_passed else 0.0,
        "dense_score": float(score),
        "pass_at_1": pass_at_1,
        "hard_gate_pass_rate": 1.0 if hard_gate_passed else 0.0,
        "oracle_pass_rate": (
            1.0 if oracle_passed else 0.0
        ) if oracle_passed is not None else 0.0,
        "n_probes": float(n),
        "runtime_s": float(runtime_s),
    }


def fill_defaults_for_dashboard(
    tier_metrics: dict[str, dict[str, float]],
    class_metrics: dict[str, float],
) -> tuple[dict[str, dict], dict[str, float]]:
    """Ensure every canonical channel is present.

    Dashboards prefer a stable shape. A tier with no probes is marked
    ``applicable=false`` with ``passed=None`` / ``score=None`` so the
    dashboard renders it as N/A rather than "failed."
    """
    tier_out: dict[str, dict] = {}
    for k, v in tier_metrics.items():
        bucket: dict = dict(v)
        n = int(bucket.get("n", 0) or 0)
        if n == 0:
            bucket["applicable"] = False
            bucket["passed"] = None
            bucket["score"] = None
            bucket.setdefault("n_passed", 0)
        else:
            bucket.setdefault("applicable", True)
        tier_out[k] = bucket
    for ch in TIER_CHANNELS:
        tier_out.setdefault(ch, {
            "applicable": False,
            "score": None,
            "passed": None,
            "n": 0,
            "n_passed": 0,
        })
    class_out = dict(class_metrics)
    for ch in CLASS_CHANNELS:
        class_out.setdefault(ch, 0.0)
    return tier_out, class_out
