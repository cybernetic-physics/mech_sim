"""Compact reward API for an RL/verifier loop.

A generation agent does not need the full ``EvalReport``; it needs a
small, JSON-serializable payload describing reward, hard-gate pass,
public feedback, and per-channel scalar signals. ``evaluate_for_rlvr``
is the one-stop call.

Design notes
------------

* Reward is **0.0** whenever ``evaluation_valid=False`` so the agent
  cannot earn credit for runs the verifier itself couldn't trust.
* When the oracle is synthetic (``fake_contact_oracle``), the result
  flags ``oracle_is_synthetic=True`` so the caller can decide whether
  to use the signal for training or only for development.
* ``retry_suggestions`` are derived from feedback codes — a thin
  mapping from machine-readable codes to short, actionable hints. The
  caller is free to ignore them.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mech_bench.feedback import FailureCode
from mech_bench.metrics import CLASS_CHANNELS, TIER_CHANNELS


@dataclass
class RLVRResult:
    """Compact RLVR-loop-friendly evaluation result.

    Keys are stable; new keys may be added but existing keys retain
    their semantics. Use :meth:`to_dict` for JSON emission.
    """

    reward: float = 0.0
    hard_gate_passed: bool = False
    evaluation_valid: bool = True
    dense_score: float = 0.0
    public_feedback: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    report_dir: Path | None = None
    retry_suggestions: list[str] = field(default_factory=list)
    scalar_channels: dict[str, float] = field(default_factory=dict)
    oracle_is_synthetic: bool = False
    # The score the run WOULD earn (valid+gate rule) before any synthetic-oracle
    # quarantine. Non-learning: for eval/dev visibility only.
    dev_reward: float = 0.0
    # True iff a synthetic oracle forced the learnable ``reward`` to 0 under the
    # train profile. Mirrors GBA-Eval's anti-hack gate: a verifier that cannot
    # prove the candidate solved the task must never mint learnable credit.
    reward_quarantined: bool = False
    # "eval" (default): report the synthetic score transparently. "train":
    # quarantine synthetic reward to 0 so the policy never learns from a fake.
    reward_profile: str = "eval"
    mode: str = "fast"
    task_id: str = ""
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["report_dir"] = str(self.report_dir) if self.report_dir else None
        return d


# --------------------------------------------------------------------- #
# Public entry point                                                    #
# --------------------------------------------------------------------- #


def evaluate_for_rlvr(
    task_dir: Path,
    submission_dir: Path,
    *,
    mode: str = "fast",
    report_dir: Path | None = None,
    reward_profile: str = "eval",
    allow_synthetic_reward: bool = False,
) -> RLVRResult:
    """Evaluate one submission and return the compact RLVR payload.

    ``mode`` is one of ``"fast"``, ``"oracle"``, ``"final"`` or empty
    (default eval config). ``report_dir`` opt-in writes the full bundle
    to disk and the path is referenced in the result.

    ``reward_profile`` selects the eval-vs-RL reward contract:

    * ``"eval"`` (default) — report the score transparently even when the
      oracle is synthetic. Use for benchmarking/leaderboards.
    * ``"train"`` — quarantine synthetic-oracle reward to 0 (anti-hack), so a
      policy never learns from a fabricated verifier. The would-be score is
      still reported as ``dev_reward``.

    ``allow_synthetic_reward=True`` overrides the quarantine even under the
    train profile (use only for deliberately training against the fake oracle).
    """
    if reward_profile not in ("eval", "train"):
        raise ValueError(
            f"reward_profile must be 'eval' or 'train', got {reward_profile!r}")
    from mech_bench.evaluator import (
        evaluate_with_evidence,
        write_run_bundle,
    )

    if mode == "final":
        return _rlvr_final(
            task_dir, submission_dir, report_dir=report_dir,
            reward_profile=reward_profile,
            allow_synthetic_reward=allow_synthetic_reward,
        )

    evidence = evaluate_with_evidence(
        Path(task_dir), Path(submission_dir), mode=mode)
    report = evidence.report

    rd_out: Path | None = None
    if report_dir is not None:
        rd_out = Path(report_dir)
        rd_out.mkdir(parents=True, exist_ok=True)
        write_run_bundle(evidence, rd_out)

    return _build_rlvr_result(
        report=report,
        evidence_cfg_visibility=evidence.cfg.visibility,
        mode=mode or "fast",
        report_dir=rd_out,
        reward_profile=reward_profile,
        allow_synthetic_reward=allow_synthetic_reward,
    )


def _quarantine(
    base_reward: float,
    synthetic: bool,
    reward_profile: str,
    allow_synthetic_reward: bool,
) -> tuple[float, bool]:
    """Apply the synthetic-oracle anti-hack gate.

    Returns ``(reward, quarantined)``. Under the ``train`` profile a synthetic
    oracle forces reward to 0 unless explicitly allowed; under ``eval`` the
    score is reported transparently.
    """
    quarantined = (
        synthetic
        and reward_profile == "train"
        and not allow_synthetic_reward
    )
    return (0.0 if quarantined else base_reward), quarantined


def _build_rlvr_result(
    *,
    report,
    evidence_cfg_visibility,
    mode: str,
    report_dir: Path | None,
    reward_profile: str = "eval",
    allow_synthetic_reward: bool = False,
) -> RLVRResult:
    valid = bool(report.evaluation_valid)
    hard_gate = bool(report.hard_gate_passed)
    dense = float(report.score)
    synthetic = bool(report.oracle_is_synthetic)

    # Score the run WOULD earn: zero when invalid or hard-gate failed.
    base_reward = dense if (valid and hard_gate) else 0.0
    # Anti-hack gate: a synthetic oracle must never mint learnable reward.
    reward, quarantined = _quarantine(
        base_reward, synthetic, reward_profile, allow_synthetic_reward)

    public_feedback = []
    codes_seen: list[str] = []
    for f in report.feedback:
        if hasattr(f, "public"):
            item = f.public()
        else:
            item = dict(f)
        code = item.get("code", "")
        if code:
            codes_seen.append(str(code))
        public_feedback.append(item)

    metrics_public = _public_metric_view(
        report.metrics, evidence_cfg_visibility,
    )

    scalar_channels: dict[str, float] = {}
    # Tier scores → scalar channels.
    for ch in TIER_CHANNELS:
        v = report.tier_results.get(ch, {})
        scalar_channels[ch] = float(v.get("score", 0.0) or 0.0)
    # Class metrics → scalar channels (alongside tiers).
    for ch in CLASS_CHANNELS:
        scalar_channels[ch] = float(report.class_metrics.get(ch, 0.0))

    return RLVRResult(
        reward=reward,
        hard_gate_passed=hard_gate,
        evaluation_valid=valid,
        dense_score=dense,
        public_feedback=public_feedback,
        metrics=metrics_public,
        report_dir=report_dir,
        retry_suggestions=_suggestions_for(codes_seen),
        scalar_channels=scalar_channels,
        oracle_is_synthetic=synthetic,
        dev_reward=base_reward,
        reward_quarantined=quarantined,
        reward_profile=reward_profile,
        mode=mode,
        task_id=str(report.task_id),
        run_id=str(report.run_id),
    )


def _rlvr_final(
    task_dir: Path,
    submission_dir: Path,
    *,
    report_dir: Path | None,
    reward_profile: str = "eval",
    allow_synthetic_reward: bool = False,
) -> RLVRResult:
    from mech_bench.evaluator import write_run_bundle
    from mech_bench.modes import run_final

    final = run_final(Path(task_dir), Path(submission_dir),
                      scratch_dir=report_dir)
    rd_out: Path | None = None
    if report_dir is not None:
        rd_out = Path(report_dir)
        rd_out.mkdir(parents=True, exist_ok=True)
        if final.fast:
            write_run_bundle(final.fast.evidence, rd_out / "fast")
        if final.oracle:
            write_run_bundle(final.oracle.evidence, rd_out / "oracle")
        (rd_out / "final.json").write_text(
            json.dumps(final.to_dict(), indent=2, default=str,
                       allow_nan=False)
        )

    # Aggregate public feedback from both modes.
    public_feedback: list[dict[str, Any]] = []
    codes_seen: list[str] = []
    for src in (final.fast, final.oracle):
        if src is None:
            continue
        for f in src.report.feedback:
            item = f.public() if hasattr(f, "public") else dict(f)
            code = item.get("code", "")
            if code:
                codes_seen.append(str(code))
            public_feedback.append(item)
    for f in final.feedback:
        item = f.public() if hasattr(f, "public") else dict(f)
        code = item.get("code", "")
        if code:
            codes_seen.append(str(code))
        public_feedback.append(item)

    # Channels: merge tier/class metrics from both, prefer oracle's.
    scalar_channels: dict[str, float] = {}
    for ch in TIER_CHANNELS:
        scalar_channels[ch] = 0.0
    for ch in CLASS_CHANNELS:
        scalar_channels[ch] = 0.0
    for src in (final.fast, final.oracle):
        if src is None:
            continue
        for ch in TIER_CHANNELS:
            v = src.report.tier_results.get(ch, {})
            if v:
                scalar_channels[ch] = max(
                    scalar_channels[ch], float(v.get("score", 0.0) or 0.0))
        for ch in CLASS_CHANNELS:
            scalar_channels[ch] = max(
                scalar_channels[ch],
                float(src.report.class_metrics.get(ch, 0.0)),
            )
    scalar_channels["agreement_score"] = float(final.agreement_score)

    synthetic = bool(final.oracle_is_synthetic)
    base_reward = float(final.final_score) if (
        final.evaluation_valid and final.hard_gate_passed) else 0.0
    reward, quarantined = _quarantine(
        base_reward, synthetic, reward_profile, allow_synthetic_reward)

    return RLVRResult(
        reward=reward,
        hard_gate_passed=bool(final.hard_gate_passed),
        evaluation_valid=bool(final.evaluation_valid),
        dense_score=float(final.final_score),
        public_feedback=public_feedback,
        metrics={
            "fast_score": float(final.fast_score),
            "oracle_score": float(final.oracle_score),
            "agreement_score": float(final.agreement_score),
            **{k: float(v) for k, v in final.agreement.items()
               if isinstance(v, (int, float)) and math.isfinite(float(v))},
        },
        report_dir=rd_out,
        retry_suggestions=_suggestions_for(codes_seen),
        scalar_channels=scalar_channels,
        oracle_is_synthetic=synthetic,
        dev_reward=base_reward,
        reward_quarantined=quarantined,
        reward_profile=reward_profile,
        mode="final",
        task_id=str(final.fast.report.task_id if final.fast
                    else (final.oracle.report.task_id if final.oracle
                          else "")),
        run_id=str(final.fast.report.run_id if final.fast
                    else (final.oracle.report.run_id if final.oracle
                          else "")),
    )


def _public_metric_view(
    metrics: dict[str, float], vis,
) -> dict[str, float]:
    from mech_bench.schema import _filter_public_metrics
    return {
        k: (float(v) if isinstance(v, (int, float)) and math.isfinite(
            float(v)) else 0.0)
        for k, v in _filter_public_metrics(
            metrics, vis.public_metrics, vis.hidden_metrics
        ).items()
        if v is not None
    }


_SUGGESTION_BY_CODE: dict[str, str] = {
    FailureCode.MISSING_PORT.value: (
        "Add the missing port to the IR — every required port must be "
        "declared with the correct kind."
    ),
    FailureCode.WRONG_MOBILITY.value: (
        "Re-check joint topology; the Grübler-Kutzbach count does not "
        "match the expected mobility."
    ),
    FailureCode.WRONG_RATIO.value: (
        "Recompute the transmission ratio from teeth/diameter/etc.; the "
        "declared value disagrees with the geometry."
    ),
    FailureCode.PATH_ERROR.value: (
        "Adjust link lengths or coupler-point offset to bring the trace "
        "closer to the target."
    ),
    FailureCode.COLLISION.value: (
        "Increase clearance for the colliding pair or move it onto the "
        "allowed_pairs list if intentional."
    ),
    FailureCode.MISSING_CONTACT.value: (
        "The required contact pair never carries load; revisit the "
        "geometry that should engage."
    ),
    FailureCode.LOCKUP.value: (
        "Output port did not move under input drive; topology or "
        "contact may be locked."
    ),
    FailureCode.EXCESSIVE_PENETRATION.value: (
        "Reduce penetration — either widen clearance or stiffen contact."
    ),
    FailureCode.EXCESSIVE_TORQUE_RIPPLE.value: (
        "Smooth the input torque trace — likely insufficient engagement "
        "or geometry artifact."
    ),
    FailureCode.POWER_BALANCE_ERROR.value: (
        "Energy in/out are inconsistent; check losses or simulation dt."
    ),
    FailureCode.INSUFFICIENT_SAFETY_FACTOR.value: (
        "Increase cross-section or change material to raise FOS."
    ),
    FailureCode.UNPRINTABLE.value: (
        "Adjust wall thickness or overhang angle for the target process."
    ),
    FailureCode.CAPABILITY_UNAVAILABLE.value: (
        "This run requires a simulator not registered in this build. "
        "Either run with the matching adapter or use --mode fast."
    ),
    FailureCode.SIMULATOR_DIVERGENCE.value: (
        "Simulator did not converge; reduce dt or check the design IR."
    ),
    FailureCode.INVALID_ARTIFACT.value: (
        "build_design did not produce a valid DesignIR — fix imports / "
        "exceptions before next attempt."
    ),
    FailureCode.INVALID_MASS_PROPERTIES.value: (
        "Mass properties are out of range; check kg vs g unit confusion."
    ),
    FailureCode.SCHEMA_ERROR.value: (
        "DesignIR shape is wrong — refer to the schema docs."
    ),
    FailureCode.WRONG_TOPOLOGY.value: (
        "Joint or body topology does not match the task requirement."
    ),
    FailureCode.INSUFFICIENT_CLEARANCE.value: (
        "Widen clearance between flagged parts."
    ),
}


def _suggestions_for(codes: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in codes:
        s = _SUGGESTION_BY_CODE.get(str(c))
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out
