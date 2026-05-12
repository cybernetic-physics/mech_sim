"""Fast / oracle / final evaluation modes.

A *mode* is a named projection of a single ``EvalConfig``:

* ``fast`` runs cheap probes (kinematics, analytic) and the cheapest
  adapter the dispatcher finds.
* ``oracle`` runs the contact-and-dynamics probes (and is intended to
  use a high-fidelity adapter such as ``chrono_contact``; tests/demos
  use ``fake_contact_oracle`` instead).
* ``final`` runs both, computes agreement metrics, and gates on:
  fast hard gate, oracle hard gate, and agreement thresholds.

The ``modes`` table on EvalConfig (see schema.ModeConfig) declares the
probe-id subset for each mode and optional adapter overrides.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.metrics import (
    compute_class_metrics,
    compute_general_metrics,
    compute_tier_metrics,
    fill_defaults_for_dashboard,
)
from mech_bench.schema import EvalConfig, EvalReport, ModeConfig, TaskSpec


KNOWN_MODES = ("fast", "oracle", "final")


@dataclass
class ModeResult:
    """One evaluated mode result (``fast`` or ``oracle``)."""

    mode: str
    report: EvalReport
    evidence: Any
    runtime_s: float = 0.0


@dataclass
class FinalResult:
    """Composite result for ``--mode final``."""

    fast: ModeResult | None
    oracle: ModeResult | None
    final_score: float
    fast_score: float
    oracle_score: float
    agreement_score: float
    hard_gate_passed: bool
    evaluation_valid: bool
    agreement: dict[str, float] = field(default_factory=dict)
    feedback: list[Failure] = field(default_factory=list)
    oracle_is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "final",
            "fast_score": float(self.fast_score),
            "oracle_score": float(self.oracle_score),
            "agreement_score": float(self.agreement_score),
            "final_score": float(self.final_score),
            "hard_gate_passed": bool(self.hard_gate_passed),
            "evaluation_valid": bool(self.evaluation_valid),
            "agreement": dict(self.agreement),
            "oracle_is_synthetic": bool(self.oracle_is_synthetic),
            "fast": (self.fast.report.to_dict()
                     if self.fast else None),
            "oracle": (self.oracle.report.to_dict()
                       if self.oracle else None),
            "feedback": [
                (f.to_dict() if hasattr(f, "to_dict") else dict(f))
                for f in self.feedback
            ],
        }


# --------------------------------------------------------------------- #
# Mode application                                                      #
# --------------------------------------------------------------------- #


def apply_mode(cfg: EvalConfig, mode: str) -> EvalConfig:
    """Return a copy of *cfg* projected through the named mode.

    For unknown modes (``"public"``, ``""``, …), the original config
    is returned unchanged. Mode application:

    * filters ``cfg.probes`` to the mode's ``enabled_probe_ids`` (when
      non-empty);
    * filters ``cfg.hard_gate_probes`` to the surviving set;
    * merges per-mode ``adapter_overrides`` on top of
      ``cfg.adapter_configs``.
    """
    if mode not in cfg.modes:
        return cfg
    mcfg: ModeConfig = cfg.modes[mode]
    enabled = set(mcfg.enabled_probe_ids)

    if enabled:
        probes = [p for p in cfg.probes if p.id in enabled]
    else:
        probes = list(cfg.probes)
    keep_ids = {p.id for p in probes}
    hard_gate = [pid for pid in cfg.hard_gate_probes if pid in keep_ids]

    adapter_configs = {
        k: dict(v) for k, v in cfg.adapter_configs.items()
    }
    for name, over in mcfg.adapter_overrides.items():
        existing = adapter_configs.setdefault(name, {})
        existing.update(over)

    out = copy.copy(cfg)
    out.probes = probes
    out.hard_gate_probes = hard_gate
    out.adapter_configs = adapter_configs
    return out


# --------------------------------------------------------------------- #
# Final-mode evaluation                                                 #
# --------------------------------------------------------------------- #


def run_mode(
    task_dir: Path,
    submission_dir: Path,
    mode: str,
    *,
    scratch_dir: Path | None = None,
    run_id: str | None = None,
) -> ModeResult:
    """Run one named mode (``fast`` or ``oracle``)."""
    import time

    from mech_bench.evaluator import evaluate_with_evidence, load_task

    if mode not in ("fast", "oracle"):
        raise ValueError(
            f"run_mode supports 'fast' or 'oracle', got {mode!r}")

    _task, cfg = load_task(task_dir)

    if mode in cfg.modes:
        # Hand the evaluator a temp dir whose eval_config.toml is the
        # mode-projected version. The simplest way to avoid copying
        # logic is to do the projection at the EvalConfig level — we
        # already do that in ``evaluate_with_evidence`` via the
        # ``mode`` keyword argument plumbed through.
        pass
    t0 = time.perf_counter()
    evidence = evaluate_with_evidence(
        task_dir, submission_dir,
        scratch_dir=scratch_dir,
        run_id=run_id,
        mode=mode,
    )
    elapsed = time.perf_counter() - t0
    return ModeResult(
        mode=mode,
        report=evidence.report,
        evidence=evidence,
        runtime_s=elapsed,
    )


def run_final(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
) -> FinalResult:
    """Run fast + oracle, compute agreement, gate on both."""
    from mech_bench.evaluator import load_task

    _task, cfg = load_task(task_dir)
    require = list(cfg.final_mode.require_modes) or ["fast", "oracle"]

    fast_r: ModeResult | None = None
    oracle_r: ModeResult | None = None
    feedback: list[Failure] = []
    valid = True

    for m in require:
        scratch = (Path(scratch_dir) / m) if scratch_dir else None
        try:
            res = run_mode(task_dir, submission_dir, m, scratch_dir=scratch)
        except Exception as e:  # noqa: BLE001 — runner is firewall
            feedback.append(Failure(
                code=FailureCode.SIMULATOR_DIVERGENCE,
                severity=Severity.CRITICAL,
                message=f"final mode: {m!r} run raised: "
                        f"{type(e).__name__}: {e}",
                where=f"final.{m}",
            ))
            valid = False
            continue
        if m == "fast":
            fast_r = res
        elif m == "oracle":
            oracle_r = res
        if not res.report.evaluation_valid:
            valid = False

    fast_score = float(fast_r.report.score) if fast_r else 0.0
    oracle_score = float(oracle_r.report.score) if oracle_r else 0.0

    # Agreement: ratio / penetration / lockup / contact-presence.
    agreement = compute_agreement_metrics(fast_r, oracle_r)
    ratio_thr = float(cfg.final_mode.ratio_delta_pct_max)
    pen_thr = float(cfg.final_mode.penetration_delta_mm_max)

    agreement_components: list[float] = []
    if "ratio_delta_pct" in agreement:
        delta = agreement["ratio_delta_pct"]
        agreement_components.append(
            max(0.0, 1.0 - delta / ratio_thr) if ratio_thr > 0 else 0.0
        )
    if "penetration_delta_mm" in agreement:
        delta = agreement["penetration_delta_mm"]
        agreement_components.append(
            max(0.0, 1.0 - delta / pen_thr) if pen_thr > 0 else 0.0
        )
    if "lockup_agreement" in agreement:
        agreement_components.append(float(agreement["lockup_agreement"]))
    if "contact_presence_agreement" in agreement:
        agreement_components.append(
            float(agreement["contact_presence_agreement"])
        )
    agreement_score = (sum(agreement_components)
                       / len(agreement_components)) if (
        agreement_components) else 1.0

    # Hard gate: both fast and oracle must pass, and agreement must be
    # within thresholds (agreement_score ≥ 0.5 is the minimum).
    hard_gate_passed = (
        bool(fast_r and fast_r.report.hard_gate_passed)
        and bool(oracle_r and oracle_r.report.hard_gate_passed)
        and agreement_score >= 0.5
    )
    if not hard_gate_passed:
        # Surface agreement failures so the agent has structured info.
        if fast_r and oracle_r and (
                fast_r.report.hard_gate_passed
                and oracle_r.report.hard_gate_passed
                and agreement_score < 0.5):
            feedback.append(Failure(
                code=FailureCode.SIMULATOR_DIVERGENCE,
                severity=Severity.MAJOR,
                message=(f"final mode: fast and oracle disagreed; "
                         f"agreement={agreement_score:.3f}. "
                         f"Details: {dict(agreement)}"),
                where="final.agreement",
            ))

    # Composite final score: weighted (oracle counts more).
    final_score = (0.4 * fast_score
                   + 0.4 * oracle_score
                   + 0.2 * agreement_score)
    if not hard_gate_passed:
        final_score = 0.0

    oracle_is_synthetic = bool(
        oracle_r and oracle_r.report.oracle_is_synthetic
    )

    return FinalResult(
        fast=fast_r,
        oracle=oracle_r,
        fast_score=fast_score,
        oracle_score=oracle_score,
        agreement_score=agreement_score,
        final_score=final_score,
        hard_gate_passed=hard_gate_passed,
        evaluation_valid=valid,
        agreement=agreement,
        feedback=feedback,
        oracle_is_synthetic=oracle_is_synthetic,
    )


# --------------------------------------------------------------------- #
# Agreement metrics                                                     #
# --------------------------------------------------------------------- #


def compute_agreement_metrics(
    fast: ModeResult | None,
    oracle: ModeResult | None,
) -> dict[str, float]:
    """Compute cross-mode agreement metrics from two ModeResults.

    Each metric reflects a comparable quantity present in both reports'
    scalar metric dicts. Missing values produce no entry.
    """
    if fast is None or oracle is None:
        return {}
    fast_m = _flatten_scalar_metrics(fast)
    or_m = _flatten_scalar_metrics(oracle)

    out: dict[str, float] = {}
    # Ratio comparison.
    r_fast = _first(fast_m, ("ratio_observed", "ratio_estimate"))
    r_or = _first(or_m, ("ratio_observed", "ratio_estimate"))
    if r_fast is not None and r_or is not None and abs(r_or) > 1e-9:
        out["ratio_delta_pct"] = float(
            abs(r_fast - r_or) / abs(r_or) * 100.0)

    # Penetration comparison.
    p_fast = _first(fast_m, ("max_penetration_mm",))
    p_or = _first(or_m, ("max_penetration_mm",))
    if p_fast is not None and p_or is not None:
        out["penetration_delta_mm"] = float(abs(p_fast - p_or))

    # Lockup agreement.
    l_fast = _first(fast_m, ("lockup_detected",))
    l_or = _first(or_m, ("lockup_detected",))
    if l_fast is not None and l_or is not None:
        out["lockup_agreement"] = 1.0 if (
            bool(l_fast > 0.5) == bool(l_or > 0.5)) else 0.0

    # Contact-presence agreement.
    c_fast = _first(fast_m, ("n_contacts_max",))
    c_or = _first(or_m, ("n_contacts_max",))
    if c_fast is not None and c_or is not None:
        agree = 1.0 if (bool(c_fast > 0) == bool(c_or > 0)) else 0.0
        out["contact_presence_agreement"] = agree

    # Torque ripple / power balance comparisons (when both supplied).
    for key, out_key in (
        ("torque_ripple_pct", "torque_ripple_delta_pct"),
        ("power_balance_error_pct", "power_balance_delta_pct"),
    ):
        a = _first(fast_m, (key,))
        b = _first(or_m, (key,))
        if a is not None and b is not None:
            out[out_key] = float(abs(a - b))

    return out


def _flatten_scalar_metrics(mr: ModeResult) -> dict[str, float]:
    """Collect numeric scalars across the report + its adapter outputs."""
    out: dict[str, float] = {}
    for k, v in (mr.report.metrics or {}).items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    # Also pull adapter scalar_metrics if present.
    sim_outputs = getattr(mr.evidence, "sim_outputs_by_adapter", {}) or {}
    for adapter_name, sim in sim_outputs.items():
        if not isinstance(sim, dict):
            continue
        for k, v in (sim.get("scalar_metrics") or {}).items():
            if isinstance(v, (int, float)):
                # Prefer report-side keys when both exist.
                out.setdefault(str(k), float(v))
    return out


def _first(d: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in d:
            return d[k]
    return None


__all__ = [
    "KNOWN_MODES",
    "ModeResult",
    "FinalResult",
    "apply_mode",
    "run_mode",
    "run_final",
    "compute_agreement_metrics",
]
