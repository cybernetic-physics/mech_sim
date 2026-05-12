"""Generic evaluator.

Pipeline (called once per submission):

  1. Load TaskSpec + EvalConfig from a task directory.
  2. Load the submission's `design.py` and call build_design() in a
     sandboxed working directory; parse the returned dict into a
     DesignIR.
  3. Run DesignIR validation. If the IR has critical structural
     failures, short-circuit before any probe runs and return a zero
     report whose feedback is the validation failures.
  4. Build an ExecutionPlan: per probe, pick the cheapest adapter
     whose capabilities cover the probe's requirements. Probes
     without an adapter (and that need one) are marked unavailable
     and emit `CAPABILITY_UNAVAILABLE` in their result.
  5. Run each adapter at most once; pass its outputs to the probes
     that need it. Topology-only probes (`Capability.NONE`) run with
     an empty sim_outputs dict.
  6. Compose the final score:
        hard_gate = AND over hard-gate probes (or probes marked
                    `hard_gate=true`).
        dense     = Σ w_i · s_i across non-gate probes, with weights
                    renormalized to sum to 1.
        final     = 0 if hard_gate fails else dense.
  7. Return EvalReport.

The runtime does **not** trust paths in the agent submission outside
the sandbox `out_dir`. Geometry-path policy is enforced in the
validation layer (see mech_bench/validation.py).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_bench.adapters import SimAdapter, all_adapters
from mech_bench.feedback import Failure, FailureCode, Severity
from mech_bench.probes import Capability, Probe, get_probe
from mech_bench.schema import (
    DesignIR,
    EvalConfig,
    EvalReport,
    FeedbackVisibility,
    ProbeResult,
    ProbeSpec,
    TaskSpec,
)
from mech_bench.validation import has_critical_failures, validate_design_ir

_LOG = logging.getLogger("mech_bench.evaluator")


# --------------------------------------------------------------------- #
# Execution planning                                                    #
# --------------------------------------------------------------------- #


@dataclass
class ProbePlan:
    probe_id: str
    probe_type: str
    capabilities: frozenset[Capability]
    adapter_type: str | None = None  # None means no adapter needed
    available: bool = True
    reason: str = ""  # explanation when unavailable or probe missing
    probe_known: bool = True


@dataclass
class ExecutionPlan:
    probes: list[ProbePlan] = field(default_factory=list)

    def adapters_to_run(self) -> list[str]:
        seen: list[str] = []
        for p in self.probes:
            if p.adapter_type and p.adapter_type not in seen:
                seen.append(p.adapter_type)
        return seen


def _pick_adapter_for(caps: frozenset[Capability]) -> type[SimAdapter] | None:
    needed = caps - {Capability.NONE}
    if not needed:
        return None
    candidates = [a for a in all_adapters()
                  if needed.issubset(a.capabilities_provided)]
    if not candidates:
        return None
    candidates.sort(key=lambda a: a.cost_tier)
    return candidates[0]


def build_execution_plan(cfg: EvalConfig) -> ExecutionPlan:
    """Plan adapter selection per probe.

    For each probe, look up its capability requirements and pick the
    cheapest registered adapter that covers them. Probes that need an
    adapter for which none is registered get `available=False`; the
    evaluator turns those into `CAPABILITY_UNAVAILABLE` failures.
    """
    plan = ExecutionPlan()
    for spec in cfg.probes:
        try:
            probe = get_probe(spec.type)
        except KeyError as e:
            plan.probes.append(ProbePlan(
                probe_id=spec.id,
                probe_type=spec.type,
                capabilities=frozenset(),
                adapter_type=None,
                available=False,
                reason=str(e),
                probe_known=False,
            ))
            continue
        caps = frozenset(probe.capabilities_required)
        if caps <= {Capability.NONE}:
            plan.probes.append(ProbePlan(
                probe_id=spec.id,
                probe_type=spec.type,
                capabilities=caps,
                adapter_type=None,
                available=True,
            ))
            continue
        adapter = _pick_adapter_for(caps)
        if adapter is None:
            missing = sorted(c.value for c in caps - {Capability.NONE})
            plan.probes.append(ProbePlan(
                probe_id=spec.id,
                probe_type=spec.type,
                capabilities=caps,
                adapter_type=None,
                available=False,
                reason=(f"No registered adapter provides the required "
                        f"capabilities: {missing}."),
            ))
            continue
        plan.probes.append(ProbePlan(
            probe_id=spec.id,
            probe_type=spec.type,
            capabilities=caps,
            adapter_type=adapter.type_name,
            available=True,
        ))
    return plan


# --------------------------------------------------------------------- #
# Tier classification                                                   #
# --------------------------------------------------------------------- #


_TIER_BY_CAP: dict[Capability, str] = {
    Capability.NONE: "topology",
    Capability.PLANAR_KINEMATICS: "kinematics",
    Capability.SPATIAL_KINEMATICS: "kinematics",
    Capability.PATH_TRACE: "kinematics",
    Capability.DOF_DETECTION: "kinematics",
    Capability.MESH_OVERLAP: "geometry",
    Capability.MESH: "geometry",
    Capability.RIGID_BODY_DYNAMICS: "dynamics",
    Capability.CONTACT_FORCES: "dynamics",
    Capability.JOINT_CONSTRAINTS: "dynamics",
    Capability.MOTOR_DRIVES: "dynamics",
    Capability.LOAD_TORQUES: "dynamics",
    Capability.POSE_TRACES: "dynamics",
    Capability.FEA_STATIC: "structural",
    Capability.SAFETY_FACTOR: "structural",
}


def _tier_for(caps: frozenset[Capability]) -> str:
    pruned = caps - {Capability.NONE}
    if not pruned:
        return "topology"
    tiers = {_TIER_BY_CAP.get(c, "other") for c in pruned}
    # Pick the most-expensive tier so a mixed probe goes with its
    # heaviest dependency. Order matches typical pipeline cost.
    order = ["topology", "kinematics", "geometry", "dynamics",
             "structural", "other"]
    for t in reversed(order):
        if t in tiers:
            return t
    return "other"


# --------------------------------------------------------------------- #
# Task / submission loading                                             #
# --------------------------------------------------------------------- #


def load_task(task_dir: Path) -> tuple[TaskSpec, EvalConfig]:
    task_dir = Path(task_dir)
    task_toml = task_dir / "task.toml"
    eval_toml = task_dir / "eval_config.toml"
    prompt_md = task_dir / "prompt.md"

    with task_toml.open("rb") as f:
        task_data = tomllib.load(f)
    with eval_toml.open("rb") as f:
        eval_data = tomllib.load(f)

    fixtures_dir = task_dir / "fixtures"
    for entry in eval_data.get("probes", []):
        if "target_csv" in entry:
            entry["target_csv"] = str(
                (fixtures_dir / entry["target_csv"]).resolve()
            )

    prompt = prompt_md.read_text() if prompt_md.exists() else ""
    task = TaskSpec.from_dict(task_data, fixtures_dir=fixtures_dir,
                              prompt=prompt)
    cfg = EvalConfig.from_dict(eval_data)
    return task, cfg


def load_submission(submission_dir: Path, scratch_dir: Path) -> DesignIR:
    submission_dir = Path(submission_dir).resolve()
    scratch_dir = Path(scratch_dir).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    design_py = submission_dir / "design.py"
    if not design_py.exists():
        raise FileNotFoundError(f"Submission missing design.py: {design_py}")

    spec = importlib.util.spec_from_file_location(
        f"_mech_submission_{abs(hash(submission_dir))}",
        design_py,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {design_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    if not hasattr(mod, "build_design"):
        raise AttributeError(
            f"{design_py} does not define build_design(out_dir)."
        )
    raw = mod.build_design(scratch_dir)
    if not isinstance(raw, dict):
        raise TypeError(
            f"build_design must return a dict, got {type(raw).__name__}"
        )
    return DesignIR.from_dict(raw)


# --------------------------------------------------------------------- #
# Probe execution                                                       #
# --------------------------------------------------------------------- #


def _run_probe(
    spec: ProbeSpec,
    plan: ProbePlan,
    ir: DesignIR,
    sim_outputs_by_adapter: dict[str, dict[str, Any]],
) -> ProbeResult:
    if not plan.probe_known:
        return ProbeResult(
            probe_id=spec.id,
            probe_type=spec.type,
            passed=False,
            score=0.0,
            metrics={},
            failures=[Failure(
                code=FailureCode.CAPABILITY_UNAVAILABLE,
                severity=Severity.CRITICAL,
                message=plan.reason or
                        f"Unknown probe type {spec.type!r}.",
            )],
            skipped_reason=plan.reason,
        )
    if not plan.available:
        return ProbeResult(
            probe_id=spec.id,
            probe_type=spec.type,
            passed=False,
            score=0.0,
            metrics={},
            failures=[Failure(
                code=FailureCode.CAPABILITY_UNAVAILABLE,
                severity=Severity.CRITICAL,
                message=plan.reason or
                        f"No adapter available for {spec.type!r}.",
                public_hint=(
                    "This probe needs a simulator that is not "
                    "registered in this evaluator build."
                ),
            )],
            skipped_reason=plan.reason,
        )
    probe: Probe = get_probe(spec.type)
    sim_outputs: dict[str, Any] = {}
    if plan.adapter_type is not None:
        sim_outputs = sim_outputs_by_adapter.get(plan.adapter_type, {})
    result = probe.run(ir, sim_outputs, spec.config)
    result.probe_id = spec.id
    result.probe_type = spec.type
    return result


def _score(
    probe_results: list[ProbeResult],
    specs_by_id: dict[str, ProbeSpec],
    hard_gate_ids: set[str],
) -> tuple[bool, float]:
    hard_gate_passed = True
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        is_gate = r.probe_id in hard_gate_ids or (
            spec is not None and spec.hard_gate)
        if is_gate and not r.passed:
            hard_gate_passed = False
            break
    if not hard_gate_passed:
        return False, 0.0

    weighted = 0.0
    total_w = 0.0
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        if spec is None:
            continue
        if spec.hard_gate or spec.id in hard_gate_ids:
            continue
        w = float(spec.weight)
        if w <= 0:
            continue
        weighted += w * float(r.score)
        total_w += w
    dense = weighted / total_w if total_w > 0 else 0.0
    return True, dense


def _tier_summary(
    probe_results: list[ProbeResult],
    plans: list[ProbePlan],
    specs_by_id: dict[str, ProbeSpec],
) -> dict[str, dict]:
    by_id = {p.probe_id: p for p in plans}
    tiers: dict[str, dict] = {}
    for r in probe_results:
        plan = by_id.get(r.probe_id)
        tier = _tier_for(plan.capabilities) if plan else "other"
        bucket = tiers.setdefault(tier, {
            "probe_ids": [],
            "passed": True,
            "score_sum": 0.0,
            "score_count": 0,
        })
        bucket["probe_ids"].append(r.probe_id)
        if not r.passed:
            bucket["passed"] = False
        bucket["score_sum"] += float(r.score)
        bucket["score_count"] += 1
    for t, b in tiers.items():
        n = b.pop("score_count")
        s = b.pop("score_sum")
        b["score"] = s / n if n else 0.0
    return tiers


# --------------------------------------------------------------------- #
# Top-level evaluate()                                                  #
# --------------------------------------------------------------------- #


def evaluate(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
    run_id: str | None = None,
) -> EvalReport:
    task_dir = Path(task_dir)
    submission_dir = Path(submission_dir)
    scratch_dir = scratch_dir or (submission_dir / "_scratch")
    rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    task, cfg = load_task(task_dir)
    timings["load_task"] = time.perf_counter() - t0

    def _empty_report(failures: list[Failure],
                      tier_results: dict | None = None) -> EvalReport:
        return EvalReport(
            task_id=task.id,
            task_family=task.family,
            difficulty=task.difficulty,
            run_id=rid,
            score=0.0,
            hard_gate_passed=False,
            probe_results=[],
            metrics={},
            feedback=failures,
            tier_results=tier_results or {},
            timings=dict(timings),
        )

    # Load submission.
    t0 = time.perf_counter()
    try:
        ir = load_submission(submission_dir, scratch_dir)
    except Exception as e:
        timings["load_submission"] = time.perf_counter() - t0
        return _empty_report([Failure(
            code=FailureCode.INVALID_ARTIFACT,
            severity=Severity.CRITICAL,
            message=f"Failed to load submission: {e}",
            where=str(submission_dir),
        )])
    timings["load_submission"] = time.perf_counter() - t0

    # Validate DesignIR.
    t0 = time.perf_counter()
    validation_failures = validate_design_ir(
        ir, task=task, build_root=Path(scratch_dir).resolve(),
    )
    timings["validate"] = time.perf_counter() - t0
    if has_critical_failures(validation_failures):
        return _empty_report(validation_failures, {
            "validation": {
                "probe_ids": [],
                "passed": False,
                "score": 0.0,
            },
        })

    # Build the per-probe execution plan.
    t0 = time.perf_counter()
    plan = build_execution_plan(cfg)
    timings["plan"] = time.perf_counter() - t0

    # Run each needed adapter once.
    sim_outputs_by_adapter: dict[str, dict[str, Any]] = {}
    for adapter_name in plan.adapters_to_run():
        adapter_cls = next(
            (a for a in all_adapters() if a.type_name == adapter_name),
            None,
        )
        if adapter_cls is None:
            continue
        adapter = adapter_cls()
        t0 = time.perf_counter()
        try:
            sim_outputs_by_adapter[adapter_name] = adapter.run(
                ir, {"samples": 360})
        except Exception as e:  # adapter shouldn't crash; report as divergence
            _LOG.warning("adapter %s raised: %s", adapter_name, e)
            sim_outputs_by_adapter[adapter_name] = {
                "__adapter_error__": str(e),
            }
        timings[f"adapter.{adapter_name}"] = time.perf_counter() - t0

    # Run probes.
    plan_by_id = {p.probe_id: p for p in plan.probes}
    probe_results: list[ProbeResult] = []
    for spec in cfg.probes:
        pplan = plan_by_id[spec.id]
        t0 = time.perf_counter()
        r = _run_probe(spec, pplan, ir, sim_outputs_by_adapter)
        timings[f"probe.{spec.id}"] = time.perf_counter() - t0
        probe_results.append(r)

    specs_by_id = {s.id: s for s in cfg.probes}
    hard_gate_passed, dense = _score(
        probe_results, specs_by_id, set(cfg.hard_gate_probes)
    )

    # Aggregate metrics / feedback.
    agg_metrics: dict[str, float] = {}
    feedback: list[Failure] = list(validation_failures)
    for r in probe_results:
        for k, v in r.metrics.items():
            agg_metrics[f"{r.probe_id}.{k}"] = float(v)
        for f in r.failures:
            f.where = f.where or r.probe_id
            feedback.append(f)

    tier_results = _tier_summary(probe_results, plan.probes, specs_by_id)
    if validation_failures:
        tier_results.setdefault("validation", {
            "probe_ids": [],
            "passed": False,
            "score": 0.0,
        })

    return EvalReport(
        task_id=task.id,
        task_family=task.family,
        difficulty=task.difficulty,
        run_id=rid,
        score=dense,
        hard_gate_passed=hard_gate_passed,
        probe_results=probe_results,
        metrics=agg_metrics,
        feedback=feedback,
        tier_results=tier_results,
        timings=timings,
    )


# --------------------------------------------------------------------- #
# Report bundle                                                         #
# --------------------------------------------------------------------- #


def _flatten_numeric(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_numeric(v, prefix=key + "."))
    return out


def write_report_bundle(
    report: EvalReport,
    out_dir: Path,
    *,
    visibility: FeedbackVisibility,
) -> dict[str, Path]:
    """Write the report bundle and return a dict of artifact paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = report.to_dict(public=False, visibility=visibility)
    public = report.to_dict(public=True, visibility=visibility)

    paths: dict[str, Path] = {}

    scorecard = out_dir / "scorecard.json"
    scorecard.write_text(json.dumps(full, indent=2, default=str))
    paths["scorecard"] = scorecard

    scorecard_pub = out_dir / "scorecard.public.json"
    scorecard_pub.write_text(json.dumps(public, indent=2, default=str))
    paths["scorecard_public"] = scorecard_pub

    metrics_blob = {
        "score": report.score,
        "hard_gate_passed": report.hard_gate_passed,
        **_flatten_numeric({"metrics": report.metrics}),
        **_flatten_numeric({"timings": report.timings}),
    }
    metrics = out_dir / "metrics.json"
    metrics.write_text(json.dumps(metrics_blob, indent=2, default=str))
    paths["metrics"] = metrics

    public_failures = [
        (f.public() if hasattr(f, "public") else dict(f))
        for f in report.feedback
    ]
    feedback_pub = out_dir / "feedback.public.json"
    feedback_pub.write_text(json.dumps(public_failures, indent=2,
                                        default=str))
    paths["feedback_public"] = feedback_pub

    return paths
