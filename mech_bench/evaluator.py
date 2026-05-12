"""Generic evaluator.

Pipeline (called once per submission):

  1. Load TaskSpec + EvalConfig from a task directory.
  2. Run the submission's ``design.py`` in an isolated subprocess
     (``mech_bench.submission_worker``) and parse the returned dict
     into a DesignIR. The evaluator process never imports the
     submission's design.py.
  3. Validate the DesignIR. If validation surfaces critical
     structural failures, short-circuit before any probe runs and
     return a zero report whose feedback is the validation failures.
  4. Build an ExecutionPlan: per probe, pick the cheapest adapter
     whose capabilities cover the probe's requirements. Probes that
     need an adapter for which none is registered produce
     CAPABILITY_UNAVAILABLE and mark the whole evaluation invalid.
  5. Run each adapter at most once; pass its outputs to the probes
     that need it. Adapter exceptions also invalidate the evaluation.
  6. Compose the final score (see ``_score`` for the rules) and
     sanitize all numeric values for strict JSON.

The runtime treats the submission as adversarial. Path-policy
enforcement lives in mech_bench.validation; out-of-process execution
lives in mech_bench.submission_worker.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_bench.adapters import (
    SimAdapter,
    all_adapters,
    normalize_sim_output,
)
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

DEFAULT_SUBMISSION_TIMEOUT = 10.0


class SubmissionError(Exception):
    """The submission subprocess failed in a structured way."""


# --------------------------------------------------------------------- #
# Execution planning                                                    #
# --------------------------------------------------------------------- #


@dataclass
class ProbePlan:
    probe_id: str
    probe_type: str
    capabilities: frozenset[Capability]
    adapter_type: str | None = None
    available: bool = True
    reason: str = ""
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


def load_submission(
    submission_dir: Path,
    scratch_dir: Path,
    *,
    timeout: float = DEFAULT_SUBMISSION_TIMEOUT,
) -> DesignIR:
    """Execute the submission's design.py in an isolated subprocess.

    The evaluator process must NOT import the agent's design.py — any
    monkeypatch the agent applies stays inside the subprocess.
    """
    submission_dir = Path(submission_dir).resolve()
    scratch_dir = Path(scratch_dir).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    design_py = submission_dir / "design.py"
    if not design_py.exists():
        raise SubmissionError(f"Submission missing design.py: {design_py}")

    result_json = scratch_dir / "_design_ir.json"
    if result_json.exists():
        try:
            result_json.unlink()
        except OSError:
            pass

    cmd = [
        sys.executable, "-I",
        "-m", "mech_bench.submission_worker",
        "--design-py", str(design_py),
        "--out-dir", str(scratch_dir),
        "--result-json", str(result_json),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SubmissionError(
            f"build_design did not finish within {timeout}s; subprocess "
            f"killed."
        )

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-1000:].strip()
        raise SubmissionError(
            f"Submission subprocess exited with code "
            f"{proc.returncode}: {stderr_tail or '<no stderr>'}"
        )

    if not result_json.exists():
        raise SubmissionError(
            "Submission subprocess exited 0 but did not write the "
            "result JSON."
        )

    try:
        text = result_json.read_text()
        raw = json.loads(text)
    except (OSError, json.JSONDecodeError) as e:
        raise SubmissionError(f"Could not parse submission JSON: {e}")

    if not isinstance(raw, dict):
        raise SubmissionError(
            f"Submission JSON root must be a dict, got "
            f"{type(raw).__name__}."
        )

    try:
        return DesignIR.from_dict(raw)
    except (KeyError, TypeError, ValueError) as e:
        raise SubmissionError(
            f"Submission JSON does not fit DesignIR schema: "
            f"{type(e).__name__}: {e}"
        )


# --------------------------------------------------------------------- #
# Sanitation                                                            #
# --------------------------------------------------------------------- #


def sanitize_metric_value(v: Any) -> float | None:
    """Coerce a metric value for strict JSON.

    Non-finite (NaN / +Inf / -Inf) becomes None. Booleans are
    preserved as 1.0 / 0.0. Strings / dicts / lists pass through
    unchanged — metrics dicts only ever hold floats today, but the
    helper stays liberal so it can be reused on report blobs.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    return v  # passthrough


def sanitize_metrics_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: sanitize_metric_value(v) for k, v in d.items()}


def sanitize_report_for_json(blob: Any) -> Any:
    """Recursively replace NaN/Inf with None so json.dumps with
    ``allow_nan=False`` succeeds.
    """
    if isinstance(blob, dict):
        return {k: sanitize_report_for_json(v) for k, v in blob.items()}
    if isinstance(blob, (list, tuple)):
        return [sanitize_report_for_json(v) for v in blob]
    if isinstance(blob, bool):
        return blob
    if isinstance(blob, (int, float)):
        f = float(blob)
        if math.isfinite(f):
            return blob
        return None
    return blob


def _strict_json_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(
        sanitize_report_for_json(obj),
        indent=indent,
        default=str,
        allow_nan=False,
    )


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


def _sanitize_probe_result(
    r: ProbeResult,
) -> tuple[ProbeResult, bool]:
    """Clamp score to [0, 1] and replace NaN/Inf metrics with None.

    Returns the (possibly mutated) result and a flag indicating
    whether anything non-finite was observed, which the caller treats
    as an evaluation_valid invalidation.
    """
    invalid = False
    if not math.isfinite(float(r.score)):
        r.failures.append(Failure(
            code=FailureCode.SIMULATOR_DIVERGENCE,
            severity=Severity.CRITICAL,
            message=(f"Probe {r.probe_id!r} returned a non-finite "
                     f"score ({r.score!r})."),
        ))
        r.score = 0.0
        r.passed = False
        invalid = True
    else:
        r.score = max(0.0, min(1.0, float(r.score)))
    # Metric sanitation: keep dict but replace bad floats with None.
    new_metrics: dict[str, float] = {}
    for k, v in r.metrics.items():
        sv = sanitize_metric_value(v)
        if sv is None and v is not None:
            invalid = True  # metric was non-finite
        new_metrics[k] = sv  # type: ignore[assignment]
    r.metrics = new_metrics
    return r, invalid


def _score(
    probe_results: list[ProbeResult],
    specs_by_id: dict[str, ProbeSpec],
    hard_gate_ids: set[str],
) -> tuple[bool, float]:
    hard_gate_passed = True
    has_gate = False
    for r in probe_results:
        spec = specs_by_id.get(r.probe_id)
        is_gate = r.probe_id in hard_gate_ids or (
            spec is not None and spec.hard_gate)
        if is_gate:
            has_gate = True
            if not r.passed:
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
    if total_w > 0:
        return True, weighted / total_w
    # No non-gate weighted probes. If the hard gate exists and it
    # passed, the task is fully satisfied: score = 1.0.
    if has_gate:
        return True, 1.0
    # No gate, no weighted dense probes — undefined; report 0 instead
    # of pretending there is a verifiable signal.
    return True, 0.0


def _tier_summary(
    probe_results: list[ProbeResult],
    plans: list[ProbePlan],
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
    for b in tiers.values():
        n = b.pop("score_count")
        s = b.pop("score_sum")
        b["score"] = s / n if n else 0.0
    return tiers


# --------------------------------------------------------------------- #
# Top-level evaluate()                                                  #
# --------------------------------------------------------------------- #


@dataclass
class RunEvidence:
    """Everything needed to package a run after evaluation.

    The CLI's --report-dir flow turns this into the on-disk
    scorecard / metrics / feedback / dashboard / trace bundle.
    """

    report: EvalReport
    task: TaskSpec
    cfg: EvalConfig
    sim_outputs_by_adapter: dict[str, Any] = field(default_factory=dict)


def evaluate(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
    run_id: str | None = None,
    submission_timeout: float = DEFAULT_SUBMISSION_TIMEOUT,
) -> EvalReport:
    """Backwards-compatible wrapper: returns only the EvalReport."""
    return evaluate_with_evidence(
        task_dir,
        submission_dir,
        scratch_dir=scratch_dir,
        run_id=run_id,
        submission_timeout=submission_timeout,
    ).report


def evaluate_with_evidence(
    task_dir: Path,
    submission_dir: Path,
    *,
    scratch_dir: Path | None = None,
    run_id: str | None = None,
    submission_timeout: float = DEFAULT_SUBMISSION_TIMEOUT,
) -> RunEvidence:
    task_dir = Path(task_dir)
    submission_dir = Path(submission_dir)
    scratch_dir = scratch_dir or (submission_dir / "_scratch")
    rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    task, cfg = load_task(task_dir)
    timings["load_task"] = time.perf_counter() - t0

    def _empty_evidence(
        failures: list[Failure],
        tier_results: dict | None = None,
        *,
        valid: bool = False,
    ) -> RunEvidence:
        return RunEvidence(
            report=EvalReport(
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
                evaluation_valid=valid,
            ),
            task=task,
            cfg=cfg,
            sim_outputs_by_adapter={},
        )

    # Load submission via isolated subprocess.
    t0 = time.perf_counter()
    try:
        ir = load_submission(submission_dir, Path(scratch_dir),
                              timeout=submission_timeout)
    except SubmissionError as e:
        timings["load_submission"] = time.perf_counter() - t0
        return _empty_evidence([Failure(
            code=FailureCode.INVALID_ARTIFACT,
            severity=Severity.CRITICAL,
            message=str(e),
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
        return _empty_evidence(validation_failures, {
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

    evaluation_valid = True

    # Run each needed adapter once. Each adapter receives the
    # ``[adapters.<name>]`` table from eval_config.toml (or an empty
    # dict). A registered default of ``samples=360`` is provided for
    # backward compatibility with adapters that expect it.
    sim_outputs_by_adapter: dict[str, dict[str, Any]] = {}
    adapter_failures: list[Failure] = []
    for adapter_name in plan.adapters_to_run():
        adapter_cls = next(
            (a for a in all_adapters() if a.type_name == adapter_name),
            None,
        )
        if adapter_cls is None:
            continue
        adapter = adapter_cls()
        adapter_cfg: dict[str, Any] = {"samples": 360}
        adapter_cfg.update(cfg.adapter_configs.get(adapter_name, {}))
        t0 = time.perf_counter()
        try:
            raw = adapter.run(ir, adapter_cfg)
            sim_outputs_by_adapter[adapter_name] = normalize_sim_output(raw)
        except Exception as e:  # noqa: BLE001 — adapter is internal-ish
            _LOG.warning("adapter %s raised: %s", adapter_name, e)
            sim_outputs_by_adapter[adapter_name] = {
                "__adapter_error__": str(e),
            }
            adapter_failures.append(Failure(
                code=FailureCode.SIMULATOR_DIVERGENCE,
                severity=Severity.CRITICAL,
                message=(f"Adapter {adapter_name!r} raised "
                         f"{type(e).__name__}: {e}"),
                where=f"adapter.{adapter_name}",
            ))
            evaluation_valid = False
        timings[f"adapter.{adapter_name}"] = time.perf_counter() - t0

    # Run probes.
    plan_by_id = {p.probe_id: p for p in plan.probes}
    probe_results: list[ProbeResult] = []
    for spec in cfg.probes:
        pplan = plan_by_id[spec.id]
        t0 = time.perf_counter()
        r = _run_probe(spec, pplan, ir, sim_outputs_by_adapter)
        r, bad = _sanitize_probe_result(r)
        if bad:
            evaluation_valid = False
        timings[f"probe.{spec.id}"] = time.perf_counter() - t0
        probe_results.append(r)
        # If any probe surfaced CAPABILITY_UNAVAILABLE, the entire
        # evaluation is structurally invalid: an agent should not be
        # able to earn reward on a partial verifier.
        if any(f.code == FailureCode.CAPABILITY_UNAVAILABLE
               for f in r.failures):
            evaluation_valid = False

    specs_by_id = {s.id: s for s in cfg.probes}
    hard_gate_passed, dense = _score(
        probe_results, specs_by_id, set(cfg.hard_gate_probes)
    )

    if not evaluation_valid:
        # An invalid evaluation can never earn reward and must surface
        # as a CLI failure.
        hard_gate_passed = False
        dense = 0.0

    # Aggregate metrics / feedback.
    agg_metrics: dict[str, float] = {}
    feedback: list[Failure] = list(validation_failures) + list(
        adapter_failures)
    for r in probe_results:
        for k, v in r.metrics.items():
            sv = sanitize_metric_value(v)
            agg_metrics[f"{r.probe_id}.{k}"] = sv  # may be None
        for f in r.failures:
            f.where = f.where or r.probe_id
            feedback.append(f)

    if not math.isfinite(float(dense)):
        feedback.append(Failure(
            code=FailureCode.SIMULATOR_DIVERGENCE,
            severity=Severity.CRITICAL,
            message=f"Aggregate score is non-finite: {dense!r}",
        ))
        dense = 0.0
        evaluation_valid = False
        hard_gate_passed = False

    tier_results = _tier_summary(probe_results, plan.probes)
    if validation_failures:
        tier_results.setdefault("validation", {
            "probe_ids": [],
            "passed": False,
            "score": 0.0,
        })

    report = EvalReport(
        task_id=task.id,
        task_family=task.family,
        difficulty=task.difficulty,
        run_id=rid,
        score=float(dense),
        hard_gate_passed=hard_gate_passed,
        probe_results=probe_results,
        metrics=agg_metrics,
        feedback=feedback,
        tier_results=tier_results,
        timings=timings,
        evaluation_valid=evaluation_valid,
    )
    return RunEvidence(
        report=report,
        task=task,
        cfg=cfg,
        sim_outputs_by_adapter=sim_outputs_by_adapter,
    )


# --------------------------------------------------------------------- #
# Report bundle                                                         #
# --------------------------------------------------------------------- #


def _flatten_numeric(d: dict, prefix: str = "") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif v is None:
            out[key] = None
        elif isinstance(v, (int, float)):
            f = float(v)
            out[key] = f if math.isfinite(f) else None
        elif isinstance(v, dict):
            out.update(_flatten_numeric(v, prefix=key + "."))
    return out


def write_report_bundle(
    report: EvalReport,
    out_dir: Path,
    *,
    visibility: FeedbackVisibility,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = report.to_dict(public=False, visibility=visibility)
    public = report.to_dict(public=True, visibility=visibility)

    paths: dict[str, Path] = {}

    (out_dir / "scorecard.json").write_text(_strict_json_dumps(full))
    paths["scorecard"] = out_dir / "scorecard.json"

    (out_dir / "scorecard.public.json").write_text(_strict_json_dumps(public))
    paths["scorecard_public"] = out_dir / "scorecard.public.json"

    metrics_blob = {
        "score": report.score,
        "hard_gate_passed": report.hard_gate_passed,
        "evaluation_valid": report.evaluation_valid,
        **_flatten_numeric({"metrics": report.metrics}),
        **_flatten_numeric({"timings": report.timings}),
    }
    (out_dir / "metrics.json").write_text(_strict_json_dumps(metrics_blob))
    paths["metrics"] = out_dir / "metrics.json"

    public_failures = [
        (f.public() if hasattr(f, "public") else dict(f))
        for f in report.feedback
    ]
    (out_dir / "feedback.public.json").write_text(
        _strict_json_dumps(public_failures))
    paths["feedback_public"] = out_dir / "feedback.public.json"

    return paths


def write_run_bundle(
    evidence: "RunEvidence",
    out_dir: Path,
) -> dict[str, Path]:
    """Write the full evidence bundle for one run.

    Always writes: scorecard.json, scorecard.public.json,
    metrics.json, feedback.public.json, dashboard_payload.json,
    media_manifest.json. Optionally writes traces.h5 (when h5py is
    installed) and dashboard.html (when plotly is installed).
    Returns a dict of the artifact paths actually written.
    """
    from mech_bench.dashboard_payload import (
        build_dashboard_payload,
        write_dashboard_payload,
    )
    from mech_bench.media import write_media_manifest
    from mech_bench.traces import (
        HAS_H5PY,
        TraceData,
        write_capability_unavailable,
        write_trace_hdf5,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_report_bundle(
        evidence.report, out_dir, visibility=evidence.cfg.visibility)

    # Pick the first non-error adapter output for trace generation.
    primary_sim: dict | None = None
    primary_adapter = ""
    for name, sim in evidence.sim_outputs_by_adapter.items():
        if isinstance(sim, dict) and "__adapter_error__" in sim:
            continue
        primary_sim = sim
        primary_adapter = name
        break

    trace = TraceData.from_sim_output(
        primary_sim or {},
        run_id=evidence.report.run_id,
        task_id=evidence.report.task_id,
        adapter=primary_adapter,
    )

    trace_path: Path | None = None
    if not trace.is_empty():
        if HAS_H5PY:
            trace_path = write_trace_hdf5(out_dir / "traces.h5", trace)
            paths["trace"] = trace_path
        else:
            stub = write_capability_unavailable(
                out_dir / "traces.unavailable.json",
                reason=("h5py is not installed; HDF5 trace not "
                        "written. Install mech-bench[traces] to enable."),
            )
            paths["trace_stub"] = stub

    payload = build_dashboard_payload(
        evidence.report, trace, task=evidence.task)
    payload_path = write_dashboard_payload(
        out_dir / "dashboard_payload.json", payload)
    paths["dashboard_payload"] = payload_path

    dashboard_path: Path | None = None
    try:
        from mech_bench.dashboard import (
            HAS_PLOTLY,
            write_static_dashboard,
        )
        if HAS_PLOTLY:
            dashboard_path = write_static_dashboard(
                payload, out_dir / "dashboard.html")
            paths["dashboard"] = dashboard_path
    except ImportError:  # pragma: no cover - defensive
        pass

    manifest_path = write_media_manifest(
        out_dir,
        evidence.report,
        trace_path=trace_path,
        dashboard_payload_path=payload_path,
        dashboard_html_path=dashboard_path,
    )
    paths["media_manifest"] = manifest_path
    return paths
