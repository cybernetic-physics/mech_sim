"""Benchmark runner: evaluates a suite of tasks and aggregates results.

Given a directory of generated tasks (see ``mech_bench.generators``),
the runner walks each task, picks a submission (the reference
solution by default, optionally a named negative control), runs the
evaluator, and writes:

* a per-task report bundle under ``<report_dir>/<task_id>/``;
* an aggregate ``benchmark_summary.json`` at the report-dir root.

Hidden / public eval split
--------------------------

If a task ships both ``eval_config.public.toml`` and
``eval_config.hidden.toml``, the runner can score the same submission
twice and emit a ``generalization_gap`` per task and aggregated
across the suite.

* ``mode="public"`` runs only the public eval (default).
* ``mode="hidden"`` runs only the hidden eval.
* ``mode="both"`` runs both and reports the gap.

Aggregate metrics
-----------------

The summary is dataclass-ish JSON: any field not produced by a given
run is set to ``None`` so consumers can pattern-match. The list of
documented fields is in :func:`build_benchmark_summary`.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mech_bench.evaluator import (
    RunEvidence,
    evaluate_with_evidence,
    load_task,
    sanitize_report_for_json,
    write_run_bundle,
)
from mech_bench.feedback import FailureCode


# --------------------------------------------------------------------- #
# Constants                                                             #
# --------------------------------------------------------------------- #


EVAL_MODES = ("public", "hidden", "both")


# --------------------------------------------------------------------- #
# Per-task / per-run helpers                                            #
# --------------------------------------------------------------------- #


def _read_task_meta(task_dir: Path) -> dict[str, Any]:
    """Load metadata.json if present; otherwise derive from task.toml."""
    meta_path = task_dir / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    # Fallback: synthesize from task.toml.
    task_toml = task_dir / "task.toml"
    if task_toml.exists():
        with task_toml.open("rb") as f:
            data = tomllib.load(f)
        t = data.get("task", {})
        return {
            "task_id": t.get("id", task_dir.name),
            "family": t.get("family", task_dir.name),
            "tier": t.get("tier", "unknown"),
            "difficulty": t.get("difficulty", 1),
        }
    return {"task_id": task_dir.name, "family": task_dir.name,
            "tier": "unknown", "difficulty": 1}


def _resolve_submission(task_dir: Path, kind: str,
                          negative: str | None) -> Path | None:
    """Resolve which submission to score.

    ``kind`` is ``"reference"`` (default; ``reference_solution/``) or
    the literal ``"negative"`` which uses ``negative`` as the case
    name under ``negative_solutions/``.
    """
    if kind == "negative":
        if not negative:
            return None
        sub = task_dir / "negative_solutions" / negative
        return sub if sub.is_dir() else None
    sub = task_dir / "reference_solution"
    return sub if sub.is_dir() else None


def _swap_eval_config(task_dir: Path, variant: str,
                       scratch_root: Path) -> Path:
    """Materialize a copy of *task_dir* with the chosen variant as the
    active ``eval_config.toml``.

    Returns the path to the new (scratch) task directory, which the
    evaluator will load as if it were the original. Copies use
    hardlinks when possible to keep the cost negligible.
    """
    src_cfg = task_dir / f"eval_config.{variant}.toml"
    if not src_cfg.exists():
        # No variant on disk — fall back to the default.
        src_cfg = task_dir / "eval_config.toml"
    scratch_root.mkdir(parents=True, exist_ok=True)
    dst = scratch_root / f"{task_dir.name}__{variant}"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir()
    # Hardlink fixtures, reference, negatives, etc.
    for child in task_dir.iterdir():
        if child.name in {"eval_config.toml", "eval_config.public.toml",
                          "eval_config.hidden.toml"}:
            continue
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target,
                            copy_function=_link_or_copy,
                            dirs_exist_ok=True)
        else:
            _link_or_copy(child, target)
    # Always provide both names: eval_config.toml is what the evaluator
    # reads; eval_config.<variant>.toml mirrors it for traceability.
    (dst / "eval_config.toml").write_text(src_cfg.read_text())
    (dst / f"eval_config.{variant}.toml").write_text(src_cfg.read_text())
    return dst


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists():
            return
        dst.hardlink_to(src)  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


# --------------------------------------------------------------------- #
# Single-task run                                                       #
# --------------------------------------------------------------------- #


@dataclass
class TaskRunResult:
    task_id: str
    family: str
    tier: str
    difficulty: int
    public_score: float | None = None
    hidden_score: float | None = None
    public_passed: bool | None = None
    hidden_passed: bool | None = None
    public_valid: bool | None = None
    hidden_valid: bool | None = None
    public_failure_codes: list[str] | None = None
    hidden_failure_codes: list[str] | None = None
    public_runtime_s: float | None = None
    hidden_runtime_s: float | None = None
    submission: str = "reference"
    report_dir: str = ""

    def generalization_gap(self) -> float | None:
        if self.public_score is None or self.hidden_score is None:
            return None
        return float(self.public_score) - float(self.hidden_score)

    def overall_score(self) -> float:
        """Score used for suite-level aggregation: hidden when both
        are present (the harder, less-overfit signal), otherwise
        whichever exists."""
        if self.hidden_score is not None:
            return float(self.hidden_score)
        if self.public_score is not None:
            return float(self.public_score)
        return 0.0

    def overall_passed(self) -> bool:
        if self.hidden_passed is not None:
            return bool(self.hidden_passed)
        if self.public_passed is not None:
            return bool(self.public_passed)
        return False

    def all_failure_codes(self) -> list[str]:
        seen: list[str] = []
        for src in (self.public_failure_codes, self.hidden_failure_codes):
            for c in src or []:
                if c not in seen:
                    seen.append(c)
        return seen


def _failure_codes_of(evidence: RunEvidence) -> list[str]:
    out: list[str] = []
    for f in evidence.report.feedback:
        code = getattr(f, "code", None)
        if hasattr(code, "value"):
            out.append(code.value)
        elif code is not None:
            out.append(str(code))
    return out


def run_task(
    task_dir: Path,
    *,
    submission: str = "reference",
    negative: str | None = None,
    eval_mode: str = "public",
    report_dir: Path | None = None,
    scratch_root: Path | None = None,
) -> TaskRunResult:
    """Evaluate one task. ``eval_mode`` ∈ {public, hidden, both}."""
    if eval_mode not in EVAL_MODES:
        raise ValueError(
            f"eval_mode must be one of {EVAL_MODES}, got {eval_mode!r}")
    meta = _read_task_meta(task_dir)
    sub_dir = _resolve_submission(task_dir, submission, negative)
    result = TaskRunResult(
        task_id=str(meta.get("task_id", task_dir.name)),
        family=str(meta.get("family", task_dir.name)),
        tier=str(meta.get("tier", "unknown")),
        difficulty=int(meta.get("difficulty", 1) or 1),
        submission=(negative or "reference"),
    )
    if sub_dir is None:
        return result

    scratch_root = scratch_root or (task_dir.parent / "_runner_scratch")
    variants_to_run: list[str] = (
        ["public", "hidden"] if eval_mode == "both" else [eval_mode]
    )

    for variant in variants_to_run:
        prepared = _swap_eval_config(task_dir, variant, scratch_root)
        scratch_subdir = scratch_root / f"_run_{prepared.name}"
        scratch_subdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            evidence = evaluate_with_evidence(
                prepared, sub_dir, scratch_dir=scratch_subdir)
        except Exception as e:  # noqa: BLE001 — runner is the firewall
            elapsed = time.perf_counter() - t0
            _record_runner_failure(result, variant, elapsed, str(e))
            continue
        elapsed = time.perf_counter() - t0
        score = float(evidence.report.score)
        passed = bool(evidence.report.hard_gate_passed)
        valid = bool(evidence.report.evaluation_valid)
        codes = _failure_codes_of(evidence)
        if variant == "public":
            result.public_score = score
            result.public_passed = passed
            result.public_valid = valid
            result.public_failure_codes = codes
            result.public_runtime_s = elapsed
        else:
            result.hidden_score = score
            result.hidden_passed = passed
            result.hidden_valid = valid
            result.hidden_failure_codes = codes
            result.hidden_runtime_s = elapsed

        # Write per-task report bundle. For "both" mode, write under
        # variant-named subdirectories so they don't overwrite.
        if report_dir is not None:
            if eval_mode == "both":
                per_task = report_dir / result.task_id / variant
            else:
                per_task = report_dir / result.task_id
            per_task.mkdir(parents=True, exist_ok=True)
            write_run_bundle(evidence, per_task)
            result.report_dir = str(report_dir / result.task_id)
    return result


def _record_runner_failure(result: TaskRunResult, variant: str,
                              elapsed: float, msg: str) -> None:
    codes = [FailureCode.SIMULATOR_DIVERGENCE.value]
    if variant == "public":
        result.public_score = 0.0
        result.public_passed = False
        result.public_valid = False
        result.public_failure_codes = codes
        result.public_runtime_s = elapsed
    else:
        result.hidden_score = 0.0
        result.hidden_passed = False
        result.hidden_valid = False
        result.hidden_failure_codes = codes
        result.hidden_runtime_s = elapsed


# --------------------------------------------------------------------- #
# Suite run + aggregation                                               #
# --------------------------------------------------------------------- #


def iter_task_dirs(suite_dir: Path) -> list[Path]:
    """List task directories under *suite_dir* (each must hold a
    task.toml).
    """
    out: list[Path] = []
    for p in sorted(Path(suite_dir).iterdir()):
        if p.is_dir() and (p / "task.toml").exists():
            out.append(p)
    return out


def run_suite(
    suite_dir: Path,
    *,
    submissions: str = "reference",
    negative: str | None = None,
    eval_mode: str = "public",
    report_dir: Path | None = None,
    families: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run every task in *suite_dir* and return the aggregate summary.

    The summary is also written to ``report_dir/benchmark_summary.json``
    when *report_dir* is supplied.
    """
    suite_dir = Path(suite_dir)
    task_dirs = iter_task_dirs(suite_dir)
    if families is not None:
        wanted = set(families)
        task_dirs = [t for t in task_dirs
                     if _read_task_meta(t).get("family") in wanted]

    report_dir_path = Path(report_dir) if report_dir is not None else None
    scratch_root = (report_dir_path / "_scratch"
                    if report_dir_path is not None
                    else suite_dir / "_runner_scratch")

    results: list[TaskRunResult] = []
    for task_dir in task_dirs:
        results.append(
            run_task(
                task_dir,
                submission=submissions,
                negative=negative,
                eval_mode=eval_mode,
                report_dir=report_dir_path,
                scratch_root=scratch_root,
            )
        )

    summary = build_benchmark_summary(results, eval_mode=eval_mode,
                                          suite_dir=suite_dir)

    if report_dir_path is not None:
        report_dir_path.mkdir(parents=True, exist_ok=True)
        (report_dir_path / "benchmark_summary.json").write_text(
            json.dumps(sanitize_report_for_json(summary), indent=2,
                        default=str, allow_nan=False)
        )
        # Best-effort dashboard payload + static HTML.
        try:
            from mech_bench.dashboard_payload import (
                build_benchmark_dashboard_payload,
                write_dashboard_payload,
            )
            payload = build_benchmark_dashboard_payload(summary)
            write_dashboard_payload(
                report_dir_path / "benchmark_dashboard_payload.json",
                payload,
            )
            try:
                from mech_bench.dashboard import (
                    HAS_PLOTLY,
                    write_benchmark_dashboard,
                )
                if HAS_PLOTLY:
                    write_benchmark_dashboard(
                        payload,
                        report_dir_path / "benchmark_dashboard.html",
                    )
            except ImportError:  # pragma: no cover
                pass
        except Exception:  # noqa: BLE001 - best effort
            pass

    return summary


# --------------------------------------------------------------------- #
# Aggregation                                                           #
# --------------------------------------------------------------------- #


def build_benchmark_summary(
    results: list[TaskRunResult],
    *,
    eval_mode: str = "public",
    suite_dir: Path | None = None,
) -> dict[str, Any]:
    """Construct the benchmark_summary.json dict."""
    n_tasks = len(results)
    overall_scores = [r.overall_score() for r in results]
    pass_flags = [r.overall_passed() for r in results]
    capability_unavail_n = sum(
        1 for r in results if "capability_unavailable" in r.all_failure_codes()
    )

    public_scores = [r.public_score for r in results
                     if r.public_score is not None]
    hidden_scores = [r.hidden_score for r in results
                     if r.hidden_score is not None]
    gen_gaps = [r.generalization_gap() for r in results
                if r.generalization_gap() is not None]

    pass_by_tier: dict[str, dict[str, float]] = {}
    score_by_tier: dict[str, dict[str, float]] = {}
    pass_by_family: dict[str, dict[str, float]] = {}
    score_by_family: dict[str, dict[str, float]] = {}
    runtime_by_tier: dict[str, dict[str, float]] = {}
    failure_hist: dict[str, int] = {}
    hard_gate_pass_n = 0

    for r in results:
        tier = r.tier or "unknown"
        family = r.family or "unknown"
        score = r.overall_score()
        passed = r.overall_passed()
        if passed:
            hard_gate_pass_n += 1

        _bump(pass_by_tier, tier, passed)
        _bump(pass_by_family, family, passed)
        _accumulate(score_by_tier, tier, score)
        _accumulate(score_by_family, family, score)

        rt = (r.public_runtime_s or 0.0) + (r.hidden_runtime_s or 0.0)
        _accumulate(runtime_by_tier, tier, rt)

        for code in r.all_failure_codes():
            failure_hist[code] = failure_hist.get(code, 0) + 1

    summary: dict[str, Any] = {
        "version": "mech_bench.benchmark_summary.v1",
        "suite_dir": str(suite_dir) if suite_dir is not None else "",
        "eval_mode": eval_mode,
        "n_tasks": n_tasks,
        "overall_score_mean": _mean(overall_scores),
        "overall_score_median": _median(overall_scores),
        "pass_rate": _frac(pass_flags),
        "hard_gate_pass_rate": (hard_gate_pass_n / n_tasks) if n_tasks else 0.0,
        "public_score_mean": _mean(public_scores) if public_scores else None,
        "hidden_score_mean": _mean(hidden_scores) if hidden_scores else None,
        "generalization_gap_mean": _mean(gen_gaps) if gen_gaps else None,
        "capability_unavailable_n": capability_unavail_n,
        "pass_by_tier": _finalize_counts(pass_by_tier),
        "score_by_tier": _finalize_means(score_by_tier),
        "pass_by_family": _finalize_counts(pass_by_family),
        "score_by_family": _finalize_means(score_by_family),
        "runtime_by_tier": _finalize_runtime(runtime_by_tier),
        "failure_code_histogram": dict(sorted(failure_hist.items())),
        "tasks": [_result_dict(r) for r in results],
    }
    return summary


def _bump(d: dict[str, dict[str, float]], key: str, passed: bool) -> None:
    bucket = d.setdefault(key, {"n": 0.0, "n_passed": 0.0})
    bucket["n"] = bucket.get("n", 0.0) + 1.0
    if passed:
        bucket["n_passed"] = bucket.get("n_passed", 0.0) + 1.0


def _accumulate(d: dict[str, dict[str, float]], key: str, v: float) -> None:
    bucket = d.setdefault(key, {"sum": 0.0, "n": 0.0})
    bucket["sum"] = bucket.get("sum", 0.0) + float(v)
    bucket["n"] = bucket.get("n", 0.0) + 1.0


def _finalize_counts(
    d: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k, v in d.items():
        n = v.get("n", 0.0)
        np_ = v.get("n_passed", 0.0)
        out[k] = {
            "n": int(n),
            "n_passed": int(np_),
            "pass_rate": (np_ / n) if n else 0.0,
        }
    return out


def _finalize_means(
    d: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k, v in d.items():
        n = v.get("n", 0.0)
        s = v.get("sum", 0.0)
        out[k] = {
            "n": int(n),
            "mean": (s / n) if n else 0.0,
        }
    return out


def _finalize_runtime(
    d: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k, v in d.items():
        n = v.get("n", 0.0)
        s = v.get("sum", 0.0)
        out[k] = {
            "n": int(n),
            "total_s": float(s),
            "mean_s": (s / n) if n else 0.0,
        }
    return out


def _mean(xs: list[float | None]) -> float:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _median(xs: list[float | None]) -> float:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def _frac(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for f in flags if f) / len(flags))


# --------------------------------------------------------------------- #
# Negative-control checker                                              #
# --------------------------------------------------------------------- #


@dataclass
class NegativeControlCheck:
    task_id: str
    control_id: str
    passed: bool
    reasons: list[str]
    expected: dict[str, Any]
    observed: dict[str, Any]


def check_negative_controls(
    suite_dir: Path,
    *,
    eval_mode: str = "public",
) -> tuple[list[NegativeControlCheck], dict[str, Any]]:
    """Run every negative control in *suite_dir* and verify the
    expectations recorded in ``expected_failures.json``.

    Returns ``(checks, summary)``. ``summary["all_passed"]`` is True
    iff every control matched its expectations exactly.
    """
    suite_dir = Path(suite_dir)
    checks: list[NegativeControlCheck] = []
    for task_dir in iter_task_dirs(suite_dir):
        exp_path = task_dir / "expected_failures.json"
        if not exp_path.exists():
            continue
        try:
            spec = json.loads(exp_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for ctrl in spec.get("controls", []) or []:
            control_id = str(ctrl.get("id") or ctrl.get("submission", ""))
            sub_rel = ctrl.get("submission") or f"negative_solutions/{control_id}"
            negative_name = Path(sub_rel).name
            scratch_root = task_dir.parent / "_runner_scratch"
            result = run_task(
                task_dir,
                submission="negative",
                negative=negative_name,
                eval_mode=eval_mode,
                scratch_root=scratch_root,
            )
            checks.append(_compare_control(result, ctrl))
    summary = {
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c.passed),
        "all_passed": all(c.passed for c in checks) if checks else True,
        "failures": [
            {
                "task_id": c.task_id,
                "control_id": c.control_id,
                "reasons": c.reasons,
                "expected": c.expected,
                "observed": c.observed,
            }
            for c in checks if not c.passed
        ],
    }
    return checks, summary


def _compare_control(
    result: TaskRunResult,
    expected: dict[str, Any],
) -> NegativeControlCheck:
    exp_codes = set(expected.get("expected_failure_codes") or [])
    exp_gate = expected.get("expected_hard_gate_passed")
    exp_score_below = expected.get("expected_score_below")
    observed_codes = set(result.all_failure_codes())
    observed_gate = result.overall_passed()
    observed_score = result.overall_score()
    reasons: list[str] = []
    if exp_codes and not exp_codes.issubset(observed_codes):
        missing = sorted(exp_codes - observed_codes)
        reasons.append(f"missing expected failure codes: {missing}")
    if exp_gate is not None and bool(exp_gate) != bool(observed_gate):
        reasons.append(
            f"hard_gate_passed expected {exp_gate} got {observed_gate}"
        )
    if exp_score_below is not None and observed_score >= float(exp_score_below):
        reasons.append(
            f"score {observed_score} not below "
            f"expected_score_below {exp_score_below}"
        )
    return NegativeControlCheck(
        task_id=result.task_id,
        control_id=str(expected.get("id", "")),
        passed=not reasons,
        reasons=reasons,
        expected={
            "failure_codes": sorted(exp_codes),
            "hard_gate_passed": exp_gate,
            "score_below": exp_score_below,
        },
        observed={
            "failure_codes": sorted(observed_codes),
            "hard_gate_passed": observed_gate,
            "score": observed_score,
        },
    )


def _result_dict(r: TaskRunResult) -> dict[str, Any]:
    return {
        "task_id": r.task_id,
        "family": r.family,
        "tier": r.tier,
        "difficulty": r.difficulty,
        "submission": r.submission,
        "public_score": r.public_score,
        "hidden_score": r.hidden_score,
        "public_hard_gate_passed": r.public_passed,
        "hidden_hard_gate_passed": r.hidden_passed,
        "public_evaluation_valid": r.public_valid,
        "hidden_evaluation_valid": r.hidden_valid,
        "public_failure_codes": list(r.public_failure_codes or []),
        "hidden_failure_codes": list(r.hidden_failure_codes or []),
        "public_runtime_s": r.public_runtime_s,
        "hidden_runtime_s": r.hidden_runtime_s,
        "generalization_gap": r.generalization_gap(),
        "overall_score": r.overall_score(),
        "report_dir": r.report_dir,
    }
