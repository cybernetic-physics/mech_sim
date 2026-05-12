"""Dashboard payload — a single JSON blob the static HTML dashboard
and the (future) MP4 renderer can both consume.

Shape:

::

    {
        "version": "mech_bench.dashboard_payload.v1",
        "run": { run_id, task_id, task_family, difficulty, adapter },
        "score": { dense, hard_gate_passed, evaluation_valid },
        "tier_results": { ... },
        "metrics": { ... public-allowlisted },
        "feedback": [ ... public failure cards ... ],
        "traces": {
            "coupler_path":   [[x, y], ...],
            "target_path":    [[x, y], ...],   # if available
            "input_angle":    { t: [...], theta: [...] },
            "output_angle":   { t: [...], theta: [...] },
            "ratio_over_time":{ t: [...], r: [...] }   # if computable
        },
        "media": { ... MediaManifest dict ... }
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mech_bench.media import MediaManifest
    from mech_bench.schema import EvalReport, TaskSpec
    from mech_bench.traces import TraceData


def _xy_pairs(arr: np.ndarray | None) -> list[list[float]]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        return []
    return [[float(x), float(y)] for x, y in a[:, :2]]


def _safe_list(arr: np.ndarray | None) -> list[float]:
    if arr is None:
        return []
    a = np.asarray(arr, dtype=float).reshape(-1)
    out: list[float] = []
    for v in a:
        f = float(v)
        out.append(f if math.isfinite(f) else 0.0)
    return out


def _load_target_path(task: "TaskSpec | None") -> list[list[float]]:
    if task is None or task.fixtures_dir is None:
        return []
    rel = task.objective.get("target_path_csv")
    if not rel:
        return []
    csv_path = Path(task.fixtures_dir) / str(rel)
    if not csv_path.exists():
        return []
    try:
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=float)
    except (OSError, ValueError):
        return []
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return _xy_pairs(arr[:, :2])


def build_dashboard_payload(
    report: "EvalReport",
    trace: "TraceData | None" = None,
    *,
    task: "TaskSpec | None" = None,
    media: "MediaManifest | None" = None,
) -> dict[str, Any]:
    """Build the dashboard JSON payload for one run."""
    from mech_bench.evaluator import sanitize_metric_value  # avoid cycle

    metrics_view: dict[str, Any] = {}
    for k, v in report.metrics.items():
        metrics_view[k] = sanitize_metric_value(v)

    feedback_cards: list[dict[str, Any]] = []
    for f in report.feedback:
        item = f.public() if hasattr(f, "public") else dict(f)
        feedback_cards.append(item)

    payload: dict[str, Any] = {
        "version": "mech_bench.dashboard_payload.v1",
        "run": {
            "run_id": report.run_id,
            "task_id": report.task_id,
            "task_family": report.task_family,
            "difficulty": report.difficulty,
            "adapter": getattr(trace, "adapter", "") if trace else "",
        },
        "score": {
            "dense": float(report.score),
            "hard_gate_passed": bool(report.hard_gate_passed),
            "evaluation_valid": bool(report.evaluation_valid),
        },
        "tier_results": dict(report.tier_results),
        "metrics": metrics_view,
        "feedback": feedback_cards,
        "probe_results": [
            {
                "probe_id": r.probe_id,
                "probe_type": r.probe_type,
                "passed": bool(r.passed),
                "score": float(r.score),
            }
            for r in report.probe_results
        ],
    }

    traces_block: dict[str, Any] = {}
    target_path = _load_target_path(task)
    if target_path:
        traces_block["target_path"] = target_path

    if trace is not None and not trace.is_empty():
        coupler = trace.port_traces.get("coupler_point")
        if coupler is not None:
            traces_block["coupler_path"] = _xy_pairs(coupler)
        output_pt = trace.port_traces.get("output_port")
        if output_pt is not None:
            traces_block["output_path"] = _xy_pairs(output_pt)

        t = _safe_list(trace.time_s) if trace.time_s.size else []
        input_theta = trace.joint_positions.get("input_port")
        output_theta = trace.joint_positions.get("output_port")
        if input_theta is not None and t:
            traces_block["input_angle"] = {
                "t": t,
                "theta": _safe_list(input_theta),
            }
        if output_theta is not None and t:
            traces_block["output_angle"] = {
                "t": t,
                "theta": _safe_list(output_theta),
            }
        if (input_theta is not None and output_theta is not None and t
                and len(t) == len(input_theta) == len(output_theta)):
            inp = np.asarray(input_theta, dtype=float)
            inv = np.gradient(inp, np.asarray(t, dtype=float))
            outp = np.asarray(output_theta, dtype=float)
            outv = np.gradient(outp, np.asarray(t, dtype=float))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(inv) > 1e-9, outv / inv, np.nan)
            ratio_clean = [
                float(r) if math.isfinite(float(r)) else None
                for r in ratio
            ]
            traces_block["ratio_over_time"] = {"t": t, "r": ratio_clean}

    payload["traces"] = traces_block

    if media is not None:
        payload["media"] = media.to_dict()

    return payload


def write_dashboard_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Write *payload* as JSON to *path* (creating parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# --------------------------------------------------------------------- #
# Benchmark-level payload (suite summary)                               #
# --------------------------------------------------------------------- #


def build_benchmark_dashboard_payload(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Project a ``benchmark_summary.json`` dict into a dashboard
    payload suitable for an overview page.

    The payload has these sections:

    * ``overview``      — cards: total tasks, overall score, pass rate
    * ``funnel``        — pipeline funnel: submitted → valid → gate → passed
    * ``tier_heatmap``  — per-tier score + pass-rate
    * ``family_heatmap``— per-family score + pass-rate
    * ``failure_histogram`` — code → count
    * ``score_distribution`` — histogram bins of overall_score
    * ``runtime_distribution`` — per-tier mean / total runtime
    * ``task_table``    — one row per task, link to per-task report
    """
    n = int(summary.get("n_tasks", 0) or 0)
    overall_mean = float(summary.get("overall_score_mean", 0.0) or 0.0)
    overall_median = float(summary.get("overall_score_median", 0.0) or 0.0)
    pass_rate = float(summary.get("pass_rate", 0.0) or 0.0)
    hard_gate_pass_rate = float(
        summary.get("hard_gate_pass_rate", 0.0) or 0.0)
    cap_unavail = int(summary.get("capability_unavailable_n", 0) or 0)

    tasks = summary.get("tasks") or []
    score_bins = _bin_scores(
        [t.get("overall_score", 0.0) or 0.0 for t in tasks]
    )

    # Funnel: submitted (n_tasks) → valid (eval valid) → passed gate → score>0.
    n_valid = sum(
        1 for t in tasks
        if (t.get("hidden_evaluation_valid")
            if t.get("hidden_evaluation_valid") is not None
            else t.get("public_evaluation_valid", True))
    )
    n_gate = sum(1 for t in tasks if t.get("public_hard_gate_passed")
                  or t.get("hidden_hard_gate_passed"))
    n_score = sum(1 for t in tasks
                  if (t.get("overall_score") or 0.0) > 0.0)

    return {
        "version": "mech_bench.benchmark_dashboard_payload.v1",
        "overview": {
            "n_tasks": n,
            "overall_score_mean": overall_mean,
            "overall_score_median": overall_median,
            "pass_rate": pass_rate,
            "hard_gate_pass_rate": hard_gate_pass_rate,
            "capability_unavailable_n": cap_unavail,
            "generalization_gap_mean":
                summary.get("generalization_gap_mean"),
            "public_score_mean": summary.get("public_score_mean"),
            "hidden_score_mean": summary.get("hidden_score_mean"),
        },
        "funnel": {
            "submitted": n,
            "evaluation_valid": n_valid,
            "hard_gate_passed": n_gate,
            "score_above_zero": n_score,
        },
        "tier_heatmap": _heatmap(summary.get("score_by_tier", {}),
                                  summary.get("pass_by_tier", {})),
        "family_heatmap": _heatmap(summary.get("score_by_family", {}),
                                    summary.get("pass_by_family", {})),
        "failure_histogram": dict(
            summary.get("failure_code_histogram", {}) or {}
        ),
        "score_distribution": score_bins,
        "runtime_distribution": dict(
            summary.get("runtime_by_tier", {}) or {}
        ),
        "task_table": [
            {
                "task_id": t.get("task_id"),
                "family": t.get("family"),
                "tier": t.get("tier"),
                "difficulty": t.get("difficulty"),
                "overall_score": t.get("overall_score"),
                "public_score": t.get("public_score"),
                "hidden_score": t.get("hidden_score"),
                "generalization_gap": t.get("generalization_gap"),
                "submission": t.get("submission"),
                "failure_codes": (t.get("public_failure_codes") or [])
                                  + (t.get("hidden_failure_codes") or []),
                "report_dir": t.get("report_dir"),
            }
            for t in tasks
        ],
    }


def _heatmap(score_by: dict[str, dict[str, Any]],
              pass_by: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(set(score_by) | set(pass_by))
    rows: list[dict[str, Any]] = []
    for k in keys:
        s = score_by.get(k, {})
        p = pass_by.get(k, {})
        rows.append({
            "key": k,
            "n": int(s.get("n") or p.get("n") or 0),
            "score_mean": float(s.get("mean") or 0.0),
            "pass_rate": float(p.get("pass_rate") or 0.0),
            "n_passed": int(p.get("n_passed") or 0),
        })
    return rows


def _bin_scores(scores: list[float]) -> list[dict[str, float]]:
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bins = [0 for _ in range(len(edges) - 1)]
    for s in scores:
        v = float(s)
        for i in range(len(edges) - 1):
            if v <= edges[i + 1]:
                bins[i] += 1
                break
        else:
            bins[-1] += 1
    return [
        {"low": edges[i], "high": edges[i + 1], "count": bins[i]}
        for i in range(len(edges) - 1)
    ]
