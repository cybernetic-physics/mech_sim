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
