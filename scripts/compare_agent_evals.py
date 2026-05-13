#!/usr/bin/env python3
"""Side-by-side diff of two agent eval summaries.

Given the JSONs that ``run_claude_on_eval.py`` and
``run_codex_on_eval.py`` emit, produce a compact comparison table:
per-tier pass rates, per-task winners, total cost & wall-clock,
and the failure-code histogram for each side.

Usage::

    python scripts/compare_agent_evals.py \
        --left  /tmp/claude_eval_full/claude_eval_summary.json \
        --right /tmp/codex_eval_full/codex_eval_summary.json \
        --out   evals/comparison_$(date -u +%Y%m%d).json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _label(summary: dict[str, Any], fallback: str) -> str:
    agent = summary.get("agent") or fallback
    model = summary.get("model")
    return f"{agent}:{model}" if model else agent


def _index_tasks(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["task_id"]: t for t in summary.get("tasks", [])}


def _passed(t: dict[str, Any]) -> bool:
    return bool(
        t.get("eval_valid")
        and t.get("eval_hard_gate_passed")
        and (t.get("eval_score") or 0.0) > 0.0
    )


def _failure_hist(summary: dict[str, Any]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for t in summary.get("tasks", []):
        for code in t.get("eval_failure_codes") or []:
            hist[code] = hist.get(code, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def _cost_of(summary: dict[str, Any]) -> float:
    return float(
        summary.get("total_cost_usd")
        or summary.get("total_cost_usd_estimate")
        or 0.0
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="compare_agent_evals")
    p.add_argument("--left", required=True, type=Path)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--left-label", default=None)
    p.add_argument("--right-label", default=None)
    args = p.parse_args(argv)

    left = _load(args.left)
    right = _load(args.right)
    L = args.left_label or _label(left, "left")
    R = args.right_label or _label(right, "right")

    L_tasks = _index_tasks(left)
    R_tasks = _index_tasks(right)
    common = sorted(set(L_tasks) & set(R_tasks))

    rows: list[dict[str, Any]] = []
    n_L_wins = 0
    n_R_wins = 0
    n_both = 0
    n_neither = 0
    n_disagree = 0
    for tid in common:
        l = L_tasks[tid]
        r = R_tasks[tid]
        lp, rp = _passed(l), _passed(r)
        if lp and rp:
            n_both += 1
            verdict = "both"
        elif lp:
            n_L_wins += 1
            verdict = f"{L}"
            n_disagree += 1
        elif rp:
            n_R_wins += 1
            verdict = f"{R}"
            n_disagree += 1
        else:
            n_neither += 1
            verdict = "neither"
        rows.append({
            "task_id": tid,
            "tier": l.get("tier") or r.get("tier"),
            "family": l.get("family") or r.get("family"),
            "L_passed": lp,
            "L_score": l.get("eval_score"),
            "L_codes": l.get("eval_failure_codes") or [],
            "L_cost_usd": l.get("agent_cost_usd"),
            "R_passed": rp,
            "R_score": r.get("eval_score"),
            "R_codes": r.get("eval_failure_codes") or [],
            "R_cost_usd": r.get("agent_cost_usd"),
            "winner": verdict,
        })

    summary = {
        "version": "mech_bench.agent_eval_comparison.v1",
        "left": {
            "label": L,
            "path": str(args.left),
            "n_tasks": left.get("n_tasks"),
            "n_passed": left.get("n_passed"),
            "pass_rate": left.get("pass_rate"),
            "total_cost_usd": _cost_of(left),
            "wall_clock_s": left.get("wall_clock_s"),
            "by_tier": left.get("by_tier"),
            "failure_histogram": _failure_hist(left),
        },
        "right": {
            "label": R,
            "path": str(args.right),
            "n_tasks": right.get("n_tasks"),
            "n_passed": right.get("n_passed"),
            "pass_rate": right.get("pass_rate"),
            "total_cost_usd": _cost_of(right),
            "wall_clock_s": right.get("wall_clock_s"),
            "by_tier": right.get("by_tier"),
            "failure_histogram": _failure_hist(right),
        },
        "head_to_head": {
            "common_tasks": len(common),
            "both_pass": n_both,
            "neither_pass": n_neither,
            f"only_{L}_passes": n_L_wins,
            f"only_{R}_passes": n_R_wins,
            "agreement_rate": (
                (n_both + n_neither) / len(common) if common else 0.0
            ),
            "disagreements": n_disagree,
        },
        "tasks": rows,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {args.out}", file=sys.stderr)

    # Pretty-print table to stdout.
    print(f"\n{'agent':<24}  {'pass':<10}  {'cost':<10}  wall")
    for side in (summary["left"], summary["right"]):
        pr = side.get("pass_rate") or 0.0
        print(
            f"{side['label']:<24}  "
            f"{side['n_passed']}/{side['n_tasks']} "
            f"({pr*100:5.1f}%)  "
            f"${side['total_cost_usd']:>7.2f}    "
            f"{side['wall_clock_s'] or 0.0:>5.0f}s"
        )

    h2h = summary["head_to_head"]
    print(
        f"\nhead-to-head (common={h2h['common_tasks']}): both="
        f"{h2h['both_pass']}, neither={h2h['neither_pass']}, "
        f"only-{L}={h2h[f'only_{L}_passes']}, "
        f"only-{R}={h2h[f'only_{R}_passes']}, "
        f"agreement={h2h['agreement_rate']*100:.1f}%"
    )

    print("\nfailure codes (left):", summary["left"]["failure_histogram"])
    print("failure codes (right):", summary["right"]["failure_histogram"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
