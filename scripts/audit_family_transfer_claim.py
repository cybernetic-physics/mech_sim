#!/usr/bin/env python3
"""Audit whether an RLVR run supports a family-held-out transfer claim.

The paper claim depends on holding out mechanism families, not only task
instances or seeds.  This script compares train/eval split files against
``task.toml`` metadata, summarizes family overlap, and writes paper-facing
artifacts that distinguish supported claims from overclaims.
"""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path
from typing import Any

from mech_bench.family_splits import canonical_mechanism_family


DEFAULT_RUN_DIR = "runs/rlvr_papergrade_20260525_142921"
DEFAULT_TASKS = "runs/rlvr_papergrade_20260525_142921/tasks_hard_v1"
DEFAULT_TRAIN = "runs/rlvr_papergrade_20260525_142921/hard_v1_train_visible_s0030_s0035.txt"
DEFAULT_EVAL = "runs/rlvr_papergrade_20260525_142921/hard_v1_eval_visible_s0038_s0039.txt"
DEFAULT_FINAL = "runs/rlvr_papergrade_20260525_142921/papergrade_final_clean_aggregate.json"
DEFAULT_MULTI = "runs/rlvr_papergrade_20260525_142921/papergrade_multiseed_aggregate_seed910_912.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--tasks-root", default=DEFAULT_TASKS)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN)
    parser.add_argument("--eval-split", default=DEFAULT_EVAL)
    parser.add_argument("--final-aggregate", default=DEFAULT_FINAL)
    parser.add_argument("--multiseed-aggregate", default=DEFAULT_MULTI)
    parser.add_argument("--out-json", default="docs/family_transfer_claim_audit.json")
    parser.add_argument("--out-csv", default="docs/family_transfer_claim_audit.csv")
    parser.add_argument("--out-md", default="docs/family_transfer_claim_audit.md")
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root)
    train = load_split(Path(args.train_split), tasks_root)
    eval_ = load_split(Path(args.eval_split), tasks_root)
    train_families = set(train["canonical_family_counts"])
    eval_families = set(eval_["canonical_family_counts"])
    overlap = sorted(train_families & eval_families)
    unseen = sorted(eval_families - train_families)

    final = load_json(Path(args.final_aggregate))
    multi = load_json(Path(args.multiseed_aggregate))
    conditions = summarize_conditions(final)
    comparisons = final.get("comparisons", {})
    family_rows = summarize_family_rows(multi)

    family_heldout = bool(eval_families) and not overlap
    payload = {
        "schema": "mech_bench.family_transfer_claim_audit.v1",
        "run_dir": str(Path(args.run_dir)),
        "tasks_root": str(tasks_root),
        "train_split": str(Path(args.train_split)),
        "eval_split": str(Path(args.eval_split)),
        "claim_status": (
            "supports_family_heldout_transfer"
            if family_heldout
            else "does_not_support_family_heldout_transfer"
        ),
        "supported_claim": (
            "family-held-out transfer"
            if family_heldout
            else "seed-heldout multi-family task generalization"
        ),
        "unsupported_claim": (
            None
            if family_heldout
            else "RLVR learns reusable mechanical reasoning on unseen mechanism families"
        ),
        "train": train,
        "eval": eval_,
        "train_eval_family_overlap": overlap,
        "eval_families_unseen_in_train": unseen,
        "conditions": conditions,
        "comparisons": comparisons,
        "family_rows": family_rows,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(out_csv, family_rows)
    out_md.write_text(render_markdown(payload))
    print(json.dumps({
        "claim_status": payload["claim_status"],
        "train_families": len(train_families),
        "eval_families": len(eval_families),
        "overlap_families": len(overlap),
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "out_md": str(out_md),
    }, indent=2, sort_keys=True))
    return 0


def load_split(path: Path, tasks_root: Path) -> dict[str, Any]:
    task_ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    raw_counts: dict[str, int] = {}
    canonical_counts: dict[str, int] = {}
    for task_id in task_ids:
        task_toml = tasks_root / task_id / "task.toml"
        data = tomllib.loads(task_toml.read_text())
        raw_family = str(data.get("task", {}).get("family", task_id))
        canonical = canonical_mechanism_family(raw_family)
        raw_counts[raw_family] = raw_counts.get(raw_family, 0) + 1
        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
    return {
        "path": str(path),
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "raw_family_counts": dict(sorted(raw_counts.items())),
        "canonical_family_counts": dict(sorted(canonical_counts.items())),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def summarize_conditions(final: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, condition in final.get("conditions", {}).items():
        runs = condition.get("runs", [])
        if not runs:
            continue
        n = len(runs)
        passed = sum(float(run.get("passed", 0)) for run in runs)
        total = sum(float(run.get("n_tasks", 0)) for run in runs)
        rows.append({
            "condition": name,
            "n_seeds": n,
            "mean_passed": passed / n,
            "mean_total": total / n,
            "mean_pass_rate": passed / total if total else 0.0,
            "sampler_error_count": sum(
                int(run.get("sampler_error_count", 0)) for run in runs
            ),
        })
    return rows


def summarize_family_rows(multi: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, counts in multi.get("family_counts", {}).items():
        n = int(counts.get("n", 0))
        baseline = int(counts.get("baseline_passed", 0))
        rlvr = int(counts.get("rlvr_passed", 0))
        rows.append({
            "family": family,
            "canonical_family": canonical_mechanism_family(family),
            "n": n,
            "baseline_passed": baseline,
            "rlvr_passed": rlvr,
            "baseline_pass_rate": baseline / n if n else 0.0,
            "rlvr_pass_rate": rlvr / n if n else 0.0,
            "improved": int(counts.get("improved", 0)),
            "regressed": int(counts.get("regressed", 0)),
            "both_pass": int(counts.get("both_pass", 0)),
            "both_fail": int(counts.get("both_fail", 0)),
        })
    return sorted(rows, key=lambda row: row["family"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "canonical_family",
        "n",
        "baseline_passed",
        "rlvr_passed",
        "baseline_pass_rate",
        "rlvr_pass_rate",
        "improved",
        "regressed",
        "both_pass",
        "both_fail",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(payload: dict[str, Any]) -> str:
    conditions = payload["conditions"]
    overlap = payload["train_eval_family_overlap"]
    unseen = payload["eval_families_unseen_in_train"]
    comparison = payload.get("comparisons", {}).get("rlvr_vs_ref_sft", {})
    baseline_comparison = payload.get("comparisons", {}).get(
        "rlvr_vs_clean_prompted_baseline", {}
    )
    lines = [
        "# Family Transfer Claim Audit",
        "",
        f"Claim status: `{payload['claim_status']}`.",
        "",
        f"Supported claim: {payload['supported_claim']}.",
    ]
    if payload.get("unsupported_claim"):
        lines.append(f"Unsupported claim: {payload['unsupported_claim']}.")
    lines.extend([
        "",
        "## Split Audit",
        "",
        f"- Train tasks: {payload['train']['task_count']}",
        f"- Eval tasks: {payload['eval']['task_count']}",
        f"- Train canonical families: {len(payload['train']['canonical_family_counts'])}",
        f"- Eval canonical families: {len(payload['eval']['canonical_family_counts'])}",
        f"- Train/eval overlapping canonical families: {len(overlap)}",
        f"- Eval canonical families unseen in train: {len(unseen)}",
        "",
    ])
    if overlap:
        lines.append(
            "The existing paper-grade run is not a family-held-out result: "
            "every eval family is also present in training."
        )
    else:
        lines.append("The split is family-held-out: no eval family appears in training.")
    lines.extend([
        "",
        "Overlapping families:",
        "",
    ])
    for family in overlap:
        lines.append(f"- `{family}`")
    if unseen:
        lines.extend(["", "Eval families unseen in train:", ""])
        for family in unseen:
            lines.append(f"- `{family}`")

    lines.extend([
        "",
        "## Existing Result",
        "",
    ])
    for row in conditions:
        lines.append(
            f"- `{row['condition']}`: {row['mean_passed']:.2f}/"
            f"{row['mean_total']:.0f} mean tasks passed "
            f"({100.0 * row['mean_pass_rate']:.2f}%), "
            f"sampler errors={row['sampler_error_count']}"
        )
    if baseline_comparison:
        lines.append(
            "- RLVR vs prompted baseline: "
            f"{baseline_comparison.get('mean_delta_tasks', 0):+.2f} tasks, "
            f"p={baseline_comparison.get('two_sided_sign_test_p')}"
        )
    if comparison:
        lines.append(
            "- RLVR vs SFT: "
            f"{comparison.get('mean_delta_tasks', 0):+.2f} tasks, "
            f"p={comparison.get('two_sided_sign_test_p')}"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This run is good evidence that RLVR improves verified multi-family "
        "mechanical task performance on held-out task instances/seeds. It is "
        "not evidence for transfer to unseen mechanism families, because the "
        "train and eval split files share all eval canonical families.",
        "",
        "The next required experiment is a fresh family-held-out run using the "
        "frozen split machinery, with training families disjoint from eval "
        "families and matched verifier budget across frozen, SFT, no-update, "
        "and RLVR/TTRL methods.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
