#!/usr/bin/env python3
"""Analyze MechanismRepair-TTRL cell-level results.

The input is a long-form CSV or JSON rows file with one row per
split/task/seed/method cell. The script computes paired TTRL-vs-no-update
statistics and writes the machine-readable artifacts required by ``goals.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.prepare_mechanism_repair_benchmark import (
    EVAL_SEEDS as LEGACY_EVAL_SEEDS,
    PRIMARY_BASELINE as LEGACY_PRIMARY_BASELINE,
    PRIMARY_BUDGET as LEGACY_PRIMARY_BUDGET,
    PRIMARY_METHOD as LEGACY_PRIMARY_METHOD,
    REQUIRED_METHODS as LEGACY_REQUIRED_METHODS,
    SUCCESS_DELTA_PCT as LEGACY_SUCCESS_DELTA_PCT,
)

SECONDARY_METRIC_FIELDS = (
    "first_valid_verifier_call",
    "strict_score_pass_rate",
    "wrong_mobility_rate",
    "missing_port_rate",
    "ungrounded_port_rate",
    "invalid_topology_rate",
    "invalid_artifact_rate",
    "cad_pass_rate",
    "chrono_real_geometry_rate",
    "no_procedural_fallback_rate",
    "lockup_rate",
    "contact_lockup_rate",
    "best_ratio_error_pct",
    "best_path_trace_error",
    "best_max_penetration_mm",
    "best_contact_force_rms_N",
    "adapter_updates",
    "trained_tokens",
    "rl_trained_tokens",
    "n_rl_datums",
    "sampler_error_count",
    "sampler_http_400_count",
    "sampler_retry_count",
    "invalid_artifact_count",
    "timeout_count",
    "audit_retry_count",
)

PHYSICS_HIDDEN_VARIANT_SPLIT = "hidden_perturbation"
PHYSICS_ANTI_SHORTCUT_SPLITS = ("hidden_perturbation", "external_style")
PHYSICS_MIN_POSITIVE_FAMILIES = 8


@dataclass(frozen=True)
class AnalysisContract:
    schema: str
    primary_method: str
    primary_baseline: str
    required_methods: tuple[str, ...]
    primary_budget: int
    eval_seeds: tuple[int, ...]
    success_delta_pct: float
    required_min_trace_pairs: int
    learning_methods: tuple[str, ...]


DEFAULT_CONTRACT = AnalysisContract(
    schema="mechanism_repair_ttrl.analysis_contract.v1",
    primary_method=LEGACY_PRIMARY_METHOD,
    primary_baseline=LEGACY_PRIMARY_BASELINE,
    required_methods=tuple(LEGACY_REQUIRED_METHODS),
    primary_budget=int(LEGACY_PRIMARY_BUDGET),
    eval_seeds=tuple(int(seed) for seed in LEGACY_EVAL_SEEDS),
    success_delta_pct=float(LEGACY_SUCCESS_DELTA_PCT),
    required_min_trace_pairs=8,
    learning_methods=(LEGACY_PRIMARY_METHOD,),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="runs/mechanism_repair_ttrl_final")
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="directory containing method_manifest.json and split manifests; "
             "when present, the claim audit rejects incomplete coverage",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    benchmark_dir = (
        Path(args.benchmark_dir).expanduser().resolve()
        if args.benchmark_dir
        else default_benchmark_dir(out_dir)
    )
    contract = (
        load_analysis_contract(benchmark_dir)
        if benchmark_dir is not None
        else DEFAULT_CONTRACT
    )
    expected = (
        build_expected_coverage(benchmark_dir, contract=contract)
        if benchmark_dir is not None
        else None
    )
    benchmark_readiness = (
        build_benchmark_readiness(benchmark_dir)
        if benchmark_dir is not None
        else {"enforced": False}
    )
    rows = normalize_rows(load_rows(Path(args.results).expanduser().resolve()))
    stats = analyze_rows(
        rows,
        bootstrap_samples=max(100, int(args.bootstrap_samples)),
        seed=int(args.seed),
        expected_coverage=expected,
        benchmark_readiness=benchmark_readiness,
        contract=contract,
    )
    failure_analysis = build_failure_analysis(rows, contract=contract)
    trace_pairs = build_trace_pairs(rows, contract=contract)
    repair_taxonomy = build_repair_taxonomy(rows)
    claim_audit = build_claim_audit(stats)
    stats["claim_status"] = claim_audit["claim_status"]
    stats["analysis_claim_audit"] = claim_audit

    write_json(out_dir / "stats.json", stats)
    write_json(out_dir / "failure_analysis.json", failure_analysis)
    write_json(out_dir / "trace_pairs.json", {"pairs": trace_pairs})
    write_json(out_dir / "repair_taxonomy.json", repair_taxonomy)
    write_json(out_dir / "claim_audit.json", claim_audit)
    print(json.dumps({
        "stats": str(out_dir / "stats.json"),
        "failure_analysis": str(out_dir / "failure_analysis.json"),
        "trace_pairs": str(out_dir / "trace_pairs.json"),
        "repair_taxonomy": str(out_dir / "repair_taxonomy.json"),
        "claim_audit": str(out_dir / "claim_audit.json"),
        "claim_status": claim_audit["claim_status"],
        "blockers": claim_audit["blockers"],
    }, indent=2, sort_keys=True))
    return 0 if claim_audit["claim_status"] == "supports_primary_hypothesis" else 2


def load_analysis_contract(benchmark_dir: Path) -> AnalysisContract:
    manifest_path = benchmark_dir / "method_manifest.json"
    if not manifest_path.is_file():
        return DEFAULT_CONTRACT
    manifest = json.loads(manifest_path.read_text())
    required = tuple(
        str(method)
        for method in manifest.get("required_methods", DEFAULT_CONTRACT.required_methods)
    )
    primary_method = str(
        manifest.get("primary_method")
        or (
            "mechanical_evolve_ttrl_tool_verified"
            if "mechanical_evolve_ttrl_tool_verified" in required
            else DEFAULT_CONTRACT.primary_method
        )
    )
    primary_baseline = str(
        manifest.get("primary_baseline") or DEFAULT_CONTRACT.primary_baseline
    )
    primary_budget = int(
        manifest.get("primary_budget_expensive_verifier_calls")
        or manifest.get("primary_budget_verifier_calls")
        or DEFAULT_CONTRACT.primary_budget
    )
    eval_seeds = tuple(
        int(seed)
        for seed in manifest.get("eval_seeds", DEFAULT_CONTRACT.eval_seeds)
    )
    threshold = manifest.get("success_threshold") or {}
    success_delta_pct = float(
        threshold.get("level23_success_abs_delta_pct")
        or threshold.get("success_delta_pct")
        or manifest.get("success_delta_pct")
        or DEFAULT_CONTRACT.success_delta_pct
    )
    is_physics = str(manifest.get("schema", "")).startswith(
        "mechanism_repair_physics."
    )
    learning_methods = tuple(
        method
        for method in required
        if method.startswith("mechanical_evolve_ttrl")
    ) or (primary_method,)
    return AnalysisContract(
        schema=(
            "mechanism_repair_physics.analysis_contract.v1"
            if is_physics
            else DEFAULT_CONTRACT.schema
        ),
        primary_method=primary_method,
        primary_baseline=primary_baseline,
        required_methods=required,
        primary_budget=primary_budget,
        eval_seeds=eval_seeds,
        success_delta_pct=success_delta_pct,
        required_min_trace_pairs=24 if is_physics else DEFAULT_CONTRACT.required_min_trace_pairs,
        learning_methods=learning_methods,
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        for key in ("cells", "rows", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value]
    raise ValueError(f"unsupported results shape in {path}")


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        method = str(row.get("method") or "")
        split = str(row.get("split") or row.get("split_name") or "")
        task_id = str(row.get("task_id") or "")
        family = str(row.get("family") or row.get("canonical_family") or "")
        seed = int(float(row.get("seed", 0)))
        verifier_level = int(float(row.get("verifier_level", 0) or 0))
        success = bool_value(
            row.get("verified_repair_success_at_32",
                    row.get("verified_repair_success",
                            row.get("verifier_valid_passed",
                                    row.get("strict_passed", False))))
        )
        reward = float_value(
            row.get("best_verified_reward_at_32",
                    row.get("best_verified_reward",
                            row.get("verified_score", 0.0)))
        )
        verifier_calls = int(float_value(
            row.get("verifier_calls", row.get("actual_verifier_calls", 0))
        ))
        cad_audits = int(float_value(
            row.get("cad_audits", row.get("actual_cad_calls", 0))
        ))
        chrono_audits = int(float_value(
            row.get("chrono_audits", row.get("actual_chrono_calls", 0))
        ))
        normalized.append({
            **row,
            "method": method,
            "split": split,
            "task_id": task_id,
            "family": family,
            "seed": seed,
            "verifier_level": verifier_level,
            "verified_repair_success_at_32": success,
            "best_verified_reward_at_32": reward,
            "verifier_calls": verifier_calls,
            "cad_audits": cad_audits,
            "chrono_audits": chrono_audits,
            "_pair_key": (split, family, task_id, seed),
        })
    return normalized


def is_physics_contract(contract: AnalysisContract) -> bool:
    return str(contract.schema).startswith("mechanism_repair_physics.")


def headline_metric_rows(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract,
) -> list[dict[str, Any]]:
    if not is_physics_contract(contract):
        return rows
    return [
        row for row in rows
        if int(row.get("verifier_level", 0) or 0) >= 2
    ]


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    expected_coverage: dict[str, Any] | None = None,
    benchmark_readiness: dict[str, Any] | None = None,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    headline_rows = headline_metric_rows(rows, contract=contract)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in headline_rows:
        by_method[row["method"]].append(row)

    method_summary = {
        method: summarize_method(items)
        for method, items in sorted(by_method.items())
    }
    paired = paired_primary_rows(headline_rows, contract=contract)
    success_deltas = [
        float(p["primary_success"]) - float(p["baseline_success"])
        for p in paired
    ]
    reward_deltas = [
        float(p["primary_reward"]) - float(p["baseline_reward"])
        for p in paired
    ]
    success_mean = mean(success_deltas)
    reward_mean = mean(reward_deltas)
    success_ci = bootstrap_ci(success_deltas, bootstrap_samples, seed)
    reward_ci = bootstrap_ci(reward_deltas, bootstrap_samples, seed + 1)
    sign_p = one_sided_sign_test(success_deltas)
    family_rows = family_delta_rows(paired)
    leave_one = leave_one_family_out(paired)
    split_rows = split_delta_rows(paired)
    anti_shortcut = anti_shortcut_comparison(paired)
    method_comparisons = paired_method_comparisons(headline_rows, contract=contract)
    budget_audit = audit_budget(rows, contract=contract)
    evidence_audit = audit_evidence(rows, contract=contract)
    learning_audit = audit_ttrl_learning(rows, contract=contract)
    coverage_audit = audit_expected_coverage(rows, expected_coverage)
    family_method_summary = summarize_family_methods(headline_rows)
    primary_result_table = build_primary_result_table(
        headline_rows,
        contract=contract,
    )
    required_present = all(
        method in by_method for method in contract.required_methods
    )
    reward_beats = reward_beats_all_required(method_summary, contract=contract)

    return {
        "schema": "mechanism_repair_ttrl.stats.v1",
        "analysis_contract": {
            "schema": contract.schema,
            "primary_method": contract.primary_method,
            "primary_baseline": contract.primary_baseline,
            "required_methods": list(contract.required_methods),
            "primary_budget": contract.primary_budget,
            "eval_seeds": list(contract.eval_seeds),
            "success_delta_pct": contract.success_delta_pct,
            "required_min_trace_pairs": contract.required_min_trace_pairs,
            "learning_methods": list(contract.learning_methods),
        },
        "primary_method": contract.primary_method,
        "primary_baseline": contract.primary_baseline,
        "required_methods": list(contract.required_methods),
        "required_methods_present": required_present,
        "missing_required_methods": [
            method for method in contract.required_methods if method not in by_method
        ],
        "primary_budget_verifier_calls": contract.primary_budget,
        "eval_seeds_expected": list(contract.eval_seeds),
        "n_rows": len(rows),
        "headline_metric_rows": len(headline_rows),
        "non_headline_metric_rows": len(rows) - len(headline_rows),
        "headline_metric_filter": (
            "verifier_level>=2"
            if is_physics_contract(contract)
            else "all_rows"
        ),
        "n_paired_cells": len(paired),
        "method_summary": method_summary,
        "family_method_summary": family_method_summary,
        "primary_result_table": primary_result_table,
        "primary_comparison": {
            "success_delta_mean": success_mean,
            "success_delta_pct": success_mean * 100.0,
            "success_delta_ci95": success_ci,
            "success_sign_test_p_one_sided": sign_p,
            "reward_delta_mean": reward_mean,
            "reward_delta_ci95": reward_ci,
            "success_threshold_pct": contract.success_delta_pct,
        },
        "family_deltas": family_rows,
        "leave_one_family_out": leave_one,
        "split_deltas": split_rows,
        "anti_shortcut_comparison": anti_shortcut,
        "paired_method_comparisons": method_comparisons,
        "budget_audit": budget_audit,
        "evidence_audit": evidence_audit,
        "learning_audit": learning_audit,
        "coverage_audit": coverage_audit,
        "benchmark_readiness_audit": (
            benchmark_readiness
            if benchmark_readiness is not None
            else {"enforced": False}
        ),
        "reward_beats_all_required_baselines": reward_beats,
    }


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "verified_repair_success_at_32": mean([
            float(row["verified_repair_success_at_32"]) for row in rows
        ]),
        "best_verified_reward_at_32": mean([
            float(row["best_verified_reward_at_32"]) for row in rows
        ]),
        "verifier_calls_mean": mean([
            float(row["verifier_calls"]) for row in rows
        ]),
        "cad_audits_mean": mean([float(row["cad_audits"]) for row in rows]),
        "chrono_audits_mean": mean([
            float(row["chrono_audits"]) for row in rows
        ]),
        "secondary_metrics": summarize_secondary_metrics(rows),
    }


def summarize_family_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["family"]))].append(row)
    return [
        {
            "method": method,
            "family": family,
            "n": len(items),
            "verified_repair_success_at_32": mean([
                float(item["verified_repair_success_at_32"])
                for item in items
            ]),
            "best_verified_reward_at_32": mean([
                float(item["best_verified_reward_at_32"]) for item in items
            ]),
            "secondary_metrics": summarize_secondary_metrics(items),
        }
        for (method, family), items in sorted(grouped.items())
    ]


def summarize_secondary_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        field: summarize_metric_field(rows, field)
        for field in SECONDARY_METRIC_FIELDS
    }


def summarize_metric_field(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return {
        "n_present": len(values),
        "mean": mean(values) if values else None,
    }


def paired_primary_rows(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> list[dict[str, Any]]:
    by_key_method: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key_method[row["_pair_key"]][row["method"]] = row
    paired: list[dict[str, Any]] = []
    for key, methods in sorted(by_key_method.items()):
        primary = methods.get(contract.primary_method)
        baseline = methods.get(contract.primary_baseline)
        if primary is None or baseline is None:
            continue
        split, family, task_id, seed = key
        paired.append({
            "split": split,
            "family": family,
            "task_id": task_id,
            "seed": seed,
            "primary_success": primary["verified_repair_success_at_32"],
            "baseline_success": baseline["verified_repair_success_at_32"],
            "primary_reward": primary["best_verified_reward_at_32"],
            "baseline_reward": baseline["best_verified_reward_at_32"],
        })
    return paired


def family_delta_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_family[str(row["family"])].append(row)
    out: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        success_delta = mean([
            float(row["primary_success"]) - float(row["baseline_success"])
            for row in rows
        ])
        reward_delta = mean([
            float(row["primary_reward"]) - float(row["baseline_reward"])
            for row in rows
        ])
        out.append({
            "family": family,
            "n": len(rows),
            "success_delta": success_delta,
            "success_delta_pct": success_delta * 100.0,
            "reward_delta": reward_delta,
            "ttrl_wins_reward": reward_delta > 0,
        })
    return out


def leave_one_family_out(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row["family"]) for row in paired})
    out: list[dict[str, Any]] = []
    for family in families:
        kept = [row for row in paired if row["family"] != family]
        success_delta = mean([
            float(row["primary_success"]) - float(row["baseline_success"])
            for row in kept
        ])
        reward_delta = mean([
            float(row["primary_reward"]) - float(row["baseline_reward"])
            for row in kept
        ])
        out.append({
            "removed_family": family,
            "n": len(kept),
            "success_delta": success_delta,
            "success_delta_pct": success_delta * 100.0,
            "keeps_positive_success_delta": success_delta > 0,
            "reward_delta": reward_delta,
            "keeps_positive_sign": reward_delta > 0,
        })
    return out


def split_delta_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_split[str(row["split"])].append(row)
    out: list[dict[str, Any]] = []
    for split, rows in sorted(by_split.items()):
        success_delta = mean([
            float(row["primary_success"]) - float(row["baseline_success"])
            for row in rows
        ])
        reward_delta = mean([
            float(row["primary_reward"]) - float(row["baseline_reward"])
            for row in rows
        ])
        out.append({
            "split": split,
            "n": len(rows),
            "success_delta": success_delta,
            "success_delta_pct": success_delta * 100.0,
            "reward_delta": reward_delta,
        })
    return out


def anti_shortcut_comparison(paired: list[dict[str, Any]]) -> dict[str, Any]:
    anti_splits = set(PHYSICS_ANTI_SHORTCUT_SPLITS)
    rows = [row for row in paired if str(row["split"]) in anti_splits]
    success_delta = mean([
        float(row["primary_success"]) - float(row["baseline_success"])
        for row in rows
    ])
    reward_delta = mean([
        float(row["primary_reward"]) - float(row["baseline_reward"])
        for row in rows
    ])
    return {
        "splits": sorted(anti_splits),
        "n_paired_cells": len(rows),
        "anti_shortcut_pass_rate_delta": success_delta,
        "anti_shortcut_pass_rate_delta_pct": success_delta * 100.0,
        "reward_delta_mean": reward_delta,
    }


def split_delta(split_deltas: list[dict[str, Any]], split: str) -> float | None:
    for row in split_deltas:
        if str(row.get("split") or "") == split:
            return float(row.get("success_delta", 0.0) or 0.0)
    return None


def paired_method_comparisons(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, dict[str, Any]]:
    by_key_method: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key_method[row["_pair_key"]][row["method"]] = row
    comparisons: dict[str, dict[str, Any]] = {}
    for method in contract.required_methods:
        if method == contract.primary_method:
            continue
        success_deltas: list[float] = []
        reward_deltas: list[float] = []
        for methods in by_key_method.values():
            primary = methods.get(contract.primary_method)
            baseline = methods.get(method)
            if primary is None or baseline is None:
                continue
            success_deltas.append(
                float(primary["verified_repair_success_at_32"])
                - float(baseline["verified_repair_success_at_32"])
            )
            reward_deltas.append(
                float(primary["best_verified_reward_at_32"])
                - float(baseline["best_verified_reward_at_32"])
            )
        success_delta = mean(success_deltas)
        reward_delta = mean(reward_deltas)
        comparisons[method] = {
            "n_paired_cells": len(success_deltas),
            "success_delta_mean": success_delta,
            "success_delta_pct": success_delta * 100.0,
            "reward_delta_mean": reward_delta,
            "primary_beats_on_success": bool(success_deltas) and success_delta > 0,
            "primary_beats_on_reward": bool(reward_deltas) and reward_delta > 0,
        }
    return comparisons


def build_primary_result_table(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    anti_splits = set(PHYSICS_ANTI_SHORTCUT_SPLITS)
    table: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        hidden_rows = [
            row for row in method_rows
            if str(row["split"]) == PHYSICS_HIDDEN_VARIANT_SPLIT
        ]
        anti_rows = [
            row for row in method_rows if str(row["split"]) in anti_splits
        ]
        table.append({
            "method": method,
            "n": len(method_rows),
            "is_primary_method": method == contract.primary_method,
            "level23_verified_repair_success_at_32": mean([
                float(row["verified_repair_success_at_32"])
                for row in method_rows
            ]),
            "hidden_variant_success_at_32": mean_or_none([
                float(row["verified_repair_success_at_32"])
                for row in hidden_rows
            ]),
            "anti_shortcut_pass_rate_at_32": mean_or_none([
                float(row["verified_repair_success_at_32"])
                for row in anti_rows
            ]),
            "best_verified_reward_at_32": mean([
                float(row["best_verified_reward_at_32"])
                for row in method_rows
            ]),
            "actual_verifier_calls": mean([
                float(row["verifier_calls"]) for row in method_rows
            ]),
            "actual_cad_calls": mean([
                float(row["cad_audits"]) for row in method_rows
            ]),
            "actual_chrono_calls": mean([
                float(row["chrono_audits"]) for row in method_rows
            ]),
        })
    return table


def reward_beats_all_required(
    method_summary: dict[str, dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> bool:
    primary = method_summary.get(contract.primary_method)
    if not primary:
        return False
    primary_reward = float(primary["best_verified_reward_at_32"])
    for method in contract.required_methods:
        if method == contract.primary_method:
            continue
        summary = method_summary.get(method)
        if summary is None:
            return False
        if primary_reward <= float(summary["best_verified_reward_at_32"]):
            return False
    return True


def audit_budget(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split, family, task_id, seed = row["_pair_key"]
        by_cell[(split, family, task_id, seed)].append(row)
    mismatches: list[dict[str, Any]] = []
    primary_expensive_budget_excesses: list[dict[str, Any]] = []
    wrong_primary_budget: list[dict[str, Any]] = []
    sampler_accounting: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "sampler_error_count": 0,
            "sampler_http_400_count": 0,
            "sampler_retry_count": 0,
            "audit_retry_count": 0,
        }
    )
    for key, cell_rows in sorted(by_cell.items()):
        verifier = {row["method"]: int(row["verifier_calls"]) for row in cell_rows}
        cad = {row["method"]: int(row["cad_audits"]) for row in cell_rows}
        chrono = {row["method"]: int(row["chrono_audits"]) for row in cell_rows}
        for row in cell_rows:
            method = str(row["method"])
            sampler_accounting[method]["sampler_error_count"] += int(
                row.get("sampler_error_count", 0) or 0
            )
            sampler_accounting[method]["sampler_http_400_count"] += int(
                row.get("sampler_http_400_count", 0) or 0
            )
            sampler_accounting[method]["sampler_retry_count"] += int(
                row.get("sampler_retry_count", 0) or 0
            )
            sampler_accounting[method]["audit_retry_count"] += int(
                row.get("audit_retry_count", 0) or 0
            )
        for method, calls in verifier.items():
            if calls != contract.primary_budget:
                wrong_primary_budget.append({
                    "cell": list(key),
                    "method": method,
                    "verifier_calls": calls,
                    "expected": contract.primary_budget,
                })
        if len(set(verifier.values())) > 1:
            mismatches.append({"cell": list(key), "kind": "verifier", "values": verifier})
        primary = contract.primary_method
        baseline = contract.primary_baseline
        if primary in cad and baseline in cad and cad[primary] > cad[baseline]:
            primary_expensive_budget_excesses.append({
                "cell": list(key),
                "kind": "cad",
                "primary_method": primary,
                "primary_calls": cad[primary],
                "baseline_method": baseline,
                "baseline_calls": cad[baseline],
                "values": cad,
            })
        if (
            primary in chrono
            and baseline in chrono
            and chrono[primary] > chrono[baseline]
        ):
            primary_expensive_budget_excesses.append({
                "cell": list(key),
                "kind": "chrono",
                "primary_method": primary,
                "primary_calls": chrono[primary],
                "baseline_method": baseline,
                "baseline_calls": chrono[baseline],
                "values": chrono,
            })
    return {
        "n_cells": len(by_cell),
        "budget_matched": not mismatches and not primary_expensive_budget_excesses,
        "primary_budget_spent": not wrong_primary_budget,
        "mismatches": mismatches[:100],
        "n_mismatches": len(mismatches),
        "wrong_primary_budget": wrong_primary_budget[:100],
        "n_wrong_primary_budget": len(wrong_primary_budget),
        "primary_expensive_budget_not_more_than_baseline": (
            not primary_expensive_budget_excesses
        ),
        "primary_expensive_budget_excesses": (
            primary_expensive_budget_excesses[:100]
        ),
        "n_primary_expensive_budget_excesses": (
            len(primary_expensive_budget_excesses)
        ),
        "sampler_accounting_by_method": dict(sorted(sampler_accounting.items())),
    }


def audit_evidence(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    missing_raw_rows: list[dict[str, Any]] = []
    missing_verifier_rows: list[dict[str, Any]] = []
    missing_raw_files: list[dict[str, Any]] = []
    missing_verifier_files: list[dict[str, Any]] = []
    verifier_count_mismatches: list[dict[str, Any]] = []
    missing_summary_files: list[dict[str, Any]] = []
    missing_ttrl_training_logs: list[dict[str, Any]] = []
    missing_ttrl_adapters: list[dict[str, Any]] = []

    for row in rows:
        ref = row_ref(row)
        raw_paths = parse_paths(row.get("raw_completion_paths", []))
        verifier_paths = parse_paths(row.get("verifier_output_paths", []))
        if not raw_paths:
            missing_raw_rows.append(ref)
        if not verifier_paths:
            missing_verifier_rows.append(ref)
        for path in raw_paths:
            if not Path(path).is_file():
                missing_raw_files.append({**ref, "path": path})
        for path in verifier_paths:
            if not Path(path).is_file():
                missing_verifier_files.append({**ref, "path": path})
        verifier_calls = int(row.get("verifier_calls", 0) or 0)
        if verifier_calls and len(verifier_paths) != verifier_calls:
            verifier_count_mismatches.append({
                **ref,
                "verifier_calls": verifier_calls,
                "verifier_output_paths": len(verifier_paths),
            })
        summary_path = str(row.get("summary_path") or "")
        if summary_path and not Path(summary_path).is_file():
            missing_summary_files.append({**ref, "path": summary_path})
        if row["method"] in contract.learning_methods:
            trace_path = str(row.get("trace_path") or "")
            if not trace_path or not Path(trace_path).is_file():
                missing_ttrl_training_logs.append({**ref, "path": trace_path})
            adapter_path = str(row.get("adapter_path") or "")
            if not adapter_path or not Path(adapter_path).exists():
                missing_ttrl_adapters.append({**ref, "path": adapter_path})

    paired_with_evidence = count_paired_trace_evidence(rows, contract=contract)
    ok = not any([
        missing_raw_rows,
        missing_verifier_rows,
        missing_raw_files,
        missing_verifier_files,
        verifier_count_mismatches,
        missing_summary_files,
        missing_ttrl_training_logs,
        missing_ttrl_adapters,
    ])
    return {
        "raw_completions_present": not missing_raw_rows and not missing_raw_files,
        "verifier_outputs_present": (
            not missing_verifier_rows
            and not missing_verifier_files
            and not verifier_count_mismatches
        ),
        "training_logs_present": not missing_ttrl_training_logs,
        "adapter_checkpoints_present": not missing_ttrl_adapters,
        "matched_ttrl_vs_no_update_trace_pairs_with_evidence": paired_with_evidence,
        "required_min_trace_pairs": contract.required_min_trace_pairs,
        "evidence_complete": (
            ok and paired_with_evidence >= contract.required_min_trace_pairs
        ),
        "n_missing_raw_rows": len(missing_raw_rows),
        "n_missing_verifier_rows": len(missing_verifier_rows),
        "n_missing_raw_files": len(missing_raw_files),
        "n_missing_verifier_files": len(missing_verifier_files),
        "n_verifier_count_mismatches": len(verifier_count_mismatches),
        "n_missing_summary_files": len(missing_summary_files),
        "n_missing_ttrl_training_logs": len(missing_ttrl_training_logs),
        "n_missing_ttrl_adapters": len(missing_ttrl_adapters),
        "missing_raw_rows": missing_raw_rows[:100],
        "missing_verifier_rows": missing_verifier_rows[:100],
        "missing_raw_files": missing_raw_files[:100],
        "missing_verifier_files": missing_verifier_files[:100],
        "verifier_count_mismatches": verifier_count_mismatches[:100],
        "missing_summary_files": missing_summary_files[:100],
        "missing_ttrl_training_logs": missing_ttrl_training_logs[:100],
        "missing_ttrl_adapters": missing_ttrl_adapters[:100],
    }


def audit_ttrl_learning(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    ttrl_rows = [
        row for row in rows if row["method"] in contract.learning_methods
    ]
    bad_rows: list[dict[str, Any]] = []
    for row in ttrl_rows:
        adapter_updates = int(row.get("adapter_updates", 0) or 0)
        trained_tokens = int(row.get("trained_tokens", 0) or 0)
        rl_trained_tokens = int(row.get("rl_trained_tokens", 0) or 0)
        n_rl_datums = int(row.get("n_rl_datums", 0) or 0)
        min_rl_datums = max(
            contract.primary_budget,
            int(row.get("verifier_calls", 0) or 0),
        )
        reasons = []
        if adapter_updates <= 0:
            reasons.append("adapter_updates<=0")
        if trained_tokens <= 0:
            reasons.append("trained_tokens<=0")
        if rl_trained_tokens <= 0:
            reasons.append("rl_trained_tokens<=0")
        if n_rl_datums < min_rl_datums:
            reasons.append(f"n_rl_datums<{min_rl_datums}")
        if reasons:
            bad_rows.append({
                **row_ref(row),
                "adapter_updates": adapter_updates,
                "trained_tokens": trained_tokens,
                "rl_trained_tokens": rl_trained_tokens,
                "n_rl_datums": n_rl_datums,
                "reasons": reasons,
            })
    return {
        "ttrl_rows": len(ttrl_rows),
        "ttrl_learning_evidence_complete": bool(ttrl_rows) and not bad_rows,
        "n_bad_ttrl_learning_rows": len(bad_rows),
        "bad_ttrl_learning_rows": bad_rows[:100],
    }


def count_paired_trace_evidence(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> int:
    by_key_method: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key_method[row["_pair_key"]][row["method"]] = row
    count = 0
    for methods in by_key_method.values():
        primary = methods.get(contract.primary_method)
        baseline = methods.get(contract.primary_baseline)
        if not primary or not baseline:
            continue
        if row_has_evidence(primary, require_training=True) and row_has_evidence(
            baseline,
            require_training=False,
        ):
            count += 1
    return count


def row_has_evidence(row: dict[str, Any], *, require_training: bool) -> bool:
    raw_paths = parse_paths(row.get("raw_completion_paths", []))
    verifier_paths = parse_paths(row.get("verifier_output_paths", []))
    if not raw_paths or not verifier_paths:
        return False
    if any(not Path(path).is_file() for path in raw_paths):
        return False
    if any(not Path(path).is_file() for path in verifier_paths):
        return False
    verifier_calls = int(row.get("verifier_calls", 0) or 0)
    if verifier_calls and len(verifier_paths) != verifier_calls:
        return False
    if require_training:
        trace_path = str(row.get("trace_path") or "")
        adapter_path = str(row.get("adapter_path") or "")
        if not trace_path or not Path(trace_path).is_file():
            return False
        if not adapter_path or not Path(adapter_path).exists():
            return False
    return True


def row_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": str(row.get("split") or ""),
        "task_id": str(row.get("task_id") or ""),
        "seed": int(row.get("seed", 0) or 0),
        "method": str(row.get("method") or ""),
    }


def build_failure_analysis(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    counts: Counter[tuple[str, str, str]] = Counter()
    method_family: Counter[tuple[str, str]] = Counter()
    for row in rows:
        method = str(row["method"])
        family = str(row["family"])
        method_family[(method, family)] += 1
        for code in parse_codes(row.get("failure_codes", "")):
            counts[(method, family, code)] += 1
    return {
        "schema": "mechanism_repair_ttrl.failure_analysis.v1",
        "counts": [
            {"method": method, "family": family, "failure_code": code, "n": n}
            for (method, family, code), n in sorted(counts.items())
        ],
        "first_to_final_attempt_changes": first_to_final_attempt_changes(rows),
        "ttrl_vs_no_update_failure_deltas": failure_code_deltas(
            counts,
            method_family,
            primary=contract.primary_method,
            baseline=contract.primary_baseline,
        ),
        "repair_dimension_deltas": repair_dimension_deltas(
            rows,
            contract=contract,
        ),
    }


def build_trace_pairs(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> list[dict[str, Any]]:
    by_key_method: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key_method[row["_pair_key"]][row["method"]] = row
    pairs: list[dict[str, Any]] = []
    for key, methods in sorted(by_key_method.items()):
        primary = methods.get(contract.primary_method)
        baseline = methods.get(contract.primary_baseline)
        if not primary or not baseline:
            continue
        split, family, task_id, seed = key
        pairs.append({
            "split": split,
            "family": family,
            "task_id": task_id,
            "seed": seed,
            "primary_success": primary["verified_repair_success_at_32"],
            "baseline_success": baseline["verified_repair_success_at_32"],
            "primary_failure_codes": parse_codes(primary.get("failure_codes", "")),
            "baseline_failure_codes": parse_codes(baseline.get("failure_codes", "")),
            "primary_trace_path": primary.get("trace_path", ""),
            "baseline_trace_path": baseline.get("trace_path", ""),
            "primary_raw_completion_paths": parse_paths(
                primary.get("raw_completion_paths", [])
            ),
            "baseline_raw_completion_paths": parse_paths(
                baseline.get("raw_completion_paths", [])
            ),
            "primary_verifier_output_paths": parse_paths(
                primary.get("verifier_output_paths", [])
            ),
            "baseline_verifier_output_paths": parse_paths(
                baseline.get("verifier_output_paths", [])
            ),
            "same_verifier_budget": (
                int(primary.get("verifier_calls", 0) or 0)
                == int(baseline.get("verifier_calls", 0) or 0)
            ),
            "verifier_calls": {
                contract.primary_method: int(primary.get("verifier_calls", 0) or 0),
                contract.primary_baseline: int(baseline.get("verifier_calls", 0) or 0),
            },
            "repair_dimension_delta": repair_dimension_delta(primary, baseline),
        })
    return pairs


def build_repair_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method_family_dimension: Counter[tuple[str, str, str]] = Counter()
    by_dimension: Counter[str] = Counter()
    resolved_by_dimension: Counter[str] = Counter()
    for row in rows:
        method = str(row.get("method") or "")
        family = str(row.get("family") or "")
        for code in parse_codes(row.get("failure_codes", "")):
            dimension = repair_dimension_for_failure_code(code)
            by_dimension[dimension] += 1
            by_method_family_dimension[(method, family, dimension)] += 1
        paths = parse_paths(row.get("verifier_output_paths", []))
        if len(paths) < 2:
            continue
        first = load_json_path(paths[0])
        final = load_json_path(paths[-1])
        if first is None or final is None:
            continue
        first_codes = set(parse_codes(first.get("failure_codes", [])))
        final_codes = set(parse_codes(final.get("failure_codes", [])))
        for code in sorted(first_codes - final_codes):
            resolved_by_dimension[repair_dimension_for_failure_code(code)] += 1
    return {
        "schema": "mechanism_repair_ttrl.repair_taxonomy.v1",
        "dimension_counts": [
            {"dimension": dimension, "n": n}
            for dimension, n in sorted(by_dimension.items())
        ],
        "resolved_dimension_counts": [
            {"dimension": dimension, "n": n}
            for dimension, n in sorted(resolved_by_dimension.items())
        ],
        "method_family_dimension_counts": [
            {
                "method": method,
                "family": family,
                "dimension": dimension,
                "n": n,
            }
            for (method, family, dimension), n in sorted(
                by_method_family_dimension.items()
            )
        ],
        "dimension_map": {
            "topology_mobility": [
                "mobility",
                "topology",
                "ground",
                "joint",
                "dof",
            ],
            "interface": ["port", "interface", "io"],
            "functional_behavior": [
                "ratio",
                "stroke",
                "path",
                "timing",
                "travel",
                "index",
            ],
            "cad_artifact": [
                "cad",
                "artifact",
                "geometry",
                "mass",
                "material",
                "watertight",
                "manifold",
            ],
            "physics_contact": [
                "chrono",
                "contact",
                "penetration",
                "lockup",
                "force",
                "torque",
                "power",
            ],
            "runtime_or_sampling": ["timeout", "sampler", "parse", "execution"],
        },
    }


def repair_dimension_for_failure_code(code: str) -> str:
    text = code.lower()
    buckets = [
        (
            "physics_contact",
            ("chrono", "contact", "penetration", "lockup", "force", "torque", "power"),
        ),
        (
            "cad_artifact",
            ("cad", "artifact", "geometry", "mass", "material", "watertight", "manifold"),
        ),
        (
            "topology_mobility",
            ("mobility", "topology", "ground", "joint", "dof"),
        ),
        ("interface", ("port", "interface", "io")),
        (
            "functional_behavior",
            ("ratio", "stroke", "path", "timing", "travel", "index"),
        ),
        ("runtime_or_sampling", ("timeout", "sampler", "parse", "execution")),
    ]
    for dimension, needles in buckets:
        if any(needle in text for needle in needles):
            return dimension
    return "other"


def first_to_final_attempt_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for row in rows:
        paths = parse_paths(row.get("verifier_output_paths", []))
        if len(paths) < 2:
            continue
        first = load_json_path(paths[0])
        final = load_json_path(paths[-1])
        if first is None or final is None:
            continue
        first_codes = parse_codes(first.get("failure_codes", []))
        final_codes = parse_codes(final.get("failure_codes", []))
        first_set = set(first_codes)
        final_set = set(final_codes)
        changes.append({
            "split": row["split"],
            "family": row["family"],
            "task_id": row["task_id"],
            "seed": row["seed"],
            "method": row["method"],
            "attempts": len(paths),
            "first_failure_codes": first_codes,
            "final_failure_codes": final_codes,
            "resolved_failure_codes": sorted(first_set - final_set),
            "new_failure_codes": sorted(final_set - first_set),
            "first_verified_score": float_value(first.get("verified_score", 0.0)),
            "final_verified_score": float_value(final.get("verified_score", 0.0)),
            "first_score": float_value(first.get("score", 0.0)),
            "final_score": float_value(final.get("score", 0.0)),
            "first_evaluation_valid": bool_value(first.get("evaluation_valid", False)),
            "final_evaluation_valid": bool_value(final.get("evaluation_valid", False)),
            "first_hard_gate_passed": bool_value(first.get("hard_gate_passed", False)),
            "final_hard_gate_passed": bool_value(final.get("hard_gate_passed", False)),
        })
    return changes


def failure_code_deltas(
    counts: Counter[tuple[str, str, str]],
    method_family: Counter[tuple[str, str]],
    *,
    primary: str,
    baseline: str,
) -> list[dict[str, Any]]:
    families = sorted({
        family
        for method, family in method_family
        if method in {primary, baseline}
    })
    codes = sorted({
        code
        for method, _family, code in counts
        if method in {primary, baseline}
    })
    out: list[dict[str, Any]] = []
    for family in families:
        n_primary = method_family.get((primary, family), 0)
        n_baseline = method_family.get((baseline, family), 0)
        for code in codes:
            primary_n = counts.get((primary, family, code), 0)
            baseline_n = counts.get((baseline, family, code), 0)
            primary_rate = primary_n / n_primary if n_primary else 0.0
            baseline_rate = baseline_n / n_baseline if n_baseline else 0.0
            out.append({
                "family": family,
                "failure_code": code,
                "primary_n": primary_n,
                "baseline_n": baseline_n,
                "primary_rate": primary_rate,
                "baseline_rate": baseline_rate,
                "rate_delta_primary_minus_baseline": (
                    primary_rate - baseline_rate
                ),
            })
    return out


def repair_dimension_deltas(
    rows: list[dict[str, Any]],
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> list[dict[str, Any]]:
    by_key_method: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key_method[row["_pair_key"]][row["method"]] = row
    paired = [
        (methods[contract.primary_method], methods[contract.primary_baseline])
        for methods in by_key_method.values()
        if contract.primary_method in methods and contract.primary_baseline in methods
    ]
    by_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    for primary, baseline in paired:
        by_family[str(primary["family"])].append(
            repair_dimension_delta(primary, baseline)
        )
    out: list[dict[str, Any]] = []
    for family, deltas in sorted(by_family.items()):
        keys = sorted({key for delta in deltas for key in delta})
        out.append({
            "family": family,
            "n": len(deltas),
            "mean_deltas": {
                key: mean([float(delta.get(key, 0.0)) for delta in deltas])
                for key in keys
            },
        })
    return out


def repair_dimension_delta(
    primary: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    lower_is_better = {
        "topology_error_rate": [
            "wrong_mobility_rate",
            "invalid_topology_rate",
        ],
        "port_error_rate": [
            "missing_port_rate",
            "ungrounded_port_rate",
        ],
        "artifact_error_rate": [
            "invalid_artifact_rate",
        ],
        "contact_error_rate": [
            "lockup_rate",
            "contact_lockup_rate",
        ],
        "ratio_error_pct": [
            "best_ratio_error_pct",
        ],
        "path_trace_error": [
            "best_path_trace_error",
        ],
        "max_penetration_mm": [
            "best_max_penetration_mm",
        ],
        "contact_force_rms_N": [
            "best_contact_force_rms_N",
        ],
    }
    higher_is_better = {
        "artifact_validity_rate": [
            "cad_pass_rate",
            "no_procedural_fallback_rate",
        ],
        "chrono_real_geometry_rate": [
            "chrono_real_geometry_rate",
        ],
        "strict_score_pass_rate": [
            "strict_score_pass_rate",
        ],
    }
    out: dict[str, float] = {}
    for name, fields in lower_is_better.items():
        p = mean_present(primary, fields)
        b = mean_present(baseline, fields)
        if p is not None and b is not None:
            out[name + "_improvement"] = b - p
    for name, fields in higher_is_better.items():
        p = mean_present(primary, fields)
        b = mean_present(baseline, fields)
        if p is not None and b is not None:
            out[name + "_improvement"] = p - b
    out["verified_reward_improvement"] = (
        float(primary.get("best_verified_reward_at_32", 0.0) or 0.0)
        - float(baseline.get("best_verified_reward_at_32", 0.0) or 0.0)
    )
    out["verified_success_improvement"] = (
        float(bool_value(primary.get("verified_repair_success_at_32", False)))
        - float(bool_value(baseline.get("verified_repair_success_at_32", False)))
    )
    return out


def build_claim_audit(stats: dict[str, Any]) -> dict[str, Any]:
    comparison = stats["primary_comparison"]
    family_deltas = stats["family_deltas"]
    leave_one = stats["leave_one_family_out"]
    split_deltas = stats.get("split_deltas") or []
    anti_shortcut = stats.get("anti_shortcut_comparison") or {}
    method_comparisons = stats.get("paired_method_comparisons") or {}
    primary_method = str(stats.get("primary_method") or DEFAULT_CONTRACT.primary_method)
    primary_baseline = str(
        stats.get("primary_baseline") or DEFAULT_CONTRACT.primary_baseline
    )
    primary_budget = int(
        stats.get("primary_budget_verifier_calls")
        or DEFAULT_CONTRACT.primary_budget
    )
    success_delta_pct = float(
        comparison.get("success_threshold_pct", DEFAULT_CONTRACT.success_delta_pct)
    )
    contract = stats.get("analysis_contract") or {}
    is_physics = str(contract.get("schema", "")).startswith(
        "mechanism_repair_physics."
    )
    blockers: list[str] = []
    if not stats["required_methods_present"]:
        blockers.append(
            "missing required methods: "
            + ", ".join(stats["missing_required_methods"])
        )
    if stats["n_paired_cells"] <= 0:
        blockers.append("no paired primary/baseline cells")
    if comparison["success_delta_pct"] < success_delta_pct:
        blockers.append(
            "success delta below threshold: "
            f"{comparison['success_delta_pct']:.3f} < {success_delta_pct:.3f}"
        )
    success_ci = comparison["success_delta_ci95"]
    sign_p = comparison["success_sign_test_p_one_sided"]
    if not (success_ci["low"] > 0 and sign_p <= 0.05):
        blockers.append(
            "primary success delta lacks statistical support: "
            f"ci_low={success_ci['low']:.6f}, p={sign_p:.6f}"
        )
    if comparison["reward_delta_mean"] <= 0:
        blockers.append("paired reward delta is not positive")
    if not stats["reward_beats_all_required_baselines"]:
        blockers.append("TTRL does not beat every required baseline on reward")
    if is_physics:
        hidden_delta = split_delta(split_deltas, PHYSICS_HIDDEN_VARIANT_SPLIT)
        if hidden_delta is None:
            blockers.append("hidden variant success delta is missing")
        elif hidden_delta <= 0:
            blockers.append(
                "hidden variant success delta is not positive: "
                f"{hidden_delta:.6f}"
            )
        anti_delta = anti_shortcut.get("anti_shortcut_pass_rate_delta")
        if anti_delta is None:
            blockers.append("anti-shortcut pass-rate delta is missing")
        elif float(anti_delta) <= 0:
            blockers.append(
                "anti-shortcut pass-rate delta is not positive: "
                f"{float(anti_delta):.6f}"
            )
        for baseline in ("adaptive_evolution", "verifier_gated_search"):
            comparison_row = method_comparisons.get(baseline) or {}
            if not comparison_row.get("primary_beats_on_success", False):
                blockers.append(
                    f"TTRL does not beat {baseline} under equal verifier budget"
                )
        positive_family_success = sum(
            1 for row in family_deltas if float(row.get("success_delta", 0.0)) > 0
        )
        if positive_family_success < PHYSICS_MIN_POSITIVE_FAMILIES:
            blockers.append(
                "TTRL success delta is positive in too few families: "
                f"{positive_family_success} < {PHYSICS_MIN_POSITIVE_FAMILIES}"
            )
        if any(not row.get("keeps_positive_success_delta", False) for row in leave_one):
            blockers.append("leave-one-family-out check flips success delta sign")
    else:
        wins = sum(1 for row in family_deltas if row["reward_delta"] > 0)
        if family_deltas and wins <= len(family_deltas) / 2:
            blockers.append("TTRL does not win in a majority of held-out families")
        if any(not row["keeps_positive_sign"] for row in leave_one):
            blockers.append("leave-one-family-out check flips reward delta sign")
    if not stats["budget_audit"]["budget_matched"]:
        blockers.append(
            "actual verifier budget or primary CAD/Chrono budget is invalid"
        )
    if not stats["budget_audit"].get("primary_budget_spent", False):
        blockers.append(
            f"not every cell spent B={primary_budget} verifier calls"
        )
    evidence = stats.get("evidence_audit") or {}
    if not evidence.get("evidence_complete", False):
        blockers.append(
            "raw/verifier/training/adapter evidence is incomplete: "
            f"missing_raw_rows={evidence.get('n_missing_raw_rows', 0)}, "
            f"missing_verifier_rows={evidence.get('n_missing_verifier_rows', 0)}, "
            f"missing_raw_files={evidence.get('n_missing_raw_files', 0)}, "
            f"missing_verifier_files={evidence.get('n_missing_verifier_files', 0)}, "
            f"trace_pairs={evidence.get('matched_ttrl_vs_no_update_trace_pairs_with_evidence', 0)}"
        )
    learning = stats.get("learning_audit") or {}
    if not learning.get("ttrl_learning_evidence_complete", False):
        blockers.append(
            "TTRL learning evidence is incomplete: "
            f"bad_rows={learning.get('n_bad_ttrl_learning_rows', 0)}, "
            f"ttrl_rows={learning.get('ttrl_rows', 0)}"
        )
    benchmark = stats.get("benchmark_readiness_audit") or {}
    if benchmark.get("enforced") and not benchmark.get("benchmark_ready", False):
        blockers.append(
            "benchmark/split/verifier readiness failed: "
            + "; ".join((benchmark.get("blockers") or [])[:5])
        )
    coverage = stats.get("coverage_audit") or {}
    if coverage.get("enforced") and not coverage.get("complete_coverage"):
        blockers.append(
            "incomplete expected split/task/seed/method coverage: "
            f"missing={coverage.get('n_missing_cells', 0)}, "
            f"extra={coverage.get('n_extra_cells', 0)}"
        )
    return {
        "schema": "mechanism_repair_ttrl.claim_audit.v1",
        "claim_status": (
            "supports_primary_hypothesis"
            if not blockers
            else "does_not_support_primary_hypothesis"
        ),
        "blockers": blockers,
        "primary_method": primary_method,
        "primary_baseline": primary_baseline,
        "primary_budget_verifier_calls": primary_budget,
        "evidence": {
            "stats": "stats.json",
            "failure_analysis": "failure_analysis.json",
            "trace_pairs": "trace_pairs.json",
        },
    }


def bootstrap_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"low": math.nan, "high": math.nan}
    if len(values) == 1:
        return {"low": values[0], "high": values[0]}
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    low_idx = max(0, int(0.025 * (samples - 1)))
    high_idx = min(samples - 1, int(0.975 * (samples - 1)))
    return {"low": means[low_idx], "high": means[high_idx]}


def one_sided_sign_test(values: list[float]) -> float:
    pos = sum(1 for value in values if value > 0)
    neg = sum(1 for value in values if value < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
    return float(tail)


def parse_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(";") if item.strip()]


def parse_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(";") if item.strip()]


def load_json_path(path: str) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "pass", "passed"}


def float_value(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def default_benchmark_dir(out_dir: Path) -> Path | None:
    required = (
        out_dir / "method_manifest.json",
        out_dir / "split_manifest_A.json",
        out_dir / "split_manifest_B.json",
    )
    if all(path.is_file() for path in required):
        return out_dir
    return None


def build_expected_coverage(
    benchmark_dir: Path,
    *,
    contract: AnalysisContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    method_manifest = json.loads(
        (benchmark_dir / "method_manifest.json").read_text()
    )
    methods = [
        str(method)
        for method in method_manifest.get("required_methods", contract.required_methods)
    ]
    seeds = [
        int(seed)
        for seed in method_manifest.get("eval_seeds", contract.eval_seeds)
    ]
    expected: set[tuple[str, str, int, str]] = set()
    split_task_counts: dict[str, int] = {}
    for split_path in sorted(benchmark_dir.glob("split_manifest_*.json")):
        split = split_path.name.removeprefix("split_manifest_").removesuffix(".json")
        manifest = json.loads(split_path.read_text())
        task_ids = [
            Path(str(task_id)).name
            for task_id in (manifest.get("splits") or {}).get("test", [])
        ]
        split_task_counts[split] = len(task_ids)
        for task_id in task_ids:
            for seed in seeds:
                for method in methods:
                    expected.add((split, task_id, seed, method))
    return {
        "methods": methods,
        "seeds": seeds,
        "split_task_counts": split_task_counts,
        "expected_cells": sorted(list(expected)),
    }


def build_benchmark_readiness(benchmark_dir: Path) -> dict[str, Any]:
    blockers: list[str] = []
    benchmark_path = benchmark_dir / "benchmark_manifest.json"
    verifier_path = benchmark_dir / "verifier_manifest.json"
    if not benchmark_path.is_file():
        blockers.append("missing benchmark_manifest.json")
        benchmark = {}
    else:
        benchmark = json.loads(benchmark_path.read_text())
    if not verifier_path.is_file():
        blockers.append("missing verifier_manifest.json")
        verifier = {}
    else:
        verifier = json.loads(verifier_path.read_text())
    is_physics = str(benchmark.get("schema", "")).startswith(
        "mechanism_repair_physics."
    )

    audit = benchmark.get("audit") or {}
    audit_tasks = audit.get("tasks") or []
    family_counts = audit.get("family_counts") or {}
    primary_families = (
        audit.get("primary_families")
        or audit.get("required_families")
        or benchmark.get("primary_families")
        or benchmark.get("required_families")
        or []
    )
    if benchmark.get("experiment_ready") is not True:
        blockers.append("benchmark_manifest experiment_ready is not true")
    if audit and audit.get("passes") is not True:
        blockers.append("benchmark audit did not pass")
    for blocker in audit.get("blockers") or []:
        blockers.append(f"benchmark audit blocker: {blocker}")
    min_task_count = 120 if is_physics else 40
    min_family_count = 12 if is_physics else 8
    min_tasks_per_family = 10 if is_physics else 5
    if int(audit.get("task_count") or benchmark.get("task_count") or 0) < min_task_count:
        blockers.append(f"benchmark has fewer than {min_task_count} tasks")
    if len(primary_families) < min_family_count:
        blockers.append(
            f"benchmark has fewer than {min_family_count} primary families"
        )
    if family_counts:
        low = {
            str(family): int(count)
            for family, count in family_counts.items()
            if int(count) < min_tasks_per_family
        }
        if low:
            blockers.append(
                f"primary families below {min_tasks_per_family} tasks: {low}"
            )
    if not audit_tasks:
        blockers.append("benchmark audit task records are missing")
    else:
        bad_constraints = []
        bad_negatives = []
        bad_reference = []
        fake_oracles = []
        for task in audit_tasks:
            task_id = str(task.get("task_id") or "")
            classes = set(str(item) for item in task.get("constraint_classes") or [])
            has_mobility = bool(
                {"topology_or_mobility", "topology_mobility"} & classes
            )
            has_functional_or_physics = bool(
                {"functional_behavior", "physics_contact"} & classes
            )
            min_constraint_classes = 3 if is_physics else 2
            if is_physics:
                if len(classes) < min_constraint_classes or not has_functional_or_physics:
                    bad_constraints.append(task_id)
            elif (
                len(classes) < min_constraint_classes
                or not has_mobility
                or "functional_behavior" not in classes
            ):
                bad_constraints.append(task_id)
            if (
                task.get("has_negative_control") is not True
                and int(task.get("negative_control_count") or 0) < 1
                and int(task.get("effective_negative_control_count") or 0) < 1
            ):
                bad_negatives.append(task_id)
            validation = task.get("validation") or {}
            if (
                validation.get("reference_passed") is not True
                or validation.get("reference_evaluation_valid") is not True
                or validation.get("reference_hard_gate_passed") is not True
            ):
                bad_reference.append(task_id)
            if task.get("uses_fake_contact_oracle") is True:
                fake_oracles.append(task_id)
        if bad_constraints:
            blockers.append(
                "tasks missing required non-toy constraint classes: "
                + ", ".join(bad_constraints[:20])
            )
        if bad_negatives:
            blockers.append(
                "tasks missing negative controls: "
                + ", ".join(bad_negatives[:20])
            )
        if bad_reference:
            blockers.append(
                "tasks missing passing reference validation: "
                + ", ".join(bad_reference[:20])
            )
        if fake_oracles:
            blockers.append(
                "tasks use fake contact oracle: " + ", ".join(fake_oracles[:20])
            )

    if verifier.get("main_claim_allows_fake_oracle") is not False:
        blockers.append("verifier manifest allows fake oracle in main claim")
    if verifier.get("fake_oracle_tasks"):
        blockers.append("verifier manifest lists fake oracle tasks")
    verifier_levels = verifier.get("verifier_levels") or {}
    if not verifier_levels:
        blockers.append("verifier levels are missing")
    if is_physics:
        level_counts = audit.get("level_counts") or benchmark.get("level_counts") or {}
        level2 = int(level_counts.get("2") or level_counts.get(2) or 0)
        level3 = int(level_counts.get("3") or level_counts.get(3) or 0)
        if level2 + level3 < min_task_count:
            blockers.append("physics benchmark has fewer than 120 Level-2/3 tasks")
        if level3 < 30:
            blockers.append("physics benchmark has fewer than 30 Level-3 tasks")

    expected_splits = (
        {
            "A": {
                "seen": {"belt", "chain", "rack_pinion", "fourbar"},
                "unseen": {"planetary", "lead_screw", "slider_crank", "cycloidal"},
            },
            "B": {
                "seen": {"planetary", "lead_screw", "fourbar", "slider_crank"},
                "unseen": {"belt", "chain", "rack_pinion", "cycloidal"},
            },
        }
        if not is_physics
        else {}
    )
    split_paths = (
        [benchmark_dir / f"split_manifest_{split}.json" for split in expected_splits]
        if expected_splits
        else sorted(benchmark_dir.glob("split_manifest_*.json"))
    )
    split_summary: dict[str, Any] = {}
    for path in split_paths:
        split = path.name.removeprefix("split_manifest_").removesuffix(".json")
        if not path.is_file():
            blockers.append(f"missing split_manifest_{split}.json")
            continue
        manifest = json.loads(path.read_text())
        splits = manifest.get("splits") or {}
        train = set(Path(str(item)).name for item in splits.get("train", []) or [])
        test = set(Path(str(item)).name for item in splits.get("test", []) or [])
        seen = set(str(item) for item in manifest.get("seen_families") or [])
        unseen = set(str(item) for item in manifest.get("unseen_families") or [])
        split_summary[split] = {
            "train_tasks": len(train),
            "test_tasks": len(test),
            "seen_families": sorted(seen),
            "unseen_families": sorted(unseen),
        }
        min_heldout = 20
        if is_physics and split in {"A", "B"}:
            min_heldout = 40
        if len(test) < min_heldout:
            blockers.append(
                f"split {split} has fewer than {min_heldout} held-out tasks"
            )
        expected = expected_splits.get(split)
        if expected is not None and seen != expected["seen"]:
            blockers.append(f"split {split} seen families mismatch: {sorted(seen)}")
        if expected is not None and unseen != expected["unseen"]:
            blockers.append(f"split {split} unseen families mismatch: {sorted(unseen)}")
        if train & test:
            blockers.append(f"split {split} has train/test overlap")

    return {
        "enforced": True,
        "benchmark_ready": not blockers,
        "blockers": blockers,
        "task_count": int(audit.get("task_count") or benchmark.get("task_count") or 0),
        "family_counts": family_counts,
        "primary_families": primary_families,
        "verifier_levels": verifier_levels,
        "split_summary": split_summary,
    }


def audit_expected_coverage(
    rows: list[dict[str, Any]],
    expected_coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    if expected_coverage is None:
        return {"enforced": False}
    expected = {
        tuple(item)
        for item in expected_coverage.get("expected_cells", []) or []
    }
    observed = {
        (
            str(row["split"]),
            str(row["task_id"]),
            int(row["seed"]),
            str(row["method"]),
        )
        for row in rows
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {
        "enforced": True,
        "complete_coverage": not missing and not extra,
        "expected_cells": len(expected),
        "observed_cells": len(observed),
        "n_missing_cells": len(missing),
        "n_extra_cells": len(extra),
        "missing_cells": [list(item) for item in missing[:100]],
        "extra_cells": [list(item) for item in extra[:100]],
        "methods": expected_coverage.get("methods", []),
        "seeds": expected_coverage.get("seeds", []),
        "split_task_counts": expected_coverage.get("split_task_counts", {}),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def mean_present(row: dict[str, Any], fields: list[str]) -> float | None:
    values: list[float] = []
    for field in fields:
        raw = row.get(field)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return mean(values) if values else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
