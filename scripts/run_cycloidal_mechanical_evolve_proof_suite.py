#!/usr/bin/env python3
"""Run multi-seed, multi-target cycloidal MechanicalEvolve proof suite.

This script is intentionally a wrapper around the stricter single-target
runner. Each trial is a complete equal-budget CAD+Chrono experiment with
procedural fallback disabled. The aggregate table reports confidence intervals
and paired deltas, so a negative or inconclusive result remains visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "mech_bench.cycloidal_mechanical_evolve_proof_suite.v1"
DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
METHOD_ORDER = (
    "verifier_gated",
    "llm_evolve_no_update",
    "mechanical_evolve_ttrl",
)
CSV_COLUMNS = (
    "method",
    "target_count",
    "trial_count",
    "chrono_audits",
    "candidate_count",
    "best_verified_reward_mean",
    "best_verified_reward_stderr",
    "best_verified_reward_ci95_low",
    "best_verified_reward_ci95_high",
    "verified_pass_rate",
    "cad_pass_rate",
    "chrono_real_geometry_rate",
    "lockup_rate",
    "best_out_omega_med_mean",
    "best_ratio_error_pct_mean",
    "best_power_balance_error_pct_mean",
    "best_torque_ripple_pct_mean",
    "best_max_penetration_mm_mean",
    "best_contact_force_rms_N_mean",
    "adapter_updates",
    "trained_tokens",
    "target_win_rate_vs_ttrl",
    "best_trial_reward",
)


@dataclass(frozen=True)
class TargetConfig:
    name: str
    kind: str
    input_speed_rad_s: float = 10.0
    output_load_Nm: float = 0.75
    young_modulus: float = 1.0e8
    normal_stiffness: float = 5.0e7
    damping: float = 250.0
    friction: float = 0.0
    min_output_speed_rad_s: float = 0.5


TARGETS: dict[str, TargetConfig] = {
    "nominal": TargetConfig(
        name="nominal",
        kind="target",
        input_speed_rad_s=10.0,
        output_load_Nm=0.75,
    ),
    "high_load": TargetConfig(
        name="high_load",
        kind="target",
        input_speed_rad_s=10.0,
        output_load_Nm=1.00,
    ),
    "high_speed": TargetConfig(
        name="high_speed",
        kind="target",
        input_speed_rad_s=14.0,
        output_load_Nm=0.75,
    ),
    "soft_contact": TargetConfig(
        name="soft_contact",
        kind="sensitivity",
        input_speed_rad_s=10.0,
        output_load_Nm=0.75,
        young_modulus=5.0e7,
        normal_stiffness=2.5e7,
        damping=180.0,
    ),
    "stiff_contact": TargetConfig(
        name="stiff_contact",
        kind="sensitivity",
        input_speed_rad_s=10.0,
        output_load_Nm=0.75,
        young_modulus=2.0e8,
        normal_stiffness=1.0e8,
        damping=350.0,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/cycloidal_mechanical_evolve_proof_suite")
    parser.add_argument("--results-json", default="docs/cycloidal_mechanical_evolve_proof_suite_results.json")
    parser.add_argument("--results-csv", default="docs/cycloidal_mechanical_evolve_proof_suite_results.csv")
    parser.add_argument("--results-md", default="docs/cycloidal_mechanical_evolve_proof_suite.md")
    parser.add_argument("--model", default=os.environ.get("MECHANICAL_EVOLVE_LORA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--seeds", default="20260525,20260526,20260527")
    parser.add_argument("--targets", default="nominal,high_load,high_speed")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--proposals-per-round", type=int, default=64)
    parser.add_argument("--audits-per-round", type=int, default=32)
    parser.add_argument("--mutation-fill", type=int, default=64)
    parser.add_argument("--baseline-audits", type=int, default=240)
    parser.add_argument("--verifier-pool", type=int, default=320)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--duration-s", type=float, default=0.15)
    parser.add_argument("--power-balance-limit-pct", type=float, default=90.0)
    parser.add_argument("--torque-ripple-limit-pct", type=float, default=1000.0)
    parser.add_argument("--contact-force-limit-N", type=float, default=3000.0)
    parser.add_argument("--max-contacts", type=float, default=128.0)
    parser.add_argument("--lora-iters", type=int, default=4)
    parser.add_argument("--lora-max-examples", type=int, default=256)
    parser.add_argument("--lora-num-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-scale", type=float, default=16.0)
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=0,
        help=(
            "Post-hoc regenerated-CAD re-audits for the TTRL best candidate. "
            "Keep this at 0 for strict equal-total-Chrono-budget comparisons."
        ),
    )
    parser.add_argument(
        "--allow-zero-reward-lora",
        action="store_true",
        help=(
            "Forward zero-reward LoRA training only for plumbing smoke tests. "
            "Leave unset for real paper-grade runs."
        ),
    )
    parser.add_argument("--trial-timeout-s", type=float, default=21600.0)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    targets = selected_targets(args.targets, include_sensitivity=args.include_sensitivity)
    trials: list[dict[str, Any]] = []
    for target in targets:
        for seed in seeds:
            trials.append(run_or_load_trial(
                args=args,
                out_dir=out_dir,
                target=target,
                seed=seed,
            ))

    method_table = aggregate_methods(trials)
    target_table = aggregate_targets(trials)
    paired = paired_deltas(trials)
    summary = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "seeds": seeds,
        "targets": [target.__dict__ for target in targets],
        "budget": {
            "chrono_audits_per_method_per_trial": int(args.budget),
            "rounds": int(args.rounds),
            "proposals_per_round": int(args.proposals_per_round),
            "audits_per_round": int(args.audits_per_round),
            "baseline_audits": int(args.baseline_audits),
            "posthoc_stability_repeats": int(args.stability_repeats),
        },
        "verifier_invariants": {
            "contact_model": "smc",
            "procedural_cycloidal_fallback": False,
            "samples": int(args.samples),
            "duration_s": float(args.duration_s),
            "power_balance_limit_pct": float(args.power_balance_limit_pct),
            "torque_ripple_limit_pct": float(args.torque_ripple_limit_pct),
            "contact_force_limit_N": float(args.contact_force_limit_N),
            "max_contacts": float(args.max_contacts),
        },
        "method_table": method_table,
        "target_table": target_table,
        "paired_deltas": paired,
        "proof_conditions": proof_conditions(method_table, paired),
        "trials": trials,
    }
    results_json = Path(args.results_json).expanduser().resolve()
    results_csv = Path(args.results_csv).expanduser().resolve()
    results_md = Path(args.results_md).expanduser().resolve()
    write_json(results_json, json_safe(summary))
    write_csv(results_csv, method_table)
    write_markdown(results_md, summary)
    print_table(method_table)
    print(f"results_json={results_json}")
    print(f"results_csv={results_csv}")
    print(f"results_md={results_md}")
    return 0


def run_or_load_trial(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    target: TargetConfig,
    seed: int,
) -> dict[str, Any]:
    trial_dir = out_dir / target.name / f"seed_{seed}"
    result_path = trial_dir / "summary.json"
    if result_path.is_file() and not args.rerun:
        summary = read_json(result_path)
        return trial_record(target, seed, trial_dir, summary, reused=True)
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_cycloidal_mechanical_evolve_ttrl.py"),
        "--out-dir",
        str(trial_dir / "run"),
        "--results-json",
        str(result_path),
        "--results-csv",
        str(trial_dir / "summary.csv"),
        "--results-md",
        str(trial_dir / "summary.md"),
        "--model",
        str(args.model),
        "--seed",
        str(int(seed)),
        "--baseline-methods",
        "verifier_gated",
        "--baseline-audits",
        str(max(int(args.baseline_audits), int(args.budget))),
        "--verifier-pool",
        str(max(int(args.verifier_pool), int(args.budget))),
        "--target-chrono-audits",
        str(max(1, int(args.budget))),
        "--rounds",
        str(max(1, int(args.rounds))),
        "--proposals-per-round",
        str(max(1, int(args.proposals_per_round))),
        "--audits-per-round",
        str(max(1, int(args.audits_per_round))),
        "--mutation-fill",
        str(max(0, int(args.mutation_fill))),
        "--samples",
        str(max(3, int(args.samples))),
        "--duration-s",
        str(max(1.0e-6, float(args.duration_s))),
        "--input-speed-rad-s",
        str(target.input_speed_rad_s),
        "--output-load-Nm",
        str(target.output_load_Nm),
        "--young-modulus",
        str(target.young_modulus),
        "--normal-stiffness",
        str(target.normal_stiffness),
        "--damping",
        str(target.damping),
        "--friction",
        str(target.friction),
        "--min-output-speed-rad-s",
        str(target.min_output_speed_rad_s),
        "--power-balance-limit-pct",
        str(float(args.power_balance_limit_pct)),
        "--torque-ripple-limit-pct",
        str(float(args.torque_ripple_limit_pct)),
        "--contact-force-limit-N",
        str(float(args.contact_force_limit_N)),
        "--max-contacts",
        str(float(args.max_contacts)),
        "--lora-iters",
        str(max(1, int(args.lora_iters))),
        "--lora-max-examples",
        str(max(1, int(args.lora_max_examples))),
        "--lora-num-layers",
        str(int(args.lora_num_layers)),
        "--lora-rank",
        str(max(1, int(args.lora_rank))),
        "--lora-scale",
        str(float(args.lora_scale)),
        "--stability-repeats",
        str(max(0, int(args.stability_repeats))),
        "--keep-out-dir",
        "--no-policy-baseline-bootstrap",
        "--policy-seed-bootstrap",
    ]
    if not args.rerun:
        run_dir = trial_dir / "run"
        baseline_json = run_dir / "baselines" / "cycloidal_optimizer_strict_matched.json"
        if baseline_json.is_file():
            command.append("--skip-baselines")
        no_update_json = run_dir / "llm_evolve_no_update" / "summary.json"
        if no_update_json.is_file():
            command.append("--skip-no-update")
    if getattr(args, "allow_zero_reward_lora", False):
        command.append("--allow-zero-reward-lora")
    result = run_command(
        command,
        cwd=SCRIPT_DIR.parent,
        log_path=trial_dir / "proof_trial.log",
        timeout_s=max(1.0, float(args.trial_timeout_s)),
    )
    summary = read_json(result_path) if result_path.is_file() else {}
    if result.get("status") != "completed" or result.get("returncode") != 0:
        raise SystemExit(
            f"proof-suite trial failed for target={target.name} seed={seed}; "
            f"see {trial_dir / 'proof_trial.log'}"
        )
    if not isinstance(summary.get("method_table"), list) or not summary["method_table"]:
        raise SystemExit(
            f"proof-suite trial produced no method table for target={target.name} "
            f"seed={seed}; see {trial_dir / 'proof_trial.log'}"
        )
    record = trial_record(target, seed, trial_dir, summary, reused=False)
    record["command"] = result
    return record


def trial_record(
    target: TargetConfig,
    seed: int,
    trial_dir: Path,
    summary: dict[str, Any],
    *,
    reused: bool,
) -> dict[str, Any]:
    return {
        "target": target.name,
        "target_kind": target.kind,
        "target_config": target.__dict__,
        "seed": seed,
        "trial_dir": str(trial_dir),
        "reused": reused,
        "win_conditions": summary.get("win_conditions", {}),
        "method_table": summary.get("method_table", []),
        "stability_audit": {
            key: value for key, value in (
                summary.get("stability_audit", {}) or {}
            ).items() if key != "rows"
        },
    }


def aggregate_methods(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for trial in trials:
        for row in trial.get("method_table", []):
            if isinstance(row, dict):
                by_method.setdefault(str(row.get("method")), []).append({
                    **row,
                    "target": trial.get("target"),
                    "seed": trial.get("seed"),
                })
    table = []
    ttrl_rows = by_method.get("mechanical_evolve_ttrl", [])
    for method in METHOD_ORDER:
        rows = by_method.get(method, [])
        rewards = [float(row.get("best_verified_reward", 0.0) or 0.0) for row in rows]
        interval = mean_ci95(rewards)
        audits = sum(int(row.get("chrono_audits", 0) or 0) for row in rows)
        candidates = sum(int(row.get("candidate_count", 0) or 0) for row in rows)
        valid = sum_rate_numer(rows, "verified_pass_rate", "candidate_count")
        cad = sum_rate_numer(rows, "cad_pass_rate", "candidate_count")
        real_geometry = sum_rate_numer(
            rows,
            "chrono_real_geometry_rate",
            "candidate_count",
        )
        lockups = sum_rate_numer(rows, "lockup_rate", "chrono_audits")
        target_wins = target_win_rate(method, rows, ttrl_rows)
        table.append({
            "method": method,
            "target_count": len({row.get("target") for row in rows}),
            "trial_count": len(rows),
            "candidate_count": candidates,
            "chrono_audits": audits,
            "best_verified_reward_mean": interval["mean"],
            "best_verified_reward_stderr": interval["stderr"],
            "best_verified_reward_ci95_low": interval["ci95_low"],
            "best_verified_reward_ci95_high": interval["ci95_high"],
            "verified_pass_rate": rate(valid, candidates),
            "cad_pass_rate": rate(cad, candidates),
            "chrono_real_geometry_rate": rate(real_geometry, candidates),
            "lockup_rate": rate(lockups, audits),
            "best_out_omega_med_mean": mean_metric(rows, "best_out_omega_med"),
            "best_ratio_error_pct_mean": mean_metric(rows, "best_ratio_error_pct"),
            "best_power_balance_error_pct_mean": mean_metric(
                rows,
                "best_power_balance_error_pct",
            ),
            "best_torque_ripple_pct_mean": mean_metric(
                rows,
                "best_torque_ripple_pct",
            ),
            "best_max_penetration_mm_mean": mean_metric(
                rows,
                "best_max_penetration_mm",
            ),
            "best_contact_force_rms_N_mean": mean_metric(
                rows,
                "best_contact_force_rms_N",
            ),
            "adapter_updates": sum(
                int(row.get("adapter_updates", 0) or 0) for row in rows
            ),
            "trained_tokens": sum(
                int(row.get("trained_tokens", 0) or 0) for row in rows
            ),
            "target_win_rate_vs_ttrl": target_wins,
            "best_trial_reward": round(max(rewards, default=0.0), 6),
        })
    return table


def aggregate_targets(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        methods = {row.get("method"): row for row in trial.get("method_table", [])}
        ttrl = methods.get("mechanical_evolve_ttrl", {})
        best_other = max(
            (
                float(row.get("best_verified_reward", 0.0) or 0.0)
                for name, row in methods.items()
                if name != "mechanical_evolve_ttrl"
            ),
            default=0.0,
        )
        rows.append({
            "target": trial.get("target"),
            "target_kind": trial.get("target_kind"),
            "seed": trial.get("seed"),
            "ttrl_best_verified_reward": ttrl.get("best_verified_reward"),
            "best_non_ttrl_verified_reward": round(best_other, 6),
            "ttrl_wins": (
                float(ttrl.get("best_verified_reward", 0.0) or 0.0) > best_other
            ),
        })
    return rows


def paired_deltas(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: dict[str, list[float]] = {}
    for trial in trials:
        methods = {row.get("method"): row for row in trial.get("method_table", [])}
        ttrl = float(
            (methods.get("mechanical_evolve_ttrl", {}) or {})
            .get("best_verified_reward", 0.0) or 0.0
        )
        for baseline in METHOD_ORDER:
            if baseline == "mechanical_evolve_ttrl" or baseline not in methods:
                continue
            base = float(methods[baseline].get("best_verified_reward", 0.0) or 0.0)
            deltas.setdefault(baseline, []).append(ttrl - base)
    out = []
    for baseline, values in sorted(deltas.items()):
        interval = mean_ci95(values)
        out.append({
            "baseline": baseline,
            "trial_count": len(values),
            "mean_delta_ttrl_minus_baseline": interval["mean"],
            "stderr": interval["stderr"],
            "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"],
            "win_rate": round(sum(1 for value in values if value > 0.0) / len(values), 6)
            if values else 0.0,
        })
    return out


def proof_conditions(
    method_table: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = {row["method"]: row for row in method_table}
    ttrl = rows.get("mechanical_evolve_ttrl", {})
    baselines = [
        row for method, row in rows.items() if method != "mechanical_evolve_ttrl"
    ]
    ttrl_mean = float(ttrl.get("best_verified_reward_mean", 0.0) or 0.0)
    mean_beats_all = bool(baselines) and ttrl_mean > max(
        float(row.get("best_verified_reward_mean", 0.0) or 0.0)
        for row in baselines
    )
    paired_positive = all(
        float(row.get("mean_delta_ttrl_minus_baseline", 0.0) or 0.0) > 0.0
        for row in paired
    ) if paired else False
    ci_positive = all(float(row.get("ci95_low", 0.0) or 0.0) > 0.0 for row in paired) if paired else False
    secondary_support = secondary_metric_support(rows)
    return {
        "ttrl_mean_beats_all_baselines": mean_beats_all,
        "paired_mean_delta_positive_vs_all": paired_positive,
        "paired_ci95_positive_vs_all": ci_positive,
        "secondary_metric_support_vs_all": secondary_support,
        "paper_grade_statistical_claim_supported": (
            mean_beats_all and paired_positive and ci_positive and secondary_support
        ),
    }


def secondary_metric_support(rows: dict[str, dict[str, Any]]) -> bool:
    ttrl = rows.get("mechanical_evolve_ttrl", {})
    baselines = [
        row for method, row in rows.items() if method != "mechanical_evolve_ttrl"
    ]
    if not ttrl or not baselines:
        return False
    ttrl_pass = numeric(ttrl.get("verified_pass_rate"))
    ttrl_lockup = numeric(ttrl.get("lockup_rate"))
    ttrl_ratio = numeric(ttrl.get("best_ratio_error_pct_mean"))
    pass_beats = (
        ttrl_pass is not None
        and all(
            ttrl_pass > numeric(row.get("verified_pass_rate"), default=-math.inf)
            for row in baselines
        )
    )
    lockup_beats = (
        ttrl_lockup is not None
        and all(
            ttrl_lockup < numeric(row.get("lockup_rate"), default=math.inf)
            for row in baselines
        )
    )
    ratio_beats = (
        ttrl_ratio is not None
        and all(
            ttrl_ratio < numeric(
                row.get("best_ratio_error_pct_mean"),
                default=math.inf,
            )
            for row in baselines
        )
    )
    return bool(pass_beats or lockup_beats or ratio_beats)


def target_win_rate(
    method: str,
    rows: list[dict[str, Any]],
    ttrl_rows: list[dict[str, Any]],
) -> float | None:
    if method == "mechanical_evolve_ttrl":
        return None
    ttrl_by_trial = {
        (row.get("target"), row.get("seed")): float(
            row.get("best_verified_reward", 0.0) or 0.0)
        for row in ttrl_rows
    }
    paired = [
        (
            ttrl_by_trial.get((row.get("target"), row.get("seed"))),
            float(row.get("best_verified_reward", 0.0) or 0.0),
        )
        for row in rows
    ]
    paired = [(a, b) for a, b in paired if a is not None]
    if not paired:
        return None
    return round(sum(1 for ttrl, base in paired if ttrl > base) / len(paired), 6)


def sum_rate_numer(rows: list[dict[str, Any]], rate_key: str, denom_key: str) -> int:
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(rate_key, 0.0) or 0.0) * float(
                row.get(denom_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return int(round(total))


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value for value in (numeric(row.get(key)) for row in rows)
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def numeric(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def mean_ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stderr": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    avg = sum(values) / len(values)
    se = stderr(values)
    half = t_critical_975(len(values)) * se
    return {
        "mean": round(avg, 6),
        "stderr": round(se, 6),
        "ci95_low": round(avg - half, 6),
        "ci95_high": round(avg + half, 6),
    }


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def t_critical_975(n: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    return table.get(n, 1.96 if n > 30 else 2.0)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    stdout = ""
    stderr_text = ""
    returncode = 0
    status = "completed"
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        stdout = completed.stdout
        stderr_text = completed.stderr
        returncode = int(completed.returncode)
        status = "completed" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        stdout = ensure_text(exc.stdout)
        stderr_text = ensure_text(exc.stderr)
        returncode = 124
        status = "timeout"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n"
        + stdout
        + "\nSTDERR\n"
        + stderr_text
    )
    return {
        "status": status,
        "returncode": returncode,
        "command": command,
        "log_path": str(log_path),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr_text[-4000:],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    conditions = summary["proof_conditions"]
    lines = [
        "# Cycloidal MechanicalEvolve Proof Suite",
        "",
        "All trials use Chrono SMC with procedural_cycloidal_fallback=false and equal real Chrono audit budgets within each trial.",
        "Within each target/seed trial, all methods use identical verifier thresholds, CAD generation, Chrono configuration, and random seed.",
        f"Budget per method per trial: {summary['budget']['chrono_audits_per_method_per_trial']}.",
        "",
        "## Method Aggregate",
        "",
        "| method | trials | audits | reward mean | 95% CI | verified pass | CAD pass | real geom | lockup | ratio err | force RMS | updates | tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_table"]:
        lines.append(
            f"| {row['method']} | {row['trial_count']} | {row['chrono_audits']} | "
            f"{row['best_verified_reward_mean']:.6g} | "
            f"[{row['best_verified_reward_ci95_low']:.6g}, {row['best_verified_reward_ci95_high']:.6g}] | "
            f"{format_value(row.get('verified_pass_rate'))} | "
            f"{format_value(row.get('cad_pass_rate'))} | "
            f"{format_value(row.get('chrono_real_geometry_rate'))} | "
            f"{format_value(row.get('lockup_rate'))} | "
            f"{format_value(row.get('best_ratio_error_pct_mean'))} | "
            f"{format_value(row.get('best_contact_force_rms_N_mean'))} | "
            f"{row.get('adapter_updates', 0)} | {row.get('trained_tokens', 0)} |"
        )
    lines.extend(["", "## Paired Deltas", "", "| baseline | trials | mean TTRL minus baseline | 95% CI | win rate |", "|---|---:|---:|---:|---:|"])
    for row in summary["paired_deltas"]:
        lines.append(
            f"| {row['baseline']} | {row['trial_count']} | "
            f"{row['mean_delta_ttrl_minus_baseline']:.6g} | "
            f"[{row['ci95_low']:.6g}, {row['ci95_high']:.6g}] | "
            f"{row['win_rate']:.6g} |"
        )
    lines.extend([
        "",
        "## Target Trials",
        "",
        "| target | kind | seed | TTRL reward | best non-TTRL reward | TTRL wins |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in summary.get("target_table", []):
        lines.append(
            f"| {row.get('target')} | {row.get('target_kind')} | {row.get('seed')} | "
            f"{format_value(row.get('ttrl_best_verified_reward'))} | "
            f"{format_value(row.get('best_non_ttrl_verified_reward'))} | "
            f"{bool(row.get('ttrl_wins'))} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if conditions["paper_grade_statistical_claim_supported"]:
        lines.append(
            "TTRL still wins under the equal Chrono audit budget. The "
            "multi-seed/multi-target suite supports the stronger claim: "
            "under equal expensive physics-verification budgets, iterative "
            "RLVR/TTRL adaptation outperforms the non-updating baselines with "
            "positive paired 95% confidence intervals."
        )
    else:
        lines.append(
            "TTRL does not yet win under the equal Chrono audit budget across "
            "the full suite. The multi-seed/multi-target suite does not "
            "support the full statistical paper claim; report this as "
            "inconclusive or failed where the paired CI crosses zero or a "
            "target-level baseline beats TTRL."
        )
        losing_trials = [
            row for row in summary.get("target_table", [])
            if not bool(row.get("ttrl_wins"))
        ]
        if losing_trials:
            losses = ", ".join(
                f"{row.get('target')}/seed_{row.get('seed')}"
                for row in losing_trials[:12]
            )
            if len(losing_trials) > 12:
                losses += f", ... ({len(losing_trials)} total losses)"
            lines.append(f"Target-level losses: {losses}.")
        failed_conditions = [
            key for key, value in conditions.items()
            if key != "paper_grade_statistical_claim_supported" and not bool(value)
        ]
        if failed_conditions:
            lines.append("Failed proof conditions: `" + ", ".join(failed_conditions) + "`.")
    lines.append("")
    lines.append("Proof conditions: `" + json.dumps(conditions, sort_keys=True) + "`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def print_table(rows: list[dict[str, Any]]) -> None:
    print(",".join(CSV_COLUMNS))
    for row in rows:
        print(",".join(str(row.get(key, "")) for key in CSV_COLUMNS))


def selected_targets(raw: str, *, include_sensitivity: bool) -> list[TargetConfig]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if include_sensitivity:
        names.extend(["soft_contact", "stiff_contact"])
    out = []
    for name in names:
        if name not in TARGETS:
            raise SystemExit(f"unknown target {name!r}; choices: {', '.join(sorted(TARGETS))}")
        out.append(TARGETS[name])
    return out


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("at least one seed is required")
    return values


def rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
