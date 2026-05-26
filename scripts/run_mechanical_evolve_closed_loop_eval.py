#!/usr/bin/env python3
"""Run repeated Qwen/LoRA MechanicalEvolve CAD+Chrono evaluations.

This is the fixed-budget paper harness for the current branch setup. For each
seed it samples candidates from the base 32B MLX policy and, when an adapter is
provided, the adapted LoRA policy. Each candidate is then audited through
``mechanical_evolve.py`` with FreeCAD/Chrono SMC and procedural fallback off.
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
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "mlx-community/Qwen3-32B-4bit"
SCHEMA = "mech_bench.mechanical_evolve.closed_loop_eval.v1"
CSV_COLUMNS = (
    "method",
    "trial_count",
    "candidate_count",
    "chrono_audits",
    "best_verified_reward_max",
    "best_verified_reward_mean",
    "best_verified_reward_stderr",
    "trial_success_rate",
    "verified_pass_rate",
    "cad_pass_rate",
    "chrono_real_geometry_rate",
    "lockup_rate",
    "mean_audits_to_first_valid",
    "best_id",
    "best_seed",
    "best_fast_reward",
    "best_out_omega_med",
    "best_ratio_observed",
    "best_ratio_error_pct",
    "best_max_penetration_mm",
    "best_contact_force_rms_N",
    "best_n_contacts_max",
    "best_power_balance_error_pct",
    "best_torque_ripple_pct",
)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    sample_method: str
    proposer: str
    adapter_path: Path | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/mechanical_evolve_closed_loop_eval")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--model", default=os.environ.get(
        "MECHANICAL_EVOLVE_LORA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--adapter-path", default=os.environ.get(
        "MECHANICAL_EVOLVE_LORA_ADAPTER"))
    parser.add_argument("--archive", default=None)
    parser.add_argument("--seeds", default="20260525,20260526,20260527")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--audit-k", type=int, default=0)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--duration-s", type=float, default=0.15)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--contact-force-limit-N", type=float, default=3000.0)
    parser.add_argument("--max-contacts", type=float, default=128.0)
    parser.add_argument("--power-balance-limit-pct", type=float, default=1.0e12)
    parser.add_argument("--torque-ripple-limit-pct", type=float, default=1.0e12)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-adapted", action="store_true")
    parser.add_argument("--dry-run-audits", action="store_true")
    parser.add_argument("--require-adapted-improvement", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = Path(args.results_json).expanduser().resolve() if (
        args.results_json
    ) else out_dir / "summary.json"
    results_csv = Path(args.results_csv).expanduser().resolve() if (
        args.results_csv
    ) else out_dir / "method_table.csv"
    seeds = parse_seeds(args.seeds)
    methods = method_configs(args)
    if not methods:
        raise SystemExit("no methods selected")

    trials: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            trials.append(run_trial(
                args=args,
                out_dir=out_dir,
                method=method,
                seed=seed,
            ))

    table = aggregate_trials(trials)
    summary = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter_path": str(Path(args.adapter_path).expanduser().resolve())
        if args.adapter_path else None,
        "seeds": seeds,
        "candidate_budget_per_seed": max(1, int(args.count)),
        "audit_budget_per_seed": audit_budget(args),
        "verifier": {
            "contact_model": "smc",
            "procedural_cycloidal_fallback": False,
            "samples": max(3, int(args.samples)),
            "duration_s": max(1.0e-6, float(args.duration_s)),
            "input_speed_rad_s": 10.0,
            "output_load_Nm": 0.75,
            "limits": {
                "max_contact_force_rms_N": args.contact_force_limit_N,
                "max_contacts": args.max_contacts,
                "max_power_balance_error_pct": (
                    args.power_balance_limit_pct
                ),
                "max_torque_ripple_pct": args.torque_ripple_limit_pct,
            },
        },
        "method_table": table,
        "trials": trials,
    }
    write_json(results_json, json_safe(summary))
    write_table(results_csv, table)
    print_table(table)
    print(f"results_json={results_json}")
    print(f"results_csv={results_csv}")
    if args.require_adapted_improvement and not adapted_improved(table):
        return 1
    return 0


def method_configs(args: argparse.Namespace) -> list[MethodConfig]:
    methods: list[MethodConfig] = []
    if not args.skip_base:
        methods.append(MethodConfig(
            method="qwen3_32b_zero_shot",
            sample_method="llm_zero_shot",
            proposer="mlx_base_policy",
            adapter_path=None,
        ))
    adapter = Path(args.adapter_path).expanduser().resolve() if (
        args.adapter_path
    ) else None
    if not args.skip_adapted:
        if adapter is None:
            raise SystemExit(
                "--adapter-path or MECHANICAL_EVOLVE_LORA_ADAPTER is required "
                "unless --skip-adapted is set"
            )
        if not (adapter / "adapters.safetensors").is_file():
            raise SystemExit(f"missing LoRA adapter file under {adapter}")
        methods.append(MethodConfig(
            method="mechanical_evolve_lora_ttrl",
            sample_method="mechanical_evolve_ttrl",
            proposer="mlx_lora_policy",
            adapter_path=adapter,
        ))
    return methods


def run_trial(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    method: MethodConfig,
    seed: int,
) -> dict[str, Any]:
    trial_dir = out_dir / method.method / f"seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    proposals_jsonl = trial_dir / "candidates.jsonl"
    raw_json = trial_dir / "raw_generations.json"
    sample_command = [
        sys.executable,
        str(SCRIPT_DIR / "sample_mechanical_evolve_mlx.py"),
        "--out-jsonl",
        str(proposals_jsonl),
        "--raw-json",
        str(raw_json),
        "--model",
        str(args.model),
        "--count",
        str(max(1, int(args.count))),
        "--batch-size",
        str(max(1, int(args.sample_batch_size))),
        "--max-tokens",
        str(max(32, int(args.max_tokens))),
        "--temp",
        str(max(0.0, float(args.temp))),
        "--top-p",
        str(max(0.0, min(1.0, float(args.top_p)))),
        "--seed",
        str(seed),
        "--method",
        method.sample_method,
        "--proposer",
        method.proposer,
    ]
    if method.adapter_path is not None:
        sample_command.extend(["--adapter-path", str(method.adapter_path)])
    if args.archive:
        sample_command.extend([
            "--archive",
            str(Path(args.archive).expanduser().resolve()),
        ])
    sample_result = run_command(
        sample_command,
        cwd=SCRIPT_DIR.parent,
        log_path=trial_dir / "sample.log",
    )
    proposal_count = count_jsonl(proposals_jsonl)
    audit_summary: dict[str, Any] = {}
    audit_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "no_valid_sampled_proposals",
    }
    if sample_result["returncode"] == 0 and proposal_count > 0:
        audit_dir = trial_dir / "audit"
        audit_count = min(audit_budget(args), proposal_count)
        audit_command = [
            sys.executable,
            str(SCRIPT_DIR / "mechanical_evolve.py"),
            "--mode",
            "evolve-only",
            "--no-seed-bootstrap",
            "--out-dir",
            str(audit_dir),
            "--seed",
            str(seed),
            "--generations",
            "1",
            "--population",
            str(proposal_count),
            "--audit-k",
            str(audit_count),
            "--proposal-jsonl",
            str(proposals_jsonl),
            "--samples",
            str(max(3, int(args.samples))),
            "--duration-s",
            str(max(1.0e-6, float(args.duration_s))),
            "--contact-force-limit-N",
            str(max(0.0, float(args.contact_force_limit_N))),
            "--max-contacts",
            str(max(1.0, float(args.max_contacts))),
            "--power-balance-limit-pct",
            str(max(0.0, float(args.power_balance_limit_pct))),
            "--torque-ripple-limit-pct",
            str(max(0.0, float(args.torque_ripple_limit_pct))),
        ]
        if args.dry_run_audits:
            audit_command.append("--dry-run")
        audit_result = run_command(
            audit_command,
            cwd=SCRIPT_DIR.parent,
            log_path=trial_dir / "audit.log",
        )
        summary_path = audit_dir / "summary.json"
        if summary_path.is_file():
            audit_summary = read_json(summary_path)
    return {
        "method": method.method,
        "seed": seed,
        "trial_dir": str(trial_dir),
        "proposal_jsonl": str(proposals_jsonl),
        "raw_json": str(raw_json),
        "sample": sample_result,
        "sampled_candidate_count": proposal_count,
        "audit": audit_result,
        "audit_summary": compact_audit_summary(audit_summary),
        "trial_metrics": trial_metrics(audit_summary),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n"
        + "STDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr
    )
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "command": command,
        "log_path": str(log_path),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def trial_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    evaluated = [
        row for row in summary.get("evaluated", [])
        if isinstance(row, dict)
    ] if isinstance(summary, dict) else []
    best = summary.get("best") if isinstance(summary, dict) else None
    if not isinstance(best, dict):
        best = best_row(evaluated)
    audited = [row for row in evaluated if row.get("metrics")]
    lockups = [
        metric_float(row, "lockup_detected", 0.0) != 0.0
        for row in audited
    ]
    first_valid = None
    for idx, row in enumerate(evaluated, start=1):
        if row.get("verified_gate_passed"):
            first_valid = idx
            break
    return {
        "candidate_count": len(evaluated),
        "chrono_audits": len(audited),
        "best_verified_reward": round(float(
            (best or {}).get("verified_reward", 0.0) or 0.0), 6),
        "best_fast_reward": round(max(
            (float(row.get("fast_reward", 0.0) or 0.0) for row in evaluated),
            default=0.0,
        ), 6),
        "verified_pass_count": sum(
            1 for row in evaluated if row.get("verified_gate_passed")),
        "cad_pass_count": sum(
            1 for row in evaluated
            if row.get("cad_generated") and row.get("cad_static_ok")
        ),
        "chrono_real_geometry_count": sum(
            1 for row in evaluated if row.get("chrono_real_geometry")),
        "lockup_count": sum(1 for value in lockups if value),
        "lockup_denominator": len(lockups),
        "audits_to_first_valid": first_valid,
        "best": compact_best(best),
    }


def aggregate_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_method.setdefault(str(trial.get("method")), []).append(trial)
    table: list[dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        metrics = [row.get("trial_metrics", {}) for row in rows]
        rewards = [
            float(item.get("best_verified_reward", 0.0) or 0.0)
            for item in metrics
        ]
        best_trial = max(
            rows,
            key=lambda row: float(
                row.get("trial_metrics", {})
                .get("best_verified_reward", 0.0) or 0.0
            ),
        )
        best = best_trial.get("trial_metrics", {}).get("best") or {}
        candidates = sum_int(metrics, "candidate_count")
        audits = sum_int(metrics, "chrono_audits")
        valid_counts = sum_int(metrics, "verified_pass_count")
        cad_counts = sum_int(metrics, "cad_pass_count")
        real_geom_counts = sum_int(metrics, "chrono_real_geometry_count")
        lockup_den = sum_int(metrics, "lockup_denominator")
        first_valid = [
            int(item["audits_to_first_valid"])
            for item in metrics
            if item.get("audits_to_first_valid") is not None
        ]
        table.append({
            "method": method,
            "trial_count": len(rows),
            "candidate_count": candidates,
            "chrono_audits": audits,
            "best_verified_reward_max": round(max(rewards, default=0.0), 6),
            "best_verified_reward_mean": round(mean(rewards), 6),
            "best_verified_reward_stderr": round(stderr(rewards), 6),
            "trial_success_rate": round(
                sum(1 for reward in rewards if reward > 0.0) / len(rows), 6
            ) if rows else 0.0,
            "verified_pass_rate": rate(valid_counts, candidates),
            "cad_pass_rate": rate(cad_counts, candidates),
            "chrono_real_geometry_rate": rate(real_geom_counts, candidates),
            "lockup_rate": rate(sum_int(metrics, "lockup_count"), lockup_den),
            "mean_audits_to_first_valid": round(mean(first_valid), 6)
            if first_valid else None,
            "best_id": best.get("id"),
            "best_seed": best_trial.get("seed"),
            "best_fast_reward": best.get("fast_reward"),
            "best_out_omega_med": best.get("metrics", {}).get("out_omega_med"),
            "best_ratio_observed": best.get("metrics", {}).get(
                "ratio_observed"),
            "best_ratio_error_pct": best.get("metrics", {}).get(
                "ratio_error_pct"),
            "best_max_penetration_mm": best.get("metrics", {}).get(
                "max_penetration_mm"),
            "best_contact_force_rms_N": best.get("metrics", {}).get(
                "contact_force_rms_N"),
            "best_n_contacts_max": best.get("metrics", {}).get(
                "n_contacts_max"),
            "best_power_balance_error_pct": best.get("metrics", {}).get(
                "power_balance_error_pct"),
            "best_torque_ripple_pct": best.get("metrics", {}).get(
                "torque_ripple_pct"),
        })
    return table


def compact_audit_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {}
    return {
        "mode": summary.get("mode"),
        "seed": summary.get("seed"),
        "archive_cell_count": summary.get("archive_cell_count"),
        "verifier": summary.get("verifier"),
        "best": compact_best(summary.get("best")),
        "evaluated_count": len(summary.get("evaluated", [])),
    }


def compact_best(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    metrics = row.get("metrics") or {}
    return {
        "id": row.get("id"),
        "method": row.get("method"),
        "params": row.get("params"),
        "fast_reward": row.get("fast_reward"),
        "verified_reward": row.get("verified_reward"),
        "verified_gate_passed": row.get("verified_gate_passed"),
        "defects": row.get("defects", []),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "lockup_detected",
                "out_omega_med",
                "ratio_observed",
                "ratio_error_pct",
                "max_penetration_mm",
                "max_constraint_error_mm",
                "contact_force_rms_N",
                "n_contacts_max",
                "power_balance_error_pct",
                "torque_ripple_pct",
            )
            if key in metrics
        },
    }


def adapted_improved(table: list[dict[str, Any]]) -> bool:
    rows = {row.get("method"): row for row in table}
    base = rows.get("qwen3_32b_zero_shot")
    adapted = rows.get("mechanical_evolve_lora_ttrl")
    if not base or not adapted:
        return False
    return (
        float(adapted.get("best_verified_reward_mean", 0.0) or 0.0)
        > float(base.get("best_verified_reward_mean", 0.0) or 0.0)
        and float(adapted.get("best_verified_reward_max", 0.0) or 0.0)
        > float(base.get("best_verified_reward_max", 0.0) or 0.0)
    )


def print_table(table: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "trial_count",
        "candidate_count",
        "best_verified_reward_max",
        "best_verified_reward_mean",
        "verified_pass_rate",
        "lockup_rate",
        "best_id",
    ]
    print(",".join(fields))
    for row in table:
        print(",".join(str(row.get(field, "")) for field in fields))


def write_table(path: Path, table: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in table:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("verified_reward", 0.0) or 0.0),
            1.0 if row.get("verified_gate_passed") else 0.0,
            float(row.get("fast_reward", 0.0) or 0.0),
        ),
    )


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise SystemExit("at least one seed is required")
    return seeds


def audit_budget(args: argparse.Namespace) -> int:
    explicit = int(args.audit_k)
    return max(1, explicit if explicit > 0 else int(args.count))


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def metric_float(row: dict[str, Any], key: str, default: float) -> float:
    try:
        return float((row.get("metrics") or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0) or 0) for row in rows))


def mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


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
