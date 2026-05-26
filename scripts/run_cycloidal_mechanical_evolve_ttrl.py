#!/usr/bin/env python3
"""Run full iterative MechanicalEvolve TTRL for cycloidal/QDD actuators.

This harness is intentionally heavier than the closed-loop smoke checks:

* strict FreeCAD/Chrono SMC verifier with procedural fallback disabled
* matched audit-budget seed/random/CMA/verifier-gated baselines
* LLM evolution without weight updates
* iterative LoRA test-time adaptation where round N+1 samples from the
  adapter trained after round N
* regenerated-CAD stability audit for the best TTRL candidate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mechanical_evolve as mech  # noqa: E402
import optimize_cycloidal_chrono_candidates as cyclo  # noqa: E402


SCHEMA = "mech_bench.cycloidal_mechanical_evolve_ttrl.v1"
DEFAULT_MODEL = "mlx-community/Qwen3-32B-4bit"
CSV_COLUMNS = (
    "method",
    "candidate_count",
    "chrono_audits",
    "best_verified_reward",
    "verified_pass_rate",
    "cad_pass_rate",
    "chrono_real_geometry_rate",
    "lockup_rate",
    "best_id",
    "best_fast_reward",
    "best_out_omega_med",
    "best_ratio_error_pct",
    "best_power_balance_error_pct",
    "best_torque_ripple_pct",
    "adapter_updates",
    "trained_tokens",
)


@dataclass(frozen=True)
class CommandResult:
    status: str
    returncode: int
    command: list[str]
    log_path: str
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "command": self.command,
            "log_path": self.log_path,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


@dataclass
class PolicyState:
    method: str
    sample_method: str
    proposer: str
    archive: mech.MapElitesArchive
    evaluated: list[dict[str, Any]]
    rounds: list[dict[str, Any]]
    trainers: list[dict[str, Any]]
    adapter_path: Path | None = None
    adapter_updates: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/cycloidal_mechanical_evolve_ttrl")
    parser.add_argument("--results-json", default="docs/cycloidal_mechanical_evolve_ttrl_results.json")
    parser.add_argument("--results-csv", default="docs/cycloidal_mechanical_evolve_ttrl_results.csv")
    parser.add_argument("--model", default=os.environ.get(
        "MECHANICAL_EVOLVE_LORA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--proposals-per-round", type=int, default=40)
    parser.add_argument("--audits-per-round", type=int, default=20)
    parser.add_argument("--mutation-fill", type=int, default=8)
    parser.add_argument("--baseline-audits", type=int, default=80)
    parser.add_argument("--verifier-pool", type=int, default=192)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--duration-s", type=float, default=0.15)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--lora-iters", type=int, default=4)
    parser.add_argument("--lora-batch-size", type=int, default=1)
    parser.add_argument("--lora-grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--lora-num-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-max-examples", type=int, default=256)
    parser.add_argument("--lora-min-reward", type=float, default=1.0)
    parser.add_argument("--contact-force-limit-N", type=float, default=3000.0)
    parser.add_argument("--max-contacts", type=float, default=128.0)
    parser.add_argument("--power-balance-limit-pct", type=float, default=90.0)
    parser.add_argument("--torque-ripple-limit-pct", type=float, default=1000.0)
    parser.add_argument("--stability-repeats", type=int, default=3)
    parser.add_argument("--baseline-results-json", default=None)
    parser.add_argument("--baseline-timeout-s", type=float, default=21600.0)
    parser.add_argument("--sample-timeout-s", type=float, default=900.0)
    parser.add_argument("--audit-timeout-s", type=float, default=7200.0)
    parser.add_argument("--train-timeout-s", type=float, default=1800.0)
    parser.add_argument("--keep-out-dir", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-no-update", action="store_true")
    parser.add_argument("--skip-ttrl", action="store_true")
    parser.add_argument("--require-ttrl-win", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and not args.keep_out_dir:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = Path(args.results_json).expanduser().resolve()
    results_csv = Path(args.results_csv).expanduser().resolve()
    limits = cyclo.VerificationLimits(
        max_contact_force_rms_N=max(0.0, float(args.contact_force_limit_N)),
        max_contacts=max(1.0, float(args.max_contacts)),
        max_power_balance_error_pct=max(
            0.0, float(args.power_balance_limit_pct)),
        max_torque_ripple_pct=max(0.0, float(args.torque_ripple_limit_pct)),
    )

    baselines = {}
    bootstrap_archive = mech.MapElitesArchive()
    baseline_command: dict[str, Any] | None = None
    baseline_results_json = (
        Path(args.baseline_results_json).expanduser().resolve()
        if args.baseline_results_json else None
    )
    if baseline_results_json is not None and baseline_results_json.is_file():
        baselines = read_json(baseline_results_json)
        bootstrap_archive = archive_from_optimizer(baselines)
        out_baseline = out_dir / "baselines" / "cycloidal_optimizer_strict_matched.json"
        out_baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_results_json, out_baseline)
        baseline_command = {
            "status": "reused",
            "results_json": str(baseline_results_json),
        }
    elif args.skip_baselines:
        existing = out_dir / "baselines" / "cycloidal_optimizer_strict_matched.json"
        if existing.is_file():
            baselines = read_json(existing)
            bootstrap_archive = archive_from_optimizer(baselines)
            baseline_command = {
                "status": "reused",
                "results_json": str(existing),
            }
    else:
        baselines, baseline_command = run_baselines(
            args=args,
            out_dir=out_dir / "baselines",
            limits=limits,
        )
        bootstrap_archive = archive_from_optimizer(baselines)

    policy_results: dict[str, Any] = {}
    if args.skip_no_update:
        existing = out_dir / "llm_evolve_no_update" / "summary.json"
        if existing.is_file():
            policy_results["llm_evolve_no_update"] = read_json(existing)
    else:
        policy_results["llm_evolve_no_update"] = run_policy_loop(
            args=args,
            out_dir=out_dir / "llm_evolve_no_update",
            limits=limits,
            bootstrap_archive=bootstrap_archive,
            train_lora=False,
        )
    if not args.skip_ttrl:
        policy_results["mechanical_evolve_ttrl"] = run_policy_loop(
            args=args,
            out_dir=out_dir / "mechanical_evolve_ttrl",
            limits=limits,
            bootstrap_archive=bootstrap_archive,
            train_lora=True,
        )

    stability = {}
    ttrl_summary = policy_results.get("mechanical_evolve_ttrl", {})
    ttrl_best = ttrl_summary.get("best")
    if isinstance(ttrl_best, dict) and ttrl_best.get("params"):
        stability = stability_audit(
            args=args,
            out_dir=out_dir / "stability",
            best=ttrl_best,
            limits=limits,
        )

    method_table = aggregate_method_table(
        baselines=baselines,
        policy_results=policy_results,
    )
    win_conditions = win_conditions_from_table(
        method_table=method_table,
        stability=stability,
    )
    summary = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "model": args.model,
        "budget": {
            "rounds": int(args.rounds),
            "proposals_per_round": int(args.proposals_per_round),
            "audits_per_round": int(args.audits_per_round),
            "mutation_fill": int(args.mutation_fill),
            "baseline_audits_per_method": int(args.baseline_audits),
        },
        "verifier": {
            "contact_model": "smc",
            "procedural_cycloidal_fallback": False,
            "samples": int(args.samples),
            "duration_s": float(args.duration_s),
            "limits": limits.__dict__,
        },
        "baseline_command": baseline_command,
        "baselines": compact_baselines(baselines),
        "policy_results": policy_results,
        "stability_audit": stability,
        "method_table": method_table,
        "win_conditions": win_conditions,
    }
    safe = json_safe(summary)
    write_json(results_json, safe)
    write_table(results_csv, method_table)
    print_table(method_table)
    print("win_conditions=" + json.dumps(win_conditions, sort_keys=True))
    print(f"results_json={results_json}")
    print(f"results_csv={results_csv}")
    if args.require_ttrl_win and not win_conditions.get("ttrl_beats_all_baselines"):
        return 1
    return 0


def run_baselines(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    limits: cyclo.VerificationLimits,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "cycloidal_optimizer_strict_matched.json"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "optimize_cycloidal_chrono_candidates.py"),
        "--out-dir",
        str(out_dir / "assets"),
        "--results-json",
        str(results_json),
        "--results-csv",
        str(out_dir / "cycloidal_optimizer_strict_matched.csv"),
        "--random-candidates",
        str(max(1, int(args.baseline_audits))),
        "--cma-candidates",
        str(max(1, int(args.baseline_audits))),
        "--verifier-pool",
        str(max(int(args.verifier_pool), int(args.baseline_audits))),
        "--verifier-audit-k",
        str(max(1, int(args.baseline_audits))),
        "--samples",
        str(max(3, int(args.samples))),
        "--duration-s",
        str(max(1.0e-6, float(args.duration_s))),
        "--contact-force-limit-N",
        str(limits.max_contact_force_rms_N),
        "--max-contacts",
        str(limits.max_contacts),
        "--power-balance-limit-pct",
        str(limits.max_power_balance_error_pct),
        "--torque-ripple-limit-pct",
        str(limits.max_torque_ripple_pct),
    ]
    result = run_command(
        command,
        cwd=SCRIPT_DIR.parent,
        log_path=out_dir / "baseline.log",
        timeout_s=max(1.0, float(args.baseline_timeout_s)),
    )
    if not results_json.is_file():
        raise SystemExit(f"baseline run did not produce {results_json}")
    data = read_json(results_json)
    return data, result.to_dict()


def archive_from_optimizer(data: dict[str, Any]) -> mech.MapElitesArchive:
    archive = mech.MapElitesArchive()
    for row in data.get("candidates", []):
        if isinstance(row, dict):
            archive.insert(row)
    return archive


def run_policy_loop(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    limits: cyclo.VerificationLimits,
    bootstrap_archive: mech.MapElitesArchive,
    train_lora: bool,
) -> dict[str, Any]:
    method = "mechanical_evolve_ttrl" if train_lora else "llm_evolve_no_update"
    state = PolicyState(
        method=method,
        sample_method="mechanical_evolve_ttrl" if train_lora else "llm_evolution_no_update",
        proposer="mlx_lora_ttrl_policy" if train_lora else "mlx_base_archive_policy",
        archive=mech.MapElitesArchive.from_dict(bootstrap_archive.to_dict()),
        evaluated=[],
        rounds=[],
        trainers=[],
        adapter_path=None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "archive.json", state.archive.to_dict())
    bootstrap_rows = list(state.archive.cells.values())

    for round_idx in range(max(1, int(args.rounds))):
        round_dir = out_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        archive_path = round_dir / "archive_input.json"
        write_json(archive_path, state.archive.to_dict())
        proposals_path = round_dir / "candidates.jsonl"
        adapter_input = str(state.adapter_path) if state.adapter_path else None
        sample_result = sample_policy_round(
            args=args,
            round_dir=round_dir,
            archive_path=archive_path,
            proposals_path=proposals_path,
            state=state,
            round_idx=round_idx,
        )
        audit_summary, audit_command = audit_policy_round(
            args=args,
            round_dir=round_dir,
            archive=state.archive,
            proposals_path=proposals_path,
            limits=limits,
            round_idx=round_idx,
        )
        evaluated = [
            row for row in audit_summary.get("evaluated", [])
            if isinstance(row, dict)
        ]
        for row in evaluated:
            state.archive.insert(row)
        state.evaluated.extend(evaluated)
        write_json(out_dir / "archive.json", state.archive.to_dict())

        trainer = {"status": "skipped", "reason": "no_lora_update"}
        if train_lora:
            dataset_path = round_dir / "grpo_dataset.jsonl"
            dataset_count = write_grpo_dataset(
                dataset_path,
                rows=[*bootstrap_rows, *state.evaluated],
                archive=state.archive,
                limits=limits,
            )
            trainer = train_lora_round(
                args=args,
                round_dir=round_dir,
                dataset_path=dataset_path,
                archive_path=out_dir / "archive.json",
                previous_adapter=state.adapter_path,
                round_idx=round_idx,
            )
            trainer["dataset_count"] = dataset_count
            next_adapter = Path(trainer.get("adapter_path", ""))
            if trainer.get("status") == "completed" and (
                next_adapter / "adapters.safetensors"
            ).is_file():
                state.adapter_path = next_adapter
                state.adapter_updates += 1
            else:
                raise SystemExit(
                    "TTRL LoRA training failed before producing an adapter; "
                    f"see {round_dir / 'train.log'}"
                )
            state.trainers.append(trainer)

        round_record = {
            "round": round_idx,
            "adapter_input": adapter_input,
            "adapter_output": str(state.adapter_path) if state.adapter_path else None,
            "sample": sample_result.to_dict(),
            "audit": audit_command.to_dict(),
            "best": compact_best(mech.best_row(evaluated)),
            "evaluated_count": len(evaluated),
            "chrono_audits": sum(1 for row in evaluated if row.get("metrics")),
            "archive_cell_count": len(state.archive.cells),
            "trainer": trainer,
        }
        state.rounds.append(round_record)
        write_json(round_dir / "round_summary.json", json_safe(round_record))

    summary = summarize_policy_state(state)
    write_json(out_dir / "summary.json", json_safe(summary))
    return summary


def sample_policy_round(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    archive_path: Path,
    proposals_path: Path,
    state: PolicyState,
    round_idx: int,
) -> CommandResult:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "sample_mechanical_evolve_mlx.py"),
        "--out-jsonl",
        str(proposals_path),
        "--raw-json",
        str(round_dir / "raw_generations.json"),
        "--archive",
        str(archive_path),
        "--model",
        str(args.model),
        "--count",
        str(max(1, int(args.proposals_per_round))),
        "--batch-size",
        str(max(1, int(args.sample_batch_size))),
        "--max-tokens",
        str(max(32, int(args.max_tokens))),
        "--temp",
        str(max(0.0, float(args.temp))),
        "--top-p",
        str(max(0.0, min(1.0, float(args.top_p)))),
        "--seed",
        str(int(args.seed) + round_idx * 1009 + (17 if state.adapter_path else 0)),
        "--method",
        state.sample_method,
        "--proposer",
        state.proposer,
    ]
    if state.adapter_path is not None:
        command.extend(["--adapter-path", str(state.adapter_path)])
    return run_command(
        command,
        cwd=SCRIPT_DIR.parent,
        log_path=round_dir / "sample.log",
        timeout_s=max(1.0, float(args.sample_timeout_s)),
    )


def audit_policy_round(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    archive: mech.MapElitesArchive,
    proposals_path: Path,
    limits: cyclo.VerificationLimits,
    round_idx: int,
) -> tuple[dict[str, Any], CommandResult]:
    audit_dir = round_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "archive.json", archive.to_dict())
    proposal_count = count_jsonl(proposals_path)
    population = proposal_count + max(0, int(args.mutation_fill))
    command = [
        sys.executable,
        str(SCRIPT_DIR / "mechanical_evolve.py"),
        "--mode",
        "evolve-only",
        "--resume",
        "--no-seed-bootstrap",
        "--out-dir",
        str(audit_dir),
        "--seed",
        str(int(args.seed) + round_idx * 1009),
        "--generations",
        "1",
        "--population",
        str(max(1, population)),
        "--audit-k",
        str(max(1, int(args.audits_per_round))),
        "--proposal-jsonl",
        str(proposals_path),
        "--samples",
        str(max(3, int(args.samples))),
        "--duration-s",
        str(max(1.0e-6, float(args.duration_s))),
        "--contact-force-limit-N",
        str(limits.max_contact_force_rms_N),
        "--max-contacts",
        str(limits.max_contacts),
        "--power-balance-limit-pct",
        str(limits.max_power_balance_error_pct),
        "--torque-ripple-limit-pct",
        str(limits.max_torque_ripple_pct),
    ]
    result = run_command(
        command,
        cwd=SCRIPT_DIR.parent,
        log_path=round_dir / "audit.log",
        timeout_s=max(1.0, float(args.audit_timeout_s)),
    )
    summary = read_json(audit_dir / "summary.json")
    return summary, result


def train_lora_round(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    dataset_path: Path,
    archive_path: Path,
    previous_adapter: Path | None,
    round_idx: int,
) -> dict[str, Any]:
    train_dir = round_dir / "mlx_lora"
    adapter_path = train_dir / "adapters"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_mechanical_evolve_lora.py"),
        "--dataset",
        str(dataset_path),
        "--archive",
        str(archive_path),
        "--out-dir",
        str(train_dir),
        "--model",
        str(args.model),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(max(1, int(args.lora_iters))),
        "--batch-size",
        str(max(1, int(args.lora_batch_size))),
        "--grad-accumulation-steps",
        str(max(1, int(args.lora_grad_accumulation_steps))),
        "--learning-rate",
        str(float(args.lora_learning_rate)),
        "--num-layers",
        str(int(args.lora_num_layers)),
        "--lora-rank",
        str(max(1, int(args.lora_rank))),
        "--lora-scale",
        str(float(args.lora_scale)),
        "--lora-dropout",
        str(max(0.0, float(args.lora_dropout))),
        "--max-examples",
        str(max(1, int(args.lora_max_examples))),
        "--min-reward",
        str(float(args.lora_min_reward)),
        "--seed",
        str(int(args.seed) + round_idx),
    ]
    if previous_adapter is not None:
        prior_file = previous_adapter / "adapters.safetensors"
        if prior_file.is_file():
            command.extend(["--resume-adapter-file", str(prior_file)])
    result = run_command(
        command,
        cwd=SCRIPT_DIR.parent,
        log_path=round_dir / "train.log",
        timeout_s=max(1.0, float(args.train_timeout_s)),
    )
    summary = read_json(train_dir / "training_summary.json")
    trainer = summary.get("trainer", {}) if isinstance(summary, dict) else {}
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "adapter_path": str(adapter_path),
        "adapter_file_exists": (adapter_path / "adapters.safetensors").is_file(),
        "resume_adapter_file": str(previous_adapter / "adapters.safetensors")
        if previous_adapter else None,
        "example_count": summary.get("example_count") if summary else None,
        "best_training_reward": summary.get("best_training_reward") if summary else None,
        "train_loss": trainer.get("train_loss"),
        "val_loss": trainer.get("val_loss"),
        "trained_tokens": trainer.get("trained_tokens"),
        "peak_mem_gb": trainer.get("peak_mem_gb"),
        "command": result.to_dict(),
    }


def write_grpo_dataset(
    path: Path,
    *,
    rows: Iterable[dict[str, Any]],
    archive: mech.MapElitesArchive,
    limits: cyclo.VerificationLimits,
) -> int:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
        parent = str(proposal.get("parent_id") or row.get("parent_id") or "root")
        groups.setdefault(parent, []).append(row)
    prompt = {
        "task": "propose cycloidal/QDD actuator parameter programs",
        "design_variables": list(mech.DESIGN_VARIABLES),
        "paper_gate": limits.__dict__,
        "elites": [mech.compact_row(row) for row in archive.elites(limit=8)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for parent_id, group_rows in groups.items():
            responses = []
            for row in group_rows:
                responses.append({
                    "candidate_id": row.get("id"),
                    "params": row.get("params"),
                    "reward": float(row.get("verified_reward", 0.0) or 0.0),
                    "defects": row.get("defects", []),
                })
            if not responses:
                continue
            f.write(json.dumps(json_safe({
                "parent_id": parent_id,
                "prompt": prompt,
                "responses": responses,
            }), sort_keys=True) + "\n")
            count += 1
    return count


def stability_audit(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    best: dict[str, Any],
    limits: cyclo.VerificationLimits,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    params = dict(best.get("params") or {})
    repeats = max(0, int(args.stability_repeats))
    for idx in range(repeats):
        candidate = cyclo.Candidate(
            id=f"stability_{idx:03d}_{best.get('id', 'best')}",
            method="mechanical_evolve_ttrl_stability",
            params=params,
            proposer="regenerated_cad_reaudit",
        )
        rows.append(cyclo._evaluate_candidate(
            candidate,
            out_dir / f"repeat_{idx:03d}",
            samples=max(3, int(args.samples)),
            duration_s=max(1.0e-6, float(args.duration_s)),
            limits=limits,
        ))
    passed = [row for row in rows if row.get("verified_gate_passed")]
    return {
        "repeat_count": repeats,
        "pass_count": len(passed),
        "pass_rate": round(len(passed) / repeats, 6) if repeats else None,
        "best_id": best.get("id"),
        "params": params,
        "rows": rows,
    }


def summarize_policy_state(state: PolicyState) -> dict[str, Any]:
    best = mech.best_row(state.evaluated)
    metrics = aggregate_rows(state.evaluated)
    return {
        "schema": f"{SCHEMA}.policy",
        "method": state.method,
        "sample_method": state.sample_method,
        "adapter_path": str(state.adapter_path) if state.adapter_path else None,
        "adapter_updates": state.adapter_updates,
        "archive_cell_count": len(state.archive.cells),
        "rounds": state.rounds,
        "trainers": state.trainers,
        "best": compact_best(best),
        "metrics": metrics,
    }


def aggregate_method_table(
    *,
    baselines: dict[str, Any],
    policy_results: dict[str, Any],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in baselines.get("method_table", []):
        method = str(row.get("method"))
        candidates = [
            c for c in baselines.get("candidates", [])
            if c.get("method") == method
        ]
        best = best_row(candidates)
        metrics = aggregate_rows(candidates)
        table.append(method_row(
            method=method,
            rows=candidates,
            best=best,
            adapter_updates=0,
            trained_tokens=0,
            override_best=row.get("best_verified_reward"),
            metrics=metrics,
        ))
    for method, summary in policy_results.items():
        rows = []
        for round_record in summary.get("rounds", []):
            audit_log = round_record.get("audit", {})
            _ = audit_log
        rows = collect_policy_rows(summary)
        table.append(method_row(
            method=method,
            rows=rows,
            best=summary.get("best"),
            adapter_updates=int(summary.get("adapter_updates", 0) or 0),
            trained_tokens=sum(
                int(t.get("trained_tokens") or 0)
                for t in summary.get("trainers", [])
                if isinstance(t, dict)
            ),
            metrics=summary.get("metrics") or aggregate_rows(rows),
        ))
    return table


def collect_policy_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for round_record in summary.get("rounds", []):
        # Full evaluated rows are intentionally not copied into each round record
        # to keep the top-level JSON tractable; metrics carry the counts.
        _ = round_record
    best = summary.get("best")
    return [best] if isinstance(best, dict) and best else []


def method_row(
    *,
    method: str,
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
    adapter_updates: int,
    trained_tokens: int,
    metrics: dict[str, Any],
    override_best: Any = None,
) -> dict[str, Any]:
    best = best or {}
    best_metrics = best.get("metrics") or {}
    best_reward = (
        float(override_best)
        if override_best is not None
        else float(best.get("verified_reward", 0.0) or 0.0)
    )
    return {
        "method": method,
        "candidate_count": metrics.get("candidate_count", len(rows)),
        "chrono_audits": metrics.get("chrono_audits"),
        "best_verified_reward": round(best_reward, 6),
        "verified_pass_rate": metrics.get("verified_pass_rate"),
        "cad_pass_rate": metrics.get("cad_pass_rate"),
        "chrono_real_geometry_rate": metrics.get("chrono_real_geometry_rate"),
        "lockup_rate": metrics.get("lockup_rate"),
        "best_id": best.get("id"),
        "best_fast_reward": best.get("fast_reward"),
        "best_out_omega_med": best_metrics.get("out_omega_med"),
        "best_ratio_error_pct": best_metrics.get("ratio_error_pct"),
        "best_power_balance_error_pct": best_metrics.get("power_balance_error_pct"),
        "best_torque_ripple_pct": best_metrics.get("torque_ripple_pct"),
        "adapter_updates": adapter_updates,
        "trained_tokens": trained_tokens,
    }


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_list = [row for row in rows if isinstance(row, dict)]
    audited = [row for row in rows_list if row.get("metrics")]
    cad_pass = [
        row for row in rows_list
        if row.get("cad_generated") and row.get("cad_static_ok")
    ]
    real = [row for row in rows_list if row.get("chrono_real_geometry")]
    valid = [row for row in rows_list if row.get("verified_gate_passed")]
    lockups = [
        row for row in audited
        if metric_float(row, "lockup_detected", 0.0) != 0.0
    ]
    return {
        "candidate_count": len(rows_list),
        "chrono_audits": len(audited),
        "verified_pass_rate": rate(len(valid), len(rows_list)),
        "cad_pass_rate": rate(len(cad_pass), len(rows_list)),
        "chrono_real_geometry_rate": rate(len(real), len(rows_list)),
        "lockup_rate": rate(len(lockups), len(audited)),
    }


def compact_baselines(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {}
    return {
        "method_table": data.get("method_table", []),
        "best_by_method": data.get("best_by_method", {}),
        "candidate_count": data.get("candidate_count"),
        "win_condition_met": data.get("win_condition_met"),
        "results_path": data.get("results_path"),
    }


def compact_best(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {
        "id": row.get("id"),
        "method": row.get("method"),
        "params": row.get("params"),
        "fast_reward": row.get("fast_reward"),
        "verified_reward": row.get("verified_reward"),
        "verified_gate_passed": row.get("verified_gate_passed"),
        "defects": row.get("defects", []),
        "metrics": {
            key: (row.get("metrics") or {}).get(key)
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
            if key in (row.get("metrics") or {})
        },
    }


def best_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rows_list = [row for row in rows if isinstance(row, dict)]
    if not rows_list:
        return None
    return max(
        rows_list,
        key=lambda row: (
            float(row.get("verified_reward", 0.0) or 0.0),
            1.0 if row.get("verified_gate_passed") else 0.0,
            float(row.get("fast_reward", 0.0) or 0.0),
        ),
    )


def win_conditions_from_table(
    *,
    method_table: list[dict[str, Any]],
    stability: dict[str, Any],
) -> dict[str, Any]:
    rows = {row["method"]: row for row in method_table}
    ttrl = rows.get("mechanical_evolve_ttrl", {})
    no_update = rows.get("llm_evolve_no_update", {})
    baseline_rewards = [
        float(row.get("best_verified_reward", 0.0) or 0.0)
        for method, row in rows.items()
        if method not in {"mechanical_evolve_ttrl"}
    ]
    ttrl_reward = float(ttrl.get("best_verified_reward", 0.0) or 0.0)
    return {
        "ttrl_beats_no_update": ttrl_reward > float(
            no_update.get("best_verified_reward", 0.0) or 0.0),
        "ttrl_beats_all_baselines": (
            bool(baseline_rewards) and ttrl_reward > max(baseline_rewards)
        ),
        "ttrl_has_adapter_updates": int(ttrl.get("adapter_updates", 0) or 0) > 0,
        "best_survives_regenerated_cad": (
            stability.get("repeat_count", 0) > 0
            and stability.get("pass_count") == stability.get("repeat_count")
        ),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_s: float | None = None,
) -> CommandResult:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    stdout = ""
    stderr = ""
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
        stderr = completed.stderr
        returncode = int(completed.returncode)
        status = "completed" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        stdout = ensure_text(exc.stdout)
        stderr = ensure_text(exc.stderr)
        returncode = 124
        status = "timeout"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n"
        + stdout
        + "\nSTDERR\n"
        + stderr
    )
    return CommandResult(
        status=status,
        returncode=returncode,
        command=command,
        log_path=str(log_path),
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
    )


def ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def metric_float(row: dict[str, Any], key: str, default: float) -> float:
    try:
        return float((row.get("metrics") or {}).get(key, default))
    except (TypeError, ValueError):
        return default


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
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_table(path: Path, table: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in table:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def print_table(table: list[dict[str, Any]]) -> None:
    print(",".join(CSV_COLUMNS))
    for row in table:
        print(",".join(str(row.get(field, "")) for field in CSV_COLUMNS))


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
