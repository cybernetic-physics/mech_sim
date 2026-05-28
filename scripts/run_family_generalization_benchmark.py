#!/usr/bin/env python3
"""Run the family-held-out MechanicalEvolve / RLVR benchmark.

This is the paper-facing wrapper for the broader claim:
mechanical reasoning should transfer across unseen mechanism families under a
matched verifier budget.

The script freezes the family split, trains an SFT baseline and an RLVR model
on the seen families, then evaluates frozen, SFT, RLVR, and no-update search
baselines on the held-out families.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SEEN = "cycloidal,belt,chain,rack_pinion,fourbar"
DEFAULT_UNSEEN = "planetary,lead_screw,cam_follower,slider_crank"
DEFAULT_RUNNER_PYTHON = "/Users/nataliakokoromyti/Projects/worldlines/.venv/bin/python"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model_path: str | None
    samples_per_task: int
    adapter_updates: int = 0
    trained_tokens: int = 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--runner-python", default=DEFAULT_RUNNER_PYTHON)
    p.add_argument("--worldlines-base-url", default="http://127.0.0.1:18100")
    p.add_argument("--api-key", default="wld-local")
    p.add_argument("--manage-worldlines", action="store_true",
                   help="start a fresh local Worldlines backend for each "
                        "training/eval phase. This works around the "
                        "single-session PEFT backend used on local MPS.")
    p.add_argument("--worldlines-root",
                   default="/Users/nataliakokoromyti/Projects/worldlines")
    p.add_argument("--worldlines-venv",
                   default="/Users/nataliakokoromyti/Projects/worldlines/.venv")
    p.add_argument("--worldlines-artifact-root",
                   default="/tmp/wld-family-artifacts")
    p.add_argument("--worldlines-launch-timeout-s", type=float, default=600.0)
    p.add_argument("--tasks-root", default="tasks")
    p.add_argument("--out-dir", default="runs/family_generalization_benchmark")
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--seen-families", default=DEFAULT_SEEN)
    p.add_argument("--unseen-families", default=DEFAULT_UNSEEN)
    p.add_argument("--split-seed", type=int, default=20260528)
    p.add_argument("--train-rounds", type=int, default=6)
    p.add_argument("--tasks-per-round", type=int, default=4)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--max-context-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--train-timeout-s", type=float, default=21600.0)
    p.add_argument("--eval-timeout-s", type=float, default=21600.0)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--allow-zero-update-models", action="store_true",
                   help="continue evaluation even if an SFT/RLVR training "
                        "phase produces zero optimizer steps; intended only "
                        "for plumbing/debug runs, not paper-facing results")
    p.add_argument("--keep-out-dir", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and not args.keep_out_dir:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path(args.docs_dir).expanduser().resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    split_dir = out_dir / "splits"
    split_json = split_dir / "split_manifest.json"
    run([
        sys.executable,
        str(SCRIPT_DIR / "freeze_mechbench_family_splits.py"),
        "--tasks-root",
        args.tasks_root,
        "--out-dir",
        str(split_dir),
        "--manifest-json",
        str(split_json),
        "--seen-families",
        args.seen_families,
        "--unseen-families",
        args.unseen_families,
        "--seed",
        str(args.split_seed),
    ], cwd=REPO_ROOT)
    split = json.loads(split_json.read_text())
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"

    eval_rows: list[dict[str, Any]] = []

    frozen_run = out_dir / "init_frozen"
    run_with_managed_worldlines(
        args,
        lambda: init_live_session(
            run_dir=frozen_run,
            run_name="family_frozen",
            base_model=args.base_model,
            runner_python=args.runner_python,
            backend_url=args.worldlines_base_url,
            api_key=args.api_key,
            tasks_root=args.tasks_root,
            max_context_tokens=args.max_context_tokens,
            lora_rank=args.lora_rank,
            timeout_s=args.train_timeout_s,
        ),
    )
    for method in [
        MethodSpec("frozen_model", None, args.samples_per_task),
        MethodSpec("no_update_search", None, args.samples_per_task),
    ]:
        report_dir = out_dir / f"eval_{method.name}"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = run_with_managed_worldlines(
            args,
            lambda method=method, report_dir=report_dir: run_sample_and_score(
                report_dir=report_dir,
                base_model=args.base_model,
                runner_python=args.runner_python,
                model_path=method.model_path,
                tasks_root=args.tasks_root,
                split_file=test_split,
                samples_per_task=method.samples_per_task,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                concurrency=args.concurrency,
                base_url=args.worldlines_base_url,
                api_key=args.api_key,
            ),
        )
        eval_rows.append(flatten_eval(method, summary))

    sft_run = out_dir / "train_sft"
    sft_model = run_with_managed_worldlines(
        args,
        lambda: train_model(
            run_dir=sft_run,
            run_name="family_sft",
            base_model=args.base_model,
            runner_python=args.runner_python,
            backend_url=args.worldlines_base_url,
            api_key=args.api_key,
            tasks_root=args.tasks_root,
            split_file=train_split,
            train_rounds=args.train_rounds,
            tasks_per_round=args.tasks_per_round,
            samples_per_task=args.samples_per_task,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_context_tokens=args.max_context_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            train_timeout_s=args.train_timeout_s,
            eval_timeout_s=args.eval_timeout_s,
            concurrency=args.concurrency,
            lora_rank=args.lora_rank,
            lr=args.lr,
            supervised_only=True,
            allow_zero_update_models=args.allow_zero_update_models,
        ),
    )
    sft_method = MethodSpec(
        "sft_model",
        sft_model["path"],
        args.samples_per_task,
        adapter_updates=int(sft_model["adapter_updates"]),
        trained_tokens=int(sft_model["trained_tokens"]),
    )
    report_dir = out_dir / "eval_sft_model"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = run_with_managed_worldlines(
        args,
        lambda: run_sample_and_score(
            report_dir=report_dir,
            base_model=args.base_model,
            runner_python=args.runner_python,
            model_path=sft_method.model_path,
            tasks_root=args.tasks_root,
            split_file=test_split,
            samples_per_task=sft_method.samples_per_task,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            concurrency=args.concurrency,
            base_url=args.worldlines_base_url,
            api_key=args.api_key,
        ),
    )
    eval_rows.append(flatten_eval(sft_method, summary))

    rlvr_run = out_dir / "train_rlvr"
    rlvr_model = run_with_managed_worldlines(
        args,
        lambda: train_model(
            run_dir=rlvr_run,
            run_name="family_rlvr",
            base_model=args.base_model,
            runner_python=args.runner_python,
            backend_url=args.worldlines_base_url,
            api_key=args.api_key,
            tasks_root=args.tasks_root,
            split_file=train_split,
            train_rounds=args.train_rounds,
            tasks_per_round=args.tasks_per_round,
            samples_per_task=args.samples_per_task,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            max_context_tokens=args.max_context_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            train_timeout_s=args.train_timeout_s,
            eval_timeout_s=args.eval_timeout_s,
            concurrency=args.concurrency,
            lora_rank=args.lora_rank,
            lr=args.lr,
            supervised_only=False,
            allow_zero_update_models=args.allow_zero_update_models,
        ),
    )
    rlvr_method = MethodSpec(
        "mechanical_evolve_ttrl",
        rlvr_model["path"],
        args.samples_per_task,
        adapter_updates=int(rlvr_model["adapter_updates"]),
        trained_tokens=int(rlvr_model["trained_tokens"]),
    )
    report_dir = out_dir / "eval_mechanical_evolve_ttrl"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = run_with_managed_worldlines(
        args,
        lambda: run_sample_and_score(
            report_dir=report_dir,
            base_model=args.base_model,
            runner_python=args.runner_python,
            model_path=rlvr_method.model_path,
            tasks_root=args.tasks_root,
            split_file=test_split,
            samples_per_task=rlvr_method.samples_per_task,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            concurrency=args.concurrency,
            base_url=args.worldlines_base_url,
            api_key=args.api_key,
        ),
    )
    eval_rows.append(flatten_eval(rlvr_method, summary))

    write_results(docs_dir, split, eval_rows)
    print(json.dumps({
        "split_manifest": str(split_json),
        "sft_model": sft_model["path"],
        "rlvr_model": rlvr_model["path"],
        "results_dir": str(docs_dir),
    }, indent=2, sort_keys=True))
    return 0


def train_model(
    *,
    run_dir: Path,
    run_name: str,
    base_model: str,
    runner_python: str,
    backend_url: str,
    api_key: str,
    tasks_root: str,
    split_file: Path,
    train_rounds: int,
    tasks_per_round: int,
    samples_per_task: int,
    max_turns: int,
    max_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    train_timeout_s: float,
    eval_timeout_s: float,
    concurrency: int,
    lora_rank: int,
    lr: float,
    supervised_only: bool,
    allow_zero_update_models: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        runner_python,
        str(REPO_ROOT / "rl" / "train_grpo.py"),
        "--base-model",
        base_model,
        "--backend-url",
        backend_url,
        "--api-key",
        api_key,
        "--rollout-backend",
        "worldlines_sampling",
        "--run-name",
        run_name,
        "--runs-root",
        str(run_dir.parent.relative_to(REPO_ROOT)),
        "--tasks-root",
        tasks_root,
        "--split-file",
        str(split_file),
        "--family-balanced-task-sampler",
        "--rounds",
        str(train_rounds),
        "--tasks-per-round",
        str(tasks_per_round),
        "--samples-per-task",
        str(samples_per_task),
        "--max-turns",
        str(max_turns),
        "--max-tokens-per-turn",
        str(max_tokens),
        "--max-context-tokens",
        str(max_context_tokens),
        "--rollout-temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--lr",
        str(lr),
        "--lora-rank",
        str(lora_rank),
        "--reference-sft-split-file",
        str(split_file),
        "--reference-sft-weight",
        "1.0",
        "--reference-sft-per-step",
        str(tasks_per_round),
        "--save-final-sampler-name",
        f"{run_name}_final",
    ]
    if supervised_only:
        cmd.extend([
            "--sft-warmup-rounds",
            str(train_rounds),
            "--positive-only-passes",
    ])
    env = dict(os.environ)
    env["WORLDLINES_BASE_URL"] = backend_url
    env["WORLDLINES_API_KEY"] = api_key
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    run(cmd, cwd=REPO_ROOT, env=env, timeout=train_timeout_s)
    sampler_manifest = run_dir.parent / run_name / "sampler_manifest.json"
    if not sampler_manifest.is_file():
        heartbeat = run_dir.parent / run_name / "heartbeat.json"
        detail = ""
        if heartbeat.is_file():
            detail = f"; last heartbeat={heartbeat.read_text()[:500]}"
        if not allow_zero_update_models:
            raise SystemExit(
                f"training produced no exported sampler for {run_name}: "
                f"{sampler_manifest}{detail}"
            )
        return {
            "path": None,
            "adapter_updates": 0,
            "trained_tokens": 0,
        }
    manifest = json.loads(sampler_manifest.read_text())
    path = manifest.get("path")
    if not path:
        raise SystemExit(f"sampler manifest missing path: {sampler_manifest}")
    if int(manifest.get("step", 0) or 0) <= 0 and not allow_zero_update_models:
        raise SystemExit(
            f"training exported sampler with zero optimizer steps for "
            f"{run_name}: {sampler_manifest}"
        )
    history_path = run_dir.parent / run_name / "history.jsonl"
    history = load_history(history_path)
    return {
        "path": str(path),
        "adapter_updates": sum(1 for row in history if row.get("kind") == "optim"),
        "trained_tokens": sum(int(row.get("trained_tokens", 0) or 0) for row in history),
    }


def init_live_session(
    *,
    run_dir: Path,
    run_name: str,
    base_model: str,
    runner_python: str,
    backend_url: str,
    api_key: str,
    tasks_root: str,
    max_context_tokens: int,
    lora_rank: int,
    timeout_s: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        runner_python,
        str(REPO_ROOT / "rl" / "train_grpo.py"),
        "--base-model",
        base_model,
        "--backend-url",
        backend_url,
        "--api-key",
        api_key,
        "--rollout-backend",
        "worldlines_sampling",
        "--run-name",
        run_name,
        "--runs-root",
        str(run_dir.parent.relative_to(REPO_ROOT)),
        "--tasks-root",
        tasks_root,
        "--rounds",
        "0",
        "--tasks-per-round",
        "1",
        "--samples-per-task",
        "1",
        "--max-turns",
        "1",
        "--max-tokens-per-turn",
        "64",
        "--max-context-tokens",
        str(max_context_tokens),
        "--lora-rank",
        str(lora_rank),
    ]
    env = dict(os.environ)
    env["WORLDLINES_BASE_URL"] = backend_url
    env["WORLDLINES_API_KEY"] = api_key
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    run(cmd, cwd=REPO_ROOT, env=env, timeout=timeout_s)


def run_sample_and_score(
    *,
    report_dir: Path,
    base_model: str,
    runner_python: str,
    model_path: str | None,
    tasks_root: str,
    split_file: Path,
    samples_per_task: int,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    concurrency: int,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    cmd = [
        runner_python,
        str(REPO_ROOT / "rl" / "sample_and_score.py"),
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--base-model",
        base_model,
        "--tasks",
        tasks_root,
        "--report-dir",
        str(report_dir),
        "--samples-per-task",
        str(samples_per_task),
        "--max-turns",
        str(max_turns),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--timeout",
        str(timeout),
        "--concurrency",
        str(concurrency),
        "--split-file",
        str(split_file),
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    run(cmd, cwd=REPO_ROOT, env=env, timeout=21600.0)
    summary_path = report_dir / "smoke_summary.json"
    return json.loads(summary_path.read_text())


def flatten_eval(method: MethodSpec, summary: dict[str, Any]) -> dict[str, Any]:
    n_samples = summary.get("n_samples")
    samples_per_task = summary.get("samples_per_task")
    n_tasks = summary.get("n_tasks") or len(summary.get("tasks", []))
    return {
        "method": method.name,
        "candidate_count": n_samples,
        "verifier_calls": n_samples,
        "chrono_audits": 0,
        "best_verified_reward": max(
            (float(t.get("verified_score", 0.0) or 0.0)
             for t in summary.get("tasks", [])),
            default=0.0,
        ),
        "verified_pass_rate": summary.get("pass_rate_best_of_k"),
        "pass_rate_raw": summary.get("pass_rate_raw"),
        "samples_per_task": samples_per_task,
        "n_tasks": n_tasks,
        "adapter_updates": method.adapter_updates,
        "trained_tokens": method.trained_tokens,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_results(docs_dir: Path, split: dict[str, Any],
                  rows: list[dict[str, Any]]) -> None:
    csv_path = docs_dir / "family_generalization_results.csv"
    json_path = docs_dir / "family_generalization_results.json"
    md_path = docs_dir / "family_generalization_results.md"
    if not rows:
        raise SystemExit("no evaluation rows to write")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    winner = max(rows, key=lambda row: float(row.get("verified_pass_rate") or 0.0))
    target = next(
        (row for row in rows if row["method"] == "mechanical_evolve_ttrl"),
        None,
    )
    beats_all = False
    if target is not None:
        target_rate = float(target.get("verified_pass_rate") or 0.0)
        beats_all = all(
            target_rate > float(row.get("verified_pass_rate") or 0.0)
            for row in rows
            if row["method"] != target["method"]
        )
    payload = {
        "schema": "mech_bench.family_generalization_results.v1",
        "split": split,
        "claim_status": (
            "supports_family_heldout_transfer"
            if beats_all else "does_not_yet_support_family_heldout_transfer"
        ),
        "winner_by_verified_pass_rate": winner["method"],
        "mechanical_evolve_ttrl_beats_all": beats_all,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_results_md(payload))


def render_results_md(payload: dict[str, Any]) -> str:
    split = payload["split"]
    rows = payload["rows"]
    lines = [
        "# Family Generalization Results",
        "",
        f"Claim status: `{payload['claim_status']}`.",
        "",
        "All rows use the same frozen family split, the same evaluator, and the "
        "same samples-per-task budget. The split is mechanism-family held out: "
        "test families are disjoint from train families.",
        "",
        f"- Seen families: {', '.join(split.get('seen_families', []))}",
        f"- Unseen families: {', '.join(split.get('unseen_families', []))}",
        f"- Train tasks: {len(split.get('splits', {}).get('train', []))}",
        f"- Test tasks: {len(split.get('splits', {}).get('test', []))}",
        "",
        "| method | candidate_count | verified_pass_rate | best_verified_reward | adapter_updates | trained_tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method']}` | {row.get('candidate_count')} | "
            f"{float(row.get('verified_pass_rate') or 0.0):.4f} | "
            f"{float(row.get('best_verified_reward') or 0.0):.4f} | "
            f"{row.get('adapter_updates', 0)} | "
            f"{row.get('trained_tokens', 0)} |"
        )
    lines.extend([
        "",
        "The headline transfer claim is valid only if "
        "`mechanical_evolve_ttrl` beats the frozen, SFT, and no-update baselines "
        "on the unseen-family rows under this matched budget.",
        "",
    ])
    return "\n".join(lines)


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None,
        timeout: float | None = None) -> None:
    subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=True,
    )


def run_with_managed_worldlines(args: argparse.Namespace, fn):
    if not args.manage_worldlines:
        return fn()
    proc = start_worldlines_backend(args)
    try:
        return fn()
    finally:
        stop_worldlines_backend(proc)


def start_worldlines_backend(args: argparse.Namespace) -> subprocess.Popen:
    parsed = urlparse(args.worldlines_base_url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 18100)
    if port_is_listening(host, port):
        raise SystemExit(
            f"--manage-worldlines requested but {host}:{port} is already "
            "listening; stop the existing backend or use another port"
        )
    log_dir = Path(args.out_dir).expanduser().resolve() / "worldlines_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worldlines_{int(time.time() * 1000)}.log"
    log_f = log_path.open("w")
    env = dict(os.environ)
    env.update({
        "REPO_ROOT": str(Path(args.worldlines_root).expanduser().resolve()),
        "WLD_VENV": str(Path(args.worldlines_venv).expanduser().resolve()),
        "WLD_ARTIFACTS": str(
            Path(args.worldlines_artifact_root).expanduser().resolve()
        ),
        "BASE_MODEL": str(args.base_model),
        "PORT": str(port),
        "HOST": host,
        "PATCHED_ENTRY": str(REPO_ROOT / "rl" / "launch_trainer_patched.py"),
    })
    cmd = ["bash", str(REPO_ROOT / "rl" / "launch_worldlines.sh")]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    # Keep the file descriptor alive on the process object so logs are not
    # closed while the backend is running.
    proc._mechbench_log_file = log_f  # type: ignore[attr-defined]
    deadline = time.time() + float(args.worldlines_launch_timeout_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"Worldlines backend exited early with code {proc.returncode}; "
                f"see {log_path}"
            )
        if port_is_listening(host, port) and worldlines_health_ok(
            args.worldlines_base_url
        ):
            print(f"managed Worldlines ready at {args.worldlines_base_url}")
            return proc
        time.sleep(1.0)
    stop_worldlines_backend(proc)
    raise SystemExit(
        f"Worldlines backend did not become ready within "
        f"{args.worldlines_launch_timeout_s}s; see {log_path}"
    )


def stop_worldlines_backend(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    log_f = getattr(proc, "_mechbench_log_file", None)
    if log_f is not None:
        log_f.close()


def port_is_listening(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/get_server_capabilities",
            timeout=1.0,
        ) as response:
            return 200 <= response.status < 500
    except urllib.error.HTTPError as exc:
        return 200 <= exc.code < 500
    except (OSError, urllib.error.URLError):
        return False


def worldlines_health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/api/v1/get_server_capabilities",
            timeout=3.0,
        ) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        return exc.code in {401, 403}
    except (OSError, urllib.error.URLError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
