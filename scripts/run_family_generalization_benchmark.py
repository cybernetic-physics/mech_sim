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
import hashlib
import json
import os
import shlex
import subprocess
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
PAPER_TTRL_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
PHYSICAL_METRIC_DIRECTIONS = {
    "out_omega_med": "max",
    "ratio_error_pct": "min",
    "power_balance_error_pct": "min",
    "torque_ripple_pct": "min",
    "max_penetration_mm": "min",
    "contact_force_rms_N": "min",
}
DEFAULT_BASE_MODEL = PAPER_TTRL_BASE_MODEL
DEFAULT_SEEN = "cycloidal,belt,chain,rack_pinion,fourbar"
DEFAULT_UNSEEN = "planetary,lead_screw,cam_follower,slider_crank"
DEFAULT_RUNNER_PYTHON = "/Users/nataliakokoromyti/Projects/worldlines/.venv/bin/python"
REQUIRED_METHODS = (
    "verifier_gated",
    "llm_evolve_no_update",
    "mechanical_evolve_ttrl",
    "frozen_model",
    "sft_model",
    "no_update_search",
)
REQUIRED_TTRL_BASELINES = (
    "verifier_gated",
    "llm_evolve_no_update",
    "frozen_model",
    "sft_model",
    "no_update_search",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model_path: str | None
    rollout_backend: str
    sglang_lora_path: str | None
    samples_per_task: int
    max_turns: int
    temperature: float
    top_p: float
    seed_offset: int
    baseline_kind: str
    base_model: str = ""
    ttrl_trainer: str = "none"
    ttrl_exact_grpo: bool = False
    adapter_updates: int = 0
    trained_tokens: int = 0
    rl_trained_tokens: int = 0
    n_rl_datums: int = 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--runner-python", default=DEFAULT_RUNNER_PYTHON)
    p.add_argument("--worldlines-base-url", default="http://127.0.0.1:18100")
    p.add_argument("--sglang-base-url", default="http://127.0.0.1:30000",
                   help="OpenAI-compatible SGLang base URL used to evaluate "
                        "local PEFT adapters from exact TRL GRPO")
    p.add_argument("--eval-rollout-backend", default="sglang_chat",
                   choices=["sglang_chat", "worldlines_sampling"],
                   help="rollout backend for frozen/no-update baselines. The "
                        "paper path uses sglang_chat so the 35B base model, "
                        "SFT adapter, and TTRL adapter are evaluated through "
                        "the same OpenAI-compatible endpoint.")
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
    p.add_argument("--materialize-paper-tasks", action="store_true",
                   help="clone --tasks-root into a paper-verifier task root "
                        "under --out-dir and use that root for the split/run")
    p.add_argument("--paper-tasks-root", default=None,
                   help="optional destination for --materialize-paper-tasks; "
                        "defaults to <out-dir>/paper_tasks")
    p.add_argument("--paper-task-suffix", default="paper_verifier")
    p.add_argument("--paper-task-overwrite", action="store_true",
                   help="replace an existing materialized paper task root")
    p.add_argument("--allow-non-paper-tasks", action="store_true",
                   help="debug only: continue even if the frozen test split "
                        "does not explicitly require trusted CAD preflight "
                        "and Chrono contact with fallback disabled")
    p.add_argument("--out-dir", default="runs/family_generalization_benchmark")
    p.add_argument("--docs-dir", default="docs")
    p.add_argument("--seen-families", default=DEFAULT_SEEN)
    p.add_argument("--unseen-families", default=DEFAULT_UNSEEN)
    p.add_argument("--split-seed", type=int, default=20260528)
    p.add_argument("--train-rounds", type=int, default=6)
    p.add_argument("--tasks-per-round", type=int, default=4)
    p.add_argument("--samples-per-task", type=int, default=4)
    p.add_argument("--match-planned-verifier-budget",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="reduce eval samples for multi-turn methods so "
                        "samples_per_task * max_turns matches one-turn "
                        "baselines. Enabled by default for paper-facing "
                        "matched-budget runs.")
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--max-context-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--train-timeout-s", type=float, default=21600.0)
    p.add_argument("--eval-timeout-s", type=float, default=21600.0)
    p.add_argument("--eval-limit", type=int, default=None,
                   help="debug only: pass --limit to sample_and_score. Any "
                        "limited evaluation is marked incomplete and cannot "
                        "support the family-transfer claim.")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--sft-trainer", default="peft_sft",
                   choices=["peft_sft", "worldlines_ce"],
                   help="trainer for sft_model. peft_sft uses "
                        "rl/train_sft_peft.py and exports a local PEFT "
                        "adapter for SGLang evaluation; worldlines_ce keeps "
                        "the legacy local debug path.")
    p.add_argument("--sft-runner",
                   default="uv run --extra training-grpo python",
                   help="command prefix for rl/train_sft_peft.py")
    p.add_argument("--sft-max-steps", type=int, default=None,
                   help="override supervised PEFT SFT optimizer steps; "
                        "defaults to --train-rounds")
    p.add_argument("--sft-max-seq-length", type=int, default=None,
                   help="override supervised PEFT SFT sequence length; "
                        "defaults to --max-context-tokens")
    p.add_argument("--sft-load-in-4bit", action="store_true")
    p.add_argument("--sft-torch-dtype", default=None,
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--sft-attn-implementation", default=None)
    p.add_argument("--sft-device-map", default=None)
    p.add_argument("--sft-trust-remote-code", action="store_true")
    p.add_argument("--ttrl-trainer", default="trl_grpo",
                   choices=["trl_grpo", "worldlines_ce"],
                   help="trainer for mechanical_evolve_ttrl. trl_grpo uses "
                        "rl/train_true_grpo_trl.py and is required for the "
                        "paper GRPO claim; worldlines_ce is a local "
                        "group-relative weighted-CE baseline/debug path.")
    p.add_argument("--ttrl-grpo-runner",
                   default="uv run --extra training-grpo python",
                   help="command prefix for rl/train_true_grpo_trl.py. Paper "
                        "runs should use an environment with the "
                        "training-grpo extra, for example: "
                        "`uv run --extra training-grpo python`.")
    p.add_argument("--ttrl-grpo-max-steps", type=int, default=None,
                   help="override exact TRL GRPO optimizer steps; defaults to "
                        "--train-rounds for matched wrapper scheduling")
    p.add_argument("--ttrl-grpo-num-generations", type=int, default=None,
                   help="override exact TRL GRPO generations per prompt; "
                        "defaults to --samples-per-task")
    p.add_argument("--ttrl-grpo-max-prompt-length", type=int, default=None,
                   help="override exact TRL GRPO prompt length; defaults to "
                        "--max-context-tokens")
    p.add_argument("--ttrl-grpo-max-completion-length", type=int, default=None,
                   help="override exact TRL GRPO completion length; defaults "
                        "to --max-tokens")
    p.add_argument("--ttrl-grpo-load-in-4bit", action="store_true")
    p.add_argument("--ttrl-grpo-torch-dtype", default=None,
                   choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--ttrl-grpo-attn-implementation", default=None)
    p.add_argument("--ttrl-grpo-device-map", default=None)
    p.add_argument("--ttrl-grpo-max-memory", default=None)
    p.add_argument("--ttrl-grpo-trust-remote-code", action="store_true")
    p.add_argument("--max-train-datums-per-step", type=int, default=2,
                   help="cap training datums per optimizer step to keep local "
                        "MPS runs within memory; capped selection preserves "
                        "verifier-derived RL datums before SFT anchors")
    p.add_argument("--rlvr-refresh-sampler-every", type=int, default=1,
                   help="export updated LoRA weights for subsequent RLVR "
                        "rollouts every N optimizer steps; 1 means true "
                        "iterative test-time adaptation")
    p.add_argument("--rlvr-verifier-pass-fallback-weight", type=float, default=1.0,
                   help="for mechanical_evolve_ttrl only, add verifier-passing "
                        "rollouts as RL datums when group-relative advantages "
                        "collapse to zero; keeps updates verifier-derived")
    p.add_argument("--allow-single-sample-rlvr", action="store_true",
                   help="allow samples-per-task < 2 for debug runs. With one "
                        "sample per task, group-relative RL has no within-task "
                        "advantage signal and is not a valid TTRL result.")
    p.add_argument("--allow-zero-update-models", action="store_true",
                   help="continue evaluation even if an SFT/RLVR training "
                        "phase produces zero optimizer steps; intended only "
                        "for plumbing/debug runs, not paper-facing results")
    p.add_argument("--keep-out-dir", action="store_true")
    p.add_argument("--resume-existing", action="store_true",
                   help="when used with --keep-out-dir, reuse completed "
                        "smoke_summary.json and sampler_manifest.json files "
                        "in --out-dir and continue interrupted phases")
    p.add_argument("--preflight-only", action="store_true",
                   help="materialize/freeze/audit the family split and write "
                        "family_generalization_preflight.json, then exit "
                        "before training or evaluation")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.samples_per_task < 2 and not args.allow_single_sample_rlvr:
        raise SystemExit(
            "--samples-per-task must be at least 2 for a paper-facing "
            "RLVR/TTRL run. Use --allow-single-sample-rlvr only for plumbing "
            "debug runs."
        )
    llm_evolve_eval_samples = matched_eval_samples_per_task(
        base_samples_per_task=args.samples_per_task,
        max_turns=args.max_turns,
        match_planned_verifier_budget=args.match_planned_verifier_budget,
        method_name="llm_evolve_no_update",
    )
    ttrl_eval_samples = matched_eval_samples_per_task(
        base_samples_per_task=args.samples_per_task,
        max_turns=args.max_turns,
        match_planned_verifier_budget=args.match_planned_verifier_budget,
        method_name="mechanical_evolve_ttrl",
    )
    if out_dir.exists() and not args.keep_out_dir:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MECH_BENCH_COMMAND_LOG_DIR"] = str(out_dir / "command_logs")
    docs_dir = Path(args.docs_dir).expanduser().resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)
    tasks_root = resolve_family_tasks_root(
        tasks_root=args.tasks_root,
        out_dir=out_dir,
        seen_families=args.seen_families,
        unseen_families=args.unseen_families,
        materialize_paper_tasks=args.materialize_paper_tasks,
        paper_tasks_root=args.paper_tasks_root,
        paper_task_suffix=args.paper_task_suffix,
        paper_task_overwrite=args.paper_task_overwrite,
    )

    split_dir = out_dir / "splits"
    split_json = split_dir / "split_manifest.json"
    split = load_or_freeze_family_split(
        split_dir=split_dir,
        split_json=split_json,
        runner_python=args.runner_python,
        tasks_root=str(tasks_root),
        seen_families=args.seen_families,
        unseen_families=args.unseen_families,
        split_seed=args.split_seed,
        resume_existing=args.resume_existing,
    )
    enforce_paper_verifier_ready_split(
        split,
        allow_non_paper_tasks=args.allow_non_paper_tasks,
    )
    if args.preflight_only:
        payload = write_preflight_report(
            docs_dir=docs_dir,
            split=split,
            split_json=split_json,
            tasks_root=tasks_root,
            recommended_full_run_command=build_preflight_full_run_command(
                args=args,
                tasks_root=tasks_root,
                out_dir=out_dir,
                docs_dir=docs_dir,
            ),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"

    eval_rows: list[dict[str, Any]] = []

    if family_run_uses_worldlines(
        eval_rollout_backend=args.eval_rollout_backend,
        sft_trainer=args.sft_trainer,
        ttrl_trainer=args.ttrl_trainer,
    ) and not can_skip_worldlines_init_for_resume(
        out_dir=out_dir,
        resume_existing=args.resume_existing,
    ):
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
                tasks_root=str(tasks_root),
                max_context_tokens=args.max_context_tokens,
                lora_rank=args.lora_rank,
                timeout_s=args.train_timeout_s,
            ),
        )
    baseline_methods = [
        MethodSpec(
            "frozen_model",
            None,
            args.eval_rollout_backend,
            None,
            args.samples_per_task,
            1,
            0.01,
            1.0,
            0,
            "base_model_deterministic_matched_budget",
            base_model=args.base_model,
        ),
        MethodSpec(
            "verifier_gated",
            None,
            args.eval_rollout_backend,
            None,
            args.samples_per_task,
            1,
            0.2,
            args.top_p,
            10000,
            "low_temperature_verifier_gated_best_of_k",
            base_model=args.base_model,
        ),
        MethodSpec(
            "no_update_search",
            None,
            args.eval_rollout_backend,
            None,
            args.samples_per_task,
            1,
            args.temperature,
            args.top_p,
            20000,
            "high_temperature_best_of_k_no_update_search",
            base_model=args.base_model,
        ),
        MethodSpec(
            "llm_evolve_no_update",
            None,
            args.eval_rollout_backend,
            None,
            llm_evolve_eval_samples,
            args.max_turns,
            args.temperature,
            args.top_p,
            30000,
            "multi_turn_verifier_feedback_no_update",
            base_model=args.base_model,
        ),
    ]
    for method in baseline_methods:
        report_dir = out_dir / f"eval_{method.name}"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = load_or_run_eval_summary(
            report_dir=report_dir,
            resume_existing=args.resume_existing,
            run_eval=lambda method=method, report_dir=report_dir: run_with_rollout_backend(
                args,
                method.rollout_backend,
                lambda method=method, report_dir=report_dir: run_sample_and_score(
                    report_dir=report_dir,
                    base_model=args.base_model,
                    runner_python=args.runner_python,
                    model_path=method.model_path,
                    rollout_backend=method.rollout_backend,
                    sglang_lora_path=method.sglang_lora_path,
                    tasks_root=str(tasks_root),
                    split_file=test_split,
                    samples_per_task=method.samples_per_task,
                    max_turns=method.max_turns,
                    max_tokens=args.max_tokens,
                    temperature=method.temperature,
                    top_p=method.top_p,
                    seed=args.split_seed + method.seed_offset,
                    timeout=args.timeout,
                    concurrency=args.concurrency,
                    limit=args.eval_limit,
                    base_url=(
                        args.sglang_base_url
                        if method.rollout_backend == "sglang_chat"
                        else args.worldlines_base_url
                    ),
                    api_key=args.api_key,
                ),
            ),
        )
        eval_rows.append(flatten_eval(method, summary))

    sft_run = out_dir / "train_sft"
    if args.sft_trainer == "peft_sft":
        sft_model = train_peft_sft_model(
            run_dir=sft_run,
            run_name="family_sft",
            base_model=args.base_model,
            runner=args.sft_runner,
            tasks_root=str(tasks_root),
            split_file=train_split,
            max_steps=(
                args.sft_max_steps
                if args.sft_max_steps is not None
                else args.train_rounds
            ),
            max_context_tokens=(
                args.sft_max_seq_length
                if args.sft_max_seq_length is not None
                else args.max_context_tokens
            ),
            train_timeout_s=args.train_timeout_s,
            lora_rank=args.lora_rank,
            lr=args.lr,
            load_in_4bit=args.sft_load_in_4bit,
            torch_dtype=args.sft_torch_dtype,
            attn_implementation=args.sft_attn_implementation,
            device_map=args.sft_device_map,
            trust_remote_code=args.sft_trust_remote_code,
            allow_zero_update_models=args.allow_zero_update_models,
            resume_existing=args.resume_existing,
        )
        sft_model_path = None
        sft_rollout_backend = "sglang_chat"
        sft_sglang_lora_path = sft_model["path"]
    else:
        sft_model = run_with_managed_worldlines(
            args,
            lambda: train_model(
                run_dir=sft_run,
                run_name="family_sft",
                base_model=args.base_model,
                runner_python=args.runner_python,
                backend_url=args.worldlines_base_url,
                api_key=args.api_key,
                tasks_root=str(tasks_root),
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
                max_train_datums_per_step=args.max_train_datums_per_step,
                supervised_only=True,
                refresh_sampler_every=0,
                verifier_pass_fallback_weight=0.0,
                allow_zero_update_models=args.allow_zero_update_models,
                resume_existing=args.resume_existing,
            ),
        )
        sft_model_path = sft_model["path"]
        sft_rollout_backend = "worldlines_sampling"
        sft_sglang_lora_path = None
    sft_method = MethodSpec(
        "sft_model",
        sft_model_path,
        sft_rollout_backend,
        sft_sglang_lora_path,
        args.samples_per_task,
        1,
        0.2,
        args.top_p,
        40000,
        "supervised_finetuned_no_verifier_updates",
        base_model=args.base_model,
        adapter_updates=int(sft_model["adapter_updates"]),
        trained_tokens=int(sft_model["trained_tokens"]),
        rl_trained_tokens=int(sft_model.get("rl_trained_tokens", 0)),
        n_rl_datums=int(sft_model.get("n_rl_datums", 0)),
    )
    report_dir = out_dir / "eval_sft_model"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = load_or_run_eval_summary(
        report_dir=report_dir,
        resume_existing=args.resume_existing,
        run_eval=lambda: run_with_rollout_backend(
            args,
            sft_method.rollout_backend,
            lambda: run_sample_and_score(
                report_dir=report_dir,
                base_model=args.base_model,
                runner_python=args.runner_python,
                model_path=sft_method.model_path,
                rollout_backend=sft_method.rollout_backend,
                sglang_lora_path=sft_method.sglang_lora_path,
                tasks_root=str(tasks_root),
                split_file=test_split,
                samples_per_task=sft_method.samples_per_task,
                max_turns=sft_method.max_turns,
                max_tokens=args.max_tokens,
                temperature=sft_method.temperature,
                top_p=sft_method.top_p,
                seed=args.split_seed + sft_method.seed_offset,
                timeout=args.timeout,
                concurrency=args.concurrency,
                limit=args.eval_limit,
                base_url=(
                    args.sglang_base_url
                    if sft_method.rollout_backend == "sglang_chat"
                    else args.worldlines_base_url
                ),
                api_key=args.api_key,
            ),
        ),
    )
    eval_rows.append(flatten_eval(sft_method, summary))

    rlvr_run = out_dir / "train_rlvr"
    if args.ttrl_trainer == "trl_grpo":
        rlvr_model = train_true_grpo_model(
            run_dir=rlvr_run,
            run_name="family_rlvr",
            base_model=args.base_model,
            runner=args.ttrl_grpo_runner,
            tasks_root=str(tasks_root),
            split_file=train_split,
            max_steps=(
                args.ttrl_grpo_max_steps
                if args.ttrl_grpo_max_steps is not None
                else args.train_rounds
            ),
            samples_per_task=(
                args.ttrl_grpo_num_generations
                if args.ttrl_grpo_num_generations is not None
                else args.samples_per_task
            ),
            max_tokens=(
                args.ttrl_grpo_max_completion_length
                if args.ttrl_grpo_max_completion_length is not None
                else args.max_tokens
            ),
            max_context_tokens=(
                args.ttrl_grpo_max_prompt_length
                if args.ttrl_grpo_max_prompt_length is not None
                else args.max_context_tokens
            ),
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            train_timeout_s=args.train_timeout_s,
            lora_rank=args.lora_rank,
            lr=args.lr,
            load_in_4bit=args.ttrl_grpo_load_in_4bit,
            torch_dtype=args.ttrl_grpo_torch_dtype,
            attn_implementation=args.ttrl_grpo_attn_implementation,
            device_map=args.ttrl_grpo_device_map,
            max_memory=args.ttrl_grpo_max_memory,
            trust_remote_code=args.ttrl_grpo_trust_remote_code,
            allow_zero_update_models=args.allow_zero_update_models,
            resume_existing=args.resume_existing,
        )
        rlvr_trainer = "trl_grpo"
        rlvr_exact_grpo = True
        rlvr_model_path = None
        rlvr_sglang_lora_path = rlvr_model["path"]
        rlvr_rollout_backend = "sglang_chat"
    else:
        rlvr_model = run_with_managed_worldlines(
            args,
            lambda: train_model(
                run_dir=rlvr_run,
                run_name="family_rlvr",
                base_model=args.base_model,
                runner_python=args.runner_python,
                backend_url=args.worldlines_base_url,
                api_key=args.api_key,
                tasks_root=str(tasks_root),
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
                max_train_datums_per_step=args.max_train_datums_per_step,
                supervised_only=False,
                refresh_sampler_every=args.rlvr_refresh_sampler_every,
                verifier_pass_fallback_weight=args.rlvr_verifier_pass_fallback_weight,
                allow_zero_update_models=args.allow_zero_update_models,
                resume_existing=args.resume_existing,
            ),
        )
        rlvr_trainer = "worldlines_group_relative_weighted_ce"
        rlvr_exact_grpo = False
        rlvr_model_path = rlvr_model["path"]
        rlvr_sglang_lora_path = None
        rlvr_rollout_backend = "worldlines_sampling"
    rlvr_method = MethodSpec(
        "mechanical_evolve_ttrl",
        rlvr_model_path,
        rlvr_rollout_backend,
        rlvr_sglang_lora_path,
        ttrl_eval_samples,
        args.max_turns,
        args.temperature,
        args.top_p,
        50000,
        "iterative_verifier_derived_lora_updates",
        base_model=args.base_model,
        ttrl_trainer=rlvr_trainer,
        ttrl_exact_grpo=rlvr_exact_grpo,
        adapter_updates=int(rlvr_model["adapter_updates"]),
        trained_tokens=int(rlvr_model["trained_tokens"]),
        rl_trained_tokens=int(rlvr_model.get("rl_trained_tokens", 0)),
        n_rl_datums=int(rlvr_model.get("n_rl_datums", 0)),
    )
    report_dir = out_dir / "eval_mechanical_evolve_ttrl"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = load_or_run_eval_summary(
        report_dir=report_dir,
        resume_existing=args.resume_existing,
        run_eval=lambda: run_with_rollout_backend(
            args,
            rlvr_method.rollout_backend,
            lambda: run_sample_and_score(
                report_dir=report_dir,
                base_model=args.base_model,
                runner_python=args.runner_python,
                model_path=rlvr_method.model_path,
                rollout_backend=rlvr_method.rollout_backend,
                sglang_lora_path=rlvr_method.sglang_lora_path,
                tasks_root=str(tasks_root),
                split_file=test_split,
                samples_per_task=rlvr_method.samples_per_task,
                max_turns=rlvr_method.max_turns,
                max_tokens=args.max_tokens,
                temperature=rlvr_method.temperature,
                top_p=rlvr_method.top_p,
                seed=args.split_seed + rlvr_method.seed_offset,
                timeout=args.timeout,
                concurrency=args.concurrency,
                limit=args.eval_limit,
                base_url=(
                    args.sglang_base_url
                    if rlvr_method.rollout_backend == "sglang_chat"
                    else args.worldlines_base_url
                ),
                api_key=args.api_key,
            ),
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
    max_train_datums_per_step: int,
    supervised_only: bool,
    refresh_sampler_every: int,
    verifier_pass_fallback_weight: float,
    allow_zero_update_models: bool,
    resume_existing: bool = False,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler_manifest = run_dir.parent / run_name / "sampler_manifest.json"
    if resume_existing and sampler_manifest.is_file():
        return _load_sampler_manifest_model(
            sampler_manifest=sampler_manifest,
            run_name=run_name,
            label="Worldlines",
            allow_zero_update_models=allow_zero_update_models,
        )
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
        "--max-train-datums-per-step",
        str(max_train_datums_per_step),
        "--reference-sft-split-file",
        str(split_file),
        "--reference-sft-weight",
        "1.0",
        "--reference-sft-per-step",
        str(tasks_per_round),
        "--save-final-sampler-name",
        f"{run_name}_final",
    ]
    if verifier_pass_fallback_weight > 0:
        cmd.extend([
            "--verifier-pass-fallback-weight",
            str(verifier_pass_fallback_weight),
        ])
    if refresh_sampler_every > 0:
        cmd.extend([
            "--refresh-sampler-every",
            str(refresh_sampler_every),
        ])
    if supervised_only:
        cmd.extend([
            "--sft-warmup-rounds",
            str(train_rounds),
            "--positive-only-passes",
    ])
    env = repo_env({
        "WORLDLINES_BASE_URL": backend_url,
        "WORLDLINES_API_KEY": api_key,
    })
    run(cmd, cwd=REPO_ROOT, env=env, timeout=train_timeout_s)
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
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
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
    optim_rows = [row for row in history if row.get("kind") == "optim"]
    return {
        "path": str(path),
        "adapter_updates": int(
            manifest.get("adapter_updates", len(optim_rows)) or 0
        ),
        "trained_tokens": int(
            manifest.get(
                "trained_tokens",
                sum(int(row.get("trained_tokens", 0) or 0) for row in optim_rows),
            )
            or 0
        ),
        "rl_trained_tokens": int(
            manifest.get(
                "rl_trained_tokens",
                sum(
                    int(row.get("rl_trained_tokens", 0) or 0)
                    for row in optim_rows
                ),
            )
            or 0
        ),
        "n_rl_datums": int(
            manifest.get(
                "n_rl_datums",
                sum(int(row.get("n_rl_datums", 0) or 0) for row in optim_rows),
            )
            or 0
        ),
    }


def build_true_grpo_cmd(
    *,
    runner: str,
    run_dir: Path,
    base_model: str,
    tasks_root: str,
    split_file: Path,
    max_steps: int,
    samples_per_task: int,
    max_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    lora_rank: int,
    lr: float,
    load_in_4bit: bool,
    torch_dtype: str | None,
    device_map: str | None,
    max_memory: str | None,
    trust_remote_code: bool,
    attn_implementation: str | None = None,
) -> list[str]:
    runner_parts = shlex.split(runner)
    if not runner_parts:
        raise ValueError("exact GRPO runner command cannot be empty")
    cmd = [
        *runner_parts,
        str(REPO_ROOT / "rl" / "train_true_grpo_trl.py"),
        "--model",
        base_model,
        "--output-dir",
        str(run_dir),
        "--tasks-root",
        tasks_root,
        "--split-file",
        str(split_file),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(lr),
        "--num-generations",
        str(samples_per_task),
        "--max-prompt-length",
        str(max_context_tokens),
        "--max-completion-length",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--reward-timeout-s",
        str(timeout),
        "--lora-rank",
        str(lora_rank),
    ]
    if load_in_4bit:
        cmd.append("--load-in-4bit")
    if torch_dtype:
        cmd.extend(["--torch-dtype", torch_dtype])
    if attn_implementation:
        cmd.extend(["--attn-implementation", attn_implementation])
    if device_map:
        cmd.extend(["--device-map", device_map])
    if max_memory:
        cmd.extend(["--max-memory", max_memory])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def build_peft_sft_cmd(
    *,
    runner: str,
    run_dir: Path,
    base_model: str,
    tasks_root: str,
    split_file: Path,
    max_steps: int,
    max_context_tokens: int,
    lora_rank: int,
    lr: float,
    load_in_4bit: bool,
    torch_dtype: str | None,
    device_map: str | None,
    trust_remote_code: bool,
    attn_implementation: str | None = None,
) -> list[str]:
    runner_parts = shlex.split(runner)
    if not runner_parts:
        raise ValueError("PEFT SFT runner command cannot be empty")
    cmd = [
        *runner_parts,
        str(REPO_ROOT / "rl" / "train_sft_peft.py"),
        "--model",
        base_model,
        "--output-dir",
        str(run_dir),
        "--tasks-root",
        tasks_root,
        "--split-file",
        str(split_file),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(lr),
        "--max-seq-length",
        str(max_context_tokens),
        "--lora-rank",
        str(lora_rank),
    ]
    if load_in_4bit:
        cmd.append("--load-in-4bit")
    if torch_dtype:
        cmd.extend(["--torch-dtype", torch_dtype])
    if attn_implementation:
        cmd.extend(["--attn-implementation", attn_implementation])
    if device_map:
        cmd.extend(["--device-map", device_map])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def _load_sampler_manifest_model(
    *,
    sampler_manifest: Path,
    run_name: str,
    label: str,
    allow_zero_update_models: bool,
) -> dict[str, Any]:
    if not sampler_manifest.is_file():
        if not allow_zero_update_models:
            raise SystemExit(
                f"{label} training produced no sampler manifest for "
                f"{run_name}: {sampler_manifest}"
            )
        return {
            "path": None,
            "adapter_updates": 0,
            "trained_tokens": 0,
            "rl_trained_tokens": 0,
            "n_rl_datums": 0,
        }
    manifest = json.loads(sampler_manifest.read_text())
    path = manifest.get("path")
    if not path:
        raise SystemExit(f"sampler manifest missing path: {sampler_manifest}")
    if int(manifest.get("step", 0) or 0) <= 0 and not allow_zero_update_models:
        raise SystemExit(
            f"{label} exported adapter with zero optimizer steps for "
            f"{run_name}: {sampler_manifest}"
        )
    return {
        "path": str(path),
        "adapter_updates": int(manifest.get("adapter_updates", 0) or 0),
        "trained_tokens": int(manifest.get("trained_tokens", 0) or 0),
        "rl_trained_tokens": int(manifest.get("rl_trained_tokens", 0) or 0),
        "n_rl_datums": int(manifest.get("n_rl_datums", 0) or 0),
    }


def train_peft_sft_model(
    *,
    run_dir: Path,
    run_name: str,
    base_model: str,
    runner: str,
    tasks_root: str,
    split_file: Path,
    max_steps: int,
    max_context_tokens: int,
    train_timeout_s: float,
    lora_rank: int,
    lr: float,
    load_in_4bit: bool,
    torch_dtype: str | None,
    device_map: str | None,
    trust_remote_code: bool,
    allow_zero_update_models: bool,
    resume_existing: bool = False,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    output_dir = run_dir.parent / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_manifest = output_dir / "sampler_manifest.json"
    if resume_existing and sampler_manifest.is_file():
        return _load_sampler_manifest_model(
            sampler_manifest=sampler_manifest,
            run_name=run_name,
            label="PEFT SFT",
            allow_zero_update_models=allow_zero_update_models,
        )
    cmd = build_peft_sft_cmd(
        runner=runner,
        run_dir=output_dir,
        base_model=base_model,
        tasks_root=tasks_root,
        split_file=split_file,
        max_steps=max_steps,
        max_context_tokens=max_context_tokens,
        lora_rank=lora_rank,
        lr=lr,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    run(cmd, cwd=REPO_ROOT, env=repo_env(), timeout=train_timeout_s)
    return _load_sampler_manifest_model(
        sampler_manifest=sampler_manifest,
        run_name=run_name,
        label="PEFT SFT",
        allow_zero_update_models=allow_zero_update_models,
    )


def train_true_grpo_model(
    *,
    run_dir: Path,
    run_name: str,
    base_model: str,
    runner: str,
    tasks_root: str,
    split_file: Path,
    max_steps: int,
    samples_per_task: int,
    max_tokens: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    train_timeout_s: float,
    lora_rank: int,
    lr: float,
    load_in_4bit: bool,
    torch_dtype: str | None,
    attn_implementation: str | None,
    device_map: str | None,
    max_memory: str | None,
    trust_remote_code: bool,
    allow_zero_update_models: bool,
    resume_existing: bool = False,
) -> dict[str, Any]:
    output_dir = run_dir.parent / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_manifest = output_dir / "sampler_manifest.json"
    if resume_existing and sampler_manifest.is_file():
        return _load_sampler_manifest_model(
            sampler_manifest=sampler_manifest,
            run_name=run_name,
            label="exact GRPO",
            allow_zero_update_models=allow_zero_update_models,
        )
    cmd = build_true_grpo_cmd(
        runner=runner,
        run_dir=output_dir,
        base_model=base_model,
        tasks_root=tasks_root,
        split_file=split_file,
        max_steps=max_steps,
        samples_per_task=samples_per_task,
        max_tokens=max_tokens,
        max_context_tokens=max_context_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        lora_rank=lora_rank,
        lr=lr,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=trust_remote_code,
    )
    run(cmd, cwd=REPO_ROOT, env=repo_env(), timeout=train_timeout_s)
    return _load_sampler_manifest_model(
        sampler_manifest=sampler_manifest,
        run_name=run_name,
        label="exact GRPO",
        allow_zero_update_models=allow_zero_update_models,
    )


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
    env = repo_env({
        "WORLDLINES_BASE_URL": backend_url,
        "WORLDLINES_API_KEY": api_key,
    })
    run(cmd, cwd=REPO_ROOT, env=env, timeout=timeout_s)


def load_or_freeze_family_split(
    *,
    split_dir: Path,
    split_json: Path,
    runner_python: str,
    tasks_root: str,
    seen_families: str,
    unseen_families: str,
    split_seed: int,
    resume_existing: bool,
) -> dict[str, Any]:
    train_split = split_dir / "train.txt"
    test_split = split_dir / "test.txt"
    if (
        resume_existing
        and split_json.is_file()
        and train_split.is_file()
        and test_split.is_file()
    ):
        return json.loads(split_json.read_text())
    run([
        runner_python,
        str(SCRIPT_DIR / "freeze_mechbench_family_splits.py"),
        "--tasks-root",
        tasks_root,
        "--out-dir",
        str(split_dir),
        "--manifest-json",
        str(split_json),
        "--seen-families",
        seen_families,
        "--unseen-families",
        unseen_families,
        "--seed",
        str(split_seed),
    ], cwd=REPO_ROOT, env=repo_env())
    return json.loads(split_json.read_text())


def can_skip_worldlines_init_for_resume(
    *,
    out_dir: Path,
    resume_existing: bool,
) -> bool:
    if not resume_existing:
        return False
    eval_names = [
        "eval_frozen_model",
        "eval_verifier_gated",
        "eval_no_update_search",
        "eval_llm_evolve_no_update",
        "eval_sft_model",
        "eval_mechanical_evolve_ttrl",
    ]
    evals_cached = all(
        (out_dir / name / "smoke_summary.json").is_file()
        for name in eval_names
    )
    train_cached = all(
        (out_dir / name / "sampler_manifest.json").is_file()
        for name in ("family_sft", "family_rlvr")
    )
    return evals_cached and train_cached


def load_or_run_eval_summary(
    *,
    report_dir: Path,
    resume_existing: bool,
    run_eval: Any,
) -> dict[str, Any]:
    summary_path = report_dir / "smoke_summary.json"
    if resume_existing and summary_path.is_file():
        return json.loads(summary_path.read_text())
    return run_eval()


def run_sample_and_score(
    *,
    report_dir: Path,
    base_model: str,
    runner_python: str,
    model_path: str | None,
    rollout_backend: str,
    sglang_lora_path: str | None,
    tasks_root: str,
    split_file: Path,
    samples_per_task: int,
    max_turns: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
    concurrency: int,
    limit: int | None,
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
        "--rollout-backend",
        rollout_backend,
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
        "--seed",
        str(seed),
        "--timeout",
        str(timeout),
        "--concurrency",
        str(concurrency),
        "--split-file",
        str(split_file),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if model_path:
        cmd.extend(["--model-path", model_path])
    if sglang_lora_path:
        cmd.extend(["--sglang-lora-path", sglang_lora_path])
    env = repo_env()
    run(cmd, cwd=REPO_ROOT, env=env, timeout=21600.0)
    summary_path = report_dir / "smoke_summary.json"
    return json.loads(summary_path.read_text())


def flatten_eval(method: MethodSpec, summary: dict[str, Any]) -> dict[str, Any]:
    n_samples = summary.get("n_samples")
    verifier_calls = summary.get("n_verifier_calls", n_samples)
    cad_audits = int(summary.get("n_cad_audits", 0) or 0)
    chrono_audits = int(summary.get("n_chrono_audits", 0) or 0)
    samples_per_task = summary.get("samples_per_task")
    n_tasks = summary.get("n_tasks") or len(summary.get("tasks", []))
    planned_max_verifier_calls = (
        int(n_samples or 0) * max(1, int(method.max_turns or 1))
    )
    task_rows = list(summary.get("tasks", []) or [])
    lockup_count = count_lockups(task_rows)
    repair_attempt_count = count_repair_attempts(task_rows)
    repair_success_count = count_repair_successes(task_rows)
    no_procedural_fallback_count = count_no_procedural_fallback(task_rows)
    physical_metrics = summarize_physical_metrics(task_rows)
    cad_pass_rate = count_positive_audits(task_rows, "cad_audits") / n_tasks if n_tasks else 0.0
    chrono_real_geometry_rate = (
        count_positive_audits(task_rows, "chrono_audits") / n_tasks
        if n_tasks else 0.0
    )
    return {
        "method": method.name,
        "baseline_kind": method.baseline_kind,
        "candidate_count": n_samples,
        "verifier_calls": verifier_calls,
        "planned_max_verifier_calls": planned_max_verifier_calls,
        "verifier_calls_per_candidate": (
            float(verifier_calls or 0) / float(n_samples or 1)
        ),
        "cad_audits": cad_audits,
        "chrono_audits": chrono_audits,
        "cad_pass_rate": cad_pass_rate,
        "chrono_real_geometry_rate": chrono_real_geometry_rate,
        "best_verified_reward": max(
            (float(t.get("verified_score", 0.0) or 0.0)
             for t in task_rows),
            default=0.0,
        ),
        "lockup_count": lockup_count,
        "lockup_rate": lockup_count / n_tasks if n_tasks else 0.0,
        "repair_attempt_count": repair_attempt_count,
        "repair_success_count": repair_success_count,
        "repair_success_rate": (
            repair_success_count / repair_attempt_count
            if repair_attempt_count else 0.0
        ),
        "no_procedural_fallback_count": no_procedural_fallback_count,
        "no_procedural_fallback_rate": (
            no_procedural_fallback_count / n_tasks if n_tasks else 0.0
        ),
        **physical_metrics,
        "verified_pass_rate": summary.get(
            "verifier_valid_pass_rate_best_of_k",
            summary.get("pass_rate_best_of_k"),
        ),
        "strict_score_pass_rate": summary.get("pass_rate_best_of_k"),
        "verifier_valid_pass_rate_raw": summary.get(
            "verifier_valid_pass_rate_raw",
            summary.get("pass_rate_raw"),
        ),
        "strict_score_pass_rate_raw": summary.get("pass_rate_raw"),
        "samples_per_task": samples_per_task,
        "max_turns": method.max_turns,
        "temperature": method.temperature,
        "top_p": method.top_p,
        "seed_offset": method.seed_offset,
        "base_model": method.base_model,
        "ttrl_trainer": method.ttrl_trainer,
        "ttrl_exact_grpo": method.ttrl_exact_grpo,
        "paper_ttrl_base_model": PAPER_TTRL_BASE_MODEL,
        "paper_ttrl_base_model_match": (
            method.name != "mechanical_evolve_ttrl"
            or method.base_model == PAPER_TTRL_BASE_MODEL
        ),
        "n_tasks": n_tasks,
        "adapter_updates": method.adapter_updates,
        "trained_tokens": method.trained_tokens,
        "rl_trained_tokens": method.rl_trained_tokens,
        "n_rl_datums": method.n_rl_datums,
        "families": summarize_family_metrics(summary),
    }


def failure_codes_of(row: dict[str, Any]) -> set[str]:
    return {str(code).lower() for code in row.get("failure_codes", []) or []}


def row_has_lockup(row: dict[str, Any]) -> bool:
    return "lockup" in failure_codes_of(row)


def row_repair_attempted(row: dict[str, Any]) -> bool:
    if "repair_attempted" in row:
        return bool(row.get("repair_attempted"))
    return int(row.get("verifier_calls", 0) or 0) > 1


def row_repair_succeeded(row: dict[str, Any]) -> bool:
    if "repair_succeeded" in row:
        return bool(row.get("repair_succeeded"))
    return bool(row_repair_attempted(row) and row.get("verifier_valid_passed"))


def count_lockups(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row_has_lockup(row))


def count_repair_attempts(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row_repair_attempted(row))


def count_repair_successes(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row_repair_succeeded(row))


def count_positive_audits(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if int(row.get(key, 0) or 0) > 0)


def row_no_procedural_fallback(row: dict[str, Any]) -> bool | None:
    raw = row.get("no_procedural_fallback")
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def count_no_procedural_fallback(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row_no_procedural_fallback(row) is True)


def physical_metric_value(row: dict[str, Any], metric: str) -> float | None:
    raw_metrics = row.get("physical_metrics")
    if isinstance(raw_metrics, dict) and metric in raw_metrics:
        raw = raw_metrics.get(metric)
    else:
        raw = row.get(metric)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return None
    return value


def summarize_physical_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric, direction in PHYSICAL_METRIC_DIRECTIONS.items():
        values = [
            value for row in rows
            if (value := physical_metric_value(row, metric)) is not None
        ]
        if not values:
            out[f"best_{metric}"] = None
        elif direction == "max":
            out[f"best_{metric}"] = max(values)
        else:
            out[f"best_{metric}"] = min(values)
    return out


def summarize_family_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in summary.get("tasks", []) or []:
        family = str(row.get("family") or "unknown")
        buckets.setdefault(family, []).append(row)

    out: list[dict[str, Any]] = []
    for family, rows in sorted(buckets.items()):
        n = len(rows)
        strict_passed = sum(1 for row in rows if row.get("strict_passed"))
        verifier_valid = sum(
            1 for row in rows if row.get("verifier_valid_passed")
        )
        scores = [
            float(row.get("verified_score", row.get("score", 0.0)) or 0.0)
            for row in rows
        ]
        lockup_count = count_lockups(rows)
        repair_attempt_count = count_repair_attempts(rows)
        repair_success_count = count_repair_successes(rows)
        no_procedural_fallback_count = count_no_procedural_fallback(rows)
        physical_metrics = summarize_physical_metrics(rows)
        out.append({
            "family": family,
            "n_tasks": n,
            "verified_pass_rate": verifier_valid / n if n else 0.0,
            "strict_score_pass_rate": strict_passed / n if n else 0.0,
            "mean_verified_reward": sum(scores) / n if n else 0.0,
            "best_verified_reward": max(scores, default=0.0),
            "cad_pass_rate": count_positive_audits(rows, "cad_audits") / n if n else 0.0,
            "chrono_real_geometry_rate": (
                count_positive_audits(rows, "chrono_audits") / n if n else 0.0
            ),
            "lockup_count": lockup_count,
            "lockup_rate": lockup_count / n if n else 0.0,
            "repair_attempt_count": repair_attempt_count,
            "repair_success_count": repair_success_count,
            "repair_success_rate": (
                repair_success_count / repair_attempt_count
                if repair_attempt_count else 0.0
            ),
            "no_procedural_fallback_count": no_procedural_fallback_count,
            "no_procedural_fallback_rate": (
                no_procedural_fallback_count / n if n else 0.0
            ),
            "failure_count": sum(
                1 for row in rows if row.get("failure_codes")
            ),
            **physical_metrics,
        })
    return out


def audit_split(split: dict[str, Any]) -> dict[str, Any]:
    task_index = split.get("task_index", {}) or {}
    splits = split.get("splits", {}) or {}

    def families_for(split_name: str) -> list[str]:
        families = set()
        for task_id in splits.get(split_name, []) or []:
            meta = task_index.get(task_id, {}) or {}
            family = meta.get("canonical_family")
            if family:
                families.add(str(family))
        return sorted(families)

    train_families = set(families_for("train"))
    val_families = set(families_for("val"))
    test_families = set(families_for("test"))
    seen = set(str(f) for f in split.get("seen_families", []) or [])
    unseen = set(str(f) for f in split.get("unseen_families", []) or [])
    return {
        "seen_unseen_overlap": sorted(seen & unseen),
        "train_test_family_overlap": sorted(train_families & test_families),
        "val_test_family_overlap": sorted(val_families & test_families),
        "train_families": sorted(train_families),
        "val_families": sorted(val_families),
        "test_families": sorted(test_families),
        "family_heldout": not (
            (seen & unseen)
            or (train_families & test_families)
            or (val_families & test_families)
        ),
    }


def audit_methods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {str(row.get("method")) for row in rows}
    missing = [method for method in REQUIRED_METHODS if method not in methods]
    unexpected = sorted(methods - set(REQUIRED_METHODS))

    verifier_calls = {
        str(row.get("method")): int(row.get("verifier_calls", 0) or 0)
        for row in rows
    }
    required_verifier_calls = {
        method: verifier_calls.get(method)
        for method in REQUIRED_METHODS
        if method in verifier_calls
    }
    equal_verifier_budget = (
        not missing
        and bool(required_verifier_calls)
        and len(set(required_verifier_calls.values())) == 1
    )

    cad_audits = {
        str(row.get("method")): int(row.get("cad_audits", 0) or 0)
        for row in rows
    }
    required_cad_audits = {
        method: cad_audits.get(method)
        for method in REQUIRED_METHODS
        if method in cad_audits
    }
    equal_cad_budget = (
        not missing
        and bool(required_cad_audits)
        and len(set(required_cad_audits.values())) == 1
    )
    positive_cad_budget = (
        equal_cad_budget
        and all(int(value or 0) > 0 for value in required_cad_audits.values())
    )

    chrono_audits = {
        str(row.get("method")): int(row.get("chrono_audits", 0) or 0)
        for row in rows
    }
    required_chrono_audits = {
        method: chrono_audits.get(method)
        for method in REQUIRED_METHODS
        if method in chrono_audits
    }
    equal_chrono_budget = (
        not missing
        and bool(required_chrono_audits)
        and len(set(required_chrono_audits.values())) == 1
    )
    positive_chrono_budget = (
        equal_chrono_budget
        and all(int(value or 0) > 0 for value in required_chrono_audits.values())
    )

    return {
        "required_methods": list(REQUIRED_METHODS),
        "required_ttrl_baselines": list(REQUIRED_TTRL_BASELINES),
        "present_methods": sorted(methods),
        "missing_required_methods": missing,
        "unexpected_methods": unexpected,
        "required_methods_present": not missing,
        "verifier_calls_by_method": verifier_calls,
        "equal_verifier_budget": equal_verifier_budget,
        "cad_audits_by_method": cad_audits,
        "equal_cad_budget": equal_cad_budget,
        "positive_cad_budget": positive_cad_budget,
        "chrono_audits_by_method": chrono_audits,
        "equal_chrono_budget": equal_chrono_budget,
        "positive_chrono_budget": positive_chrono_budget,
    }


def audit_eval_coverage(
    split: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = len(split.get("splits", {}).get("test", []) or [])
    n_tasks_by_method = {
        str(row.get("method")): int(row.get("n_tasks", 0) or 0)
        for row in rows
    }
    incomplete_required_methods = [
        method for method in REQUIRED_METHODS
        if n_tasks_by_method.get(method, 0) != expected
    ]
    return {
        "expected_test_tasks": expected,
        "n_tasks_by_method": n_tasks_by_method,
        "complete_required_eval_coverage": not incomplete_required_methods,
        "incomplete_required_methods": incomplete_required_methods,
    }


def audit_split_task_verifiers(split: dict[str, Any]) -> dict[str, Any]:
    task_index = split.get("task_index", {}) or {}
    test_ids = [str(item) for item in split.get("splits", {}).get("test", []) or []]
    missing_chrono = []
    missing_no_fallback = []
    missing_trusted_preflight = []
    missing_trusted_mass = []
    for task_id in test_ids:
        meta = task_index.get(task_id, {}) or {}
        if not bool(meta.get("has_chrono_contact_config")):
            missing_chrono.append(task_id)
        if not bool(meta.get("chrono_procedural_fallback_disabled")):
            missing_no_fallback.append(task_id)
        if not bool(meta.get("has_trusted_asset_preflight")):
            missing_trusted_preflight.append(task_id)
        if not bool(meta.get("requires_trusted_mass_properties")):
            missing_trusted_mass.append(task_id)
    complete = not (
        missing_chrono
        or missing_no_fallback
        or missing_trusted_preflight
        or missing_trusted_mass
    )
    return {
        "n_test_tasks": len(test_ids),
        "paper_verifier_ready_test_tasks": complete,
        "missing_chrono_contact_config": missing_chrono,
        "missing_chrono_no_procedural_fallback_false": missing_no_fallback,
        "missing_trusted_asset_preflight": missing_trusted_preflight,
        "missing_trusted_mass_properties_requirement": missing_trusted_mass,
    }


def enforce_paper_verifier_ready_split(
    split: dict[str, Any],
    *,
    allow_non_paper_tasks: bool,
) -> dict[str, Any]:
    audit = audit_split_task_verifiers(split)
    if audit["paper_verifier_ready_test_tasks"] or allow_non_paper_tasks:
        return audit
    raise SystemExit(
        "family benchmark test split is not paper-verifier-ready; use "
        "--materialize-paper-tasks for paper runs or --allow-non-paper-tasks "
        "for debug-only analytic/fake-oracle plumbing. Audit: "
        f"{json.dumps(audit, sort_keys=True)}"
    )


def write_preflight_report(
    *,
    docs_dir: Path,
    split: dict[str, Any],
    split_json: Path,
    tasks_root: Path,
    recommended_full_run_command: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "mech_bench.family_generalization_preflight.v1",
        "tasks_root": str(tasks_root),
        "split_manifest": str(split_json),
        "split_audit": audit_split(split),
        "split_task_verifier_audit": audit_split_task_verifiers(split),
        "train_tasks": len(split.get("splits", {}).get("train", []) or []),
        "val_tasks": len(split.get("splits", {}).get("val", []) or []),
        "test_tasks": len(split.get("splits", {}).get("test", []) or []),
        "seen_families": list(split.get("seen_families", []) or []),
        "unseen_families": list(split.get("unseen_families", []) or []),
    }
    if recommended_full_run_command:
        payload["recommended_full_run_command"] = recommended_full_run_command
    out = docs_dir / "family_generalization_preflight.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def build_preflight_full_run_command(
    *,
    args: argparse.Namespace,
    tasks_root: Path,
    out_dir: Path,
    docs_dir: Path,
) -> str:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_family_generalization_benchmark.py",
        "--tasks-root",
        str(tasks_root),
        "--out-dir",
        str(out_dir),
        "--docs-dir",
        str(docs_dir),
        "--keep-out-dir",
        "--resume-existing",
        "--base-model",
        args.base_model,
        "--sft-trainer",
        args.sft_trainer,
        "--ttrl-trainer",
        args.ttrl_trainer,
        "--eval-rollout-backend",
        args.eval_rollout_backend,
        "--seen-families",
        args.seen_families,
        "--unseen-families",
        args.unseen_families,
        "--split-seed",
        str(args.split_seed),
        "--train-rounds",
        str(args.train_rounds),
        "--tasks-per-round",
        str(args.tasks_per_round),
        "--samples-per-task",
        str(args.samples_per_task),
        "--max-turns",
        str(args.max_turns),
        "--max-tokens",
        str(args.max_tokens),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--timeout",
        str(args.timeout),
        "--train-timeout-s",
        str(args.train_timeout_s),
        "--eval-timeout-s",
        str(args.eval_timeout_s),
        "--concurrency",
        str(args.concurrency),
        "--lora-rank",
        str(args.lora_rank),
        "--lr",
        str(args.lr),
        "--sft-runner",
        args.sft_runner,
        "--ttrl-grpo-runner",
        args.ttrl_grpo_runner,
        "--max-train-datums-per-step",
        str(args.max_train_datums_per_step),
        "--rlvr-refresh-sampler-every",
        str(args.rlvr_refresh_sampler_every),
        "--rlvr-verifier-pass-fallback-weight",
        str(args.rlvr_verifier_pass_fallback_weight),
        "--worldlines-base-url",
        args.worldlines_base_url,
        "--sglang-base-url",
        args.sglang_base_url,
        "--api-key",
        args.api_key,
        "--runner-python",
        args.runner_python,
        "--worldlines-root",
        args.worldlines_root,
        "--worldlines-venv",
        args.worldlines_venv,
        "--worldlines-artifact-root",
        args.worldlines_artifact_root,
        "--worldlines-launch-timeout-s",
        str(args.worldlines_launch_timeout_s),
    ]
    if not args.match_planned_verifier_budget:
        cmd.append("--no-match-planned-verifier-budget")
    if args.eval_limit is not None:
        cmd.extend(["--eval-limit", str(args.eval_limit)])
    if args.sft_max_steps is not None:
        cmd.extend(["--sft-max-steps", str(args.sft_max_steps)])
    if args.sft_load_in_4bit:
        cmd.append("--sft-load-in-4bit")
    sft_max_seq_length = getattr(args, "sft_max_seq_length", None)
    if sft_max_seq_length is not None:
        cmd.extend(["--sft-max-seq-length", str(sft_max_seq_length)])
    sft_torch_dtype = getattr(args, "sft_torch_dtype", None)
    if sft_torch_dtype is not None:
        cmd.extend(["--sft-torch-dtype", sft_torch_dtype])
    sft_attn_implementation = getattr(args, "sft_attn_implementation", None)
    if sft_attn_implementation is not None:
        cmd.extend(["--sft-attn-implementation", sft_attn_implementation])
    if args.sft_device_map is not None:
        cmd.extend(["--sft-device-map", args.sft_device_map])
    if args.sft_trust_remote_code:
        cmd.append("--sft-trust-remote-code")
    if args.ttrl_grpo_max_steps is not None:
        cmd.extend(["--ttrl-grpo-max-steps", str(args.ttrl_grpo_max_steps)])
    if args.ttrl_grpo_num_generations is not None:
        cmd.extend([
            "--ttrl-grpo-num-generations",
            str(args.ttrl_grpo_num_generations),
        ])
    if args.ttrl_grpo_load_in_4bit:
        cmd.append("--ttrl-grpo-load-in-4bit")
    ttrl_grpo_max_prompt_length = getattr(
        args,
        "ttrl_grpo_max_prompt_length",
        None,
    )
    if ttrl_grpo_max_prompt_length is not None:
        cmd.extend([
            "--ttrl-grpo-max-prompt-length",
            str(ttrl_grpo_max_prompt_length),
        ])
    ttrl_grpo_max_completion_length = getattr(
        args,
        "ttrl_grpo_max_completion_length",
        None,
    )
    if ttrl_grpo_max_completion_length is not None:
        cmd.extend([
            "--ttrl-grpo-max-completion-length",
            str(ttrl_grpo_max_completion_length),
        ])
    if args.ttrl_grpo_torch_dtype is not None:
        cmd.extend(["--ttrl-grpo-torch-dtype", args.ttrl_grpo_torch_dtype])
    ttrl_grpo_attn_implementation = getattr(
        args,
        "ttrl_grpo_attn_implementation",
        None,
    )
    if ttrl_grpo_attn_implementation is not None:
        cmd.extend([
            "--ttrl-grpo-attn-implementation",
            ttrl_grpo_attn_implementation,
        ])
    if args.ttrl_grpo_device_map is not None:
        cmd.extend(["--ttrl-grpo-device-map", args.ttrl_grpo_device_map])
    ttrl_grpo_max_memory = getattr(args, "ttrl_grpo_max_memory", None)
    if ttrl_grpo_max_memory is not None:
        cmd.extend(["--ttrl-grpo-max-memory", ttrl_grpo_max_memory])
    if args.ttrl_grpo_trust_remote_code:
        cmd.append("--ttrl-grpo-trust-remote-code")
    if args.manage_worldlines:
        cmd.append("--manage-worldlines")
    if args.allow_single_sample_rlvr:
        cmd.append("--allow-single-sample-rlvr")
    if args.allow_zero_update_models:
        cmd.append("--allow-zero-update-models")
    return shlex.join(cmd)


def audit_physical_metric_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = [f"best_{metric}" for metric in PHYSICAL_METRIC_DIRECTIONS]
    missing_by_method: dict[str, list[str]] = {}
    for method in REQUIRED_METHODS:
        row = next((item for item in rows if item.get("method") == method), None)
        if row is None:
            missing_by_method[method] = list(required_fields)
            continue
        missing = [
            field for field in required_fields
            if row.get(field) is None
        ]
        if missing:
            missing_by_method[method] = missing
    return {
        "required_physical_metric_fields": required_fields,
        "missing_required_physical_metrics_by_method": missing_by_method,
        "complete_required_physical_metrics": not missing_by_method,
    }


def audit_no_procedural_fallback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_or_failed = []
    rates_by_method: dict[str, float] = {}
    for method in REQUIRED_METHODS:
        row = next((item for item in rows if item.get("method") == method), None)
        rate = 0.0
        if row is not None:
            rate = float(row.get("no_procedural_fallback_rate") or 0.0)
        rates_by_method[method] = rate
        if rate < 1.0:
            missing_or_failed.append(method)
    return {
        "no_procedural_fallback_rate_by_method": rates_by_method,
        "methods_missing_no_procedural_fallback_evidence": missing_or_failed,
        "complete_no_procedural_fallback_evidence": not missing_or_failed,
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


def resolve_family_tasks_root(
    *,
    tasks_root: str,
    out_dir: Path,
    seen_families: str,
    unseen_families: str,
    materialize_paper_tasks: bool,
    paper_tasks_root: str | None,
    paper_task_suffix: str,
    paper_task_overwrite: bool,
) -> Path:
    source = Path(tasks_root).expanduser().resolve()
    if not materialize_paper_tasks:
        return source
    destination = (
        Path(paper_tasks_root).expanduser().resolve()
        if paper_tasks_root
        else out_dir / "paper_tasks"
    )
    from scripts.materialize_paper_family_tasks import (
        materialize_paper_family_tasks,
    )

    families = ",".join([seen_families, unseen_families])
    count = materialize_paper_family_tasks(
        tasks_root=source,
        out_root=destination,
        families=families,
        suffix=paper_task_suffix,
        overwrite=paper_task_overwrite,
    )
    if count <= 0:
        raise SystemExit(
            "materialized paper task root is empty; check --tasks-root and "
            "family filters"
        )
    return destination


def repo_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    if extra:
        env.update(extra)
    return env


def matched_eval_samples_per_task(
    *,
    base_samples_per_task: int,
    max_turns: int,
    match_planned_verifier_budget: bool,
    method_name: str,
) -> int:
    base = int(base_samples_per_task)
    turns = max(1, int(max_turns))
    if base <= 0:
        raise ValueError("base_samples_per_task must be positive")
    if not match_planned_verifier_budget or turns <= 1:
        return base
    if base % turns != 0:
        raise SystemExit(
            f"{method_name} cannot match planned verifier budget exactly: "
            f"--samples-per-task={base} is not divisible by max_turns={turns}. "
            "Use a divisible sample count or --no-match-planned-verifier-budget "
            "for a debug run that cannot support the matched-budget claim."
        )
    matched = base // turns
    if matched < 1:
        raise SystemExit(
            f"{method_name} cannot receive at least one sample under the "
            f"matched planned verifier budget: samples={base}, turns={turns}"
        )
    return matched


def family_run_uses_worldlines(
    *,
    eval_rollout_backend: str,
    sft_trainer: str,
    ttrl_trainer: str,
) -> bool:
    return (
        eval_rollout_backend == "worldlines_sampling"
        or sft_trainer == "worldlines_ce"
        or ttrl_trainer == "worldlines_ce"
    )


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("cad_audits", 0)
    normalized.setdefault("chrono_audits", 0)
    normalized.setdefault("cad_pass_rate", 0.0)
    normalized.setdefault("chrono_real_geometry_rate", 0.0)
    normalized.setdefault("lockup_count", 0)
    normalized.setdefault("lockup_rate", 0.0)
    normalized.setdefault("repair_attempt_count", 0)
    normalized.setdefault("repair_success_count", 0)
    normalized.setdefault("repair_success_rate", 0.0)
    normalized.setdefault("no_procedural_fallback_count", 0)
    normalized.setdefault("no_procedural_fallback_rate", 0.0)
    for metric in PHYSICAL_METRIC_DIRECTIONS:
        normalized.setdefault(f"best_{metric}", None)
    normalized.setdefault("families", [])
    for family_row in normalized.get("families", []) or []:
        family_row.setdefault("cad_pass_rate", 0.0)
        family_row.setdefault("chrono_real_geometry_rate", 0.0)
        family_row.setdefault("lockup_count", 0)
        family_row.setdefault("lockup_rate", 0.0)
        family_row.setdefault("repair_attempt_count", 0)
        family_row.setdefault("repair_success_count", 0)
        family_row.setdefault("repair_success_rate", 0.0)
        family_row.setdefault("no_procedural_fallback_count", 0)
        family_row.setdefault("no_procedural_fallback_rate", 0.0)
        for metric in PHYSICAL_METRIC_DIRECTIONS:
            family_row.setdefault(f"best_{metric}", None)
    return normalized


def split_role_for_family(split: dict[str, Any], family: str) -> str:
    seen = {str(item) for item in split.get("seen_families", []) or []}
    unseen = {str(item) for item in split.get("unseen_families", []) or []}
    if family in seen:
        return "seen"
    if family in unseen:
        return "unseen"
    return "unknown"


def summarize_split_role_metrics(
    split: dict[str, Any],
    family_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in family_rows:
        method = str(row.get("method") or "")
        family = str(row.get("family") or "")
        role = split_role_for_family(split, family)
        buckets.setdefault((method, role), []).append(row)

    out: list[dict[str, Any]] = []
    for (method, role), rows in sorted(buckets.items()):
        n_tasks = sum(int(row.get("n_tasks", 0) or 0) for row in rows)
        best_scores = [
            float(row.get("best_verified_reward", 0.0) or 0.0)
            for row in rows
        ]
        weighted_reward = sum(
            float(row.get("mean_verified_reward", 0.0) or 0.0)
            * int(row.get("n_tasks", 0) or 0)
            for row in rows
        )
        weighted_pass = sum(
            float(row.get("verified_pass_rate", 0.0) or 0.0)
            * int(row.get("n_tasks", 0) or 0)
            for row in rows
        )
        weighted_lockup = sum(
            float(row.get("lockup_rate", 0.0) or 0.0)
            * int(row.get("n_tasks", 0) or 0)
            for row in rows
        )
        weighted_no_procedural_fallback = sum(
            float(row.get("no_procedural_fallback_rate", 0.0) or 0.0)
            * int(row.get("n_tasks", 0) or 0)
            for row in rows
        )
        repair_attempts = sum(
            int(row.get("repair_attempt_count", 0) or 0) for row in rows
        )
        repair_successes = sum(
            int(row.get("repair_success_count", 0) or 0) for row in rows
        )
        out.append({
            "method": method,
            "split_role": role,
            "n_families": len(rows),
            "n_tasks": n_tasks,
            "verified_pass_rate": weighted_pass / n_tasks if n_tasks else 0.0,
            "best_verified_reward": max(best_scores, default=0.0),
            "best_verified_reward_mean": (
                weighted_reward / n_tasks if n_tasks else 0.0
            ),
            "lockup_rate": weighted_lockup / n_tasks if n_tasks else 0.0,
            "no_procedural_fallback_rate": (
                weighted_no_procedural_fallback / n_tasks if n_tasks else 0.0
            ),
            "repair_success_rate": (
                repair_successes / repair_attempts if repair_attempts else 0.0
            ),
        })
    return out


def ttrl_beats_required_baselines_on_unseen(
    split_role_rows: list[dict[str, Any]],
) -> bool:
    unseen_rows = [
        row for row in split_role_rows if row.get("split_role") == "unseen"
    ]
    by_method = {str(row.get("method")): row for row in unseen_rows}
    target = by_method.get("mechanical_evolve_ttrl")
    if target is None:
        return False
    if any(method not in by_method for method in REQUIRED_TTRL_BASELINES):
        return False
    target_score = float(target.get("best_verified_reward_mean") or 0.0)
    return all(
        target_score > float(by_method[method].get("best_verified_reward_mean") or 0.0)
        for method in REQUIRED_TTRL_BASELINES
    )


def write_results(docs_dir: Path, split: dict[str, Any],
                  rows: list[dict[str, Any]]) -> None:
    csv_path = docs_dir / "family_generalization_results.csv"
    family_csv_path = docs_dir / "family_generalization_family_results.csv"
    json_path = docs_dir / "family_generalization_results.json"
    md_path = docs_dir / "family_generalization_results.md"
    if not rows:
        raise SystemExit("no evaluation rows to write")
    rows = [normalize_result_row(row) for row in rows]
    with csv_path.open("w", newline="") as f:
        flat_rows = [
            {k: v for k, v in row.items() if k != "families"}
            for row in rows
        ]
        fieldnames = sorted({key for row in flat_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    family_rows = []
    for row in rows:
        for family_row in row.get("families", []) or []:
            family_rows.append({
                "method": row["method"],
                **family_row,
            })
    if family_rows:
        with family_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(family_rows[0].keys()))
            writer.writeheader()
            writer.writerows(family_rows)
    split_role_rows = summarize_split_role_metrics(split, family_rows)
    ttrl_beats_unseen_baselines = ttrl_beats_required_baselines_on_unseen(
        split_role_rows
    )
    winner = max(rows, key=lambda row: float(row.get("verified_pass_rate") or 0.0))
    target = next(
        (row for row in rows if row["method"] == "mechanical_evolve_ttrl"),
        None,
    )
    beats_all = False
    beats_required_baselines = False
    ttrl_adaptation_valid = False
    ttrl_paper_model_valid = False
    ttrl_algorithm_valid = False
    if target is not None:
        target_rate = float(target.get("verified_pass_rate") or 0.0)
        beats_all = all(
            target_rate > float(row.get("verified_pass_rate") or 0.0)
            for row in rows
            if row["method"] != target["method"]
        )
        baseline_rows = [
            row for row in rows
            if row.get("method") in REQUIRED_TTRL_BASELINES
        ]
        beats_required_baselines = (
            len(baseline_rows) == len(REQUIRED_TTRL_BASELINES)
            and all(
                target_rate > float(row.get("verified_pass_rate") or 0.0)
                for row in baseline_rows
            )
        )
        ttrl_adaptation_valid = (
            int(target.get("adapter_updates") or 0) > 0
            and int(target.get("trained_tokens") or 0) > 0
            and int(target.get("rl_trained_tokens") or 0) > 0
            and int(target.get("n_rl_datums") or 0) > 0
        )
        ttrl_paper_model_valid = (
            str(target.get("base_model") or "") == PAPER_TTRL_BASE_MODEL
        )
        ttrl_algorithm_valid = (
            str(target.get("ttrl_trainer") or "") == "trl_grpo"
            and bool(target.get("ttrl_exact_grpo"))
        )
    split_audit = audit_split(split)
    method_audit = audit_methods(rows)
    eval_coverage_audit = audit_eval_coverage(split, rows)
    split_task_verifier_audit = audit_split_task_verifiers(split)
    physical_metric_audit = audit_physical_metric_coverage(rows)
    procedural_fallback_audit = audit_no_procedural_fallback(rows)
    supports_claim = bool(
        split_audit["family_heldout"]
        and method_audit["required_methods_present"]
        and method_audit["equal_verifier_budget"]
        and method_audit["equal_cad_budget"]
        and method_audit["positive_cad_budget"]
        and method_audit["equal_chrono_budget"]
        and method_audit["positive_chrono_budget"]
        and eval_coverage_audit["complete_required_eval_coverage"]
        and split_task_verifier_audit["paper_verifier_ready_test_tasks"]
        and physical_metric_audit["complete_required_physical_metrics"]
        and procedural_fallback_audit["complete_no_procedural_fallback_evidence"]
        and beats_required_baselines
        and ttrl_beats_unseen_baselines
        and ttrl_adaptation_valid
        and ttrl_paper_model_valid
        and ttrl_algorithm_valid
    )
    claim_blockers: list[str] = []
    if not split_audit["family_heldout"]:
        claim_blockers.append("split_is_not_family_heldout")
    if not method_audit["required_methods_present"]:
        claim_blockers.append("missing_required_methods")
    if not method_audit["equal_verifier_budget"]:
        claim_blockers.append("unequal_verifier_budget")
    if not method_audit["equal_cad_budget"]:
        claim_blockers.append("unequal_cad_budget")
    if not method_audit["positive_cad_budget"]:
        claim_blockers.append("missing_positive_cad_budget")
    if not method_audit["equal_chrono_budget"]:
        claim_blockers.append("unequal_chrono_budget")
    if not method_audit["positive_chrono_budget"]:
        claim_blockers.append("missing_positive_chrono_budget")
    if not eval_coverage_audit["complete_required_eval_coverage"]:
        claim_blockers.append("incomplete_eval_task_coverage")
    if not split_task_verifier_audit["paper_verifier_ready_test_tasks"]:
        claim_blockers.append("split_tasks_missing_paper_verifiers")
    if not physical_metric_audit["complete_required_physical_metrics"]:
        claim_blockers.append("missing_required_physical_metrics")
    if not procedural_fallback_audit["complete_no_procedural_fallback_evidence"]:
        claim_blockers.append("procedural_fallback_not_proven_false")
    if not beats_required_baselines:
        claim_blockers.append("ttrl_does_not_beat_required_baselines")
    if not ttrl_beats_unseen_baselines:
        claim_blockers.append("ttrl_does_not_beat_required_baselines_on_unseen")
    if not ttrl_adaptation_valid:
        claim_blockers.append("ttrl_missing_verifier_derived_updates")
    if not ttrl_paper_model_valid:
        claim_blockers.append("ttrl_wrong_base_model_for_paper_goal")
    if not ttrl_algorithm_valid:
        claim_blockers.append("ttrl_not_exact_trl_grpo")
    payload = {
        "schema": "mech_bench.family_generalization_results.v1",
        "split": split,
        "split_audit": split_audit,
        "method_audit": method_audit,
        "eval_coverage_audit": eval_coverage_audit,
        "split_task_verifier_audit": split_task_verifier_audit,
        "physical_metric_audit": physical_metric_audit,
        "procedural_fallback_audit": procedural_fallback_audit,
        "claim_status": (
            "supports_family_heldout_transfer"
            if supports_claim else "does_not_yet_support_family_heldout_transfer"
        ),
        "claim_blockers": claim_blockers,
        "winner_by_verified_pass_rate": winner["method"],
        "mechanical_evolve_ttrl_beats_all": beats_all,
        "mechanical_evolve_ttrl_beats_required_baselines": beats_required_baselines,
        "mechanical_evolve_ttrl_beats_required_baselines_on_unseen": (
            ttrl_beats_unseen_baselines
        ),
        "mechanical_evolve_ttrl_adaptation_valid": ttrl_adaptation_valid,
        "mechanical_evolve_ttrl_paper_model_valid": ttrl_paper_model_valid,
        "mechanical_evolve_ttrl_algorithm_valid": ttrl_algorithm_valid,
        "paper_ttrl_base_model": PAPER_TTRL_BASE_MODEL,
        "rows": rows,
        "family_rows": family_rows,
        "split_role_rows": split_role_rows,
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
        f"- Split audit family-held-out: {payload['split_audit']['family_heldout']}",
        "- Train/test family overlap: "
        f"{payload['split_audit']['train_test_family_overlap']}",
        "- Val/test family overlap: "
        f"{payload['split_audit']['val_test_family_overlap']}",
        "- Required methods present: "
        f"{payload['method_audit']['required_methods_present']}",
        "- Missing required methods: "
        f"{payload['method_audit']['missing_required_methods']}",
        "- Equal verifier budget: "
        f"{payload['method_audit']['equal_verifier_budget']}",
        "- Equal CAD budget: "
        f"{payload['method_audit']['equal_cad_budget']}",
        "- Positive CAD budget: "
        f"{payload['method_audit']['positive_cad_budget']}",
        "- Equal Chrono budget: "
        f"{payload['method_audit']['equal_chrono_budget']}",
        "- Positive Chrono budget: "
        f"{payload['method_audit']['positive_chrono_budget']}",
        "- Complete required eval coverage: "
        f"{payload['eval_coverage_audit']['complete_required_eval_coverage']}",
        "- Test tasks paper-verifier ready: "
        f"{payload['split_task_verifier_audit']['paper_verifier_ready_test_tasks']}",
        "- Complete required physical metrics: "
        f"{payload['physical_metric_audit']['complete_required_physical_metrics']}",
        "- No procedural fallback proven: "
        f"{payload['procedural_fallback_audit']['complete_no_procedural_fallback_evidence']}",
        "- Eval tasks by method: "
        f"{payload['eval_coverage_audit']['n_tasks_by_method']}",
        "- MechanicalEvolve/TTRL paper base model: "
        f"`{payload['paper_ttrl_base_model']}`",
        "- MechanicalEvolve/TTRL base-model gate passed: "
        f"{payload['mechanical_evolve_ttrl_paper_model_valid']}",
        "- MechanicalEvolve/TTRL exact-GRPO gate passed: "
        f"{payload['mechanical_evolve_ttrl_algorithm_valid']}",
        "- MechanicalEvolve/TTRL unseen-family baseline gate passed: "
        f"{payload['mechanical_evolve_ttrl_beats_required_baselines_on_unseen']}",
        f"- Claim blockers: {payload['claim_blockers']}",
        f"- Train tasks: {len(split.get('splits', {}).get('train', []))}",
        f"- Test tasks: {len(split.get('splits', {}).get('test', []))}",
        "",
        "| method | baseline_kind | base_model | ttrl_trainer | candidate_count | verifier_calls | cad_audits | chrono_audits | planned_max_verifier_calls | verified_pass_rate | cad_pass_rate | chrono_real_geometry_rate | no_procedural_fallback_rate | lockup_rate | best_verified_reward | best_out_omega_med | best_ratio_error_pct | best_power_balance_error_pct | best_torque_ripple_pct | best_max_penetration_mm | best_contact_force_rms_N | adapter_updates | trained_tokens | rl_trained_tokens | n_rl_datums |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['method']}` | `{row.get('baseline_kind', '')}` | "
            f"`{row.get('base_model', '')}` | "
            f"`{row.get('ttrl_trainer', '')}` | "
            f"{row.get('candidate_count')} | "
            f"{row.get('verifier_calls')} | "
            f"{row.get('cad_audits', 0)} | "
            f"{row.get('chrono_audits', 0)} | "
            f"{row.get('planned_max_verifier_calls')} | "
            f"{float(row.get('verified_pass_rate') or 0.0):.4f} | "
            f"{float(row.get('cad_pass_rate') or 0.0):.4f} | "
            f"{float(row.get('chrono_real_geometry_rate') or 0.0):.4f} | "
            f"{float(row.get('no_procedural_fallback_rate') or 0.0):.4f} | "
            f"{float(row.get('lockup_rate') or 0.0):.4f} | "
            f"{float(row.get('best_verified_reward') or 0.0):.4f} | "
            f"{format_optional_metric(row.get('best_out_omega_med'))} | "
            f"{format_optional_metric(row.get('best_ratio_error_pct'))} | "
            f"{format_optional_metric(row.get('best_power_balance_error_pct'))} | "
            f"{format_optional_metric(row.get('best_torque_ripple_pct'))} | "
            f"{format_optional_metric(row.get('best_max_penetration_mm'))} | "
            f"{format_optional_metric(row.get('best_contact_force_rms_N'))} | "
            f"{row.get('adapter_updates', 0)} | "
            f"{row.get('trained_tokens', 0)} | "
            f"{row.get('rl_trained_tokens', 0)} | "
            f"{row.get('n_rl_datums', 0)} |"
        )
    lines.extend([
        "",
        "## Family Rows",
        "",
        "| method | family | n_tasks | verified_pass_rate | strict_score_pass_rate | mean_verified_reward | best_verified_reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload.get("family_rows", []) or []:
        lines.append(
            f"| `{row['method']}` | `{row['family']}` | "
            f"{row.get('n_tasks')} | "
            f"{float(row.get('verified_pass_rate') or 0.0):.4f} | "
            f"{float(row.get('strict_score_pass_rate') or 0.0):.4f} | "
            f"{float(row.get('mean_verified_reward') or 0.0):.4f} | "
            f"{float(row.get('best_verified_reward') or 0.0):.4f} |"
        )
    lines.extend([
        "",
        "## Split Role Rows",
        "",
        "| method | split_role | n_families | n_tasks | verified_pass_rate | best_verified_reward_mean | best_verified_reward | no_procedural_fallback_rate | lockup_rate | repair_success_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload.get("split_role_rows", []) or []:
        lines.append(
            f"| `{row.get('method')}` | `{row.get('split_role')}` | "
            f"{row.get('n_families')} | "
            f"{row.get('n_tasks')} | "
            f"{float(row.get('verified_pass_rate') or 0.0):.4f} | "
            f"{float(row.get('best_verified_reward_mean') or 0.0):.4f} | "
            f"{float(row.get('best_verified_reward') or 0.0):.4f} | "
            f"{float(row.get('no_procedural_fallback_rate') or 0.0):.4f} | "
            f"{float(row.get('lockup_rate') or 0.0):.4f} | "
            f"{float(row.get('repair_success_rate') or 0.0):.4f} |"
        )
    lines.extend([
        "",
        "`verified_pass_rate` counts evaluator-valid submissions with all hard "
        "gates satisfied and no failure codes. `best_verified_reward` remains "
        "the continuous verifier score, so contact-quality penalties such as "
        "bounded penetration still affect reward comparisons even when the "
        "artifact is verifier-valid.",
        "",
        "`supports_family_heldout_transfer` additionally requires a disjoint "
        "family split, all required methods from `goals.md`, equal verifier "
        "CAD and Chrono budgets, full held-out test coverage for every required "
        "method, test tasks that explicitly require trusted CAD preflight and "
        "Chrono contact with `procedural_cycloidal_fallback=false`, complete "
        "required physical metric columns for every required method, explicit "
        "evidence that no procedural fallback was used for every required "
        "method, `mechanical_evolve_ttrl` to beat every required baseline by "
        "`verified_pass_rate` overall and by `best_verified_reward_mean` on "
        "explicit unseen split-role rows, nonzero TTRL "
        "`adapter_updates`, `trained_tokens`, `rl_trained_tokens`, and "
        "`n_rl_datums`, the required paper base model, and the exact TRL "
        "`GRPOTrainer` path.",
        "",
        "The headline transfer claim is valid only if "
        "`mechanical_evolve_ttrl` beats the frozen, SFT, and no-update baselines "
        "on the unseen-family rows under this matched budget.",
        "",
    ])
    return "\n".join(lines)


def format_optional_metric(raw: Any) -> str:
    try:
        return f"{float(raw):.4f}"
    except (TypeError, ValueError):
        return ""


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None,
        timeout: float | None = None) -> None:
    log_root = os.environ.get("MECH_BENCH_COMMAND_LOG_DIR")
    if not log_root:
        subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=True,
        )
        return

    log_dir = Path(log_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(" ".join(cmd).encode("utf-8")).hexdigest()[:10]
    stem = Path(cmd[1] if len(cmd) > 1 else cmd[0]).stem
    log_path = log_dir / f"{int(time.time() * 1000)}_{stem}_{digest}.log"
    print(f"[run] {shlex.join(cmd[:4])} ... > {log_path}")
    try:
        with log_path.open("w") as f:
            subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                timeout=timeout,
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
    except subprocess.CalledProcessError as e:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        except OSError:
            pass
        raise SystemExit(
            f"command failed with exit code {e.returncode}: "
            f"{shlex.join(cmd)}\nlog: {log_path}\n{tail}"
        ) from e


def run_with_managed_worldlines(args: argparse.Namespace, fn):
    if not args.manage_worldlines:
        return fn()
    proc = start_worldlines_backend(args)
    try:
        return fn()
    finally:
        stop_worldlines_backend(proc)


def run_with_rollout_backend(
    args: argparse.Namespace,
    rollout_backend: str,
    fn,
):
    if rollout_backend == "worldlines_sampling":
        return run_with_managed_worldlines(args, fn)
    return fn()


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
