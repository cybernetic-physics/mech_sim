#!/usr/bin/env python3
"""Train an MLX LoRA adapter from MechanicalEvolve verifier rewards.

The input is the MechanicalEvolve GRPO-style JSONL dataset. This script turns
each verifier group into reward-weighted imitation examples, writes MLX-LM
``train/valid/test.jsonl`` files, and optionally launches ``mlx_lm lora``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "mech_bench.mechanical_evolve.mlx_lora.v1"
DEFAULT_MODEL = "mlx-community/Qwen3-32B-4bit"
DESIGN_VARIABLES = (
    "pins",
    "eccentricity",
    "clearance",
    "driver_circle_diameter",
    "driver_pin_collision_shrink_mm",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--resume-adapter-file",
        default=None,
        help=(
            "Optional prior adapters.safetensors file to initialize this "
            "LoRA update. Used by iterative test-time adaptation."
        ),
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-zero-reward", action="store_true")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    archive_path = (
        Path(args.archive).expanduser().resolve() if args.archive else None
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    data_dir = out_dir / "mlx_lora_data"
    adapter_path = (
        Path(args.adapter_path).expanduser().resolve()
        if args.adapter_path else out_dir / "adapters"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    archive = read_json(archive_path) if archive_path and archive_path.exists() else {}
    records = read_jsonl(dataset_path)
    examples = build_training_examples(
        records,
        archive=archive,
        min_reward=float(args.min_reward),
        allow_zero_reward=bool(args.allow_zero_reward),
        max_examples=max(1, int(args.max_examples)),
    )
    if not examples:
        raise SystemExit(
            "no verifier-positive examples available for LoRA training; "
            "run more audits or use --allow-zero-reward for a plumbing smoke"
        )

    split = write_mlx_dataset(data_dir, examples)
    config = mlx_lora_config(
        model=str(args.model),
        data_dir=data_dir,
        adapter_path=adapter_path,
        resume_adapter_file=Path(args.resume_adapter_file).expanduser().resolve()
        if args.resume_adapter_file else None,
        iters=max(1, int(args.iters)),
        batch_size=max(1, int(args.batch_size)),
        grad_accumulation_steps=max(1, int(args.grad_accumulation_steps)),
        learning_rate=float(args.learning_rate),
        num_layers=int(args.num_layers),
        lora_rank=max(1, int(args.lora_rank)),
        lora_scale=float(args.lora_scale),
        lora_dropout=max(0.0, float(args.lora_dropout)),
        max_seq_length=max(64, int(args.max_seq_length)),
        seed=int(args.seed),
        grad_checkpoint=not bool(args.no_grad_checkpoint),
    )
    config_path = out_dir / "mlx_lora_config.yaml"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema": SCHEMA,
        "dataset": str(dataset_path),
        "archive": str(archive_path) if archive_path else None,
        "model": str(args.model),
        "adapter_path": str(adapter_path),
        "resume_adapter_file": str(
            Path(args.resume_adapter_file).expanduser().resolve()
        ) if args.resume_adapter_file else None,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "example_count": len(examples),
        "split": split,
        "best_training_reward": max(
            float(example["metadata"]["reward"]) for example in examples
        ),
        "prepare_only": bool(args.prepare_only),
    }

    if args.prepare_only:
        summary["trainer"] = {"status": "prepared"}
        write_json(out_dir / "training_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        str(config_path),
    ]
    if shutil.which(command[0]) is None:
        raise SystemExit(f"python executable not found: {command[0]}")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    log_path = out_dir / "mlx_lora_train.log"
    completed = run_streamed(command, cwd=out_dir, env=env, log_path=log_path)
    quality = training_log_quality(log_path)
    trainer_ok = completed.returncode == 0 and quality.get("ok") is True
    summary["trainer"] = {
        "status": "completed" if trainer_ok else "failed",
        "returncode": int(completed.returncode if completed.returncode else (
            0 if trainer_ok else 3)),
        "command": command,
        "log_path": str(log_path),
        "adapter_file": str(adapter_path / "adapters.safetensors"),
        "adapter_file_exists": (adapter_path / "adapters.safetensors").is_file(),
        **quality,
    }
    write_json(out_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if completed.returncode != 0:
        return int(completed.returncode)
    return 0 if trainer_ok else 3


def build_training_examples(
    records: list[dict[str, Any]],
    *,
    archive: dict[str, Any],
    min_reward: float,
    allow_zero_reward: bool,
    max_examples: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    elites = compact_archive_elites(archive)
    for record in records:
        responses = [
            response for response in record.get("responses", [])
            if isinstance(response, dict)
        ]
        if not responses:
            continue
        ranked = sorted(
            responses,
            key=lambda row: float(row.get("reward", 0.0) or 0.0),
            reverse=True,
        )
        best = ranked[0]
        reward = float(best.get("reward", 0.0) or 0.0)
        if reward < min_reward and not allow_zero_reward:
            continue
        defect_feedback = summarize_defects(ranked)
        prompt = training_prompt(record, elites=elites, defects=defect_feedback)
        completion = training_completion(best)
        repeats = reward_repeats(reward)
        for _ in range(repeats):
            examples.append({
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "parent_id": record.get("parent_id"),
                    "candidate_id": best.get("candidate_id"),
                    "reward": reward,
                    "defects": best.get("defects", []),
                },
            })
            if len(examples) >= max_examples:
                return examples
    return examples[:max_examples]


def training_prompt(
    record: dict[str, Any],
    *,
    elites: list[dict[str, Any]],
    defects: list[str],
) -> str:
    prompt_record = record.get("prompt", {})
    paper_gate = prompt_record.get("paper_gate", {})
    gate = {
        key: paper_gate.get(key)
        for key in (
            "min_output_speed_rad_s",
            "max_ratio_error_pct",
            "max_penetration_mm",
            "max_contact_force_rms_N",
            "max_contacts",
            "max_power_balance_error_pct",
            "max_torque_ripple_pct",
        )
        if key in paper_gate
    }
    compact_elites = [
        {
            "params": compact_params(row.get("params", {})),
            "reward": row.get("verified_reward"),
            "defects": row.get("defects", []),
        }
        for row in elites[:3]
    ]
    return "\n".join([
        "You are the MechanicalEvolve actuator proposal policy.",
        "Return only a JSON object with keys params and notes.",
        "Set cycloidal/QDD params: " + ", ".join(DESIGN_VARIABLES) + ".",
        "Optimize CAD+Chrono SMC verifier reward with fallback=false.",
        "Gate: " + json.dumps(gate, sort_keys=True, separators=(",", ":")),
        "Elites: " + json.dumps(
            compact_elites, sort_keys=True, separators=(",", ":")),
        "Avoid defects: " + json.dumps(
            defects[:8], sort_keys=True, separators=(",", ":")),
    ])


def training_completion(response: dict[str, Any]) -> str:
    params = {
        key: response.get("params", {}).get(key)
        for key in DESIGN_VARIABLES
        if key in response.get("params", {})
    }
    reward = float(response.get("reward", 0.0) or 0.0)
    defects = response.get("defects", [])
    payload = {
        "params": params,
        "notes": (
            f"selected by CAD+Chrono verifier reward {reward:.6f}; "
            f"defects={defects}"
        ),
    }
    return json.dumps(payload, sort_keys=True)


def summarize_defects(responses: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for response in responses:
        for defect in response.get("defects", []):
            name = str(defect)
            counts[name] = counts.get(name, 0) + 1
    return [
        name for name, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def reward_repeats(reward: float) -> int:
    if not math.isfinite(reward) or reward <= 0.0:
        return 1
    return max(1, min(5, 1 + int(reward // 25.0)))


def compact_archive_elites(archive: dict[str, Any]) -> list[dict[str, Any]]:
    cells = archive.get("cells", {})
    if not isinstance(cells, dict):
        return []
    rows = [row for row in cells.values() if isinstance(row, dict)]
    rows.sort(
        key=lambda row: (
            float(row.get("verified_reward", 0.0) or 0.0),
            float(row.get("fast_reward", 0.0) or 0.0),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in rows[:8]:
        out.append({
            "id": row.get("id"),
            "params": compact_params(row.get("params", {})),
            "verified_reward": row.get("verified_reward"),
            "fast_reward": row.get("fast_reward"),
            "defects": row.get("defects", []),
        })
    return out


def compact_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in DESIGN_VARIABLES:
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, float):
            value = round(value, 6)
        out[key] = value
    return out


def write_mlx_dataset(
    data_dir: Path,
    examples: list[dict[str, Any]],
) -> dict[str, int]:
    if len(examples) == 1:
        train = examples
        valid = examples
        test = examples
    else:
        valid_count = max(1, min(len(examples) // 5, 8))
        valid = examples[:valid_count]
        test = examples[:valid_count]
        train = examples[valid_count:] or examples
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        with (data_dir / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps({
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                }, sort_keys=True) + "\n")
    metadata_path = data_dir / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for row in examples:
            f.write(json.dumps(row["metadata"], sort_keys=True) + "\n")
    return {"train": len(train), "valid": len(valid), "test": len(test)}


def mlx_lora_config(
    *,
    model: str,
    data_dir: Path,
    adapter_path: Path,
    resume_adapter_file: Path | None,
    iters: int,
    batch_size: int,
    grad_accumulation_steps: int,
    learning_rate: float,
    num_layers: int,
    lora_rank: int,
    lora_scale: float,
    lora_dropout: float,
    max_seq_length: int,
    seed: int,
    grad_checkpoint: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "train": True,
        "test": False,
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "data": str(data_dir),
        "seed": seed,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "iters": iters,
        "val_batches": 1,
        "learning_rate": learning_rate,
        "resume_adapter_file": (
            str(resume_adapter_file) if resume_adapter_file is not None else None
        ),
        "steps_per_report": 1,
        "steps_per_eval": max(1, iters),
        "grad_accumulation_steps": grad_accumulation_steps,
        "adapter_path": str(adapter_path),
        "save_every": max(1, iters),
        "max_seq_length": max_seq_length,
        "grad_checkpoint": grad_checkpoint,
        "clear_cache_threshold": "2GB",
        "lora_parameters": {
            "rank": lora_rank,
            "dropout": lora_dropout,
            "scale": lora_scale,
        },
        "mask_prompt": True,
    }


def run_streamed(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        returncode = process.wait()
    return subprocess.CompletedProcess(command, returncode)


def training_log_quality(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text() if log_path.exists() else ""
    train_loss = None
    val_loss = None
    trained_tokens = None
    peak_mem_gb = None
    train_match = None
    val_match = None
    for line in text.splitlines():
        if "Train loss" in line:
            train_match = re.search(
                r"Train loss ([^,]+).*Trained Tokens ([0-9]+), "
                r"Peak mem ([0-9.]+) GB",
                line,
            )
        if "Val loss" in line:
            val_match = re.search(r"Val loss ([^,]+)", line)
    if train_match:
        train_loss = parse_loss(train_match.group(1))
        trained_tokens = int(train_match.group(2))
        peak_mem_gb = float(train_match.group(3))
    if val_match:
        val_loss = parse_loss(val_match.group(1))
    ok = (
        train_loss is not None
        and math.isfinite(train_loss)
        and trained_tokens is not None
        and trained_tokens > 0
    )
    return {
        "ok": ok,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "trained_tokens": trained_tokens,
        "peak_mem_gb": peak_mem_gb,
    }


def parse_loss(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
