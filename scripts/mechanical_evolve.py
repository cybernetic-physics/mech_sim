#!/usr/bin/env python3
"""MechanicalEvolve: verifier-driven cycloidal actuator discovery.

The runner follows an AlphaEvolve-style loop for the current branch baseline:
generate actuator parameter programs, score them cheaply, audit elites through
FreeCAD + Chrono SMC with procedural fallback disabled, and keep a MAP-Elites
archive with lineage and structured verifier defects.

Large model serving is external, but local MLX LoRA training is supported as a
first-class backend on Apple Silicon. The runner still keeps command hooks for
veRL/TRL clusters and model-serving processes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimize_cycloidal_chrono_candidates as cyclo  # noqa: E402


SCHEMA = "mech_bench.mechanical_evolve.v1"
DESIGN_VARIABLES = (
    "pins",
    "eccentricity",
    "clearance",
    "driver_circle_diameter",
    "driver_pin_collision_shrink_mm",
)
MUTATION_OPERATORS = (
    "pins_step",
    "eccentricity_step",
    "clearance_step",
    "driver_circle_step",
    "shrink_step",
    "boundary_refine",
    "random_restart",
)
DEFAULT_LORA_MODEL = "mlx-community/Qwen3-32B-4bit"


@dataclass(frozen=True)
class MlxLoraTrainerConfig:
    model: str = DEFAULT_LORA_MODEL
    adapter_path: str | None = None
    iters: int = 10
    batch_size: int = 1
    grad_accumulation_steps: int = 1
    learning_rate: float = 1.0e-5
    num_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    max_seq_length: int = 768
    min_reward: float = 1.0
    max_examples: int = 256
    prepare_only: bool = False


@dataclass(frozen=True)
class Proposal:
    id: str
    method: str
    params: dict[str, Any]
    proposer: str
    parent_id: str | None = None
    operator: str = "seed"
    island: int = 0
    patch: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "method": self.method,
            "params": dict(self.params),
            "proposer": self.proposer,
            "parent_id": self.parent_id,
            "operator": self.operator,
            "island": self.island,
            "patch": list(self.patch),
            "prompt": self.prompt,
            "notes": self.notes,
        }


@dataclass
class PolicyState:
    weights: dict[str, float] = field(default_factory=lambda: {
        name: 1.0 for name in MUTATION_OPERATORS
    })
    updates: int = 0

    def choose(self, rng: random.Random) -> str:
        total = sum(max(0.0, weight) for weight in self.weights.values())
        if total <= 0.0:
            return rng.choice(list(MUTATION_OPERATORS))
        cursor = rng.random() * total
        for name, weight in self.weights.items():
            cursor -= max(0.0, weight)
            if cursor <= 0.0:
                return name
        return next(reversed(self.weights))

    def update(self, evaluated: Iterable[dict[str, Any]], lr: float) -> None:
        for row in evaluated:
            proposal = row.get("proposal", {})
            operator = str(proposal.get("operator", ""))
            if operator not in self.weights:
                continue
            reward = float(row.get("verified_reward", 0.0) or 0.0) / 100.0
            defects = float(row.get("defect_count", 0.0) or 0.0)
            signal = reward - 0.05 * defects
            self.weights[operator] = max(
                0.05, self.weights[operator] * math.exp(float(lr) * signal))
        self.updates += 1

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self.weights), "updates": int(self.updates)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyState":
        state = cls()
        raw = data.get("weights")
        if isinstance(raw, dict):
            for name in MUTATION_OPERATORS:
                if name in raw:
                    state.weights[name] = max(0.05, float(raw[name]))
        state.updates = int(data.get("updates", 0) or 0)
        return state


class MapElitesArchive:
    def __init__(self) -> None:
        self.cells: dict[str, dict[str, Any]] = {}

    def insert(self, row: dict[str, Any]) -> bool:
        key = archive_key(row.get("params") or {})
        incumbent = self.cells.get(key)
        if incumbent is None or archive_score(row) > archive_score(incumbent):
            self.cells[key] = row
            return True
        return False

    def elites(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = sorted(
            self.cells.values(),
            key=archive_score,
            reverse=True,
        )
        return rows if limit is None else rows[:max(0, int(limit))]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA}.archive",
            "cell_count": len(self.cells),
            "cells": self.cells,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapElitesArchive":
        archive = cls()
        cells = data.get("cells")
        if isinstance(cells, dict):
            archive.cells = {
                str(key): value for key, value in cells.items()
                if isinstance(value, dict)
            }
        return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("evolve-only", "rlvr-train", "test-time-adapt"),
        default="evolve-only",
    )
    parser.add_argument("--out-dir", default="runs/mechanical_evolve/latest")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--audit-k", type=int, default=4)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--duration-s", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-seed-bootstrap",
        action="store_true",
        help="Do not prepend current seed/refinement proposals in generation 0.",
    )
    parser.add_argument(
        "--proposal-jsonl",
        default=None,
        help="Optional JSONL proposals from a Kimi/Qwen/vLLM/SGLang process.",
    )
    parser.add_argument(
        "--model-command",
        default=None,
        help=(
            "Optional command. Receives a JSON prompt on stdin and returns a "
            "JSON list/object with candidate params."
        ),
    )
    parser.add_argument(
        "--trainer-command",
        default=None,
        help=(
            "Optional veRL/TRL launcher. Called with MECHANICAL_EVOLVE_DATASET "
            "and MECHANICAL_EVOLVE_ARCHIVE env vars in rlvr-train/adapt modes."
        ),
    )
    parser.add_argument(
        "--trainer-backend",
        choices=("none", "command", "mlx-lora"),
        default="none",
        help="Use command hook or local MLX LoRA training for policy updates.",
    )
    parser.add_argument("--lora-model", default=os.environ.get(
        "MECHANICAL_EVOLVE_LORA_MODEL", DEFAULT_LORA_MODEL))
    parser.add_argument("--lora-adapter-path", default=None)
    parser.add_argument("--lora-iters", type=int, default=10)
    parser.add_argument("--lora-batch-size", type=int, default=1)
    parser.add_argument("--lora-grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--lora-num-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-max-seq-length", type=int, default=768)
    parser.add_argument("--lora-min-reward", type=float, default=1.0)
    parser.add_argument("--lora-max-examples", type=int, default=256)
    parser.add_argument("--lora-prepare-only", action="store_true")
    parser.add_argument("--policy-lr", type=float, default=1.0)
    parser.add_argument("--target-id", default="cycloidal_qdd_default")
    parser.add_argument("--contact-force-limit-N", type=float, default=3000.0)
    parser.add_argument("--max-contacts", type=float, default=128.0)
    parser.add_argument("--power-balance-limit-pct", type=float, default=1.0e12)
    parser.add_argument("--torque-ripple-limit-pct", type=float, default=1.0e12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = cyclo.VerificationLimits(
        max_contact_force_rms_N=max(0.0, float(args.contact_force_limit_N)),
        max_contacts=max(1.0, float(args.max_contacts)),
        max_power_balance_error_pct=max(
            0.0, float(args.power_balance_limit_pct)),
        max_torque_ripple_pct=max(0.0, float(args.torque_ripple_limit_pct)),
    )
    runner = MechanicalEvolveRunner(
        out_dir=out_dir,
        seed=int(args.seed),
        limits=limits,
        samples=max(3, int(args.samples)),
        duration_s=max(1.0e-6, float(args.duration_s)),
        dry_run=bool(args.dry_run),
        proposal_jsonl=Path(args.proposal_jsonl).expanduser().resolve()
        if args.proposal_jsonl else None,
        model_command=args.model_command,
        trainer_command=args.trainer_command,
        trainer_backend=args.trainer_backend,
        mlx_lora=MlxLoraTrainerConfig(
            model=str(args.lora_model),
            adapter_path=args.lora_adapter_path,
            iters=max(1, int(args.lora_iters)),
            batch_size=max(1, int(args.lora_batch_size)),
            grad_accumulation_steps=max(
                1, int(args.lora_grad_accumulation_steps)),
            learning_rate=float(args.lora_learning_rate),
            num_layers=int(args.lora_num_layers),
            lora_rank=max(1, int(args.lora_rank)),
            lora_scale=float(args.lora_scale),
            lora_dropout=max(0.0, float(args.lora_dropout)),
            max_seq_length=max(64, int(args.lora_max_seq_length)),
            min_reward=float(args.lora_min_reward),
            max_examples=max(1, int(args.lora_max_examples)),
            prepare_only=bool(args.lora_prepare_only),
        ),
        seed_bootstrap=not bool(args.no_seed_bootstrap),
    )
    if args.resume:
        runner.load_state()
    if args.mode == "evolve-only":
        summary = runner.evolve_only(
            generations=max(1, int(args.generations)),
            population=max(1, int(args.population)),
            audit_k=max(0, int(args.audit_k)),
        )
    elif args.mode == "rlvr-train":
        summary = runner.rlvr_train(
            generations=max(1, int(args.generations)),
            population=max(1, int(args.population)),
            audit_k=max(0, int(args.audit_k)),
        )
    else:
        summary = runner.test_time_adapt(
            target_id=str(args.target_id),
            rounds=max(1, int(args.generations)),
            population=max(1, int(args.population)),
            audit_k=max(0, int(args.audit_k)),
            policy_lr=float(args.policy_lr),
        )
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps(
        json_safe(compact_summary(summary)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


class MechanicalEvolveRunner:
    def __init__(
        self,
        *,
        out_dir: Path,
        seed: int,
        limits: cyclo.VerificationLimits,
        samples: int,
        duration_s: float,
        dry_run: bool,
        proposal_jsonl: Path | None,
        model_command: str | None,
        trainer_command: str | None,
        trainer_backend: str,
        mlx_lora: MlxLoraTrainerConfig,
        seed_bootstrap: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.assets_dir = out_dir / "assets"
        self.archive_path = out_dir / "archive.json"
        self.policy_path = out_dir / "policy_state.json"
        self.lineage_path = out_dir / "lineage.jsonl"
        self.dataset_path = out_dir / "grpo_dataset.jsonl"
        self.rng = random.Random(seed)
        self.seed = int(seed)
        self.limits = limits
        self.samples = int(samples)
        self.duration_s = float(duration_s)
        self.dry_run = bool(dry_run)
        self.proposal_jsonl = proposal_jsonl
        self.model_command = model_command
        self.trainer_command = trainer_command
        self.trainer_backend = trainer_backend
        self.mlx_lora = mlx_lora
        self.seed_bootstrap = bool(seed_bootstrap)
        self.archive = MapElitesArchive()
        self.policy = PolicyState()
        self.seen_ids: set[str] = set()
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> None:
        if self.archive_path.exists():
            self.archive = MapElitesArchive.from_dict(
                json.loads(self.archive_path.read_text()))
            for row in self.archive.cells.values():
                if row.get("id"):
                    self.seen_ids.add(str(row["id"]))
        if self.policy_path.exists():
            self.policy = PolicyState.from_dict(
                json.loads(self.policy_path.read_text()))

    def evolve_only(
        self,
        *,
        generations: int,
        population: int,
        audit_k: int,
    ) -> dict[str, Any]:
        evaluated_all: list[dict[str, Any]] = []
        for generation in range(generations):
            proposals = self._propose_generation(
                generation=generation,
                population=population,
                update_policy=False,
            )
            evaluated = self._evaluate_generation(
                generation=generation,
                proposals=proposals,
                audit_k=audit_k,
                mode="evolve-only",
            )
            evaluated_all.extend(evaluated)
            self._persist_generation(generation, evaluated)
        return self._summary(
            mode="evolve-only",
            evaluated=evaluated_all,
            extra={"generations": generations, "policy_updated": False},
        )

    def rlvr_train(
        self,
        *,
        generations: int,
        population: int,
        audit_k: int,
    ) -> dict[str, Any]:
        evaluated = self.evolve_only(
            generations=generations,
            population=population,
            audit_k=audit_k,
        )["evaluated"]
        dataset_count = self._write_grpo_dataset(evaluated)
        trainer = self._run_trainer()
        return self._summary(
            mode="rlvr-train",
            evaluated=evaluated,
            extra={
                "dataset_path": str(self.dataset_path),
                "dataset_count": dataset_count,
                "trainer": trainer,
            },
        )

    def test_time_adapt(
        self,
        *,
        target_id: str,
        rounds: int,
        population: int,
        audit_k: int,
        policy_lr: float,
    ) -> dict[str, Any]:
        evaluated_all: list[dict[str, Any]] = []
        round_summaries: list[dict[str, Any]] = []
        for round_idx in range(rounds):
            proposals = self._propose_generation(
                generation=round_idx,
                population=population,
                update_policy=True,
            )
            evaluated = self._evaluate_generation(
                generation=round_idx,
                proposals=proposals,
                audit_k=audit_k,
                mode="test-time-adapt",
            )
            self.policy.update(evaluated, lr=policy_lr)
            evaluated_all.extend(evaluated)
            self._persist_generation(round_idx, evaluated)
            round_summaries.append({
                "round": round_idx,
                "best_verified_reward": best_reward(evaluated),
                "policy": self.policy.to_dict(),
            })
            if self.trainer_command:
                self._write_grpo_dataset(evaluated_all)
                self._run_trainer()
        return self._summary(
            mode="test-time-adapt",
            evaluated=evaluated_all,
            extra={
                "target_id": target_id,
                "rounds": round_summaries,
                "policy": self.policy.to_dict(),
            },
        )

    def _propose_generation(
        self,
        *,
        generation: int,
        population: int,
        update_policy: bool,
    ) -> list[Proposal]:
        proposals: list[Proposal] = []
        if self.seed_bootstrap and generation == 0 and not self.archive.cells:
            proposals.extend(seed_proposals())
        proposals.extend(self._external_proposals(generation))
        parents = self.archive.elites(limit=max(1, min(8, population)))
        while len(proposals) < population:
            operator = (
                self.policy.choose(self.rng)
                if update_policy else self.rng.choice(list(MUTATION_OPERATORS))
            )
            parent = self.rng.choice(parents) if parents else None
            proposals.append(self._mutated_proposal(
                generation=generation,
                index=len(proposals),
                parent=parent,
                operator=operator,
            ))
        return unique_proposals(proposals)[:population]

    def _external_proposals(self, generation: int) -> list[Proposal]:
        proposals: list[Proposal] = []
        if self.proposal_jsonl and self.proposal_jsonl.exists():
            proposals.extend(read_jsonl_proposals(
                self.proposal_jsonl,
                method="llm_zero_shot" if generation == 0 else "llm_evolution",
            ))
        if self.model_command:
            prompt = self._model_prompt(generation)
            proposals.extend(command_proposals(
                self.model_command,
                prompt=prompt,
                method="llm_zero_shot" if generation == 0 else "llm_evolution",
            ))
        return proposals

    def _model_prompt(self, generation: int) -> dict[str, Any]:
        return {
            "task": "propose cycloidal/QDD actuator parameter programs",
            "generation": generation,
            "design_variables": list(DESIGN_VARIABLES),
            "paper_gate": self.limits.__dict__,
            "elites": [
                compact_row(row) for row in self.archive.elites(limit=8)
            ],
            "required_output": {
                "candidates": [
                    {
                        "params": {
                            "pins": 11,
                            "eccentricity": 1.982,
                            "clearance": 0.336,
                            "driver_circle_diameter": 49.5,
                            "driver_pin_collision_shrink_mm": 0.129,
                        },
                        "notes": "why this should improve verified reward",
                    }
                ]
            },
        }

    def _mutated_proposal(
        self,
        *,
        generation: int,
        index: int,
        parent: dict[str, Any] | None,
        operator: str,
    ) -> Proposal:
        if parent is None or operator == "random_restart":
            params = random_params(self.rng)
            parent_id = None
        else:
            params = dict(parent.get("params") or cyclo.BASELINE_PARAMS)
            parent_id = str(parent.get("id", "")) or None
        if operator == "boundary_refine":
            base = self.rng.choice(cyclo._verifier_refinement_candidates())
            params.update(base.params)
        elif operator == "pins_step":
            params["pins"] = int(clamp(
                int(round(float(params.get("pins", 10))))
                + self.rng.choice([-1, 1]),
                8,
                14,
            ))
        elif operator == "eccentricity_step":
            params["eccentricity"] = round(clamp(
                float(params.get("eccentricity", 2.0))
                + self.rng.gauss(0.0, 0.10),
                1.5,
                3.0,
            ), 3)
        elif operator == "clearance_step":
            params["clearance"] = round(clamp(
                float(params.get("clearance", 0.6))
                + self.rng.gauss(0.0, 0.05),
                0.25,
                1.15,
            ), 3)
        elif operator == "driver_circle_step":
            params["driver_circle_diameter"] = round(clamp(
                float(params.get("driver_circle_diameter", 50.0))
                + self.rng.gauss(0.0, 1.0),
                36.0,
                58.0,
            ), 3)
        elif operator == "shrink_step":
            params["driver_pin_collision_shrink_mm"] = round(clamp(
                float(params.get("driver_pin_collision_shrink_mm", 0.45))
                + self.rng.gauss(0.0, 0.035),
                0.0,
                0.82,
            ), 3)
        params["line_segment_count"] = max(
            42, int(round(float(params.get("pins", 10)))) * 4)
        patch = param_patch(parent.get("params") if parent else {}, params)
        return Proposal(
            id=f"g{generation:03d}_{index:03d}_{operator}",
            method="mechanical_evolve_ttrl",
            params=params,
            proposer="policy_mutation",
            parent_id=parent_id,
            operator=operator,
            island=archive_island(params),
            patch=patch,
        )

    def _evaluate_generation(
        self,
        *,
        generation: int,
        proposals: list[Proposal],
        audit_k: int,
        mode: str,
    ) -> list[dict[str, Any]]:
        fast_rows = [fast_row(proposal) for proposal in proposals]
        ranked = sorted(
            fast_rows,
            key=lambda row: float(row.get("fast_reward", 0.0)),
            reverse=True,
        )
        audit_budget = max(0, int(audit_k))
        external_ids = [
            proposal.id for proposal in proposals
            if proposal.proposer != "policy_mutation"
        ]
        audit_ids: set[str] = set(external_ids[:audit_budget])
        for row in ranked:
            if len(audit_ids) >= audit_budget:
                break
            audit_ids.add(str(row["id"]))
        evaluated: list[dict[str, Any]] = []
        by_id = {proposal.id: proposal for proposal in proposals}
        for row in ranked:
            proposal = by_id[str(row["id"])]
            if proposal.id in audit_ids:
                evaluated_row = self._audit_proposal(
                    proposal=proposal,
                    generation=generation,
                    mode=mode,
                )
            else:
                evaluated_row = dict(row)
                evaluated_row["defects"] = ["not_audited"]
                evaluated_row["defect_count"] = 1
            evaluated.append(evaluated_row)
            self.archive.insert(evaluated_row)
            self._append_lineage(evaluated_row)
        return evaluated

    def _audit_proposal(
        self,
        *,
        proposal: Proposal,
        generation: int,
        mode: str,
    ) -> dict[str, Any]:
        if self.dry_run:
            row = dry_verified_row(proposal)
        else:
            candidate = cyclo.Candidate(
                id=proposal.id,
                method=proposal.method,
                params=proposal.params,
                proposer=proposal.proposer,
            )
            row = cyclo._evaluate_candidate(
                candidate,
                self.assets_dir / mode / f"g{generation:03d}" / proposal.id,
                samples=self.samples,
                duration_s=self.duration_s,
                limits=self.limits,
            )
        row["proposal"] = proposal.to_dict()
        row["parent_id"] = proposal.parent_id
        row["operator"] = proposal.operator
        row["island"] = proposal.island
        return row

    def _persist_generation(
        self,
        generation: int,
        evaluated: list[dict[str, Any]],
    ) -> None:
        write_json(self.archive_path, self.archive.to_dict())
        write_json(self.policy_path, self.policy.to_dict())
        write_json(self.out_dir / f"generation_{generation:03d}.json", {
            "schema": f"{SCHEMA}.generation",
            "generation": generation,
            "evaluated": evaluated,
            "best": best_row(evaluated),
        })

    def _append_lineage(self, row: dict[str, Any]) -> None:
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lineage_path.open("a") as f:
            f.write(json.dumps(
                json_safe(compact_row(row)),
                sort_keys=True,
                allow_nan=False,
            ) + "\n")

    def _write_grpo_dataset(self, evaluated: list[dict[str, Any]]) -> int:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in evaluated:
            proposal = row.get("proposal", {})
            parent_id = str(proposal.get("parent_id") or "root")
            groups.setdefault(parent_id, []).append(row)
        count = 0
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_path.open("w") as f:
            for parent_id, rows in groups.items():
                record = {
                    "parent_id": parent_id,
                    "prompt": self._model_prompt(generation=0),
                    "responses": [
                        {
                            "candidate_id": row.get("id"),
                            "params": row.get("params"),
                            "reward": float(row.get("verified_reward", 0.0) or 0.0),
                            "defects": row.get("defects", []),
                        }
                        for row in rows
                    ],
                }
                f.write(json.dumps(
                    json_safe(record),
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n")
                count += 1
        return count

    def _run_trainer(self) -> dict[str, Any]:
        backend = self.trainer_backend
        if backend == "none" and self.trainer_command:
            backend = "command"
        if backend == "mlx-lora":
            return self._run_mlx_lora_trainer()
        if backend == "command":
            return self._run_command_trainer()
        return {
            "status": "skipped",
            "reason": "trainer_backend_none",
        }

    def _run_command_trainer(self) -> dict[str, Any]:
        if not self.trainer_command:
            return {
                "status": "skipped",
                "reason": "trainer_command_not_configured",
            }
        env = os.environ.copy()
        env.update({
            "MECHANICAL_EVOLVE_DATASET": str(self.dataset_path),
            "MECHANICAL_EVOLVE_ARCHIVE": str(self.archive_path),
            "MECHANICAL_EVOLVE_POLICY": str(self.policy_path),
        })
        completed = subprocess.run(
            shlex.split(self.trainer_command),
            cwd=self.out_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }

    def _run_mlx_lora_trainer(self) -> dict[str, Any]:
        script = SCRIPT_DIR / "train_mechanical_evolve_lora.py"
        trainer_out = self.out_dir / "mlx_lora"
        adapter_path = (
            Path(self.mlx_lora.adapter_path).expanduser().resolve()
            if self.mlx_lora.adapter_path else trainer_out / "adapters"
        )
        base_args = [
            str(script),
            "--dataset",
            str(self.dataset_path),
            "--archive",
            str(self.archive_path),
            "--out-dir",
            str(trainer_out),
            "--model",
            self.mlx_lora.model,
            "--adapter-path",
            str(adapter_path),
            "--iters",
            str(self.mlx_lora.iters),
            "--batch-size",
            str(self.mlx_lora.batch_size),
            "--grad-accumulation-steps",
            str(self.mlx_lora.grad_accumulation_steps),
            "--learning-rate",
            str(self.mlx_lora.learning_rate),
            "--num-layers",
            str(self.mlx_lora.num_layers),
            "--lora-rank",
            str(self.mlx_lora.lora_rank),
            "--lora-scale",
            str(self.mlx_lora.lora_scale),
            "--lora-dropout",
            str(self.mlx_lora.lora_dropout),
            "--max-seq-length",
            str(self.mlx_lora.max_seq_length),
            "--min-reward",
            str(self.mlx_lora.min_reward),
            "--max-examples",
            str(self.mlx_lora.max_examples),
            "--seed",
            str(self.seed),
        ]
        if self.mlx_lora.prepare_only:
            base_args.append("--prepare-only")
        if importlib.util.find_spec("mlx_lm") is not None:
            command = [sys.executable, *base_args]
        elif shutil.which("uv"):
            command = [
                "uv",
                "run",
                "--with",
                "mlx-lm",
                "--with",
                "huggingface_hub",
                "python",
                *base_args,
            ]
        else:
            return {
                "status": "failed",
                "reason": "mlx_lm_not_installed_and_uv_not_found",
                "model": self.mlx_lora.model,
            }

        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.update({
            "MECHANICAL_EVOLVE_DATASET": str(self.dataset_path),
            "MECHANICAL_EVOLVE_ARCHIVE": str(self.archive_path),
            "MECHANICAL_EVOLVE_POLICY": str(self.policy_path),
        })
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        child_summary = {}
        child_summary_path = trainer_out / "training_summary.json"
        if child_summary_path.exists():
            try:
                child_summary = json.loads(child_summary_path.read_text())
            except json.JSONDecodeError:
                child_summary = {}
        child_trainer = child_summary.get("trainer", {})
        if not isinstance(child_trainer, dict):
            child_trainer = {}
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "backend": "mlx-lora",
            "returncode": completed.returncode,
            "model": self.mlx_lora.model,
            "out_dir": str(trainer_out),
            "adapter_path": str(adapter_path),
            "adapter_file_exists": (adapter_path / "adapters.safetensors").is_file(),
            "ok": child_trainer.get("ok"),
            "train_loss": child_trainer.get("train_loss"),
            "val_loss": child_trainer.get("val_loss"),
            "trained_tokens": child_trainer.get("trained_tokens"),
            "peak_mem_gb": child_trainer.get("peak_mem_gb"),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    def _summary(
        self,
        *,
        mode: str,
        evaluated: list[dict[str, Any]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "seed": self.seed,
            "dry_run": self.dry_run,
            "design_variables": list(DESIGN_VARIABLES),
            "verifier": {
                "contact_model": "smc",
                "procedural_cycloidal_fallback": False,
                "input_speed_rad_s": 10.0,
                "output_load_Nm": 0.75,
                "samples": self.samples,
                "duration_s": self.duration_s,
                "limits": self.limits.__dict__,
            },
            "archive_path": str(self.archive_path),
            "lineage_path": str(self.lineage_path),
            "archive_cell_count": len(self.archive.cells),
            "best": best_row(self.archive.elites(limit=1)),
            "evaluated": evaluated,
            **extra,
        }


def seed_proposals() -> list[Proposal]:
    proposals = [
        Proposal(
            id="seed_current",
            method="seed",
            params=dict(cyclo.BASELINE_PARAMS),
            proposer="current_branch_seed",
            operator="seed",
        )
    ]
    for candidate in cyclo._verifier_refinement_candidates():
        proposals.append(Proposal(
            id=f"seed_{candidate.id}",
            method="mechanical_evolve_ttrl",
            params=dict(candidate.params),
            proposer="known_boundary_seed",
            operator="boundary_refine",
            patch=param_patch(cyclo.BASELINE_PARAMS, candidate.params),
        ))
    return proposals


def random_params(rng: random.Random) -> dict[str, Any]:
    return cyclo._params_from_vector((
        rng.uniform(8.0, 13.5),
        rng.uniform(0.25, 1.15),
        rng.uniform(36.0, 58.0),
        rng.uniform(0.0, 0.82),
        rng.uniform(1.5, 3.0),
    ))


def fast_row(proposal: Proposal) -> dict[str, Any]:
    fast = cyclo.fast_cps_actuator_reward(proposal.params)
    return {
        "id": proposal.id,
        "method": proposal.method,
        "proposer": proposal.proposer,
        "params": dict(proposal.params),
        "fast_reward": float(fast["score"]),
        "fast_reward_detail": fast,
        "verified_reward": 0.0,
        "verified_gate_passed": False,
        "cad_generated": False,
        "cad_static_ok": False,
        "chrono_real_geometry": False,
        "defects": ["not_audited"],
        "defect_count": 1,
        "proposal": proposal.to_dict(),
        "parent_id": proposal.parent_id,
        "operator": proposal.operator,
        "island": proposal.island,
    }


def dry_verified_row(proposal: Proposal) -> dict[str, Any]:
    row = fast_row(proposal)
    score = float(row["fast_reward"])
    likely_valid = (
        int(round(float(proposal.params.get("pins", 0)))) == 11
        and 1.85 <= float(proposal.params.get("eccentricity", 0.0)) <= 2.10
        and 0.30 <= float(proposal.params.get("clearance", 0.0)) <= 0.42
        and 0.12 <= float(proposal.params.get(
            "driver_pin_collision_shrink_mm", 0.0)) <= 0.20
    )
    row.update({
        "cad_generated": True,
        "cad_static_ok": True,
        "chrono_real_geometry": True,
        "metrics": {
            "lockup_detected": 0.0 if likely_valid else 1.0,
            "out_omega_med": 1.0 if likely_valid else 0.05,
            "ratio_observed": 9.0 if likely_valid else 100.0,
            "ratio_error_pct": 10.0 if likely_valid else 900.0,
            "max_penetration_mm": 0.4,
            "contact_force_rms_N": 500.0,
            "n_contacts_max": 36.0,
        },
        "verified_gate_passed": likely_valid,
        "verified_reward": round(score * 0.85, 6) if likely_valid else 0.0,
        "defects": [] if likely_valid else ["lockup", "ratio_error_over_gate"],
        "defect_count": 0 if likely_valid else 2,
    })
    return row


def archive_key(params: dict[str, Any]) -> str:
    pins = int(round(float(params.get("pins", 0) or 0)))
    ecc_bin = int(float(params.get("eccentricity", 0.0)) / 0.25)
    clearance_bin = int(float(params.get("clearance", 0.0)) / 0.10)
    circle_bin = int(float(params.get("driver_circle_diameter", 0.0)) / 2.0)
    shrink_bin = int(float(params.get(
        "driver_pin_collision_shrink_mm", 0.0)) / 0.05)
    return (
        f"pins={pins}|ecc={ecc_bin}|clearance={clearance_bin}|"
        f"circle={circle_bin}|shrink={shrink_bin}"
    )


def archive_island(params: dict[str, Any]) -> int:
    return int(round(float(params.get("pins", 10)))) % 4


def archive_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("verified_reward", 0.0) or 0.0),
        1.0 if row.get("verified_gate_passed") else 0.0,
        float(row.get("fast_reward", 0.0) or 0.0),
        -float(row.get("defect_count", 0.0) or 0.0),
    )


def best_reward(rows: Iterable[dict[str, Any]]) -> float:
    return max(
        (float(row.get("verified_reward", 0.0) or 0.0) for row in rows),
        default=0.0,
    )


def best_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    rows_list = list(rows)
    if not rows_list:
        return None
    return max(rows_list, key=archive_score)


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    metrics = row.get("metrics") or {}
    return {
        "id": row.get("id"),
        "method": row.get("method"),
        "params": row.get("params"),
        "fast_reward": row.get("fast_reward"),
        "verified_reward": row.get("verified_reward"),
        "verified_gate_passed": row.get("verified_gate_passed"),
        "defects": row.get("defects", []),
        "operator": row.get("operator"),
        "parent_id": row.get("parent_id"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "out_omega_med",
                "ratio_observed",
                "ratio_error_pct",
                "max_penetration_mm",
                "contact_force_rms_N",
                "n_contacts_max",
                "lockup_detected",
                "power_balance_error_pct",
                "torque_ripple_pct",
            )
            if key in metrics
        },
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    trainer = summary.get("trainer") or {}
    return {
        "mode": summary.get("mode"),
        "archive_cell_count": summary.get("archive_cell_count"),
        "best": compact_row(summary.get("best")),
        "archive_path": summary.get("archive_path"),
        "lineage_path": summary.get("lineage_path"),
        "dataset_path": summary.get("dataset_path"),
        "dry_run": summary.get("dry_run"),
        "trainer": {
            key: trainer.get(key)
            for key in (
                "status",
                "backend",
                "model",
                "out_dir",
                "adapter_file_exists",
                "ok",
                "train_loss",
                "val_loss",
                "trained_tokens",
                "peak_mem_gb",
            )
            if key in trainer
        },
    }


def unique_proposals(proposals: Iterable[Proposal]) -> list[Proposal]:
    seen: set[tuple[Any, ...]] = set()
    out: list[Proposal] = []
    for proposal in proposals:
        key = cyclo._candidate_key(proposal.params)
        if key in seen:
            continue
        seen.add(key)
        out.append(proposal)
    return out


def read_jsonl_proposals(path: Path, *, method: str) -> list[Proposal]:
    proposals: list[Proposal] = []
    for idx, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        proposals.extend(parse_model_payload(
            raw,
            method=method,
            id_prefix=f"jsonl_{idx:04d}",
            proposer="jsonl_model_proposals",
        ))
    return proposals


def command_proposals(
    command: str,
    *,
    prompt: dict[str, Any],
    method: str,
) -> list[Proposal]:
    completed = subprocess.run(
        shlex.split(command),
        input=json.dumps(prompt),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    return parse_model_payload(
        raw,
        method=method,
        id_prefix="cmd",
        proposer="command_model_proposals",
    )


def parse_model_payload(
    raw: Any,
    *,
    method: str,
    id_prefix: str,
    proposer: str,
) -> list[Proposal]:
    if isinstance(raw, dict) and isinstance(raw.get("candidates"), list):
        items = raw["candidates"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    proposals: list[Proposal] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        params = item.get("params", item)
        if not isinstance(params, dict):
            continue
        clean = normalize_params(params)
        proposal_method = str(item.get("method") or method)
        proposals.append(Proposal(
            id=str(item.get("id") or f"{id_prefix}_{idx:03d}"),
            method=proposal_method,
            params=clean,
            proposer=proposer,
            parent_id=item.get("parent_id"),
            operator=str(item.get("operator", "model_mutation")),
            patch=list(item.get("patch", []))
            if isinstance(item.get("patch", []), list) else [],
            prompt=str(item.get("prompt", "")),
            notes=str(item.get("notes", "")),
        ))
    return proposals


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cyclo.BASELINE_PARAMS)
    for key in DESIGN_VARIABLES:
        if key in params:
            merged[key] = params[key]
    pins = int(clamp(round(float(merged["pins"])), 8, 14))
    merged["pins"] = pins
    merged["line_segment_count"] = max(42, pins * 4)
    merged["eccentricity"] = round(clamp(float(merged["eccentricity"]), 1.5, 3.0), 3)
    merged["clearance"] = round(clamp(float(merged["clearance"]), 0.25, 1.15), 3)
    merged["driver_circle_diameter"] = round(
        clamp(float(merged["driver_circle_diameter"]), 36.0, 58.0), 3)
    merged["driver_pin_collision_shrink_mm"] = round(
        clamp(float(merged["driver_pin_collision_shrink_mm"]), 0.0, 0.82), 3)
    return merged


def param_patch(
    parent: dict[str, Any] | None,
    child: dict[str, Any],
) -> list[dict[str, Any]]:
    parent = parent or {}
    patch: list[dict[str, Any]] = []
    for key in DESIGN_VARIABLES:
        if parent.get(key) != child.get(key):
            patch.append({
                "op": "set_param",
                "path": f"/params/{key}",
                "value": child.get(key),
            })
    return patch


def clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
