#!/usr/bin/env python3
"""Exact TRL GRPO training for mech_bench verifier rewards.

This is the canonical GRPO path. It uses Hugging Face TRL's ``GRPOTrainer``
instead of the legacy Worldlines group-relative weighted-CE loop in
``rl/train_grpo.py``. The reward function executes the existing deterministic
``mech_bench`` verifier for each sampled completion and returns verified reward
only.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rl"))

import mech_env as env  # noqa: E402
from rl.mech_bench_reward import extract_design_py, score_completion  # noqa: E402
from rl.chat_rollout import _load_sglang_lora_adapter  # noqa: E402
from rl.chat_rollout import _maybe_filter_sglang_lora_adapter  # noqa: E402
from rl.train_grpo import _build_user_prompt  # noqa: E402
from rl.train_grpo import _contract_from_task  # noqa: E402


SYSTEM_PROMPT_PATH = REPO_ROOT / "rl" / "agent_prompt_rl.md"
SCHEMA = "mech_bench.true_grpo_trl.v1"
REWARD_CHANNELS = ("verified_score", "score", "artifact_progress")
STRICT_FENCED_OUTPUT_INSTRUCTION = (
    "/no_think\n"
    "You must reply with exactly one fenced ```python code block containing "
    "the complete design.py implementation. Use compact valid Python only: "
    "no comments, no analysis, and no explanations. Return a complete "
    "build_design(out_dir: Path) implementation with all dictionaries, lists, "
    "braces, parentheses, and the final code fence closed. Do not include "
    "additional file writes, additional write_text calls, STEP contents, "
    "markdown outside that one code block, or any text before or after it. "
    "If an assistant code prefill is present, continue the already-open "
    "`parts` list after `frame(out_dir),`; do not repeat imports, helper "
    "functions, schema_version, materials, parts, or def build_design. "
    "After closing the parts list, add joints, ports, and params keys. "
    "For timing-belt and chain-sprocket analytic tasks, never create a "
    "part, joint, role, or id containing `belt` or `chain`; represent the "
    "belt/chain only by top-level params and use exactly two moving rotating "
    "parts plus the prefilled frame. "
    "Start your response with exactly ```python followed by the design.py "
    "source."
)
ASSISTANT_CODE_PREFILL = (
    "```python\n"
    "from pathlib import Path\n\n"
    "STEP='ISO-10303-21;END-ISO-10303-21;\\n'\n"
    "def cad(out_dir,n): (out_dir/n).write_text(STEP); return n\n"
    "I=((1e-4,0.0,0.0),(0.0,1e-4,0.0),(0.0,0.0,1e-4))\n"
    "def mp(m,c,s=None):\n"
    "    p={'cad_mass_properties':{'mass_kg':m,'com_local_mm':c,'inertia_kg_m2':I}}\n"
    "    if s is not None: p['chrono_collision']=s\n"
    "    return p\n\n"
    "def cyl(r,l): return {'shape':'cylinder','radius_mm':r,'length_mm':l}\n"
    "def box(x,y,z): return {'shape':'box','size_mm':(x,y,z)}\n"
    "def cm(m,c,s): return mp(m,c,s)\n\n"
    "M={'steel':{'density_kg_m3':7850.0,'provenance':'datasheet'},'aluminum':{'density_kg_m3':2700.0,'provenance':'datasheet'},'rubber':{'density_kg_m3':1100.0,'provenance':'datasheet'}}\n\n"
    "def frame(out_dir): return {'id':'frame','role':'ground','mass_kg':0.001,'fixed':True,'com_local_mm':(0.0,0.0,0.0),'geometry':{'cad':cad(out_dir,'frame.step')},'material':'steel','params':cm(0.001,(0.0,0.0,0.0),box(20,20,5))}\n\n"
    "def build_design(out_dir: Path) -> dict:\n"
    "    return {'schema_version':'design_ir.v2','units':'mm','materials':M,\n"
    "        'parts':[frame(out_dir),\n"
)


def _full_eval_contract_suffix(task: env.TaskInfo) -> str:
    eval_config_path = task.task_dir / "eval_config.toml"
    if not eval_config_path.exists():
        return ""
    try:
        eval_config = eval_config_path.read_text()
    except OSError:
        return ""
    if "trusted_asset_preflight" not in eval_config and "chrono_contact" not in eval_config:
        return ""

    contract = _contract_from_task(task.prompt, task.task_toml, eval_config)
    trusted_hint = (
        "For paper-verifier tasks, every positive-mass part must include "
        "`geometry={\"cad\": cad(out_dir,\"<part>.step\")}`, `material` as a "
        "string id referencing a top-level `materials` record with "
        "`density_kg_m3` and `provenance`, "
        "and `params` set directly to `cm(mass, com_tuple, cyl(...) or box(...))` "
        "so each moving/contact part has both `cad_mass_properties` and "
        "`chrono_collision` primitive geometry. Positive-mass parts need "
        "positive `mass_kg`, `com_local_mm`, and 3x3 positive diagonal "
        "`inertia_kg_m2` inside each positive-mass part. The prefilled "
        "`frame(out_dir)` must remain the first part. Include top-level "
        "`params.cad_source={\"kernel\": \"FreeCAD/OCCT\"}`. Do not inline "
        "STEP contents or call `write_text`; use the prefilled `cad` helper. "
        "For timing-belt and chain-sprocket analytic tasks, use only frame "
        "plus input/output pulley or sprocket parts, two revolute joints, and "
        "no extra belt, chain, or contact_pair part unless required_pairs is "
        "nonempty. In these analytic tasks, any part, role, joint, or id "
        "containing `belt` or `chain` is invalid because it changes mobility."
    )
    return "\n\n## full private verifier contract\n" + contract + "\n- " + trusted_hint


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if completion is None else str(completion)


def _read_split_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _format_prompt(task: env.TaskInfo, system_prompt: str) -> str:
    return "\n\n".join([
        "### System",
        system_prompt.strip(),
        "### User",
        _build_user_prompt(task).strip(),
        "### Assistant",
    ])


def _apply_chat_template_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return rows
    templated: list[dict[str, Any]] = []
    for row in rows:
        system_prompt = row.get("_system_prompt")
        user_prompt = row.get("_user_prompt")
        if system_prompt is None or user_prompt is None:
            templated.append(dict(row))
            continue
        next_row = dict(row)
        next_row["prompt"] = apply_chat_template(
            [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        templated.append(next_row)
    return templated


def _build_rows(
    *,
    tasks_root: Path,
    split_file: Path | None,
    families: set[str] | None,
    tiers: set[str] | None,
    system_prompt: str,
    limit: int,
) -> list[dict[str, Any]]:
    tasks = env.list_tasks(
        root=tasks_root,
        split_file=split_file,
        families=families,
        tiers=tiers,
    )
    if limit > 0:
        tasks = tasks[:limit]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        user_prompt = (
            _build_user_prompt(task).strip()
            + _full_eval_contract_suffix(task)
        )
        rows.append({
            "prompt": _format_prompt(task, system_prompt),
            "_system_prompt": system_prompt.strip(),
            "_user_prompt": user_prompt,
            "task_id": task.task_id,
            "task_dir": str(task.task_dir.resolve()),
            "family": task.family,
            "tier": task.tier,
        })
    return rows


def _repeat_rows_for_grpo_sampler(
    rows: list[dict[str, Any]],
    *,
    generation_batch_size: int,
    num_generations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return rows, {
            "expanded": False,
            "source_rows": 0,
            "trainer_rows": 0,
            "unique_prompts_per_generation": 0,
            "repeat_factor": 0,
        }
    generation_batch_size = int(generation_batch_size)
    num_generations = int(num_generations)
    if generation_batch_size <= 0:
        raise SystemExit("GRPO generation_batch_size must be positive")
    if num_generations <= 0:
        raise SystemExit("--num-generations must be positive")
    if generation_batch_size % num_generations:
        raise SystemExit(
            "GRPO generation_batch_size must divide evenly by "
            f"num_generations: {generation_batch_size} vs {num_generations}"
        )
    unique_prompts = generation_batch_size // num_generations
    if len(rows) >= unique_prompts:
        return rows, {
            "expanded": False,
            "source_rows": len(rows),
            "trainer_rows": len(rows),
            "unique_prompts_per_generation": unique_prompts,
            "repeat_factor": 1,
        }

    expanded: list[dict[str, Any]] = []
    for repeat_index in range(unique_prompts):
        source_index = repeat_index % len(rows)
        row = dict(rows[source_index])
        row["_trainer_source_index"] = source_index
        row["_trainer_repeat_index"] = repeat_index
        expanded.append(row)
    return expanded, {
        "expanded": True,
        "source_rows": len(rows),
        "trainer_rows": len(expanded),
        "unique_prompts_per_generation": unique_prompts,
        "repeat_factor": (len(expanded) + len(rows) - 1) // len(rows),
    }


def _filtered_config(cls: type, values: dict[str, Any]) -> Any:
    params = inspect.signature(cls.__init__).parameters
    accepted = {
        key: value for key, value in values.items()
        if key in params and value is not None
    }
    return cls(**accepted)


def _disable_intermediate_checkpoint_config(save_steps: int) -> dict[str, Any]:
    return {
        "save_strategy": "no",
        "save_steps": int(save_steps),
        "save_total_limit": 1,
    }


def _remove_intermediate_checkpoints(out_dir: Path) -> list[str]:
    removed: list[str] = []
    for checkpoint in sorted(out_dir.glob("checkpoint-*")):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
            removed.append(str(checkpoint))
    return removed


def _finite_named_tensor_audit(
    named_tensors: Any,
    *,
    max_examples: int = 8,
) -> dict[str, Any]:
    import torch

    total_values = 0
    nonfinite_values = 0
    checked_tensors = 0
    examples: list[dict[str, Any]] = []
    for name, tensor in named_tensors:
        if tensor is None or not hasattr(tensor, "detach"):
            continue
        detached = tensor.detach()
        if not (torch.is_floating_point(detached) or torch.is_complex(detached)):
            continue
        checked_tensors += 1
        n_values = int(detached.numel())
        total_values += n_values
        finite = torch.isfinite(detached)
        n_nonfinite = n_values - int(finite.sum().item())
        nonfinite_values += n_nonfinite
        if n_nonfinite and len(examples) < max_examples:
            examples.append({
                "name": str(name),
                "shape": list(detached.shape),
                "nonfinite": n_nonfinite,
                "numel": n_values,
            })
    return {
        "checked_tensors": checked_tensors,
        "total_values": total_values,
        "nonfinite_values": nonfinite_values,
        "examples": examples,
    }


def _finite_trainable_parameter_audit(model: Any) -> dict[str, Any]:
    return _finite_named_tensor_audit(
        (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if getattr(parameter, "requires_grad", False)
        )
    )


def _raise_if_nonfinite_trainable_parameters(
    model: Any,
    *,
    label: str,
) -> dict[str, Any]:
    audit = _finite_trainable_parameter_audit(model)
    if int(audit["nonfinite_values"]):
        raise RuntimeError(
            f"{label} contains non-finite trainable adapter weights: "
            + json.dumps(audit, sort_keys=True)
        )
    return audit


class _GuardedAdamW:
    """AdamW wrapper that refuses to persist non-finite LoRA updates."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch

        class GuardedAdamW(torch.optim.AdamW):  # type: ignore[misc]
            def __init__(
                self,
                params: Any,
                *,
                parameter_names: dict[int, str] | None = None,
                **optimizer_kwargs: Any,
            ) -> None:
                super().__init__(params, **optimizer_kwargs)
                self.parameter_names = parameter_names or {}
                self.attempted_steps = 0
                self.successful_steps = 0
                self.skipped_nonfinite_gradient_steps = 0
                self.skipped_all_nonfinite_gradient_steps = 0
                self.sanitized_nonfinite_gradient_steps = 0
                self.sanitized_nonfinite_gradient_values = 0
                self.rolled_back_nonfinite_update_steps = 0
                self.nonfinite_gradient_values = 0
                self.nonfinite_parameter_values_after_update = 0
                self.last_step_status = "not_started"
                self.last_step_audit: dict[str, Any] = {}

            def _iter_params(self) -> list[Any]:
                params: list[Any] = []
                for group in self.param_groups:
                    params.extend(group.get("params", []))
                return params

            def _named_params(self) -> list[tuple[str, Any]]:
                out: list[tuple[str, Any]] = []
                for index, param in enumerate(self._iter_params()):
                    name = self.parameter_names.get(id(param), f"param_{index}")
                    out.append((name, param))
                return out

            def _finite_param_audit(self) -> dict[str, Any]:
                return _finite_named_tensor_audit(self._named_params())

            def _finite_grad_audit(self) -> dict[str, Any]:
                return _finite_named_tensor_audit(
                    (
                        (name, getattr(param, "grad", None))
                        for name, param in self._named_params()
                    )
                )

            def _clear_state(self) -> None:
                for value in self.state.values():
                    if isinstance(value, dict):
                        value.clear()
                self.state.clear()

            def _zero_nonfinite_gradients(self) -> None:
                for _name, param in self._named_params():
                    grad = getattr(param, "grad", None)
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        grad = grad.coalesce()
                        values = grad._values()
                        finite = torch.isfinite(values)
                        if not bool(finite.all()):
                            values.masked_fill_(~finite, 0.0)
                        param.grad = grad
                        continue
                    finite = torch.isfinite(grad)
                    if not bool(finite.all()):
                        grad.masked_fill_(~finite, 0.0)

            def step(self, closure: Any = None) -> Any:  # noqa: ANN401
                self.attempted_steps += 1
                loss = closure() if closure is not None else None
                sanitized_this_step = False
                grad_audit = self._finite_grad_audit()
                if int(grad_audit["nonfinite_values"]):
                    self.nonfinite_gradient_values += int(
                        grad_audit["nonfinite_values"]
                    )
                    all_nonfinite = (
                        int(grad_audit["total_values"]) > 0
                        and int(grad_audit["nonfinite_values"])
                        >= int(grad_audit["total_values"])
                    )
                    if all_nonfinite:
                        self.skipped_nonfinite_gradient_steps += 1
                        self.skipped_all_nonfinite_gradient_steps += 1
                        self.last_step_status = "skipped_all_nonfinite_gradient"
                        self.last_step_audit = grad_audit
                        self.zero_grad(set_to_none=True)
                        return loss
                    self._zero_nonfinite_gradients()
                    clean_grad_audit = self._finite_grad_audit()
                    if int(clean_grad_audit["nonfinite_values"]):
                        self.skipped_nonfinite_gradient_steps += 1
                        self.last_step_status = "skipped_unsanitized_gradient"
                        self.last_step_audit = clean_grad_audit
                        self.zero_grad(set_to_none=True)
                        return loss
                    self.sanitized_nonfinite_gradient_steps += 1
                    self.sanitized_nonfinite_gradient_values += int(
                        grad_audit["nonfinite_values"]
                    )
                    sanitized_this_step = True
                    grad_audit = clean_grad_audit

                backups = [
                    (param, param.detach().clone())
                    for param in self._iter_params()
                    if hasattr(param, "detach")
                ]
                result = super().step()
                param_audit = self._finite_param_audit()
                if int(param_audit["nonfinite_values"]):
                    for param, backup in backups:
                        param.data.copy_(backup)
                    self._clear_state()
                    self.rolled_back_nonfinite_update_steps += 1
                    self.nonfinite_parameter_values_after_update += int(
                        param_audit["nonfinite_values"]
                    )
                    self.last_step_status = "rolled_back_nonfinite_update"
                    self.last_step_audit = param_audit
                    self.zero_grad(set_to_none=True)
                    return loss if loss is not None else result

                self.successful_steps += 1
                self.last_step_status = (
                    "successful_after_gradient_sanitize"
                    if sanitized_this_step
                    else "successful"
                )
                self.last_step_audit = param_audit
                return loss if loss is not None else result

        return GuardedAdamW(*args, **kwargs)


def _make_guarded_adamw(
    model: Any,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> Any:
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if getattr(parameter, "requires_grad", False)
    ]
    if not trainable:
        raise RuntimeError("cannot build optimizer: no trainable parameters")
    return _GuardedAdamW(
        [{"params": [parameter for _, parameter in trainable],
          "weight_decay": float(weight_decay)}],
        parameter_names={id(parameter): name for name, parameter in trainable},
        lr=float(learning_rate),
    )


def _guarded_optimizer_manifest(optimizer: Any) -> dict[str, Any]:
    return {
        "attempted_steps": int(getattr(optimizer, "attempted_steps", 0) or 0),
        "successful_steps": int(getattr(optimizer, "successful_steps", 0) or 0),
        "skipped_nonfinite_gradient_steps": int(
            getattr(optimizer, "skipped_nonfinite_gradient_steps", 0) or 0
        ),
        "skipped_all_nonfinite_gradient_steps": int(
            getattr(optimizer, "skipped_all_nonfinite_gradient_steps", 0) or 0
        ),
        "sanitized_nonfinite_gradient_steps": int(
            getattr(optimizer, "sanitized_nonfinite_gradient_steps", 0) or 0
        ),
        "sanitized_nonfinite_gradient_values": int(
            getattr(optimizer, "sanitized_nonfinite_gradient_values", 0) or 0
        ),
        "rolled_back_nonfinite_update_steps": int(
            getattr(optimizer, "rolled_back_nonfinite_update_steps", 0) or 0
        ),
        "nonfinite_gradient_values": int(
            getattr(optimizer, "nonfinite_gradient_values", 0) or 0
        ),
        "nonfinite_parameter_values_after_update": int(
            getattr(optimizer, "nonfinite_parameter_values_after_update", 0) or 0
        ),
        "last_step_status": str(
            getattr(optimizer, "last_step_status", "unknown")
        ),
        "last_step_audit": getattr(optimizer, "last_step_audit", {}),
    }


def _prepare_model_for_kbit_training_lightweight(
    model: Any,
    *,
    use_gradient_checkpointing: bool,
) -> Any:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not use_gradient_checkpointing:
        return model
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):
        def make_inputs_require_grad(_module: Any, _input: Any, output: Any) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad
        )
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable):
        params = inspect.signature(enable).parameters
        if "gradient_checkpointing_kwargs" in params:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            enable()
    return model


def _parse_device_map(value: str | None) -> str | dict[str, int] | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized in {"single", "cuda", "cuda:0", "0"}:
        return {"": 0}
    return value


def _parse_max_memory(value: str | None) -> dict[int | str, str] | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "none":
        return None
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("--max-memory JSON must be an object")
        parsed: dict[int | str, str] = {}
        for key, memory in payload.items():
            device: int | str = int(key) if str(key).isdigit() else str(key)
            parsed[device] = str(memory)
        return parsed

    parsed = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--max-memory entries must be DEVICE:MEMORY, "
                "for example 0:32GiB,1:44GiB"
            )
        key, memory = item.split(":", 1)
        key = key.strip()
        memory = memory.strip()
        if not key or not memory:
            raise ValueError("--max-memory entries require device and memory")
        device = int(key) if key.isdigit() else key
        parsed[device] = memory
    return parsed or None


def _make_reward_func(
    *,
    log_path: Path,
    scratch_root: Path,
    timeout_s: float,
    reward_scale: float,
    reward_channel: str,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def reward_func(
        prompts: list[Any],
        completions: list[Any],
        task_dir: list[str],
        task_id: list[str] | None = None,
        rollout_completion_text: list[str] | None = None,
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        rows: list[dict[str, Any]] = []
        ids = task_id or [""] * len(completions)
        for idx, completion in enumerate(completions):
            if rollout_completion_text and idx < len(rollout_completion_text):
                text = str(rollout_completion_text[idx])
            else:
                text = _completion_text(completion)
            task_path = Path(task_dir[idx]).resolve()
            candidate_scratch = scratch_root / f"reward_{time.time_ns()}_{idx}"
            candidate_scratch.mkdir(parents=True, exist_ok=True)
            result = score_completion(
                text,
                task_path,
                scratch_root=candidate_scratch,
                timeout_s=timeout_s,
            )
            reward_base, reward_features = _reward_base(
                text,
                result,
                reward_channel=reward_channel,
            )
            reward = reward_base * float(reward_scale)
            rewards.append(reward)
            rows.append({
                "ts": time.time(),
                "task_id": ids[idx],
                "task_dir": str(task_path),
                "reward": reward,
                "reward_base": reward_base,
                "reward_channel": reward_channel,
                "reward_features": reward_features,
                "completion_text": text,
                "completion_preview": text[:4000],
                **result.to_dict(),
            })
        with log_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return rewards

    return reward_func


def _reward_base(
    completion_text: str,
    result: Any,
    *,
    reward_channel: str,
) -> tuple[float, dict[str, Any]]:
    if reward_channel == "verified_score":
        return float(result.verified_score), {}
    if reward_channel == "score":
        return float(result.score), {}
    if reward_channel != "artifact_progress":
        raise ValueError(f"unknown reward channel: {reward_channel}")
    return _artifact_progress_reward(completion_text, result)


def _artifact_progress_reward(
    completion_text: str,
    result: Any,
) -> tuple[float, dict[str, Any]]:
    source, extracted = extract_design_py(completion_text)
    try:
        ast.parse(source)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False

    lowered = source.lower()
    codes = {str(code).lower() for code in (result.failure_codes or [])}
    required_terms = {
        "schema_version": "schema_version" in lowered,
        "design_ir_v2": "design_ir.v2" in lowered,
        "parts": "parts" in lowered,
        "joints": "joints" in lowered,
        "ports": "ports" in lowered,
        "params": "params" in lowered,
        "input_port": "input_port" in lowered,
        "output_port": "output_port" in lowered,
    }
    features: dict[str, Any] = {
        "design_py_extracted": bool(extracted),
        "closed_code_fence": completion_text.count("```") >= 2,
        "syntax_ok": syntax_ok,
        "has_build_design": "def build_design" in lowered,
        "has_no_invalid_artifact_code": "invalid_artifact" not in codes,
        "evaluation_valid": bool(result.evaluation_valid),
        "hard_gate_passed": bool(result.hard_gate_passed),
        "required_terms": required_terms,
        "score": float(result.score),
        "verified_score": float(result.verified_score),
    }

    reward = 0.0
    reward += 0.02 if features["design_py_extracted"] else 0.0
    reward += 0.04 if features["closed_code_fence"] else 0.0
    reward += 0.08 if features["has_build_design"] else 0.0
    reward += 0.18 if features["syntax_ok"] else 0.0
    reward += 0.025 * sum(1 for present in required_terms.values() if present)
    reward += 0.10 if features["has_no_invalid_artifact_code"] else 0.0
    reward += 0.20 if features["evaluation_valid"] else 0.0
    reward += 0.15 if features["hard_gate_passed"] else 0.0
    reward += 0.35 * max(0.0, min(1.0, float(result.score)))
    reward += 0.10 * max(0.0, min(1.0, float(result.verified_score)))
    return max(0.0, min(1.0, reward)), features


def _post_openai_chat_completion(
    *,
    requests_mod: Any,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests_mod.post(url, json=body, timeout=timeout_s, headers=headers)
    try:
        response.raise_for_status()
    except Exception as exc:
        preview = ""
        try:
            preview = f" response={response.text[:500]}"
        except Exception:
            pass
        raise RuntimeError(f"{exc}{preview}") from exc
    return response.json()


def _chat_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chat_rows: list[dict[str, Any]] = []
    for row in rows:
        system_prompt = row.get("_system_prompt")
        user_prompt = row.get("_user_prompt")
        if system_prompt is None or user_prompt is None:
            chat_rows.append(dict(row))
            continue
        next_row = dict(row)
        next_row["prompt"] = [
            {"role": "system", "content": str(system_prompt)},
            {"role": "user", "content": str(user_prompt)},
        ]
        chat_rows.append(next_row)
    return chat_rows


def _append_strict_fenced_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        return [{"role": "user", "content": STRICT_FENCED_OUTPUT_INSTRUCTION}]

    strict_messages = [dict(item) for item in messages]
    for idx in range(len(strict_messages) - 1, -1, -1):
        if strict_messages[idx].get("role") == "user":
            content = strict_messages[idx].get("content") or ""
            if STRICT_FENCED_OUTPUT_INSTRUCTION not in content:
                strict_messages[idx]["content"] = (
                    content.rstrip()
                    + "\n\n"
                    + STRICT_FENCED_OUTPUT_INSTRUCTION
                )
            return strict_messages
    strict_messages.append({
        "role": "user",
        "content": STRICT_FENCED_OUTPUT_INSTRUCTION,
    })
    return strict_messages


def _messages_for_rollout(
    prompt: Any,
    *,
    require_fenced_output: bool = True,
    include_assistant_prefill: bool = False,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(prompt, list):
        for item in prompt:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif content is not None:
                messages.append({"role": role, "content": str(content)})
    if not messages:
        messages = [{"role": "user", "content": str(prompt)}]
    if require_fenced_output:
        messages = _append_strict_fenced_instruction(messages)
    if include_assistant_prefill:
        messages = [
            item for item in messages
            if item.get("role") != "assistant"
        ]
        messages.append({
            "role": "assistant",
            "content": ASSISTANT_CODE_PREFILL,
        })
    return messages


def _prompt_text_for_rollout(
    tokenizer: Any,
    prompt: Any,
    *,
    require_fenced_output: bool = True,
    include_assistant_prefill: bool = False,
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if isinstance(prompt, list) and callable(apply_chat_template):
        messages = _messages_for_rollout(
            prompt,
            require_fenced_output=require_fenced_output,
            include_assistant_prefill=include_assistant_prefill,
        )
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": not include_assistant_prefill,
        }
        if include_assistant_prefill:
            try:
                return apply_chat_template(
                    messages,
                    continue_final_message=True,
                    **kwargs,
                )
            except TypeError:
                pass
        return apply_chat_template(
            messages,
            **kwargs,
        )
    return str(prompt)


def _truncate_token_ids(input_ids: list[int], max_length: int) -> list[int]:
    if max_length <= 0 or len(input_ids) <= max_length:
        return list(input_ids)
    head_tokens = max(1, max_length // 2)
    tail_tokens = max_length - head_tokens
    if tail_tokens <= 0:
        return list(input_ids[:max_length])
    return list(input_ids[:head_tokens] + input_ids[-tail_tokens:])


def _sanitize_token_ids(
    input_ids: list[Any],
    *,
    tokenizer: Any,
) -> list[int]:
    vocab_size = None
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = getattr(tokenizer, "vocab_size", None)
    cleaned: list[int] = []
    for raw in input_ids:
        try:
            token_id = int(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if token_id < 0:
            continue
        if vocab_size is not None and token_id >= int(vocab_size):
            continue
        cleaned.append(token_id)
    return cleaned


def _make_openai_chat_rollout_func(
    *,
    tokenizer: Any,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    lora_path: str | None,
):
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("OpenAI-compatible rollout requires requests") from exc

    filtered_lora_path = (
        _maybe_filter_sglang_lora_adapter(lora_path) if lora_path else None
    )

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        reward_texts: list[str] = []
        for idx, prompt in enumerate(prompts):
            prompt_text = _prompt_text_for_rollout(
                tokenizer,
                prompt,
                include_assistant_prefill=True,
            )
            messages = _messages_for_rollout(
                prompt,
                include_assistant_prefill=True,
            )
            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "stream": False,
                "continue_final_message": True,
                "separate_reasoning": False,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False,
                },
            }
            if filtered_lora_path:
                body["lora_path"] = filtered_lora_path
            seed = int(getattr(trainer.args, "seed", 0) or 0)
            body["seed"] = seed + int(getattr(trainer.state, "global_step", 0) or 0) + idx
            try:
                response = _post_openai_chat_completion(
                    requests_mod=requests,
                    base_url=base_url,
                    api_key=api_key,
                    body=body,
                    timeout_s=timeout_s,
                )
            except RuntimeError as exc:
                if (
                    filtered_lora_path
                    and "never been loaded" in str(exc)
                ):
                    _load_sglang_lora_adapter(
                        base_url=base_url,
                        lora_path=filtered_lora_path,
                        timeout_s=timeout_s,
                    )
                    response = _post_openai_chat_completion(
                        requests_mod=requests,
                        base_url=base_url,
                        api_key=api_key,
                        body=body,
                        timeout_s=timeout_s,
                    )
                else:
                    raise
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = str(message.get("content") or "")
            score_text = (
                text
                if text.startswith(ASSISTANT_CODE_PREFILL)
                else ASSISTANT_CODE_PREFILL + text
            )
            reward_texts.append(score_text)
            encoded_prompt = tokenizer(prompt_text, add_special_tokens=False)
            encoded_completion = tokenizer(text, add_special_tokens=False)
            max_prompt_length = int(
                getattr(trainer.args, "max_prompt_length", 0) or 0
            )
            ids_prompt = _truncate_token_ids(
                _sanitize_token_ids(
                    _extract_input_ids(encoded_prompt),
                    tokenizer=tokenizer,
                ),
                max_prompt_length,
            )
            prompt_ids.append(ids_prompt)
            ids = _truncate_token_ids(
                _sanitize_token_ids(
                    _extract_input_ids(encoded_completion),
                    tokenizer=tokenizer,
                ),
                int(max_tokens),
            )
            eos_token_id = int(getattr(tokenizer, "eos_token_id", 0) or 0)
            completion_ids.append(ids or [eos_token_id])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "rollout_completion_text": reward_texts,
            "logprobs": None,
        }

    return rollout_func


def _estimate_prompt_tokens(processor: Any, rows: list[dict[str, Any]]) -> int:
    if processor is None:
        return 0
    total = 0
    for row in rows:
        try:
            encoded = processor(str(row["prompt"]))
            input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
            if input_ids is not None:
                total += len(input_ids)
        except Exception:
            return 0
    return total


def _extract_input_ids(encoded: Any) -> list[int]:
    getter = getattr(encoded, "get", None)
    input_ids = getter("input_ids") if callable(getter) else None
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        return []
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def _prompt_token_lengths(tokenizer: Any, rows: list[dict[str, Any]]) -> list[int]:
    lengths: list[int] = []
    for row in rows:
        encoded = tokenizer(str(row["prompt"]), add_special_tokens=False)
        lengths.append(len(_extract_input_ids(encoded)))
    return lengths


def _truncate_prompt_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before = _prompt_token_lengths(tokenizer, rows)
    if max_prompt_length <= 0:
        return rows, {
            "enabled": False,
            "max_prompt_length": int(max_prompt_length),
            "max_before": max(before) if before else 0,
            "max_after": max(before) if before else 0,
        }

    truncated: list[dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer(str(row["prompt"]), add_special_tokens=False)
        input_ids = _extract_input_ids(encoded)
        if input_ids is None or len(input_ids) <= max_prompt_length:
            truncated.append(dict(row))
            continue
        kept = _truncate_token_ids(input_ids, max_prompt_length)
        next_row = dict(row)
        next_row["prompt"] = tokenizer.decode(kept, skip_special_tokens=False)
        truncated.append(next_row)
    after = _prompt_token_lengths(tokenizer, truncated)
    return truncated, {
        "enabled": any(a < b for a, b in zip(after, before, strict=False)),
        "max_prompt_length": int(max_prompt_length),
        "max_before": max(before) if before else 0,
        "max_after": max(after) if after else 0,
    }


def _load_text_tokenizer(auto_tokenizer: Any, model: str, *, trust_remote_code: bool) -> Any:
    tokenizer = auto_tokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_optional_path(path: str | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks-root", default="tasks")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--families", default=None)
    parser.add_argument("--tiers", default=None)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--steps-per-generation", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument(
        "--sanitize-generation-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask Transformers generation to remove invalid values and "
             "renormalize logits before sampling",
    )
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="comma-separated PEFT LoRA target modules",
    )
    parser.add_argument(
        "--init-adapter",
        default=None,
        help=(
            "optional PEFT adapter to load as the initial trainable policy "
            "before exact GRPO updates"
        ),
    )
    parser.add_argument(
        "--rollout-openai-base-url",
        default=None,
        help="optional OpenAI-compatible /v1/completions base URL for GRPO rollouts",
    )
    parser.add_argument("--rollout-openai-model", default=None)
    parser.add_argument("--rollout-openai-api-key", default="dummy")
    parser.add_argument("--rollout-openai-lora-path", default=None)
    parser.add_argument("--rollout-openai-timeout-s", type=float, default=240.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument(
        "--reward-channel",
        default="artifact_progress",
        choices=REWARD_CHANNELS,
        help=(
            "training reward channel. artifact_progress shapes syntax, "
            "DesignIR contract validity, evaluator validity, dense score, "
            "and strict verified score; final benchmark metrics remain strict"
        ),
    )
    parser.add_argument("--reward-timeout-s", type=float, default=60.0)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="load the base model with bitsandbytes 4-bit")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="load the base model with bitsandbytes 8-bit")
    parser.add_argument(
        "--skip-prepare-kbit-training",
        action="store_true",
        help="do not call PEFT prepare_model_for_kbit_training before loading "
             "--init-adapter",
    )
    parser.add_argument(
        "--kbit-prepare-mode",
        default="peft",
        choices=("peft", "lightweight", "none"),
        help=(
            "preparation path for quantized trainable adapter loading. "
            "`lightweight` avoids PEFT's fp32 upcast pass."
        ),
    )
    parser.add_argument("--torch-dtype", default=None,
                        choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default=None,
                        help="optional Transformers attention implementation")
    parser.add_argument("--device-map", default=None,
                        help="optional from_pretrained device_map, e.g. auto")
    parser.add_argument(
        "--max-memory",
        default=None,
        help=(
            "optional from_pretrained max_memory map, for example "
            "0:32GiB,1:44GiB or JSON"
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write dataset/config metadata without training")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_absolute():
        tasks_root = REPO_ROOT / tasks_root
    split_file = Path(args.split_file).resolve() if args.split_file else None
    families = (
        {item.strip() for item in args.families.split(",") if item.strip()}
        if args.families else None
    )
    tiers = (
        {item.strip() for item in args.tiers.split(",") if item.strip()}
        if args.tiers else None
    )
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    rows = _build_rows(
        tasks_root=tasks_root,
        split_file=split_file,
        families=families,
        tiers=tiers,
        system_prompt=system_prompt,
        limit=max(0, int(args.limit_tasks)),
    )
    if not rows:
        print("error: no tasks matched", file=sys.stderr)
        return 2
    source_task_count = len(rows)
    init_adapter = _resolve_optional_path(args.init_adapter)
    generation_batch_size = (
        max(1, int(args.per_device_train_batch_size))
        * int(args.steps_per_generation)
        if args.steps_per_generation is not None
        else int(args.num_generations)
    )
    effective_steps_per_generation = (
        int(args.steps_per_generation)
        if args.steps_per_generation is not None
        else max(
            1,
            generation_batch_size
            // max(1, int(args.per_device_train_batch_size)),
        )
    )
    expected_verifier_calls = (
        (
            int(args.max_steps)
            + effective_steps_per_generation
            - 1
        )
        // effective_steps_per_generation
        * generation_batch_size
    )
    rows, trainer_dataset_expansion = _repeat_rows_for_grpo_sampler(
        rows,
        generation_batch_size=generation_batch_size,
        num_generations=int(args.num_generations),
    )
    dataset_jsonl = out_dir / "train_prompts.jsonl"
    with dataset_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA,
        "argv": sys.argv,
        "model": args.model,
        "task_count": source_task_count,
        "trainer_prompt_count": len(rows),
        "dataset_jsonl": str(dataset_jsonl),
        "split_file": str(split_file) if split_file else None,
        "families": sorted(families) if families else None,
        "tiers": sorted(tiers) if tiers else None,
        "trainer_dataset_expansion": trainer_dataset_expansion,
        "algorithm": "trl.GRPOTrainer",
        "uses_policy_ratio_clipping": True,
        "uses_value_head": False,
        "reward": f"mech_bench {args.reward_channel} * reward_scale",
        "reward_channel": args.reward_channel,
        "reward_scale": float(args.reward_scale),
        "training_budget": {
            "max_steps": int(args.max_steps),
            "num_generations": int(args.num_generations),
            "generation_batch_size": generation_batch_size,
            "steps_per_generation": args.steps_per_generation,
            "effective_steps_per_generation": effective_steps_per_generation,
            "expected_verifier_calls": expected_verifier_calls,
        },
        "init_adapter": str(init_adapter) if init_adapter else None,
        "rollout_openai_base_url": args.rollout_openai_base_url,
        "rollout_openai_model": args.rollout_openai_model,
        "rollout_openai_lora_path": args.rollout_openai_lora_path,
        "model_init": {
            "load_in_4bit": bool(args.load_in_4bit),
            "load_in_8bit": bool(args.load_in_8bit),
            "skip_prepare_kbit_training": bool(
                args.skip_prepare_kbit_training
                or args.rollout_openai_base_url
                or args.kbit_prepare_mode == "none"
            ),
            "kbit_prepare_mode": args.kbit_prepare_mode,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "device_map": args.device_map,
            "trust_remote_code": bool(args.trust_remote_code),
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import LoraConfig  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from peft import prepare_model_for_kbit_training  # type: ignore[import-not-found]
        import peft.utils.save_and_load as peft_save_and_load  # type: ignore[import-not-found]
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        from transformers import TrainerCallback  # type: ignore[import-not-found]
        if args.load_in_4bit or args.load_in_8bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
        else:
            BitsAndBytesConfig = None  # type: ignore[assignment]
    except ImportError as exc:
        print(
            "error: exact GRPO requires the training-grpo extra. Run "
            "`uv sync --extra training-grpo` or invoke with "
            "`uv run --extra training-grpo ...`.\n"
            f"missing import: {exc}",
            file=sys.stderr,
        )
        return 2

    tokenizer = _load_text_tokenizer(
        AutoTokenizer,
        args.model,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if args.rollout_openai_base_url:
        rows = _chat_prompt_rows(rows)
        prompt_truncation = {
            "enabled": False,
            "max_prompt_length": int(args.max_prompt_length),
            "reason": "external_chat_rollout_preserves_messages",
        }
    else:
        rows = _apply_chat_template_rows(rows, tokenizer)
        rows, prompt_truncation = _truncate_prompt_rows(
            rows,
            tokenizer,
            max_prompt_length=int(args.max_prompt_length),
        )
    with dataset_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest["prompt_truncation"] = prompt_truncation
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    train_dataset = Dataset.from_list(rows)
    lora_target_modules = [
        item.strip()
        for item in str(args.lora_target_modules).split(",")
        if item.strip()
    ]
    peft_config = None
    if init_adapter is None:
        peft_config = LoraConfig(
            r=max(1, int(args.lora_rank)),
            lora_alpha=max(1, int(args.lora_alpha)),
            lora_dropout=max(0.0, float(args.lora_dropout)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
    model_init_kwargs: dict[str, Any] = {}
    if args.load_in_4bit or args.load_in_8bit:
        if BitsAndBytesConfig is None:
            print(
                "error: bitsandbytes quantization requested but "
                "BitsAndBytesConfig is unavailable",
                file=sys.stderr,
            )
            return 2
        compute_dtype = None
        if args.torch_dtype in ("bfloat16", "float16", "float32"):
            import torch
            compute_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[args.torch_dtype]
        quant_kwargs: dict[str, Any] = {
            "load_in_4bit": bool(args.load_in_4bit),
            "load_in_8bit": bool(args.load_in_8bit),
        }
        if args.load_in_4bit:
            quant_kwargs.update({
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            })
            if compute_dtype is not None:
                quant_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            **quant_kwargs
        )
    if args.torch_dtype:
        model_init_kwargs["torch_dtype"] = args.torch_dtype
    if args.attn_implementation:
        model_init_kwargs["attn_implementation"] = args.attn_implementation
    device_map = _parse_device_map(args.device_map)
    if device_map is not None:
        model_init_kwargs["device_map"] = device_map
    max_memory = _parse_max_memory(args.max_memory)
    if max_memory is not None:
        model_init_kwargs["max_memory"] = max_memory
    if args.trust_remote_code:
        model_init_kwargs["trust_remote_code"] = True

    generation_kwargs = None
    if args.sanitize_generation_logits:
        generation_kwargs = {
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }

    config = _filtered_config(GRPOConfig, {
        "output_dir": str(out_dir),
        "learning_rate": float(args.learning_rate),
        "max_grad_norm": float(args.max_grad_norm),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "num_generations": int(args.num_generations),
        "generation_batch_size": (
            None
            if args.steps_per_generation is not None
            else generation_batch_size
        ),
        "steps_per_generation": args.steps_per_generation,
        "max_prompt_length": int(args.max_prompt_length),
        "max_completion_length": int(args.max_completion_length),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": args.top_k,
        "min_p": args.min_p,
        "generation_kwargs": generation_kwargs,
        "beta": float(args.beta),
        "epsilon": float(args.epsilon),
        "max_steps": int(args.max_steps),
        **_disable_intermediate_checkpoint_config(int(args.save_steps)),
        "logging_steps": int(args.logging_steps),
        "seed": int(args.seed),
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "model_init_kwargs": None if init_adapter else (model_init_kwargs or None),
        "remove_unused_columns": False,
        "report_to": [],
    })
    model: str | Any = args.model
    if init_adapter is not None:
        if not init_adapter.exists():
            print(
                f"error: --init-adapter does not exist: {init_adapter}",
                file=sys.stderr,
            )
            return 2
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_init_kwargs,
        )
        should_prepare_adapter_base = not (
            args.skip_prepare_kbit_training
            or args.rollout_openai_base_url
            or args.kbit_prepare_mode == "none"
        ) and (
            args.load_in_4bit
            or args.load_in_8bit
            or args.kbit_prepare_mode == "lightweight"
        )
        if should_prepare_adapter_base:
            if args.kbit_prepare_mode == "peft":
                base = prepare_model_for_kbit_training(base)
            else:
                base = _prepare_model_for_kbit_training_lightweight(
                    base,
                    use_gradient_checkpointing=bool(args.gradient_checkpointing),
                )
        try:
            model = PeftModel.from_pretrained(
                base,
                str(init_adapter),
                is_trainable=True,
            )
        except TypeError as exc:
            if "distributed_operation" not in str(exc):
                raise
            previous = getattr(peft_save_and_load, "is_transformers_ge_v5", None)
            peft_save_and_load.is_transformers_ge_v5 = False
            try:
                model = PeftModel.from_pretrained(
                    base,
                    str(init_adapter),
                    is_trainable=True,
                )
            finally:
                if previous is not None:
                    peft_save_and_load.is_transformers_ge_v5 = previous
    scratch_root = Path(tempfile.mkdtemp(prefix="mech_true_grpo_"))
    reward_func = _make_reward_func(
        log_path=out_dir / "reward_log.jsonl",
        scratch_root=scratch_root,
        timeout_s=float(args.reward_timeout_s),
        reward_scale=float(args.reward_scale),
        reward_channel=str(args.reward_channel),
    )
    rollout_func = None
    if args.rollout_openai_base_url:
        rollout_func = _make_openai_chat_rollout_func(
            tokenizer=tokenizer,
            base_url=args.rollout_openai_base_url,
            model=args.rollout_openai_model or args.model,
            api_key=args.rollout_openai_api_key,
            max_tokens=int(args.max_completion_length),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            timeout_s=float(args.rollout_openai_timeout_s),
            lora_path=args.rollout_openai_lora_path,
        )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=rollout_func,
    )
    guarded_optimizer = _make_guarded_adamw(
        trainer.model,
        learning_rate=float(args.learning_rate),
    )
    trainer.optimizer = guarded_optimizer
    manifest["initial_trainable_finite_audit"] = (
        _raise_if_nonfinite_trainable_parameters(
            trainer.model,
            label="initial GRPO adapter",
        )
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )

    class FiniteTrainableParameterCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step_model = kwargs.get("model")
            if step_model is not None:
                _raise_if_nonfinite_trainable_parameters(
                    step_model,
                    label=f"GRPO adapter after step {getattr(state, 'global_step', '?')}",
                )
            return control

    trainer.add_callback(FiniteTrainableParameterCallback())
    trainer.train()
    manifest["final_trainable_finite_audit"] = (
        _raise_if_nonfinite_trainable_parameters(
            trainer.model,
            label="final GRPO adapter",
        )
    )
    final_adapter = out_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    removed_checkpoints = _remove_intermediate_checkpoints(out_dir)
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    n_rl_datums = 0
    reward_log = out_dir / "reward_log.jsonl"
    if reward_log.is_file():
        n_rl_datums = sum(1 for line in reward_log.read_text().splitlines()
                          if line.strip())
    trained_tokens = int(
        getattr(trainer.state, "num_input_tokens_seen", 0) or 0
    )
    if trained_tokens <= 0:
        processor = (
            getattr(trainer, "processing_class", None)
            or getattr(trainer, "tokenizer", None)
        )
        prompt_tokens = _estimate_prompt_tokens(processor, rows)
        trained_tokens = (
            prompt_tokens
            * max(global_step, 0)
            * max(int(args.num_generations), 1)
        )
    optimizer_guard = _guarded_optimizer_manifest(guarded_optimizer)
    adapter_updates = int(optimizer_guard["successful_steps"])
    manifest["completed_ts"] = time.time()
    manifest["final_adapter"] = str(final_adapter)
    manifest["checkpoint_policy"] = {
        "intermediate_checkpoints": "disabled",
        "save_strategy": "no",
        "removed_intermediate_checkpoints": removed_checkpoints,
    }
    manifest["adapter_updates"] = adapter_updates
    manifest["trainer_global_step"] = global_step
    manifest["trained_tokens"] = trained_tokens
    manifest["rl_trained_tokens"] = trained_tokens
    manifest["n_rl_datums"] = n_rl_datums
    manifest["optimizer_guard"] = optimizer_guard
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    sampler_manifest = {
        "ts": manifest["completed_ts"],
        "kind": "final_sampler",
        "name": final_adapter.name,
        "path": str(final_adapter),
        "step": global_step,
        "adapter_updates": adapter_updates,
        "trainer_global_step": global_step,
        "trained_tokens": trained_tokens,
        "rl_trained_tokens": trained_tokens,
        "n_rl_datums": n_rl_datums,
        "base_model": args.model,
        "init_adapter": str(init_adapter) if init_adapter else None,
        "rollout_openai_base_url": args.rollout_openai_base_url,
        "rollout_openai_model": args.rollout_openai_model,
        "rollout_openai_lora_path": args.rollout_openai_lora_path,
        "lora_rank": int(args.lora_rank),
        "lora_target_modules": lora_target_modules,
        "rollout_backend": "sglang_chat",
        "algorithm": "trl.GRPOTrainer",
        "uses_policy_ratio_clipping": True,
        "ttrl_exact_grpo": True,
    }
    (out_dir / "sampler_manifest.json").write_text(
        json.dumps(sampler_manifest, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
