#!/usr/bin/env python3
"""PEFT LoRA SFT trainer for mech_bench reference solutions.

This is the supervised baseline companion to ``train_true_grpo_trl.py`` for
family-held-out paper runs. It trains only on seen-family reference
``design.py`` solutions and exports a local PEFT adapter that can be evaluated
through the same SGLang LoRA path as exact GRPO adapters.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "rl"))

from rl.train_true_grpo_trl import (  # noqa: E402
    SYSTEM_PROMPT_PATH,
    _build_rows,
    _disable_intermediate_checkpoint_config,
    _estimate_prompt_tokens,
    _guarded_optimizer_manifest,
    _make_guarded_adamw,
    _parse_device_map,
    _remove_intermediate_checkpoints,
)


SCHEMA = "mech_bench.peft_sft.v1"


def _filtered_config(cls: type, values: dict[str, Any]) -> Any:
    params = inspect.signature(cls.__init__).parameters
    accepted = {
        key: value for key, value in values.items()
        if key in params and value is not None
    }
    return cls(**accepted)


def _completion_from_reference(task_dir: Path) -> str | None:
    path = task_dir / "reference_solution" / "design.py"
    if not path.is_file():
        return None
    return "```python\n" + path.read_text().rstrip() + "\n```"


def build_sft_rows(
    *,
    tasks_root: Path,
    split_file: Path | None,
    families: set[str] | None,
    tiers: set[str] | None,
    system_prompt: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = _build_rows(
        tasks_root=tasks_root,
        split_file=split_file,
        families=families,
        tiers=tiers,
        system_prompt=system_prompt,
        limit=0,
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        completion = _completion_from_reference(Path(row["task_dir"]))
        if completion is None:
            continue
        out.append({**row, "completion": completion})
        if limit > 0 and len(out) >= limit:
            break
    return out


def _tokenize_row(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    *,
    eos_token_id: int | None = None,
) -> dict[str, Any]:
    prompt = str(row["prompt"])
    completion = str(row["completion"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    completion_ids = list(completion_ids)
    if eos is not None:
        completion_ids.append(int(eos))

    max_length = max(1, int(max_length))
    if len(completion_ids) >= max_length:
        completion_budget = max(1, max_length // 2)
        completion_ids = completion_ids[:completion_budget]
        if eos is not None and int(eos) not in completion_ids:
            completion_ids[-1] = int(eos)

    prompt_budget = max(0, max_length - len(completion_ids))
    prompt_ids = list(prompt_ids)[-prompt_budget:] if prompt_budget else []
    input_ids = list(prompt_ids) + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _sanitize_tokenized_rows(
    tokenized: list[dict[str, Any]],
    *,
    input_vocab_size: int,
    output_vocab_size: int | None,
    fallback_token_id: int,
) -> dict[str, int]:
    audit = {
        "input_ids_replaced": 0,
        "labels_masked": 0,
        "rows_with_replacements": 0,
    }
    max_label_id = input_vocab_size if output_vocab_size is None else output_vocab_size
    fallback = int(fallback_token_id)
    if fallback < 0 or fallback >= input_vocab_size:
        fallback = 0
    for row in tokenized:
        row_changed = False
        clean_input_ids: list[int] = []
        for token_id in row["input_ids"]:
            token_id = int(token_id)
            if token_id < 0 or token_id >= input_vocab_size:
                clean_input_ids.append(fallback)
                audit["input_ids_replaced"] += 1
                row_changed = True
            else:
                clean_input_ids.append(token_id)
        clean_labels: list[int] = []
        for token_id in row["labels"]:
            token_id = int(token_id)
            if token_id == -100:
                clean_labels.append(token_id)
            elif token_id < 0 or token_id >= max_label_id:
                clean_labels.append(-100)
                audit["labels_masked"] += 1
                row_changed = True
            else:
                clean_labels.append(token_id)
        row["input_ids"] = clean_input_ids
        row["labels"] = clean_labels
        if row_changed:
            audit["rows_with_replacements"] += 1
    return audit


def _count_invalid_supervised_labels(labels: Any, vocab_size: int) -> int:
    supervised = labels != -100
    if not bool(supervised.any()):
        return 0
    invalid = supervised & ((labels < 0) | (labels >= int(vocab_size)))
    return int(invalid.sum().item())


def _mask_invalid_supervised_labels(labels: Any, vocab_size: int) -> Any:
    supervised = labels != -100
    if not bool(supervised.any()):
        return labels
    invalid = supervised & ((labels < 0) | (labels >= int(vocab_size)))
    if not bool(invalid.any()):
        return labels
    labels = labels.clone()
    labels[invalid] = -100
    return labels


def _causal_lm_loss_from_logits(
    logits: Any,
    labels: Any,
    *,
    invalid_label_policy: str = "mask",
    sanitize_nonfinite_logits: bool = True,
) -> Any:
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3:
        raise RuntimeError(f"expected 3D causal-LM logits, got shape {tuple(logits.shape)}")
    finite_logits = torch.isfinite(logits)
    if not bool(finite_logits.all()):
        if not sanitize_nonfinite_logits:
            raise RuntimeError("SFT logits contain non-finite values")
        logits = logits.masked_fill(~finite_logits, 0.0)
    vocab_size = int(logits.shape[-1])
    labels = labels.to(logits.device)
    invalid_count = _count_invalid_supervised_labels(labels, vocab_size)
    if invalid_count:
        supervised_labels = labels[labels != -100]
        min_label = int(supervised_labels.min().item())
        max_label = int(supervised_labels.max().item())
        if invalid_label_policy == "raise":
            raise RuntimeError(
                "SFT labels are outside the model logits vocabulary: "
                f"min_label={min_label} max_label={max_label} "
                f"logits_vocab_size={vocab_size}"
            )
        labels = _mask_invalid_supervised_labels(labels, vocab_size)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if not bool((shift_labels != -100).any()):
        return shift_logits.sum() * 0.0
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("SFT loss is non-finite")
    return loss


def _count_nonfinite_values(tensor: Any) -> int:
    import torch

    return int((~torch.isfinite(tensor)).sum().item())


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
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="comma-separated PEFT LoRA target modules",
    )
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--prepare-kbit-training",
        action="store_true",
        help="run PEFT prepare_model_for_kbit_training before adding LoRA",
    )
    parser.add_argument(
        "--prepare-kbit-training-mode",
        default="peft",
        choices=("peft", "lightweight"),
        help=(
            "k-bit preparation implementation. `peft` uses PEFT's full helper; "
            "`lightweight` freezes the base and enables input gradients without "
            "upcasting model weights to fp32."
        ),
    )
    parser.add_argument("--torch-dtype", default=None,
                        choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    rows = build_sft_rows(
        tasks_root=tasks_root,
        split_file=split_file,
        families=families,
        tiers=tiers,
        system_prompt=SYSTEM_PROMPT_PATH.read_text(),
        limit=max(0, int(args.limit_tasks)),
    )
    if not rows:
        print("error: no reference_solution/design.py rows matched", file=sys.stderr)
        return 2

    dataset_jsonl = out_dir / "train_sft.jsonl"
    with dataset_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": SCHEMA,
        "argv": sys.argv,
        "model": args.model,
        "task_count": len(rows),
        "dataset_jsonl": str(dataset_jsonl),
        "split_file": str(split_file) if split_file else None,
        "families": sorted(families) if families else None,
        "tiers": sorted(tiers) if tiers else None,
        "algorithm": "transformers.Trainer.peft_sft",
        "reward": "none_supervised_reference_solution",
        "model_init": {
            "load_in_4bit": bool(args.load_in_4bit),
            "load_in_8bit": bool(args.load_in_8bit),
            "prepare_kbit_training": bool(args.prepare_kbit_training),
            "prepare_kbit_training_mode": args.prepare_kbit_training_mode,
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
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
        if args.load_in_4bit or args.load_in_8bit:
            from peft import prepare_model_for_kbit_training  # type: ignore[import-not-found]
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
        else:
            prepare_model_for_kbit_training = None  # type: ignore[assignment]
            BitsAndBytesConfig = None  # type: ignore[assignment]
    except ImportError as exc:
        print(
            "error: PEFT SFT requires the training-grpo extra. Run "
            "`uv sync --extra training-grpo` or invoke with "
            "`uv run --extra training-grpo ...`.\n"
            f"missing import: {exc}",
            file=sys.stderr,
        )
        return 2

    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if args.load_in_4bit or args.load_in_8bit:
        if args.load_in_4bit and args.load_in_8bit:
            raise SystemExit("choose at most one of --load-in-4bit or --load-in-8bit")
        quant_kwargs: dict[str, Any] = {
            "load_in_4bit": bool(args.load_in_4bit),
            "load_in_8bit": bool(args.load_in_8bit),
        }
        if args.load_in_4bit:
            quant_kwargs.update({
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            })
            if args.torch_dtype in ("bfloat16", "float16", "float32"):
                quant_kwargs["bnb_4bit_compute_dtype"] = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }[args.torch_dtype]
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)
    if args.torch_dtype:
        model_kwargs["torch_dtype"] = args.torch_dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    device_map = _parse_device_map(args.device_map)
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    text_config = getattr(model.config, "text_config", model.config)
    model_eos_token_id = getattr(text_config, "eos_token_id", None)
    if isinstance(model_eos_token_id, (list, tuple)):
        model_eos_token_id = model_eos_token_id[0] if model_eos_token_id else None
    input_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    output_embeddings = model.get_output_embeddings()
    output_vocab_size = (
        int(output_embeddings.weight.shape[0])
        if output_embeddings is not None and hasattr(output_embeddings, "weight")
        else None
    )
    fallback_token_id = (
        int(model_eos_token_id)
        if model_eos_token_id is not None
        else int(tokenizer.pad_token_id)
    )
    if (
        prepare_model_for_kbit_training is not None
        and bool(args.prepare_kbit_training)
    ):
        if args.prepare_kbit_training_mode == "peft":
            model = prepare_model_for_kbit_training(model)
        else:
            model = _prepare_model_for_kbit_training_lightweight(
                model,
                use_gradient_checkpointing=bool(args.gradient_checkpointing),
            )
    peft_config = LoraConfig(
        r=max(1, int(args.lora_rank)),
        lora_alpha=max(1, int(args.lora_alpha)),
        lora_dropout=max(0.0, float(args.lora_dropout)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            item.strip()
            for item in str(args.lora_target_modules).split(",")
            if item.strip()
        ],
    )
    model = get_peft_model(model, peft_config)
    manifest["initial_trainable_finite_audit"] = (
        _raise_if_nonfinite_trainable_parameters(
            model,
            label="initial SFT adapter",
        )
    )
    tokenized = [
        _tokenize_row(
            tokenizer,
            row,
            max_length=int(args.max_seq_length),
            eos_token_id=(
                int(model_eos_token_id)
                if model_eos_token_id is not None
                else None
            ),
        )
        for row in rows
    ]
    token_audit = _sanitize_tokenized_rows(
        tokenized,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        fallback_token_id=fallback_token_id,
    )
    manifest["token_audit"] = {
        **token_audit,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "model_eos_token_id": model_eos_token_id,
        "input_vocab_size": input_vocab_size,
        "output_vocab_size": output_vocab_size,
        "fallback_token_id": fallback_token_id,
        "max_input_id": max(max(row["input_ids"]) for row in tokenized),
        "max_label_id": max(
            max((item for item in row["labels"] if item != -100), default=-100)
            for row in tokenized
        ),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    train_dataset = Dataset.from_list(tokenized)

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = int(tokenizer.pad_token_id)
        out: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in batch:
            pad = max_len - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append(item["attention_mask"] + [0] * pad)
            out["labels"].append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                out["attention_mask"],
                dtype=torch.long,
            ),
            "labels": torch.tensor(out["labels"], dtype=torch.long),
        }

    training_args = _filtered_config(TrainingArguments, {
        "output_dir": str(out_dir),
        "learning_rate": float(args.learning_rate),
        "max_grad_norm": float(args.max_grad_norm),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        **_disable_intermediate_checkpoint_config(int(args.save_steps)),
        "logging_steps": int(args.logging_steps),
        "seed": int(args.seed),
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "remove_unused_columns": False,
        "logging_nan_inf_filter": False,
        "report_to": [],
    })

    class FiniteTrainableParameterCallback(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            step_model = kwargs.get("model")
            if step_model is not None:
                _raise_if_nonfinite_trainable_parameters(
                    step_model,
                    label=f"SFT adapter after step {getattr(state, 'global_step', '?')}",
                )
            return control

    class CausalLMTrainer(Trainer):
        runtime_labels_masked = 0
        runtime_nonfinite_logits_sanitized = 0

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            self.runtime_labels_masked += _count_invalid_supervised_labels(
                labels.to(outputs.logits.device),
                int(outputs.logits.shape[-1]),
            )
            self.runtime_nonfinite_logits_sanitized += _count_nonfinite_values(
                outputs.logits
            )
            loss = _causal_lm_loss_from_logits(outputs.logits, labels)
            if return_outputs:
                return loss, outputs
            return loss

    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate,
        callbacks=[FiniteTrainableParameterCallback()],
    )
    guarded_optimizer = _make_guarded_adamw(
        trainer.model,
        learning_rate=float(args.learning_rate),
    )
    trainer.optimizer = guarded_optimizer
    trainer.train()
    manifest["final_trainable_finite_audit"] = (
        _raise_if_nonfinite_trainable_parameters(
            model,
            label="final SFT adapter",
        )
    )
    final_adapter = out_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    removed_checkpoints = _remove_intermediate_checkpoints(out_dir)
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    trained_tokens = int(
        getattr(trainer.state, "num_input_tokens_seen", 0) or 0
    )
    if trained_tokens <= 0:
        trained_tokens = sum(
            len(row["input_ids"]) for row in tokenized
        ) * max(global_step, 0)
    if trained_tokens <= 0:
        trained_tokens = _estimate_prompt_tokens(tokenizer, rows) * max(
            global_step,
            0,
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
    manifest["rl_trained_tokens"] = 0
    manifest["n_rl_datums"] = 0
    manifest["optimizer_guard"] = optimizer_guard
    manifest["runtime_labels_masked"] = int(
        getattr(trainer, "runtime_labels_masked", 0) or 0
    )
    manifest["runtime_nonfinite_logits_sanitized"] = int(
        getattr(trainer, "runtime_nonfinite_logits_sanitized", 0) or 0
    )
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
        "rl_trained_tokens": 0,
        "n_rl_datums": 0,
        "base_model": args.model,
        "lora_rank": int(args.lora_rank),
        "lora_target_modules": [
            item.strip()
            for item in str(args.lora_target_modules).split(",")
            if item.strip()
        ],
        "rollout_backend": "sglang_chat",
        "algorithm": "transformers.Trainer.peft_sft",
    }
    (out_dir / "sampler_manifest.json").write_text(
        json.dumps(sampler_manifest, indent=2, sort_keys=True, default=str)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
