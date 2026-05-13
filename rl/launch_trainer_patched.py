#!/usr/bin/env python3
"""Worldlines backend entrypoint with an in-process monkey-patch.

The PEFT LoRA trainer in worldlines 0.16.1 has a tensor-device hygiene
bug in ``_compute_token_logprobs``: after PEFT wraps the model, the
chain ``target_tokens.to(logits.device).long().unsqueeze(-1)`` can
return a CPU index tensor while logits are on cuda:0, and
``torch.gather`` insists on tensor-identity-equal devices. Splitting
the chain into ``.long().to(device).unsqueeze(-1)`` fixes it.

We patch in-process before delegating to
``scripts/launch_trainer.py`` so the worldlines repo itself stays
unmodified.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _apply_patch() -> None:
    from worldlines_backend.runtimes import peft_lora_trainer as plt

    # --- Patch 1: tensor-device hygiene in token-logprob gather. ---
    @staticmethod
    def _patched_logprobs(model, input_ids, target_tokens):
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[0].float()
        index = target_tokens.long().to(device=logits.device).unsqueeze(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, index).squeeze(-1)

    plt.PeftLoRATrainer._compute_token_logprobs = _patched_logprobs

    # --- Patch 2: force the model fully onto one CUDA device after
    # PEFT wraps it. device_map="cuda:0" in load_kwargs DOES place the
    # base on cuda:0, but PEFT's `get_peft_model` sometimes returns a
    # wrapped model whose embedding weights remain on CPU. An explicit
    # `.to(device)` after wrapping fixes the index_select(cpu, cuda)
    # crash inside the embedding lookup. ---
    original_load = plt.PeftLoRATrainer._load_model_and_adapter

    def _patched_load(self, *args, **kwargs):
        model, tokenizer, primary_device, target_modules = (
            original_load(self, *args, **kwargs)
        )
        # _infer_primary_device returns cpu when next(model.parameters())
        # is on cpu — which happens with PEFT-wrapped models where the
        # adapter (newly-allocated) is on cpu even though the base is
        # on cuda. Override: prefer CUDA when available.
        if torch.cuda.is_available():
            target = torch.device("cuda:0")
        else:
            target = primary_device
        try:
            sys.stderr.write(
                f"[launch_trainer_patched] forcing PEFT model to "
                f"{target} (was primary={primary_device})\n"
            )
            model = model.to(target)
            primary_device = target
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(
                f"[launch_trainer_patched] model.to() failed: {e}\n"
            )
        # Verify all parameters are now on the target device.
        bad = []
        for n, p in model.named_parameters():
            if p.device != target:
                bad.append((n, str(p.device)))
                if len(bad) >= 3:
                    break
        if bad:
            sys.stderr.write(
                f"[launch_trainer_patched] WARNING: {len(bad)}+ "
                f"params not on {target}: {bad[:3]}\n"
            )
        else:
            sys.stderr.write(
                f"[launch_trainer_patched] all parameters on {target}\n"
            )
        return model, tokenizer, primary_device, target_modules

    plt.PeftLoRATrainer._load_model_and_adapter = _patched_load

    sys.stderr.write(
        "[launch_trainer_patched] monkey-patched "
        "PeftLoRATrainer._compute_token_logprobs + "
        "_load_model_and_adapter\n"
    )


def main() -> None:
    _apply_patch()
    # Delegate to the worldlines launch script's main().
    wld_root = os.environ.get(
        "WORLDLINES_ROOT", "/home/freiza/worldlines"
    )
    scripts_dir = Path(wld_root) / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import launch_trainer  # noqa: E402  pyright: ignore
    launch_trainer.main()


if __name__ == "__main__":
    main()
