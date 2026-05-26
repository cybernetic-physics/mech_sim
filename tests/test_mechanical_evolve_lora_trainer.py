from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_trainer():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "train_mechanical_evolve_lora.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_mechanical_evolve_lora", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lora_config_can_resume_previous_adapter(tmp_path):
    mod = _load_trainer()
    resume = tmp_path / "round_000" / "adapters" / "adapters.safetensors"
    resume.parent.mkdir(parents=True)
    resume.write_bytes(b"adapter")

    config = mod.mlx_lora_config(
        model="mlx-community/Qwen3-32B-4bit",
        data_dir=tmp_path / "data",
        adapter_path=tmp_path / "round_001" / "adapters",
        resume_adapter_file=resume,
        iters=4,
        batch_size=1,
        grad_accumulation_steps=1,
        learning_rate=1.0e-5,
        num_layers=8,
        lora_rank=8,
        lora_scale=20.0,
        lora_dropout=0.0,
        max_seq_length=768,
        seed=20260525,
        grad_checkpoint=True,
    )

    assert config["resume_adapter_file"] == str(resume)
    assert config["adapter_path"].endswith("round_001/adapters")
