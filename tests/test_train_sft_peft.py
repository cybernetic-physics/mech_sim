from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl.train_sft_peft import (
    _causal_lm_loss_from_logits,
    _sanitize_tokenized_rows,
    _tokenize_row,
    build_sft_rows,
)


def test_build_sft_rows_uses_reference_solution(tmp_path: Path) -> None:
    task = tmp_path / "fourbar_path_t001"
    ref = task / "reference_solution"
    ref.mkdir(parents=True)
    (task / "prompt.md").write_text("Make a fourbar.")
    (task / "task.toml").write_text(
        '[task]\nfamily = "fourbar_path"\ntier = "transmission_analytic"\n'
        '[requirements]\nexpected_mobility = 1\n'
    )
    (ref / "design.py").write_text("def build():\n    return {}\n")
    split = tmp_path / "train.txt"
    split.write_text("fourbar_path_t001\n")

    rows = build_sft_rows(
        tasks_root=tmp_path,
        split_file=split,
        families=None,
        tiers=None,
        system_prompt="system",
        limit=0,
    )

    assert len(rows) == 1
    assert rows[0]["task_id"] == "fourbar_path_t001"
    assert rows[0]["completion"].startswith("```python\n")
    assert "def build()" in rows[0]["completion"]


def test_build_sft_rows_skips_tasks_without_reference_solution(tmp_path: Path) -> None:
    task = tmp_path / "fourbar_path_t001"
    task.mkdir()
    (task / "prompt.md").write_text("Make a fourbar.")
    (task / "task.toml").write_text(
        '[task]\nfamily = "fourbar_path"\ntier = "transmission_analytic"\n'
    )

    assert build_sft_rows(
        tasks_root=tmp_path,
        split_file=None,
        families=None,
        tiers=None,
        system_prompt="system",
        limit=0,
    ) == []


class _TinyTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [int(item) for item in text.split()]}


def test_tokenize_row_trims_prompt_before_completion() -> None:
    row = {
        "prompt": "1 2 3 4 5 6 7 8",
        "completion": "20 21 22",
    }

    tokenized = _tokenize_row(_TinyTokenizer(), row, max_length=6)

    assert tokenized["input_ids"] == [7, 8, 20, 21, 22, 99]
    assert tokenized["labels"] == [-100, -100, 20, 21, 22, 99]


def test_tokenize_row_keeps_eos_when_completion_is_capped() -> None:
    row = {
        "prompt": "1 2 3",
        "completion": "20 21 22 23 24",
    }

    tokenized = _tokenize_row(_TinyTokenizer(), row, max_length=4)

    assert tokenized["input_ids"] == [2, 3, 20, 99]
    assert tokenized["labels"] == [-100, -100, 20, 99]


def test_tokenize_row_accepts_model_eos_override() -> None:
    row = {
        "prompt": "1 2",
        "completion": "20",
    }

    tokenized = _tokenize_row(_TinyTokenizer(), row, max_length=4, eos_token_id=42)

    assert tokenized["input_ids"] == [1, 2, 20, 42]
    assert tokenized["labels"] == [-100, -100, 20, 42]


def test_sanitize_tokenized_rows_masks_invalid_ids() -> None:
    rows = [{
        "input_ids": [1, 2, 99],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 2, 99],
    }]

    audit = _sanitize_tokenized_rows(
        rows,
        input_vocab_size=10,
        output_vocab_size=8,
        fallback_token_id=3,
    )

    assert rows[0]["input_ids"] == [1, 2, 3]
    assert rows[0]["labels"] == [-100, 2, -100]
    assert audit == {
        "input_ids_replaced": 1,
        "labels_masked": 1,
        "rows_with_replacements": 1,
    }


def test_causal_lm_loss_uses_runtime_logits_vocab() -> None:
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    labels = torch.tensor([[-100, 3, 4]])

    loss = _causal_lm_loss_from_logits(logits, labels)

    assert torch.isfinite(loss)


def test_causal_lm_loss_rejects_labels_outside_logits_vocab() -> None:
    logits = torch.zeros((1, 3, 5), dtype=torch.float32)
    labels = torch.tensor([[-100, 3, 5]])

    with pytest.raises(RuntimeError, match="outside the model logits vocabulary"):
        _causal_lm_loss_from_logits(logits, labels)
