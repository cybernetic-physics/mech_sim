from __future__ import annotations

import pytest

from rl.train_true_grpo_trl import (
    _estimate_prompt_tokens,
    _load_text_tokenizer,
    _parse_max_memory,
    _truncate_prompt_rows,
)


class FakeProcessor:
    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"
    pad_token = None


class FakeAutoTokenizer:
    calls: list[tuple[str, bool]] = []

    @classmethod
    def from_pretrained(cls, model: str, *, trust_remote_code: bool) -> FakeTokenizer:
        cls.calls.append((model, trust_remote_code))
        return FakeTokenizer()


class WhitespaceTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}

    def decode(self, input_ids: list[str], *, skip_special_tokens: bool = False) -> str:
        return " ".join(input_ids)


class DictLikeEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids

    def get(self, key: str) -> list[int] | None:
        if key == "input_ids":
            return self.input_ids
        return None


class DictLikeTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> DictLikeEncoding:
        return DictLikeEncoding(list(range(len(text.split()))))

    def decode(self, input_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return " ".join(f"tok{i}" for i in input_ids)


def test_estimate_prompt_tokens_counts_processor_input_ids() -> None:
    rows = [
        {"prompt": "one two three"},
        {"prompt": "four five"},
    ]

    assert _estimate_prompt_tokens(FakeProcessor(), rows) == 5


def test_estimate_prompt_tokens_returns_zero_without_processor() -> None:
    assert _estimate_prompt_tokens(None, [{"prompt": "one two"}]) == 0


def test_load_text_tokenizer_sets_pad_without_auto_processor() -> None:
    FakeAutoTokenizer.calls = []

    tokenizer = _load_text_tokenizer(
        FakeAutoTokenizer,
        "Qwen/Qwen3.6-35B-A3B",
        trust_remote_code=True,
    )

    assert FakeAutoTokenizer.calls == [("Qwen/Qwen3.6-35B-A3B", True)]
    assert tokenizer.pad_token == "<eos>"


def test_parse_max_memory_accepts_comma_map_and_json() -> None:
    assert _parse_max_memory("0:32GiB,1:44GiB,cpu:128GiB") == {
        0: "32GiB",
        1: "44GiB",
        "cpu": "128GiB",
    }
    assert _parse_max_memory('{"0": "30GiB", "1": "45GiB"}') == {
        0: "30GiB",
        1: "45GiB",
    }


def test_parse_max_memory_rejects_malformed_entry() -> None:
    with pytest.raises(ValueError):
        _parse_max_memory("0:32GiB,missing-memory")


def test_truncate_prompt_rows_keeps_prompt_tail() -> None:
    rows = [{"prompt": "zero one two three", "task_id": "t1"}]

    truncated, audit = _truncate_prompt_rows(
        rows,
        WhitespaceTokenizer(),
        max_prompt_length=2,
    )

    assert truncated == [{"prompt": "two three", "task_id": "t1"}]
    assert audit == {
        "enabled": True,
        "max_prompt_length": 2,
        "max_before": 4,
        "max_after": 2,
    }


def test_truncate_prompt_rows_accepts_batchencoding_like_output() -> None:
    rows = [{"prompt": "zero one two three", "task_id": "t1"}]

    truncated, audit = _truncate_prompt_rows(
        rows,
        DictLikeTokenizer(),
        max_prompt_length=2,
    )

    assert truncated == [{"prompt": "tok2 tok3", "task_id": "t1"}]
    assert audit["enabled"] is True
    assert audit["max_before"] == 4
