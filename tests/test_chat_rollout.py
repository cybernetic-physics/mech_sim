from __future__ import annotations

import json
from typing import Any

import torch
from safetensors.torch import save_file

from rl import chat_rollout


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        *,
        text: str = "bad request",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeRequests:
    calls: list[dict[str, Any]] = []

    @classmethod
    def post(
        cls,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
        headers: dict[str, str],
    ) -> FakeResponse:
        cls.calls.append(json)
        if len(cls.calls) == 1:
            return FakeResponse(400)
        return FakeResponse(200, {"ok": True})


def test_chat_completion_retries_without_seed_on_bad_request(monkeypatch) -> None:
    FakeRequests.calls = []
    monkeypatch.setattr(chat_rollout, "requests", FakeRequests)

    result = chat_rollout._chat_completion(
        base_url="http://localhost:30000",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        timeout_s=1.0,
        seed=123,
    )

    assert result == {"ok": True}
    assert FakeRequests.calls[0]["seed"] == 123
    assert "seed" not in FakeRequests.calls[1]


class FakeLoRARequests:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def post(
        cls,
        url: str,
        *,
        json: dict[str, Any],
        timeout: float,
        headers: dict[str, str],
    ) -> FakeResponse:
        cls.calls.append((url, json))
        if url.endswith("/load_lora_adapter"):
            return FakeResponse(200, {"success": True})
        if len([u for u, _ in cls.calls if u.endswith("/v1/chat/completions")]) == 1:
            return FakeResponse(
                400,
                {"object": "error"},
                text="Got LoRA adapter that has never been loaded",
            )
        return FakeResponse(200, {"ok": True})


def test_chat_completion_loads_unloaded_lora_then_retries(monkeypatch) -> None:
    FakeLoRARequests.calls = []
    chat_rollout._LOADED_LORA_ADAPTERS.clear()
    monkeypatch.setattr(chat_rollout, "requests", FakeLoRARequests)

    result = chat_rollout._chat_completion(
        base_url="http://localhost:30000",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        timeout_s=1.0,
        lora_path="/tmp/adapter",
    )

    assert result == {"ok": True}
    assert [url.rsplit("/", 1)[-1] for url, _ in FakeLoRARequests.calls] == [
        "completions",
        "load_lora_adapter",
        "completions",
    ]
    assert FakeLoRARequests.calls[1][1] == {
        "lora_name": "/tmp/adapter",
        "lora_path": "/tmp/adapter",
        "pinned": False,
    }


def test_lora_load_is_cached(monkeypatch) -> None:
    FakeLoRARequests.calls = []
    chat_rollout._LOADED_LORA_ADAPTERS.clear()
    monkeypatch.setattr(chat_rollout, "requests", FakeLoRARequests)

    chat_rollout._load_sglang_lora_adapter(
        base_url="http://localhost:30000",
        lora_path="/tmp/adapter",
        timeout_s=1.0,
    )
    chat_rollout._load_sglang_lora_adapter(
        base_url="http://localhost:30000",
        lora_path="/tmp/adapter",
        timeout_s=1.0,
    )

    assert [url.rsplit("/", 1)[-1] for url, _ in FakeLoRARequests.calls] == [
        "load_lora_adapter",
    ]


def test_lora_filter_removes_q_proj_for_sglang(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }))
    save_file({
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 3),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        "base_model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.ones(2, 3),
        "base_model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.ones(1, 2),
        "base_model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.ones(2, 3),
        "base_model.model.layers.0.self_attn.o_proj.lora_B.weight": torch.ones(1, 2),
    }, str(adapter / "adapter_model.safetensors"))

    chat_rollout._FILTERED_LORA_ADAPTERS.clear()
    filtered = chat_rollout._maybe_filter_sglang_lora_adapter(str(adapter))

    assert filtered != str(adapter)
    filtered_dir = tmp_path / "adapter_sglang_o"
    assert filtered == str(filtered_dir)
    config = json.loads((filtered_dir / "adapter_config.json").read_text())
    assert config["target_modules"] == ["o_proj"]

    from safetensors.torch import safe_open

    with safe_open(
        filtered_dir / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        keys = list(handle.keys())
    assert all(".q_proj." not in key for key in keys)
    assert all(".k_proj." not in key for key in keys)
    assert all(".v_proj." not in key for key in keys)
    assert any(".o_proj." in key for key in keys)
