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


class FakeOkRequests:
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
        return FakeResponse(200, {"ok": True})


class FakeOptionalChatRequests:
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
        optional = {
            "continue_final_message",
            "separate_reasoning",
            "chat_template_kwargs",
        }
        if optional.intersection(json):
            return FakeResponse(400, text="unsupported chat template options")
        return FakeResponse(200, {"ok": True})


class FakeTransientBadRequest:
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
            return FakeResponse(400, text="transient bad request")
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

    assert result["ok"] is True
    assert result["_sglang_retry_stats"]["sampler_http_400_count"] == 1
    assert result["_sglang_retry_stats"]["sampler_retry_count"] == 1
    assert result["_sglang_retry_stats"]["seed_retry_count"] == 1
    assert FakeRequests.calls[0]["seed"] == 123
    assert "seed" not in FakeRequests.calls[1]


def test_chat_completion_retries_without_optional_sglang_chat_fields(
    monkeypatch,
) -> None:
    FakeOptionalChatRequests.calls = []
    monkeypatch.setattr(chat_rollout, "requests", FakeOptionalChatRequests)

    result = chat_rollout._chat_completion(
        base_url="http://localhost:30000",
        model="model",
        messages=[{"role": "assistant", "content": "```python\n"}],
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        timeout_s=1.0,
        seed=123,
        continue_final_message=True,
        extra_body={
            "separate_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert result["ok"] is True
    assert result["_sglang_retry_stats"]["sampler_http_400_count"] == 2
    assert result["_sglang_retry_stats"]["sampler_retry_count"] == 2
    assert result["_sglang_retry_stats"]["seed_retry_count"] == 1
    assert result["_sglang_retry_stats"]["optional_chat_field_retry_count"] == 1
    assert FakeOptionalChatRequests.calls[0]["seed"] == 123
    assert "seed" not in FakeOptionalChatRequests.calls[1]
    assert "continue_final_message" not in FakeOptionalChatRequests.calls[2]
    assert "separate_reasoning" not in FakeOptionalChatRequests.calls[2]
    assert "chat_template_kwargs" not in FakeOptionalChatRequests.calls[2]


def test_chat_completion_retries_transient_bad_request(monkeypatch) -> None:
    FakeTransientBadRequest.calls = []
    monkeypatch.setattr(chat_rollout, "requests", FakeTransientBadRequest)
    monkeypatch.setattr(chat_rollout.time, "sleep", lambda _: None)

    result = chat_rollout._chat_completion(
        base_url="http://localhost:30000",
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        timeout_s=1.0,
    )

    assert result["ok"] is True
    assert result["_sglang_retry_stats"]["sampler_http_400_count"] == 1
    assert result["_sglang_retry_stats"]["sampler_retry_count"] == 1
    assert result["_sglang_retry_stats"]["transient_bad_request_retry_count"] == 1
    assert len(FakeTransientBadRequest.calls) == 2


def test_chat_completion_forwards_continue_final_message(monkeypatch) -> None:
    FakeOkRequests.calls = []
    monkeypatch.setattr(chat_rollout, "requests", FakeOkRequests)

    chat_rollout._chat_completion(
        base_url="http://localhost:30000",
        model="model",
        messages=[{"role": "assistant", "content": "```python\n"}],
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        timeout_s=1.0,
        continue_final_message=True,
        extra_body={"separate_reasoning": False},
    )

    assert FakeOkRequests.calls[0]["continue_final_message"] is True
    assert FakeOkRequests.calls[0]["separate_reasoning"] is False


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

    assert result["ok"] is True
    assert result["_sglang_retry_stats"]["sampler_http_400_count"] == 1
    assert result["_sglang_retry_stats"]["sampler_retry_count"] == 1
    assert result["_sglang_retry_stats"]["lora_load_retry_count"] == 1
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


def test_verifier_feedback_includes_soft_physics_hint_for_submaximal_pass() -> None:
    turn = chat_rollout.TurnTrace(
        turn_idx=0,
        assistant_text="```python\npass\n```",
        score=50.0,
        dense_pct=50.0,
        passed=True,
        parsed_ok=True,
        failure_codes=[],
        completion_tokens=12,
        stop_reason="stop",
        physical_metrics={
            "max_penetration_mm": 0.025,
            "contact_force_rms_N": 16836.5,
            "out_omega_med": 50.3677,
        },
    )

    feedback = chat_rollout._format_verifier_feedback(turn)

    assert "soft_objective" in feedback
    assert "dense physical reward is only 50.0/100" in feedback
    assert "max_penetration_mm=0.025" in feedback
    assert "contact_force_rms_N=16836.5" in feedback
    assert "reduce penetration" in feedback
