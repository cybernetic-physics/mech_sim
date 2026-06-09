from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl.mech_env import TaskInfo
from rl.mech_bench_reward import RewardResult
from rl.train_true_grpo_trl import (
    STRICT_FENCED_OUTPUT_INSTRUCTION,
    _chat_prompt_rows,
    _cast_adapter_safetensors,
    _disable_intermediate_checkpoint_config,
    _estimate_prompt_tokens,
    _finite_named_tensor_audit,
    _filtered_config,
    _full_eval_contract_suffix,
    _guarded_optimizer_manifest,
    _load_text_tokenizer,
    _make_guarded_adamw,
    _messages_for_rollout,
    _parse_max_memory,
    _post_openai_chat_completion,
    _prepare_model_for_kbit_training_lightweight,
    _prompt_text_for_rollout,
    _raise_if_nonfinite_trainable_parameters,
    _remove_intermediate_checkpoints,
    _repeat_rows_for_grpo_sampler,
    _reward_base,
    _sanitize_token_ids,
    _truncate_prompt_rows,
    _truncate_token_ids,
)


class FakeProcessor:
    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}


class FakeTokenizer:
    pad_token_id = None
    eos_token = "<eos>"
    pad_token = None

    def __len__(self) -> int:
        return 10


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


class ChatTemplateTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<{item['role']}>{item['content']}</{item['role']}>"
            for item in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class FakeChatResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, object] | None = None,
        *,
        text: str = "bad request",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")

    def json(self) -> dict[str, object]:
        return self._payload


class FakeOpenAIRequests:
    calls: list[dict[str, object]] = []

    @classmethod
    def post(
        cls,
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
        headers: dict[str, str],
    ) -> FakeChatResponse:
        cls.calls.append(dict(json))
        optional = {
            "continue_final_message",
            "separate_reasoning",
            "chat_template_kwargs",
        }
        if "seed" in json:
            return FakeChatResponse(400, text="seed unsupported")
        if optional.intersection(json):
            return FakeChatResponse(400, text="chat options unsupported")
        return FakeChatResponse(200, {"ok": True})


class DummyTrainingArgs:
    def __init__(
        self,
        *,
        save_strategy: str | None = None,
        save_steps: int | None = None,
        save_total_limit: int | None = None,
    ) -> None:
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit


def test_openai_chat_post_retries_without_seed_and_optional_sglang_fields() -> None:
    FakeOpenAIRequests.calls = []
    result = _post_openai_chat_completion(
        requests_mod=FakeOpenAIRequests,
        base_url="http://localhost:30000",
        api_key="dummy",
        body={
            "model": "model",
            "messages": [{"role": "assistant", "content": "```python\n"}],
            "max_tokens": 8,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False,
            "continue_final_message": True,
            "separate_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": 123,
        },
        timeout_s=1.0,
    )

    assert result == {"ok": True}
    assert FakeOpenAIRequests.calls[0]["seed"] == 123
    assert "seed" not in FakeOpenAIRequests.calls[1]
    assert "continue_final_message" not in FakeOpenAIRequests.calls[2]
    assert "separate_reasoning" not in FakeOpenAIRequests.calls[2]
    assert "chat_template_kwargs" not in FakeOpenAIRequests.calls[2]


def test_intermediate_checkpoint_config_disables_trainer_saves() -> None:
    cfg = _filtered_config(
        DummyTrainingArgs,
        {
            **_disable_intermediate_checkpoint_config(25),
            "unsupported": "ignored",
        },
    )

    assert cfg.save_strategy == "no"
    assert cfg.save_steps == 25
    assert cfg.save_total_limit == 1


def test_remove_intermediate_checkpoints_preserves_final_adapter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").write_text("large optimizer state")
    checkpoint_file = tmp_path / "checkpoint-note"
    checkpoint_file.write_text("not a trainer directory")
    final_adapter = tmp_path / "final_adapter"
    final_adapter.mkdir()

    removed = _remove_intermediate_checkpoints(tmp_path)

    assert removed == [str(checkpoint)]
    assert not checkpoint.exists()
    assert checkpoint_file.exists()
    assert final_adapter.exists()


def test_cast_adapter_safetensors_to_bfloat16(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    adapter = tmp_path / "final_adapter"
    adapter.mkdir()
    path = adapter / "adapter_model.safetensors"
    save_file({"w": torch.ones((8, 8), dtype=torch.float32)}, str(path))
    before = path.stat().st_size

    manifest = _cast_adapter_safetensors(adapter, "bfloat16")
    tensors = load_file(str(path), device="cpu")

    assert manifest["applied"] is True
    assert manifest["requested_dtype"] == "bfloat16"
    assert manifest["before_bytes"] == before
    assert manifest["after_bytes"] < before
    assert tensors["w"].dtype == torch.bfloat16


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


def test_reward_base_keeps_strict_verified_channel_binary() -> None:
    result = RewardResult(
        score=0.75,
        verified_score=0.0,
        hard_gate_passed=False,
        evaluation_valid=True,
        failure_codes=["missing_port"],
    )

    reward, features = _reward_base(
        "```python\nfrom pathlib import Path\n```",
        result,
        reward_channel="verified_score",
    )

    assert reward == 0.0
    assert features == {}


def test_artifact_progress_reward_orders_syntax_and_schema_progress() -> None:
    truncated = (
        "```python\n"
        "from pathlib import Path\n"
        "def build_design(out_dir: Path) -> dict:\n"
        "    return {'schema_version':'design_ir.v2','parts':["
    )
    complete = (
        "```python\n"
        "from pathlib import Path\n"
        "def build_design(out_dir: Path) -> dict:\n"
        "    return {'schema_version':'design_ir.v2','parts':[],"
        "'joints':[],'ports':{'input_port':{},'output_port':{}},"
        "'params':{}}\n"
        "```"
    )
    invalid_result = RewardResult(
        score=0.0,
        verified_score=0.0,
        hard_gate_passed=False,
        evaluation_valid=False,
        failure_codes=["invalid_artifact"],
        design_py_extracted=True,
    )
    schema_result = RewardResult(
        score=0.0,
        verified_score=0.0,
        hard_gate_passed=False,
        evaluation_valid=False,
        failure_codes=["schema_error"],
        design_py_extracted=True,
    )

    truncated_reward, truncated_features = _reward_base(
        truncated,
        invalid_result,
        reward_channel="artifact_progress",
    )
    complete_reward, complete_features = _reward_base(
        complete,
        schema_result,
        reward_channel="artifact_progress",
    )

    assert truncated_reward > 0.0
    assert complete_reward > truncated_reward
    assert truncated_features["syntax_ok"] is False
    assert complete_features["syntax_ok"] is True
    assert complete_features["closed_code_fence"] is True


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


def test_truncate_prompt_rows_keeps_prompt_head_and_tail() -> None:
    rows = [{"prompt": "zero one two three", "task_id": "t1"}]

    truncated, audit = _truncate_prompt_rows(
        rows,
        WhitespaceTokenizer(),
        max_prompt_length=2,
    )

    assert truncated == [{"prompt": "zero three", "task_id": "t1"}]
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

    assert truncated == [{"prompt": "tok0 tok3", "task_id": "t1"}]
    assert audit["enabled"] is True
    assert audit["max_before"] == 4


def test_truncate_token_ids_keeps_head_and_tail() -> None:
    assert _truncate_token_ids([0, 1, 2, 3, 4], 3) == [0, 3, 4]
    assert _truncate_token_ids([0, 1, 2], 3) == [0, 1, 2]
    assert _truncate_token_ids([0, 1, 2], 0) == [0, 1, 2]


def test_sanitize_token_ids_drops_invalid_and_out_of_vocab_ids() -> None:
    assert _sanitize_token_ids(
        [0, "3", -1, 10, 11, None, 9],
        tokenizer=FakeTokenizer(),
    ) == [0, 3, 9]


def test_grpo_finite_tensor_audit_reports_nonfinite_values() -> None:
    audit = _finite_named_tensor_audit([
        ("good", torch.tensor([1.0, 2.0])),
        ("bad", torch.tensor([float("nan"), float("inf")])),
    ])

    assert audit["checked_tensors"] == 2
    assert audit["total_values"] == 4
    assert audit["nonfinite_values"] == 2
    assert audit["examples"][0]["name"] == "bad"


def test_grpo_nonfinite_trainable_adapter_weights_raise() -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite trainable adapter weights"):
        _raise_if_nonfinite_trainable_parameters(module, label="test adapter")


def test_guarded_adamw_counts_successful_finite_update() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = _make_guarded_adamw(module, learning_rate=1e-3)
    loss = module(torch.ones(1, 2)).sum()

    loss.backward()
    optimizer.step()

    manifest = _guarded_optimizer_manifest(optimizer)
    assert manifest["attempted_steps"] == 1
    assert manifest["successful_steps"] == 1
    assert manifest["skipped_nonfinite_gradient_steps"] == 0
    assert manifest["rolled_back_nonfinite_update_steps"] == 0
    assert torch.isfinite(module.weight).all()


def test_guarded_adamw_skips_all_nonfinite_gradient_update() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = _make_guarded_adamw(module, learning_rate=1e-3)
    before = module.weight.detach().clone()
    module.weight.grad = torch.full_like(module.weight, float("nan"))

    optimizer.step()

    manifest = _guarded_optimizer_manifest(optimizer)
    assert manifest["successful_steps"] == 0
    assert manifest["skipped_nonfinite_gradient_steps"] == 1
    assert manifest["skipped_all_nonfinite_gradient_steps"] == 1
    assert torch.equal(module.weight, before)


def test_guarded_adamw_sanitizes_partial_nonfinite_gradient_update() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = _make_guarded_adamw(module, learning_rate=1e-3)
    before = module.weight.detach().clone()
    module.weight.grad = torch.tensor([[float("nan"), 1.0]])

    optimizer.step()

    manifest = _guarded_optimizer_manifest(optimizer)
    assert manifest["successful_steps"] == 1
    assert manifest["sanitized_nonfinite_gradient_steps"] == 1
    assert manifest["sanitized_nonfinite_gradient_values"] == 1
    assert manifest["last_step_status"] == "successful_after_gradient_sanitize"
    assert torch.isfinite(module.weight).all()
    assert torch.equal(module.weight[:, :1], before[:, :1])
    assert not torch.equal(module.weight[:, 1:], before[:, 1:])


def test_guarded_adamw_rolls_back_nonfinite_post_update_weights() -> None:
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = _make_guarded_adamw(module, learning_rate=float("inf"))
    before = module.weight.detach().clone()
    module.weight.grad = torch.ones_like(module.weight)

    optimizer.step()

    manifest = _guarded_optimizer_manifest(optimizer)
    assert manifest["successful_steps"] == 0
    assert manifest["rolled_back_nonfinite_update_steps"] == 1
    assert torch.equal(module.weight, before)
    assert torch.isfinite(module.weight).all()


def test_grpo_lightweight_kbit_prepare_freezes_base_weights() -> None:
    module = torch.nn.Linear(2, 2, bias=False)

    prepared = _prepare_model_for_kbit_training_lightweight(
        module,
        use_gradient_checkpointing=False,
    )

    assert prepared is module
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_chat_prompt_rows_preserves_system_and_user_messages() -> None:
    rows = [{
        "prompt": "ignored",
        "_system_prompt": "system text",
        "_user_prompt": "user text",
        "task_id": "t1",
    }]

    assert _chat_prompt_rows(rows) == [{
        "prompt": [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ],
        "_system_prompt": "system text",
        "_user_prompt": "user text",
        "task_id": "t1",
    }]


def test_repeat_rows_for_grpo_sampler_expands_one_task_generation_batch() -> None:
    rows = [{"prompt": "p", "task_id": "task"}]

    expanded, audit = _repeat_rows_for_grpo_sampler(
        rows,
        generation_batch_size=32,
        num_generations=4,
    )

    assert len(expanded) == 8
    assert {row["task_id"] for row in expanded} == {"task"}
    assert [row["_trainer_source_index"] for row in expanded] == [0] * 8
    assert [row["_trainer_repeat_index"] for row in expanded] == list(range(8))
    assert audit == {
        "expanded": True,
        "source_rows": 1,
        "trainer_rows": 8,
        "unique_prompts_per_generation": 8,
        "repeat_factor": 8,
    }


def test_repeat_rows_for_grpo_sampler_keeps_sufficient_dataset() -> None:
    rows = [{"prompt": f"p{i}", "task_id": f"task{i}"} for i in range(8)]

    expanded, audit = _repeat_rows_for_grpo_sampler(
        rows,
        generation_batch_size=32,
        num_generations=4,
    )

    assert expanded is rows
    assert audit == {
        "expanded": False,
        "source_rows": 8,
        "trainer_rows": 8,
        "unique_prompts_per_generation": 8,
        "repeat_factor": 1,
    }


def test_repeat_rows_for_grpo_sampler_rejects_fractional_prompt_groups() -> None:
    with pytest.raises(SystemExit, match="divide evenly"):
        _repeat_rows_for_grpo_sampler(
            [{"prompt": "p", "task_id": "task"}],
            generation_batch_size=10,
            num_generations=4,
        )


def test_rollout_prompt_text_uses_chat_template_for_messages() -> None:
    messages = [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]

    assert _prompt_text_for_rollout(ChatTemplateTokenizer(), messages) == (
        "<system>system text</system><user>user text\n\n"
        f"{STRICT_FENCED_OUTPUT_INSTRUCTION}</user><assistant>"
    )
    rollout_messages = _messages_for_rollout(messages)
    assert rollout_messages[0] == {"role": "system", "content": "system text"}
    assert rollout_messages[1]["role"] == "user"
    assert rollout_messages[1]["content"] == (
        "user text\n\n" + STRICT_FENCED_OUTPUT_INSTRUCTION
    )
    assert _messages_for_rollout(
        messages,
        require_fenced_output=False,
    ) == messages


def test_full_eval_contract_suffix_includes_hidden_paper_verifier_requirements(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "eval_config.toml").write_text(
        """
[adapters.chrono_contact]
procedural_cycloidal_fallback = false

[[probes]]
id = "trusted_asset_preflight"
type = "trusted_asset_preflight"
require_geometry_roles = ["cad"]
require_materials = true
require_trusted_mass_properties = true

[[probes]]
id = "chrono_contact_smoke"
type = "contact_engagement"
adapter = "chrono_contact"
"""
    )
    task = TaskInfo(
        task_id="t1",
        family="chain",
        tier="paper",
        prompt="prompt",
        task_toml="[requirements]\nrequired_ports = []\n",
        task_dir=task_dir,
    )

    suffix = _full_eval_contract_suffix(task)

    assert "full private verifier contract" in suffix
    assert "trusted_cad_preflight" in suffix
    assert "trusted_mass_properties_required" in suffix
    assert "chrono_contact_required" in suffix
    assert "params.cad_mass_properties" in suffix
    assert "chrono_collision" in suffix
    assert "frame(out_dir)" in suffix
    assert "containing `belt` or `chain` is invalid" in suffix
