from __future__ import annotations

import json
from types import SimpleNamespace

from rl import sample_and_score
from rl.mech_bench_reward import (
    RewardResult,
    extract_no_procedural_fallback,
    extract_physical_metrics,
)
from rl.sample_and_score import (
    ASSISTANT_CODE_PREFILL,
    SampleOutcome,
    STRICT_FENCED_OUTPUT_INSTRUCTION,
    _needs_audit_retry,
    _reward_from_rollout_final,
    _rollout_audit_totals,
    _rollout_verifier_calls,
    _sglang_one_turn_messages,
    _task_required_audits,
    archive_feedback_text,
)
from rl.train_true_grpo_trl import ASSISTANT_CODE_PREFILL as TRAIN_ASSISTANT_CODE_PREFILL
from rl.verifier_audits import cad_audit_count, chrono_audit_count


def _outcome(reward: RewardResult) -> SampleOutcome:
    return SampleOutcome(
        task_id="cam_follower_contact_stub_s0001",
        family="cam_follower",
        tier="contact_dynamics",
        sample_idx=0,
        sample_duration_s=0.0,
        sample_tokens_in=0,
        sample_tokens_out=0,
        completion_chars=0,
        reward=reward,
        pass_threshold=1.0,
    )


def test_sample_and_score_prefers_absolute_split_task_dir(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks"
    root_task = tasks / "shared_task"
    root_task.mkdir(parents=True)
    (root_task / "prompt.md").write_text("public prompt")
    (root_task / "task.toml").write_text(
        'family = "public_family"\n'
        'tier = "public"\n'
    )
    variant_task = tmp_path / "variants" / "shared_task"
    variant_task.mkdir(parents=True)
    (variant_task / "prompt.md").write_text("variant prompt")
    (variant_task / "task.toml").write_text(
        'family = "variant_family"\n'
        'tier = "hidden"\n'
    )
    split = tmp_path / "hidden.txt"
    split.write_text(f"{variant_task}\n")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"
    seen_task_dirs = []

    def fake_run_one(task_dir, **_kwargs):
        seen_task_dirs.append(task_dir)
        outcome = _outcome(
            RewardResult(
                score=1.0,
                verified_score=1.0,
                hard_gate_passed=True,
                evaluation_valid=True,
                failure_codes=[],
            )
        )
        outcome.task_id = task_dir.name
        outcome.family = "variant_family"
        outcome.tier = "hidden"
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    rc = sample_and_score.main([
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--split-file", str(split),
        "--samples-per-task", "1",
    ])

    assert rc == 0
    assert seen_task_dirs == [variant_task.resolve()]
    summary = json.loads((report_dir / "smoke_summary.json").read_text())
    assert summary["all_samples"][0]["family"] == "variant_family"


def test_verifier_valid_pass_can_have_subunit_continuous_score() -> None:
    outcome = _outcome(
        RewardResult(
            score=0.8,
            verified_score=0.8,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )

    assert not outcome.passed()
    assert outcome.verifier_valid_passed()
    assert outcome.to_dict()["strict_passed"] is False
    assert outcome.to_dict()["verifier_valid_passed"] is True


def test_sglang_one_turn_messages_continue_code_prefill() -> None:
    messages = _sglang_one_turn_messages(
        system_prompt="sys",
        user_prompt="task",
    )

    assert messages == [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": "task\n\n" + STRICT_FENCED_OUTPUT_INSTRUCTION,
        },
        {"role": "assistant", "content": ASSISTANT_CODE_PREFILL},
    ]


def test_local_transformers_sampler_continues_code_prefill(monkeypatch) -> None:
    import torch

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, **_kwargs):
            assert messages == [{"role": "user", "content": "task"}]
            return "<chat>"

        def __call__(self, text, return_tensors):
            assert text == "<chat>" + ASSISTANT_CODE_PREFILL
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens is True
            assert list(ids) == [4, 5]
            return "continuation"

    class FakeModel:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.tensor(0.0))])

        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 3, 4, 5]])

    monkeypatch.setattr(
        sample_and_score,
        "_get_local_transformers",
        lambda **_kwargs: (FakeTokenizer(), FakeModel(), torch),
    )

    text, usage = sample_and_score.sample_from_local_transformers(
        base_model="base",
        lora_path=None,
        local_device="cpu",
        local_torch_dtype="auto",
        local_trust_remote_code=False,
        messages=[{"role": "user", "content": "task"}],
        assistant_prefill=ASSISTANT_CODE_PREFILL,
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        seed=123,
    )

    assert text == ASSISTANT_CODE_PREFILL + "continuation"
    assert usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "stop_reason": "stop",
    }


def test_run_one_uses_local_transformers_backend(monkeypatch, tmp_path) -> None:
    task_dir = tmp_path / "task_a"
    task_dir.mkdir()
    (task_dir / "prompt.md").write_text("prompt")
    (task_dir / "task.toml").write_text('family = "fam"\ntier = "tier"\n')
    seen: dict[str, object] = {}

    def fake_sample(**kwargs):
        seen.update(kwargs)
        return "```python\nx = 1\n```", {
            "input_tokens": 7,
            "output_tokens": 3,
        }

    monkeypatch.setattr(
        sample_and_score,
        "sample_from_local_transformers",
        fake_sample,
    )
    monkeypatch.setattr(
        sample_and_score,
        "score_completion",
        lambda *_args, **_kwargs: RewardResult(
            score=0.5,
            verified_score=0.0,
            hard_gate_passed=False,
            evaluation_valid=True,
            failure_codes=["missing_port"],
            design_py_extracted=True,
        ),
    )

    outcome = sample_and_score.run_one(
        task_dir,
        base_url="unused",
        api_key="unused",
        base_model="base",
        model_path=None,
        sglang_lora_path=None,
        rollout_backend="transformers_local",
        local_device="cpu",
        local_torch_dtype="auto",
        local_trust_remote_code=False,
        system_prompt="system",
        out_root=tmp_path / "out",
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        seed=123,
        timeout_s=1.0,
        pass_threshold=1.0,
        max_turns=1,
        sample_idx=0,
    )

    assert outcome.error == ""
    assert outcome.verifier_calls == 1
    assert outcome.sample_tokens_in == 7
    assert outcome.sample_tokens_out == 3
    assert seen["base_model"] == "base"
    assert seen["local_device"] == "cpu"
    assert seen["assistant_prefill"] == ASSISTANT_CODE_PREFILL


def test_prefill_mass_properties_accept_optional_chrono_shape() -> None:
    for prefill in (ASSISTANT_CODE_PREFILL, TRAIN_ASSISTANT_CODE_PREFILL):
        namespace: dict[str, object] = {}
        helper_source = prefill.removeprefix("```python\n").split(
            "def build_design", 1
        )[0]
        exec(helper_source, namespace)

        mp = namespace["mp"]
        cyl = namespace["cyl"]

        mass_only = mp(0.1, (1.0, 2.0, 3.0))
        collision = mp(0.1, (1.0, 2.0, 3.0), cyl(4, 5))

        assert "cad_mass_properties" in mass_only
        assert "chrono_collision" not in mass_only
        assert collision["cad_mass_properties"]["mass_kg"] == 0.1
        assert collision["chrono_collision"]["shape"] == "cylinder"


def test_verifier_valid_pass_rejects_failures_even_with_score() -> None:
    outcome = _outcome(
        RewardResult(
            score=0.8,
            verified_score=0.8,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=["contact_missing"],
        )
    )

    assert not outcome.verifier_valid_passed()


def test_sample_outcome_serializes_verifier_calls() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )
    outcome.verifier_calls = 3

    assert outcome.to_dict()["verifier_calls"] == 3


def test_sample_outcome_marks_repair_success_after_feedback_turns() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )
    outcome.verifier_calls = 2

    assert outcome.repair_attempted() is True
    assert outcome.repair_succeeded() is True
    assert outcome.to_dict()["repair_attempted"] is True
    assert outcome.to_dict()["repair_succeeded"] is True


def test_sample_outcome_serializes_chrono_audits() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
            chrono_audits=1,
        )
    )
    outcome.chrono_audits = 1

    assert outcome.to_dict()["chrono_audits"] == 1


def test_sample_outcome_serializes_cad_audits() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
            cad_audits=1,
        )
    )
    outcome.cad_audits = 2

    assert outcome.to_dict()["cad_audits"] == 2


def test_audit_retries_count_actual_budget(monkeypatch, tmp_path) -> None:
    tasks = tmp_path / "tasks"
    task_dir = tasks / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    (task_dir / "eval_config.toml").write_text(
        """
[adapters.chrono_contact]
procedural_cycloidal_fallback = false

[[probes]]
id = "trusted_asset_preflight"
type = "trusted_asset_preflight"

[[probes]]
id = "contact"
type = "contact_engagement"
adapter = "chrono_contact"
"""
    )
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"

    def fake_run_one(task_dir, **kwargs):
        is_retry = "audit_retry" in str(kwargs["out_root"])
        outcome = _outcome(
            RewardResult(
                score=1.0,
                verified_score=1.0,
                hard_gate_passed=True,
                evaluation_valid=True,
                failure_codes=[],
                cad_audits=1 if is_retry else 0,
                chrono_audits=1 if is_retry else 0,
            )
        )
        outcome.task_id = task_dir.name
        outcome.verifier_calls = 1
        outcome.cad_audits = 1 if is_retry else 0
        outcome.chrono_audits = 1 if is_retry else 0
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    rc = sample_and_score.main([
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--samples-per-task", "1",
        "--audit-retries", "1",
    ])

    assert rc == 0
    summary = json.loads((report_dir / "smoke_summary.json").read_text())
    sample = summary["all_samples"][0]
    assert sample["verifier_calls"] == 2
    assert sample["cad_audits"] == 1
    assert sample["chrono_audits"] == 1
    assert sample["audit_retry_count"] == 1


def test_max_verifier_calls_per_task_caps_multiturn_budget(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks"
    task_dir = tasks / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"
    observed_turns: list[int] = []

    def fake_run_one(task_dir, **kwargs):
        max_turns = int(kwargs["max_turns"])
        observed_turns.append(max_turns)
        outcome = _outcome(
            RewardResult(
                score=0.0,
                verified_score=0.0,
                hard_gate_passed=False,
                evaluation_valid=True,
                failure_codes=["wrong_ratio"],
            )
        )
        outcome.task_id = task_dir.name
        outcome.sample_idx = len(observed_turns) - 1
        outcome.verifier_calls = max_turns
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    rc = sample_and_score.main([
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--samples-per-task", "32",
        "--max-turns", "4",
        "--max-verifier-calls-per-task", "6",
    ])

    assert rc == 0
    summary = json.loads((report_dir / "smoke_summary.json").read_text())
    assert observed_turns == [4, 2]
    assert len(summary["all_samples"]) == 2
    assert sum(item["verifier_calls"] for item in summary["all_samples"]) == 6


def test_resume_existing_reuses_completed_samples_and_continues_missing_tasks(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks"
    task_a = tasks / "task_a"
    task_b = tasks / "task_b"
    for task_dir in (task_a, task_b):
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            'family = "cycloidal"\n'
            'tier = "contact_dynamics"\n'
        )
    split_a = tmp_path / "split_a.txt"
    split_a.write_text("task_a\n")
    split_all = tmp_path / "split_all.txt"
    split_all.write_text("task_a\ntask_b\n")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"
    calls: list[str] = []

    def fake_run_one(task_dir, **kwargs):
        calls.append(task_dir.name)
        outcome = _outcome(
            RewardResult(
                score=1.0,
                verified_score=1.0,
                hard_gate_passed=True,
                evaluation_valid=True,
                failure_codes=[],
            )
        )
        outcome.task_id = task_dir.name
        outcome.sample_idx = int(kwargs["sample_idx"])
        outcome.completion_chars = 101 if task_dir.name == "task_a" else 202
        outcome.verifier_calls = 1
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    base_args = [
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--samples-per-task", "1",
        "--seed", "20260610",
    ]
    assert sample_and_score.main([*base_args, "--split-file", str(split_a)]) == 0
    partial_summary_path = report_dir / "smoke_summary.json"
    partial_summary = json.loads(partial_summary_path.read_text())
    partial_summary["complete"] = False
    partial_summary_path.write_text(json.dumps(partial_summary))

    calls.clear()
    assert sample_and_score.main([
        *base_args,
        "--split-file", str(split_all),
        "--resume-existing",
    ]) == 0

    summary = json.loads(partial_summary_path.read_text())
    assert calls == ["task_b"]
    assert summary["complete"] is True
    assert {row["task_id"] for row in summary["all_samples"]} == {
        "task_a",
        "task_b",
    }
    by_task = {row["task_id"]: row for row in summary["all_samples"]}
    assert by_task["task_a"]["completion_chars"] == 101
    assert by_task["task_b"]["completion_chars"] == 202


def test_resume_existing_recovers_from_sample_outcome_checkpoint_without_summary(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks"
    task_a = tasks / "task_a"
    task_b = tasks / "task_b"
    for task_dir in (task_a, task_b):
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            'family = "cycloidal"\n'
            'tier = "contact_dynamics"\n'
        )
    split_a = tmp_path / "split_a.txt"
    split_a.write_text("task_a\n")
    split_all = tmp_path / "split_all.txt"
    split_all.write_text("task_a\ntask_b\n")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"
    calls: list[str] = []

    def fake_run_one(task_dir, **kwargs):
        calls.append(task_dir.name)
        outcome = _outcome(
            RewardResult(
                score=1.0,
                verified_score=1.0,
                hard_gate_passed=True,
                evaluation_valid=True,
                failure_codes=[],
            )
        )
        outcome.task_id = task_dir.name
        outcome.sample_idx = int(kwargs["sample_idx"])
        outcome.completion_chars = 101 if task_dir.name == "task_a" else 202
        outcome.verifier_calls = 1
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    base_args = [
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--samples-per-task", "1",
        "--seed", "20260610",
    ]
    assert sample_and_score.main([*base_args, "--split-file", str(split_a)]) == 0
    checkpoint = report_dir / "sample_0" / "task_a" / "sample_outcome.json"
    assert checkpoint.is_file()
    (report_dir / "smoke_summary.json").unlink()

    calls.clear()
    assert sample_and_score.main([
        *base_args,
        "--split-file", str(split_all),
        "--resume-existing",
    ]) == 0

    summary = json.loads((report_dir / "smoke_summary.json").read_text())
    assert calls == ["task_b"]
    by_task = {row["task_id"]: row for row in summary["all_samples"]}
    assert by_task["task_a"]["completion_chars"] == 101
    assert by_task["task_b"]["completion_chars"] == 202


def test_archive_feedback_text_summarizes_prior_candidates() -> None:
    good = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )
    good.sample_idx = 3
    weak = _outcome(
        RewardResult(
            score=0.2,
            verified_score=0.2,
            hard_gate_passed=False,
            evaluation_valid=True,
            failure_codes=["wrong_mobility"],
        )
    )
    weak.sample_idx = 1

    text = archive_feedback_text([weak, good])

    assert "adaptive evolution archive" in text
    assert "sample=3 score=1.000" in text
    assert "wrong_mobility" in text


def test_sampler_retry_preserves_spent_verifier_call_traces(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks"
    task_dir = tasks / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"
    calls = 0

    def trace(turn_idx: int, text: str) -> dict:
        return {
            "turn_idx": turn_idx,
            "assistant_text": text,
            "dense_pct": 0.0,
            "score": 0.0,
            "passed": False,
            "parsed_ok": True,
            "evaluation_valid": True,
            "failure_codes": ["wrong_ratio"],
        }

    def fake_run_one(task_dir, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SampleOutcome(
                task_id=task_dir.name,
                family="cycloidal",
                tier="contact_dynamics",
                sample_idx=0,
                sample_duration_s=0.0,
                sample_tokens_in=0,
                sample_tokens_out=0,
                completion_chars=0,
                reward=None,
                verifier_calls=3,
                pass_threshold=1.0,
                error="[sampler_error: context length]",
                turn_traces=[
                    trace(0, "failed attempt turn 0"),
                    trace(1, "failed attempt turn 1"),
                    trace(2, "failed attempt turn 2"),
                ],
            )
        outcome = _outcome(
            RewardResult(
                score=0.0,
                verified_score=0.0,
                hard_gate_passed=False,
                evaluation_valid=True,
                failure_codes=["wrong_ratio"],
            )
        )
        outcome.task_id = task_dir.name
        outcome.verifier_calls = 1
        outcome.turn_traces = [trace(0, "retry turn 0")]
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    rc = sample_and_score.main([
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--samples-per-task", "1",
        "--max-turns", "4",
        "--sampler-retries", "1",
        "--max-verifier-calls-per-task", "4",
    ])

    assert rc == 0
    summary = json.loads((report_dir / "smoke_summary.json").read_text())
    sample = summary["all_samples"][0]
    assert sample["verifier_calls"] == 4
    assert len(sample["turn_traces"]) == 4
    assert [
        turn["verifier_call_idx_within_sample"]
        for turn in sample["turn_traces"]
    ] == [0, 1, 2, 3]
    assert [
        turn["sampler_attempt"] for turn in sample["turn_traces"]
    ] == [0, 0, 0, 1]


def test_sampler_retry_synthesizes_terminal_trace_when_retry_uses_final_call(
    monkeypatch,
    tmp_path,
) -> None:
    tasks = tmp_path / "tasks"
    task_dir = tasks / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("[task]\n")
    system_prompt = tmp_path / "system.md"
    system_prompt.write_text("system")
    report_dir = tmp_path / "report"
    calls = 0

    def trace(turn_idx: int, text: str) -> dict:
        return {
            "turn_idx": turn_idx,
            "assistant_text": text,
            "dense_pct": 0.0,
            "score": 0.0,
            "passed": False,
            "parsed_ok": True,
            "evaluation_valid": True,
            "failure_codes": ["wrong_ratio"],
        }

    def fake_run_one(task_dir, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SampleOutcome(
                task_id=task_dir.name,
                family="cycloidal",
                tier="contact_dynamics",
                sample_idx=0,
                sample_duration_s=0.0,
                sample_tokens_in=0,
                sample_tokens_out=0,
                completion_chars=0,
                reward=None,
                verifier_calls=3,
                pass_threshold=1.0,
                error="[sampler_error: context length]",
                turn_traces=[
                    trace(0, "failed attempt turn 0"),
                    trace(1, "failed attempt turn 1"),
                    trace(2, "failed attempt turn 2"),
                ],
            )
        per_task = kwargs["out_root"] / task_dir.name
        per_task.mkdir(parents=True)
        (per_task / "completion.txt").write_text("terminal design")
        outcome = _outcome(
            RewardResult(
                score=0.0,
                verified_score=0.0,
                hard_gate_passed=False,
                evaluation_valid=True,
                failure_codes=["wrong_ratio"],
            )
        )
        outcome.task_id = task_dir.name
        outcome.verifier_calls = 1
        return outcome

    monkeypatch.setattr(sample_and_score, "run_one", fake_run_one)

    rc = sample_and_score.main([
        "--tasks", str(tasks),
        "--report-dir", str(report_dir),
        "--system-prompt-file", str(system_prompt),
        "--samples-per-task", "1",
        "--max-turns", "4",
        "--sampler-retries", "1",
        "--max-verifier-calls-per-task", "4",
    ])

    assert rc == 0
    summary = json.loads((report_dir / "smoke_summary.json").read_text())
    sample = summary["all_samples"][0]
    assert sample["verifier_calls"] == 4
    assert len(sample["turn_traces"]) == 4
    assert sample["turn_traces"][-1]["assistant_text"] == "terminal design"
    assert [
        turn["verifier_call_idx_within_sample"]
        for turn in sample["turn_traces"]
    ] == [0, 1, 2, 3]
    assert [
        turn["sampler_attempt"] for turn in sample["turn_traces"]
    ] == [0, 0, 0, 1]


def test_rollout_verifier_calls_excludes_sampler_errors() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(failure_codes=[]),
        SimpleNamespace(failure_codes=["wrong_ratio"]),
        SimpleNamespace(failure_codes=["sampler_error"]),
    ])

    assert _rollout_verifier_calls(rollout) == 2


def test_reward_from_rollout_final_uses_best_scored_turn() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(
            dense_pct=25.0,
            score=25.0,
            passed=False,
            evaluation_valid=True,
            failure_codes=["wrong_ratio"],
            feedback=[],
            parsed_ok=True,
            cad_audits=1,
            chrono_audits=1,
            no_procedural_fallback=True,
        ),
        SimpleNamespace(
            dense_pct=75.0,
            score=50.0,
            passed=True,
            evaluation_valid=True,
            failure_codes=[],
            feedback=[],
            parsed_ok=True,
            cad_audits=1,
            chrono_audits=1,
            no_procedural_fallback=True,
        )
    ])

    reward = _reward_from_rollout_final(rollout)

    assert reward is not None
    assert reward.score == 0.75
    assert reward.verified_score == 0.5
    assert reward.hard_gate_passed is True
    assert reward.evaluation_valid is True
    assert reward.design_py_extracted is True
    assert reward.cad_audits == 1
    assert reward.chrono_audits == 1
    assert reward.no_procedural_fallback is True


def test_rollout_audit_totals_sum_non_sampler_error_turns() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(failure_codes=[], cad_audits=1, chrono_audits=1),
        SimpleNamespace(
            failure_codes=["wrong_ratio"],
            cad_audits=1,
            chrono_audits=0,
        ),
        SimpleNamespace(
            failure_codes=["sampler_error"],
            cad_audits=1,
            chrono_audits=1,
        ),
    ])

    assert _rollout_audit_totals(rollout) == (2, 1)


def test_needs_audit_retry_requires_planned_cad_and_chrono_budget() -> None:
    outcome = _outcome(
        RewardResult(
            score=1.0,
            verified_score=1.0,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
        )
    )
    outcome.cad_audits = 1
    outcome.chrono_audits = 1

    assert _needs_audit_retry(
        outcome,
        required_cad_audits=2,
        required_chrono_audits=2,
    )

    outcome.cad_audits = 2
    outcome.chrono_audits = 2
    assert not _needs_audit_retry(
        outcome,
        required_cad_audits=2,
        required_chrono_audits=2,
    )


def test_task_required_audits_are_level_specific(tmp_path) -> None:
    level2 = tmp_path / "level2"
    level2.mkdir()
    (level2 / "eval_config.toml").write_text(
        """
[[probes]]
id = "trusted_asset_preflight"
type = "trusted_asset_preflight"
"""
    )

    level3 = tmp_path / "level3"
    level3.mkdir()
    (level3 / "eval_config.toml").write_text(
        """
[adapters.chrono_contact]
procedural_cycloidal_fallback = false

[[probes]]
id = "trusted_asset_preflight"
type = "trusted_asset_preflight"

[[probes]]
id = "contact"
type = "contact_engagement"
adapter = "chrono_contact"
"""
    )

    plain = tmp_path / "plain"
    plain.mkdir()

    assert _task_required_audits(level2, max_turns=4) == (4, 0)
    assert _task_required_audits(level3, max_turns=4) == (4, 4)
    assert _task_required_audits(plain, max_turns=4) == (0, 0)


def test_reward_from_rollout_final_ignores_sampler_error() -> None:
    rollout = SimpleNamespace(turns=[
        SimpleNamespace(failure_codes=["sampler_error"])
    ])

    assert _reward_from_rollout_final(rollout) is None


def test_chrono_audit_count_requires_real_chrono_attempt() -> None:
    assert chrono_audit_count({
        "timings": {"adapter.chrono_contact": 1.2},
        "feedback": [],
    }) == 1

    assert chrono_audit_count({
        "timings": {"adapter": {"chrono_contact": 1.2}},
        "feedback": [],
    }) == 1

    assert chrono_audit_count({
        "timings": {"adapter.fake_contact_oracle": 1.2},
        "feedback": [],
    }) == 0


def test_cad_audit_count_requires_trusted_cad_evidence() -> None:
    assert cad_audit_count({
        "metrics": {
            "trusted_asset_preflight.trusted_mass_properties_recomputed": 1.0,
        },
    }) == 1

    assert cad_audit_count({
        "metrics": {
            "trusted_asset_preflight.parts_with_trusted_mass_properties": 2.0,
        },
    }) == 1

    assert cad_audit_count({
        "metrics": {
            "trusted_asset_preflight": {
                "trusted_mass_properties_recomputed": 1.0,
            },
        },
    }) == 1

    assert cad_audit_count({
        "metrics": {
            "parts_with_trusted_mass_properties": 1.0,
        },
    }) == 1

    assert cad_audit_count({
        "timings": {"load_submission": 0.2},
        "metrics": {},
    }) == 0


def test_extract_physical_metrics_uses_canonical_aliases() -> None:
    metrics = extract_physical_metrics({
        "metrics": {
            "ratio.observed": 9.0,
            "ratio.error_pct": 1.5,
            "out_omega_med": 0.8,
            "max_penetration_mm": 0.2,
            "contact_force_rms_N": 12.0,
            "power_balance_error_pct": 4.0,
            "torque_ripple_pct": 8.0,
        }
    })

    assert metrics["ratio_observed"] == 9.0
    assert metrics["ratio_error_pct"] == 1.5
    assert metrics["out_omega_med"] == 0.8
    assert metrics["max_penetration_mm"] == 0.2
    assert metrics["contact_force_rms_N"] == 12.0

    prefixed = extract_physical_metrics({
        "metrics": {
            "chrono_contact.ratio_error_pct": 2.5,
            "chrono_contact.out_omega_med": 1.2,
            "chrono_contact.max_penetration_mm": 0.1,
        }
    })
    assert prefixed["ratio_error_pct"] == 2.5
    assert prefixed["out_omega_med"] == 1.2
    assert prefixed["max_penetration_mm"] == 0.1

    nested = extract_physical_metrics({
        "scalar_metrics": {
            "chrono_contact": {
                "contact_force_rms_N": 8.0,
            }
        }
    })
    assert nested["contact_force_rms_N"] == 8.0

    report_level = extract_physical_metrics({
        "metrics": {
            "contact.contact.cam:follower.rms_N": 16836.5,
            "swept_collision.max_penetration_mm": 0.025,
            "lockup.output_motion_rad": 50.0,
        }
    })
    assert report_level["contact_force_rms_N"] == 16836.5
    assert report_level["max_penetration_mm"] == 0.025
    assert report_level["out_omega_med"] == 50.0


def test_extract_no_procedural_fallback_uses_canonical_aliases() -> None:
    assert extract_no_procedural_fallback({
        "metrics": {"procedural_cycloidal_fallback": False}
    }) is True
    assert extract_no_procedural_fallback({
        "metrics": {"chrono.procedural_cycloidal_fallback": True}
    }) is False
    assert extract_no_procedural_fallback({
        "procedural_cycloidal_fallback": False,
    }) is True
    assert extract_no_procedural_fallback({
        "metrics": {"chrono_contact": {"procedural_cycloidal_fallback": False}}
    }) is True
    assert extract_no_procedural_fallback({
        "timings": {"adapter.chrono_contact": 1.2},
    }) is True
    assert extract_no_procedural_fallback({"metrics": {}}) is None

    assert chrono_audit_count({
        "timings": {"adapter.chrono_contact": 0.01},
        "feedback": [
            {
                "code": "capability_unavailable",
                "where": "adapter.chrono_contact",
                "message": "Adapter 'chrono_contact' unavailable.",
            }
        ],
    }) == 0
