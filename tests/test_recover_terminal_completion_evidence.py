from __future__ import annotations

import json
from pathlib import Path

from rl.mech_bench_reward import RewardResult
from scripts import recover_terminal_completion_evidence as recover


def test_recover_terminal_completion_writes_nonresumable_evidence(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        'family = "cam_follower"\n'
        'tier = "contact_dynamics"\n',
        encoding="utf-8",
    )
    report_dir = tmp_path / "report"
    completion_dir = report_dir / "sample_0" / "task_a"
    completion_dir.mkdir(parents=True)
    (completion_dir / "completion.txt").write_text("```python\npass\n```")
    calls: list[tuple[str, Path]] = []

    def fake_scorer(text: str, task: Path, **_kwargs) -> RewardResult:
        calls.append((text, task))
        return RewardResult(
            score=0.8,
            verified_score=0.8,
            hard_gate_passed=True,
            evaluation_valid=True,
            failure_codes=[],
            cad_audits=1,
            chrono_audits=1,
        )

    summary = recover.recover_terminal_completion_evidence(
        report_dir=report_dir,
        tasks_root=tasks_root,
        scorer=fake_scorer,
    )

    assert calls == [("```python\npass\n```", task_dir.resolve())]
    assert summary["complete"] is False
    assert summary["resumable_checkpoint_count"] == 0
    assert summary["recovered_terminal_completion_count"] == 1
    assert summary["newly_recovered_terminal_completion_count"] == 1
    assert not (completion_dir / "sample_outcome.json").exists()
    recovery_path = completion_dir / "terminal_recovery.json"
    assert recovery_path.is_file()
    payload = json.loads(recovery_path.read_text())
    assert payload["resumable_checkpoint"] is False
    assert payload["outcome"]["task_id"] == "task_a"
    assert payload["outcome"]["verified_score"] == 0.8
    assert (report_dir / "terminal_recovery_summary.json").is_file()


def test_recover_terminal_completion_prefers_absolute_split_task(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    public = tasks_root / "same_id"
    public.mkdir(parents=True)
    (public / "task.toml").write_text('family = "public"\n')
    variant = tmp_path / "variants" / "same_id"
    variant.mkdir(parents=True)
    (variant / "task.toml").write_text(
        'family = "variant"\n'
        'tier = "hidden"\n',
        encoding="utf-8",
    )
    split_file = tmp_path / "split.txt"
    split_file.write_text(str(variant) + "\n", encoding="utf-8")
    report_dir = tmp_path / "report"
    completion_dir = report_dir / "sample_2" / "same_id"
    completion_dir.mkdir(parents=True)
    (completion_dir / "completion.txt").write_text("design")
    seen_tasks: list[Path] = []

    def fake_scorer(_text: str, task: Path, **_kwargs) -> RewardResult:
        seen_tasks.append(task)
        return RewardResult(
            score=0.0,
            verified_score=0.0,
            hard_gate_passed=False,
            evaluation_valid=True,
            failure_codes=["wrong_ratio"],
        )

    recover.recover_terminal_completion_evidence(
        report_dir=report_dir,
        tasks_root=tasks_root,
        split_file=split_file,
        scorer=fake_scorer,
    )

    assert seen_tasks == [variant.resolve()]
    payload = json.loads((completion_dir / "terminal_recovery.json").read_text())
    assert payload["outcome"]["family"] == "variant"


def test_recover_terminal_completion_skips_exact_checkpoints(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('family = "x"\n')
    report_dir = tmp_path / "report"
    completion_dir = report_dir / "sample_0" / "task_a"
    completion_dir.mkdir(parents=True)
    (completion_dir / "completion.txt").write_text("design")
    (completion_dir / "sample_outcome.json").write_text("{}")

    summary = recover.recover_terminal_completion_evidence(
        report_dir=report_dir,
        tasks_root=tasks_root,
        scorer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not score exact checkpoints")
        ),
    )

    assert summary["recovered_terminal_completion_count"] == 0
    assert summary["skipped_existing_checkpoint_count"] == 1
    assert not (completion_dir / "terminal_recovery.json").exists()


def test_recover_terminal_completion_counts_existing_recoveries(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text('family = "x"\n')
    report_dir = tmp_path / "report"
    completion_dir = report_dir / "sample_0" / "task_a"
    completion_dir.mkdir(parents=True)
    (completion_dir / "completion.txt").write_text("design")
    (completion_dir / "terminal_recovery.json").write_text(json.dumps({
        "version": recover.RECOVERY_VERSION,
        "task_id": "task_a",
        "sample_idx": 0,
        "outcome": {
            "task_id": "task_a",
            "sample_idx": 0,
            "verified_score": 1.0,
            "evaluation_valid": True,
            "hard_gate_passed": True,
            "failure_codes": [],
        },
    }))

    summary = recover.recover_terminal_completion_evidence(
        report_dir=report_dir,
        tasks_root=tasks_root,
        scorer=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not rescore existing recovery")
        ),
    )

    assert summary["recovered_terminal_completion_count"] == 1
    assert summary["newly_recovered_terminal_completion_count"] == 0
    assert summary["skipped_existing_recovery_count"] == 1
    assert summary["recovered"][0]["verified_score"] == 1.0
