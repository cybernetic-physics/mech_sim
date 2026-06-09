from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_mechanism_repair_shards import main as merge_main


def test_merge_mechanism_repair_shards_copies_evidence(
    monkeypatch,
    tmp_path: Path,
    ) -> None:
    source = tmp_path / "shard"
    source.mkdir()
    raw = source / "raw.txt"
    verifier = source / "verifier.json"
    summary = source / "summary.json"
    adapter = source / "final_adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}\n')
    raw.write_text("completion")
    verifier.write_text('{"ok": true}\n')
    summary.write_text('{"all_samples": []}\n')
    row = {
        "method": "frozen_model",
        "split": "A",
        "task_id": "task_a",
        "family": "cycloidal",
        "seed": 20260607,
        "verified_repair_success_at_32": False,
        "best_verified_reward_at_32": 0.0,
        "verifier_calls": 32,
        "cad_audits": 0,
        "chrono_audits": 0,
        "actual_budget_matches_primary": True,
        "raw_completion_paths": [str(raw)],
        "verifier_output_paths": [str(verifier)],
        "summary_path": str(summary),
        "adapter_path": str(adapter),
    }
    (source / "cell_results.jsonl").write_text(json.dumps(row) + "\n")
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    for name in (
        "benchmark_manifest.json",
        "split_manifest_A.json",
        "split_manifest_B.json",
        "verifier_manifest.json",
        "method_manifest.json",
        "online_experiment_plan.json",
    ):
        (benchmark / name).write_text('{"ok": true}\n')
    (benchmark / "tasks" / "task_a").mkdir(parents=True)
    (benchmark / "tasks" / "task_a" / "metadata.json").write_text(
        '{"family": "cycloidal"}\n'
    )
    (benchmark / "splits_A").mkdir()
    (benchmark / "splits_A" / "test.txt").write_text("task_a\n")
    (benchmark / "splits_B").mkdir()
    (benchmark / "splits_B" / "test.txt").write_text("task_b\n")
    out_dir = tmp_path / "final"

    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--out-dir",
            str(out_dir),
            "--benchmark-dir",
            str(benchmark),
            "--source-dir",
            str(source),
        ],
    )

    assert merge_main() == 0
    merged = [
        json.loads(line)
        for line in (out_dir / "cell_results.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(merged) == 1
    raw_path = Path(merged[0]["raw_completion_paths"][0])
    verifier_path = Path(merged[0]["verifier_output_paths"][0])
    summary_path = Path(merged[0]["summary_path"])
    adapter_path = Path(merged[0]["adapter_path"])
    assert raw_path.is_file()
    assert verifier_path.is_file()
    assert summary_path.is_file()
    assert (adapter_path / "adapter_config.json").is_file()
    assert raw_path.is_relative_to(out_dir / "raw_completions")
    assert verifier_path.is_relative_to(out_dir / "verifier_outputs")
    assert summary_path.is_relative_to(out_dir / "verifier_outputs")
    assert adapter_path.is_relative_to(out_dir / "adapter_checkpoints")
    index = json.loads((out_dir / "raw_completions" / "index.json").read_text())
    assert index["count"] == 1
    assert (out_dir / "benchmark_manifest.json").is_file()
    assert (out_dir / "method_manifest.json").is_file()
    assert (out_dir / "verifier_manifest.json").is_file()
    assert (out_dir / "split_manifest_A.json").is_file()
    assert (out_dir / "split_manifest_B.json").is_file()
    assert (out_dir / "online_experiment_plan.json").is_file()
    assert (out_dir / "tasks" / "task_a" / "metadata.json").is_file()
    assert (out_dir / "splits_A" / "test.txt").is_file()
    assert (out_dir / "splits_B" / "test.txt").is_file()
