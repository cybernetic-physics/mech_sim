from __future__ import annotations

import os
from pathlib import Path

from scripts.dedupe_mechanism_repair_tokenizers import dedupe_run


def test_dedupe_run_hardlinks_completed_checkpoint_tokenizers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    shard_root = run_dir / "shard_runs" / "shard_0000" / "adapter_checkpoints"
    first = shard_root / "A" / "20260610" / "method" / "task_a"
    second = shard_root / "A" / "20260610" / "method" / "task_b"
    incomplete = shard_root / "A" / "20260610" / "method" / "task_c"
    for path in (first, second, incomplete):
        path.mkdir(parents=True)
        (path / "tokenizer.json").write_text('{"shared": true}\n')
    (first / "checkpoint_manifest.json").write_text("{}\n")
    (second / "checkpoint_manifest.json").write_text("{}\n")

    result = dedupe_run(run_dir)

    assert result["considered"] == 2
    assert result["deduped"] == 2
    assert result["skipped_incomplete"] == 1
    shared = list((run_dir / "shared_tokenizers").glob("tokenizer.*.json"))
    assert len(shared) == 1
    assert os.path.samefile(first / "tokenizer.json", shared[0])
    assert os.path.samefile(second / "tokenizer.json", shared[0])
    assert not os.path.samefile(incomplete / "tokenizer.json", shared[0])
