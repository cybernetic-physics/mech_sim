#!/usr/bin/env python3
"""Merge MechanismRepair-TTRL shard outputs into one final artifact bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.run_mechanism_repair_online_experiment import (
    row_key,
    run_analysis,
    write_results_bundle,
)


PATH_LIST_FIELDS = (
    ("raw_completion_paths", "raw_completions"),
    ("verifier_output_paths", "verifier_outputs"),
)

BENCHMARK_FILE_NAMES = (
    "benchmark_manifest.json",
    "split_manifest_A.json",
    "split_manifest_B.json",
    "verifier_manifest.json",
    "method_manifest.json",
    "online_experiment_plan.json",
)

BENCHMARK_DIR_NAMES = (
    "splits_A",
    "splits_B",
    "tasks",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--source-dir",
        action="append",
        required=True,
        help="run directory containing cell_results.jsonl; may be repeated",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="benchmark manifest directory for final analysis; defaults to --out-dir",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace duplicate split/task/seed/method rows with later sources",
    )
    parser.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    benchmark_dir = (
        Path(args.benchmark_dir).expanduser().resolve()
        if args.benchmark_dir
        else out_dir
    )
    source_dirs = [Path(item).expanduser().resolve() for item in args.source_dir]
    merged: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    source_by_key: dict[tuple[str, str, int, str], str] = {}
    duplicates: list[dict[str, Any]] = []

    for source_dir in source_dirs:
        rows_path = source_dir / "cell_results.jsonl"
        if not rows_path.is_file():
            continue
        for row in load_jsonl(rows_path):
            normalized = materialize_row_evidence(row, out_dir=out_dir)
            key = row_key(normalized)
            if key in merged:
                duplicates.append({
                    "key": list(key),
                    "kept_source": source_by_key[key],
                    "duplicate_source": str(source_dir),
                    "replaced": bool(args.replace_existing),
                })
                if not args.replace_existing:
                    continue
            merged[key] = normalized
            source_by_key[key] = str(source_dir)

    rows = [merged[key] for key in sorted(merged)]
    copy_benchmark_artifacts(benchmark_dir=benchmark_dir, out_dir=out_dir)
    write_jsonl(out_dir / "cell_results.jsonl", rows)
    write_results_bundle(out_dir, rows)
    write_merge_manifest(
        out_dir=out_dir,
        source_dirs=source_dirs,
        rows=rows,
        duplicates=duplicates,
        replace_existing=bool(args.replace_existing),
    )
    if args.run_analysis:
        run_analysis(out_dir=out_dir, benchmark_dir=benchmark_dir)
    print(json.dumps({
        "out_dir": str(out_dir),
        "rows": len(rows),
        "sources": [str(path) for path in source_dirs],
        "duplicates": len(duplicates),
        "merge_manifest": str(out_dir / "merge_manifest.json"),
    }, indent=2, sort_keys=True))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def materialize_row_evidence(
    row: dict[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    normalized = dict(row)
    split = str(normalized.get("split"))
    seed = str(normalized.get("seed"))
    method = str(normalized.get("method"))
    task_id = str(normalized.get("task_id"))
    for field, artifact_dir in PATH_LIST_FIELDS:
        copied: list[str] = []
        for raw_path in normalized.get(field, []) or []:
            source = Path(str(raw_path)).expanduser()
            copied.append(str(copy_artifact(
                source=source,
                dest_dir=out_dir / artifact_dir / split / seed / method / task_id,
            )))
        normalized[field] = copied

    summary_path = normalized.get("summary_path")
    if summary_path:
        copied = copy_artifact(
            source=Path(str(summary_path)).expanduser(),
            dest_dir=out_dir / "verifier_outputs" / split / seed / method / task_id,
            dest_name="summary.json",
        )
        normalized["summary_path"] = str(copied)

    trace_path = normalized.get("trace_path")
    if trace_path and str(trace_path).endswith("reward_log.jsonl"):
        copied = copy_artifact(
            source=Path(str(trace_path)).expanduser(),
            dest_dir=out_dir / "training_logs" / split / seed / method / task_id,
            dest_name="reward_log.jsonl",
        )
        normalized["trace_path"] = str(copied)

    adapter_path = normalized.get("adapter_path")
    if adapter_path:
        copied = copy_path(
            source=Path(str(adapter_path)).expanduser(),
            dest_dir=out_dir / "adapter_checkpoints" / split / seed / method / task_id,
        )
        normalized["adapter_path"] = str(copied)
    return normalized


def copy_artifact(
    *,
    source: Path,
    dest_dir: Path,
    dest_name: str | None = None,
) -> Path:
    if not source.is_file():
        return source
    resolved_source = source.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (dest_name or source.name)
    if dest.exists() and dest.resolve() == resolved_source:
        return dest
    if dest.exists() and dest.read_bytes() != source.read_bytes():
        stem = dest.stem
        suffix = dest.suffix
        index = 1
        while True:
            candidate = dest_dir / f"{stem}_{index:03d}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            if candidate.read_bytes() == source.read_bytes():
                return candidate
            index += 1
    if dest.exists():
        return dest
    shutil.copy2(source, dest)
    return dest


def copy_path(*, source: Path, dest_dir: Path) -> Path:
    if source.is_file():
        return copy_artifact(source=source, dest_dir=dest_dir)
    if not source.is_dir():
        return source
    resolved_source = source.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    if dest.exists() and dest.resolve() == resolved_source:
        return dest
    if dest.exists():
        return dest
    shutil.copytree(source, dest)
    return dest


def copy_benchmark_artifacts(*, benchmark_dir: Path, out_dir: Path) -> None:
    if benchmark_dir == out_dir:
        return
    for name in BENCHMARK_FILE_NAMES:
        source = benchmark_dir / name
        if source.is_file():
            copy_artifact(source=source, dest_dir=out_dir, dest_name=name)
    for name in BENCHMARK_DIR_NAMES:
        source_dir = benchmark_dir / name
        dest_dir = out_dir / name
        if not source_dir.is_dir() or dest_dir.exists():
            continue
        shutil.copytree(source_dir, dest_dir)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_merge_manifest(
    *,
    out_dir: Path,
    source_dirs: list[Path],
    rows: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    replace_existing: bool,
) -> None:
    payload = {
        "schema": "mechanism_repair_ttrl.merge_manifest.v1",
        "sources": [str(path) for path in source_dirs],
        "rows": len(rows),
        "replace_existing": bool(replace_existing),
        "duplicates": duplicates,
    }
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
