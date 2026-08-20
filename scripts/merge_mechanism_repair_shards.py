#!/usr/bin/env python3
"""Merge MechanismRepair shard outputs into one artifact bundle.

Two modes are supported:

* legacy ``--source-dir`` mode copies source evidence into ``--out-dir``;
* physics ``--shard-root`` mode merges ``shard_XXXX/cell_results.jsonl`` rows
  without copying evidence, preserving the shard-run paths for final audits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.run_mechanism_repair_online_experiment import (
    row_key,
    run_analysis,
    write_results_bundle,
)


LEGACY_PATH_LIST_FIELDS = (
    ("raw_completion_paths", "raw_completions"),
    ("verifier_output_paths", "verifier_outputs"),
)
PHYSICS_PATH_LIST_FIELDS = (
    "raw_completion_paths",
    "verifier_output_paths",
    "cad_artifact_paths",
    "chrono_output_paths",
    "training_log_paths",
    "adapter_checkpoint_paths",
)
PHYSICS_PATH_SCALAR_FIELDS = (
    "summary_path",
    "trace_path",
    "adapter_path",
)
MISSING_CHRONO_COLLISION_GEOMETRY_TEXT = (
    "required contact bodies lack Chrono collision geometry"
)
CAPABILITY_UNAVAILABLE_CODE = "capability_unavailable"
MISSING_COLLISION_RECLASSIFIED_CODE = "invalid_artifact"
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
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--source-dir",
        action="append",
        help="legacy run directory containing cell_results.jsonl; may repeat",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help="benchmark manifest directory; defaults to --out-dir in legacy "
             "mode and runs/mechanism_repair_physics_final in shard mode",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="legacy mode: replace duplicate rows with later sources",
    )
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--shard-root", default=None)
    parser.add_argument("--require-all-shards", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-physics-audit", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "when running the physics audit after merge, require the final "
            "claim audit to prove goals.md completion"
        ),
    )
    args = parser.parse_args()

    if args.source_dir:
        return run_legacy_source_merge(args)
    return run_physics_shard_merge(args)


def run_legacy_source_merge(args: argparse.Namespace) -> int:
    if not args.out_dir:
        raise SystemExit("--out-dir is required with --source-dir")
    out_dir = Path(args.out_dir).expanduser().resolve()
    benchmark_dir = (
        Path(args.benchmark_dir).expanduser().resolve()
        if args.benchmark_dir
        else out_dir
    )
    source_dirs = [Path(item).expanduser().resolve() for item in args.source_dir]
    merged: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    source_by_key: dict[tuple[str, str, int, str, int], str] = {}
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
    write_legacy_merge_manifest(
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


def run_physics_shard_merge(args: argparse.Namespace) -> int:
    benchmark_dir = (
        Path(args.benchmark_dir).expanduser().resolve()
        if args.benchmark_dir
        else Path("runs/mechanism_repair_physics_final").resolve()
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else benchmark_dir
    )
    shard_root = (
        Path(args.shard_root).expanduser().resolve()
        if args.shard_root
        else out_dir / "shard_runs"
    )
    rows, shard_summaries = load_shard_rows(
        shard_root,
        require_all_shards=max(0, int(args.require_all_shards)),
        expected_shard_dir=(
            benchmark_dir / "experiment_shards"
            if int(args.require_all_shards) > 0
            else None
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "cell_results.jsonl", rows)
    write_results_bundle(out_dir, rows)
    write_shard_merge_manifest(out_dir, shard_root, rows, shard_summaries)
    if not args.skip_analysis:
        run_analysis(out_dir=out_dir, benchmark_dir=benchmark_dir)
    if not args.skip_physics_audit:
        audit_cmd = [
            sys.executable,
            str(Path(__file__).with_name("run_mechanism_repair_physics_experiment.py")),
            "--benchmark-dir",
            str(benchmark_dir),
            "--out-dir",
            str(out_dir),
        ]
        if bool(getattr(args, "require_complete", False)):
            audit_cmd.append("--require-complete")
        subprocess.run(audit_cmd, check=True)
    print(json.dumps({
        "rows": len(rows),
        "jsonl": str(out_dir / "cell_results.jsonl"),
        "shards": len(shard_summaries),
        "merge_manifest": str(out_dir / "shard_merge_manifest.json"),
    }, indent=2, sort_keys=True))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_shard_rows(
    shard_root: Path,
    *,
    require_all_shards: int = 0,
    expected_shard_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if require_all_shards:
        expected = {f"shard_{idx:04d}" for idx in range(require_all_shards)}
        observed = {path.name for path in shard_dirs}
        missing = sorted(expected - observed)
        if missing:
            raise SystemExit(f"missing shard output directories: {missing}")
    expected_by_shard = load_expected_shard_keys(
        expected_shard_dir,
        require_all_shards=require_all_shards,
    )
    rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], str] = {}
    duplicates: list[dict[str, Any]] = []
    coverage_errors: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        rows_path = shard_dir / "cell_results.jsonl"
        if not rows_path.is_file():
            if require_all_shards:
                raise SystemExit(f"missing shard cell_results.jsonl: {rows_path}")
            continue
        shard_rows = [
            normalize_design_preflight_capability_code(
                absolutize_row_paths(json.loads(line), base_dir=shard_dir)
            )
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        observed_keys = {physics_cell_key(row) for row in shard_rows}
        expected_keys = expected_by_shard.get(shard_dir.name)
        missing_keys: list[tuple[Any, ...]] = []
        unexpected_keys: list[tuple[Any, ...]] = []
        if expected_keys is not None:
            missing_keys = sorted(expected_keys - observed_keys)
            unexpected_keys = sorted(observed_keys - expected_keys)
            if missing_keys or unexpected_keys:
                coverage_errors.append({
                    "shard": shard_dir.name,
                    "expected_rows": len(expected_keys),
                    "observed_rows": len(observed_keys),
                    "missing_cell_count": len(missing_keys),
                    "unexpected_cell_count": len(unexpected_keys),
                    "sample_missing_cells": [list(key) for key in missing_keys[:10]],
                    "sample_unexpected_cells": [
                        list(key) for key in unexpected_keys[:10]
                    ],
                })
        for row in shard_rows:
            key = physics_cell_key(row)
            if key in seen:
                duplicates.append({
                    "key": list(key),
                    "first_shard": seen[key],
                    "duplicate_shard": shard_dir.name,
                })
                continue
            seen[key] = shard_dir.name
            rows.append(row)
        shard_summaries.append({
            "shard": shard_dir.name,
            "rows": len(shard_rows),
            "expected_rows": (
                len(expected_keys) if expected_keys is not None else None
            ),
            "missing_cell_count": len(missing_keys),
            "unexpected_cell_count": len(unexpected_keys),
            "cell_results": str(rows_path),
        })
    if duplicates:
        raise SystemExit(
            "duplicate split/task/seed/method/budget rows across shards: "
            + json.dumps(duplicates[:20], sort_keys=True)
        )
    if coverage_errors:
        raise SystemExit(
            "incomplete or mismatched shard coverage: "
            + json.dumps(coverage_errors[:20], sort_keys=True)
        )
    rows.sort(key=lambda row: physics_cell_key(row))
    return rows, shard_summaries


def load_expected_shard_keys(
    expected_shard_dir: Path | None,
    *,
    require_all_shards: int,
) -> dict[str, set[tuple[Any, ...]]]:
    if expected_shard_dir is None:
        return {}
    if not expected_shard_dir.is_dir():
        raise SystemExit(f"missing experiment shard manifests: {expected_shard_dir}")
    expected_by_shard: dict[str, set[tuple[Any, ...]]] = {}
    missing_manifests: list[str] = []
    for shard_index in range(require_all_shards):
        shard_name = f"shard_{shard_index:04d}"
        path = expected_shard_dir / f"{shard_name}.json"
        if not path.is_file():
            missing_manifests.append(path.name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_by_shard[shard_name] = {
            physics_cell_key(cell)
            for cell in payload.get("cells", []) or []
        }
    if missing_manifests:
        raise SystemExit(
            "missing experiment shard manifests: "
            + json.dumps(missing_manifests, sort_keys=True)
        )
    return expected_by_shard


def physics_cell_key(row: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(row["split"]),
        str(row["task_id"]),
        int(row["seed"]),
        str(row["method"]),
        int(row.get("budget", 0) or 0),
    )


def absolutize_row_paths(
    row: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    out = dict(row)
    for field in PHYSICS_PATH_LIST_FIELDS:
        values = out.get(field)
        if not values:
            continue
        if not isinstance(values, list):
            values = [values]
        out[field] = [absolutize_path(value, base_dir) for value in values]
    for field in PHYSICS_PATH_SCALAR_FIELDS:
        value = out.get(field)
        if value:
            out[field] = absolutize_path(value, base_dir)
    return out


def absolutize_path(value: Any, base_dir: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def normalize_design_preflight_capability_code(
    row: dict[str, Any],
) -> dict[str, Any]:
    codes = failure_code_list(row.get("failure_codes", []))
    if CAPABILITY_UNAVAILABLE_CODE not in codes:
        return row
    if not row_has_missing_collision_geometry_evidence(row):
        return row
    out = dict(row)
    recoded = [
        code for code in codes if code != CAPABILITY_UNAVAILABLE_CODE
    ]
    if MISSING_COLLISION_RECLASSIFIED_CODE not in recoded:
        recoded.append(MISSING_COLLISION_RECLASSIFIED_CODE)
    out["failure_codes"] = recoded
    out["capability_unavailable_reclassified"] = True
    out["capability_unavailable_reclassification_reason"] = (
        MISSING_CHRONO_COLLISION_GEOMETRY_TEXT
    )
    return out


def failure_code_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def row_has_missing_collision_geometry_evidence(row: dict[str, Any]) -> bool:
    for path in candidate_evidence_paths(row):
        if path_contains(path, MISSING_CHRONO_COLLISION_GEOMETRY_TEXT):
            return True
    return False


def candidate_evidence_paths(row: dict[str, Any]) -> list[Path]:
    values: list[Any] = []
    for field in (
        "summary_path",
        "verifier_output_path",
        "verifier_output_paths",
        "chrono_output_path",
        "chrono_output_paths",
    ):
        raw = row.get(field)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw:
            values.append(raw)
    out: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path = Path(str(value))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def path_contains(path: Path, needle: str) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), ""):
                if needle in chunk:
                    return True
    except OSError:
        return False
    return False


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
    for field, artifact_dir in LEGACY_PATH_LIST_FIELDS:
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
            omit_redundant_adapter_files=True,
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


def copy_path(
    *,
    source: Path,
    dest_dir: Path,
    omit_redundant_adapter_files: bool = False,
) -> Path:
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
    ignore = None
    if omit_redundant_adapter_files:
        ignore = lambda _dir, names: {"tokenizer.json"} & set(names)
    shutil.copytree(source, dest, ignore=ignore)
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_legacy_merge_manifest(
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_shard_merge_manifest(
    out_dir: Path,
    shard_root: Path,
    rows: list[dict[str, Any]],
    shard_summaries: list[dict[str, Any]],
) -> None:
    payload = {
        "schema": "mechanism_repair_physics.shard_merge_manifest.v1",
        "shard_root": str(shard_root),
        "rows": len(rows),
        "unique_cells": len({physics_cell_key(row) for row in rows}),
        "shards": shard_summaries,
    }
    (out_dir / "shard_merge_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
