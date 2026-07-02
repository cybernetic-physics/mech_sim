#!/usr/bin/env python3
"""Recover verifier evidence for legacy terminal-only completions.

Older shard code could leave ``sample_N/task_id/completion.txt`` files without
the exact ``sample_outcome.json`` checkpoint that ``sample_and_score`` needs
for safe resume. This helper re-scores those terminal completions and writes
``terminal_recovery.json`` plus ``terminal_recovery_summary.json`` artifacts.

The recovered artifacts are intentionally not named ``sample_outcome.json``:
they do not contain the intermediate verifier traces or exact per-turn CAD /
Chrono accounting needed for final paper rows. They are triage evidence only.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from rl.mech_bench_reward import RewardResult, score_completion
from rl.sample_and_score import SampleOutcome


RECOVERY_VERSION = "mech_bench.terminal_completion_recovery.v1"
SUMMARY_VERSION = "mech_bench.terminal_completion_recovery_summary.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--tasks-root", required=True)
    parser.add_argument(
        "--split-file",
        default=None,
        help=(
            "optional task id/path allowlist. Absolute paths are honored, "
            "matching sample_and_score split behavior."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-turns", type=int, default=1)
    parser.add_argument("--pass-threshold", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = recover_terminal_completion_evidence(
        report_dir=Path(args.report_dir).expanduser().resolve(),
        tasks_root=Path(args.tasks_root).expanduser().resolve(),
        split_file=(
            Path(args.split_file).expanduser().resolve()
            if args.split_file
            else None
        ),
        timeout_s=float(args.timeout_s),
        max_turns=max(1, int(args.max_turns)),
        pass_threshold=float(args.pass_threshold),
        limit=max(0, int(args.limit)),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def recover_terminal_completion_evidence(
    *,
    report_dir: Path,
    tasks_root: Path,
    split_file: Path | None = None,
    timeout_s: float = 60.0,
    max_turns: int = 1,
    pass_threshold: float = 1.0,
    limit: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
    scorer: Callable[..., RewardResult] = score_completion,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_map = build_task_map(tasks_root=tasks_root, split_file=split_file)
    candidates = terminal_completion_candidates(report_dir)
    if limit:
        candidates = candidates[:limit]

    recovered: list[dict[str, Any]] = []
    newly_recovered = 0
    skipped_existing_checkpoint = 0
    skipped_existing_recovery = 0
    skipped_missing_task = 0
    errors: list[dict[str, Any]] = []
    for candidate in candidates:
        task_id = candidate["task_id"]
        sample_dir = Path(candidate["sample_dir"])
        if (sample_dir / "sample_outcome.json").is_file():
            skipped_existing_checkpoint += 1
            continue
        recovery_path = sample_dir / "terminal_recovery.json"
        if recovery_path.is_file() and not overwrite:
            try:
                payload = json.loads(recovery_path.read_text(encoding="utf-8"))
                recovered.append(
                    compact_recovery_row(payload, recovery_path=recovery_path)
                )
            except json.JSONDecodeError:
                errors.append({
                    "task_id": task_id,
                    "sample_idx": candidate["sample_idx"],
                    "reason": "invalid_existing_terminal_recovery",
                })
            skipped_existing_recovery += 1
            continue
        task_dir = task_map.get(task_id)
        if task_dir is None:
            skipped_missing_task += 1
            errors.append({
                "task_id": task_id,
                "sample_idx": candidate["sample_idx"],
                "reason": "missing_task_dir",
            })
            continue
        if dry_run:
            recovered.append({
                "task_id": task_id,
                "sample_idx": candidate["sample_idx"],
                "completion_path": str(candidate["completion_path"]),
                "task_dir": str(task_dir),
                "dry_run": True,
            })
            continue
        try:
            record = recover_one_completion(
                completion_path=Path(candidate["completion_path"]),
                task_dir=task_dir,
                sample_idx=int(candidate["sample_idx"]),
                timeout_s=timeout_s,
                max_turns=max_turns,
                pass_threshold=pass_threshold,
                scorer=scorer,
            )
        except Exception as exc:  # noqa: BLE001 - recovery should keep going
            errors.append({
                "task_id": task_id,
                "sample_idx": candidate["sample_idx"],
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        write_json_atomic(recovery_path, record)
        newly_recovered += 1
        recovered.append(compact_recovery_row(record, recovery_path=recovery_path))

    summary = {
        "version": SUMMARY_VERSION,
        "complete": False,
        "resumable_checkpoint_count": 0,
        "recovered_terminal_completion_count": len(recovered),
        "newly_recovered_terminal_completion_count": newly_recovered,
        "candidate_completion_count": len(candidates),
        "skipped_existing_checkpoint_count": skipped_existing_checkpoint,
        "skipped_existing_recovery_count": skipped_existing_recovery,
        "skipped_missing_task_count": skipped_missing_task,
        "error_count": len(errors),
        "errors": errors[:25],
        "recovered": recovered[:100],
        "report_dir": str(report_dir),
        "tasks_root": str(tasks_root),
        "split_file": str(split_file) if split_file else None,
        "max_turns": int(max_turns),
        "wall_clock_s": time.perf_counter() - started,
        "paper_claim_note": (
            "Recovered terminal evidence is not a final result row and is not "
            "resume-compatible because intermediate verifier traces and exact "
            "CAD/Chrono call accounting are unavailable."
        ),
    }
    if not dry_run:
        write_json_atomic(report_dir / "terminal_recovery_summary.json", summary)
    return summary


def recover_one_completion(
    *,
    completion_path: Path,
    task_dir: Path,
    sample_idx: int,
    timeout_s: float,
    max_turns: int,
    pass_threshold: float,
    scorer: Callable[..., RewardResult],
) -> dict[str, Any]:
    text = completion_path.read_text(encoding="utf-8", errors="replace")
    reward = scorer(
        text,
        task_dir,
        scratch_root=completion_path.parent / "terminal_recovery_scratch",
        timeout_s=timeout_s,
    )
    family, tier = read_task_meta(task_dir)
    outcome = SampleOutcome(
        task_id=task_dir.name,
        family=family,
        tier=tier,
        sample_idx=int(sample_idx),
        sample_duration_s=0.0,
        sample_tokens_in=0,
        sample_tokens_out=0,
        completion_chars=len(text),
        reward=reward,
        verifier_calls=1,
        cad_audits=int(reward.cad_audits or 0),
        chrono_audits=int(reward.chrono_audits or 0),
        pass_threshold=float(pass_threshold),
    )
    return {
        "version": RECOVERY_VERSION,
        "task_id": task_dir.name,
        "sample_idx": int(sample_idx),
        "completion_path": str(completion_path),
        "task_dir": str(task_dir),
        "scored_terminal_verifier_calls": 1,
        "original_intermediate_verifier_calls_unknown": True,
        "resumable_checkpoint": False,
        "outcome": outcome.to_dict(),
        "paper_claim_note": (
            "This re-scores only the terminal completion. It must not be "
            "converted into a final cell row without the original exact "
            "sample_outcome checkpoint or smoke_summary."
        ),
    }


def terminal_completion_candidates(report_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("sample_*/**/completion.txt")):
        sample_dir = path.parent
        sample_parent = sample_dir.parent
        match = re.fullmatch(r"sample_(\d+)", sample_parent.name)
        if not match:
            continue
        out.append({
            "sample_idx": int(match.group(1)),
            "task_id": sample_dir.name,
            "sample_dir": str(sample_dir),
            "completion_path": str(path),
        })
    return out


def build_task_map(*, tasks_root: Path, split_file: Path | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if split_file and split_file.is_file():
        for raw in split_file.read_text(encoding="utf-8").splitlines():
            item = raw.strip()
            if not item or item.startswith("#"):
                continue
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = tasks_root / item
            if path.is_dir():
                out[path.name] = path.resolve()
            else:
                out[item] = (tasks_root / item).resolve()
    if tasks_root.is_dir():
        for child in sorted(tasks_root.iterdir()):
            if child.is_dir():
                out.setdefault(child.name, child.resolve())
    return out


def read_task_meta(task_dir: Path) -> tuple[str, str]:
    text = (task_dir / "task.toml").read_text(encoding="utf-8", errors="replace")
    family = ""
    tier = ""
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key == "family":
            family = value
        elif key == "tier":
            tier = value
    return family, tier


def compact_recovery_row(
    record: dict[str, Any],
    *,
    recovery_path: Path,
) -> dict[str, Any]:
    outcome = dict(record.get("outcome") or {})
    return {
        "task_id": record.get("task_id"),
        "sample_idx": record.get("sample_idx"),
        "recovery_path": str(recovery_path),
        "verified_score": outcome.get("verified_score"),
        "evaluation_valid": outcome.get("evaluation_valid"),
        "hard_gate_passed": outcome.get("hard_gate_passed"),
        "failure_codes": outcome.get("failure_codes", []),
        "cad_audits": outcome.get("cad_audits", 0),
        "chrono_audits": outcome.get("chrono_audits", 0),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{id(payload)}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
