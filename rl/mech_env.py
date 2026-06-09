"""mech_bench environment for the worldlines RL training loop.

Mirrors rl-spark/worldlines-engdesign/src/engdesign_env.py in shape so a
hermes-agent / atropos driver could later swap in transparently.

What it does:
    - lists tasks under ``tasks/`` (the materialized 51-task suite +
      hand-written t001), filterable by family / tier / split file;
    - for each task, builds a ``TaskInfo`` carrying ``task_id``,
      ``family``, ``tier``, ``prompt`` (prompt.md), and ``task_toml``;
    - exposes ``score(task, raw_text) -> EpisodeResult`` which extracts
      a ``design.py`` from the model's output, runs
      ``python -m mech_bench evaluate --full --allow-partial``, and
      returns a normalized 0-100 score + pass flag + failure codes.

The reward is the verified mech_bench score *only*: ``score`` is in
[0, 100] when ``hard_gate_passed and evaluation_valid``, else 0.
``parsed_ok`` mirrors whether we found a triple-backticked Python
block. Subscores expose per-probe results.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rl.mech_bench_reward import (
    extract_no_procedural_fallback,
    extract_physical_metrics,
)
from rl.verifier_audits import cad_audit_count, chrono_audit_count


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"

_PARSE_BONUS_DEFAULT = 5.0  # points awarded just for emitting a parseable design.py.


# --------------------------------------------------------------------- #
# Data classes (kept compatible-ish with engdesign_env.TaskInfo)         #
# --------------------------------------------------------------------- #


@dataclass
class TaskInfo:
    task_id: str
    family: str
    tier: str
    prompt: str
    task_toml: str
    task_dir: Path
    # The mech_bench evaluator is structured; we don't need a pydantic
    # schema_cls like EngDesign. Kept as None for API parity.
    response_schema: type | None = None
    prm_gt: object | None = None


@dataclass
class EpisodeResult:
    task_id: str
    parsed_ok: bool
    passed: bool
    score: float
    max_score: float
    details: str
    raw_text: str
    error: str = ""
    subscores: list[float] = field(default_factory=list)
    parse_bonus: float = 0.0
    completion_tokens: int = 0
    failure_codes: list[str] = field(default_factory=list)
    feedback: list[dict[str, str]] = field(default_factory=list)
    evaluation_valid: bool = False
    cad_audits: int = 0
    chrono_audits: int = 0
    physical_metrics: dict[str, float] = field(default_factory=dict)
    no_procedural_fallback: bool | None = None
    # Dense per-probe reward — mean(probe.score) * 100. Always
    # defined whether or not the hard gate passed, so the RL loop
    # has a continuous signal even on hard-gate failures.
    dense_pct: float = 0.0


# --------------------------------------------------------------------- #
# Task discovery                                                         #
# --------------------------------------------------------------------- #


def _read_metadata(task_dir: Path) -> dict:
    meta = task_dir / "metadata.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _read_task_toml_lite(task_dir: Path) -> tuple[str, str]:
    """Return (family, tier) from task.toml without pulling tomllib."""
    text = (task_dir / "task.toml").read_text() if (task_dir / "task.toml").exists() else ""
    family = ""
    tier = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("family") and "=" in s:
            family = s.split("=", 1)[1].strip().strip('"').strip("'")
        elif s.startswith("tier") and "=" in s:
            tier = s.split("=", 1)[1].strip().strip('"').strip("'")
    return family, tier


def list_tasks(
    root: Path | None = None,
    *,
    families: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
    split_file: Path | None = None,
) -> list[TaskInfo]:
    """Walk ``tasks/`` and return TaskInfo objects.

    ``families`` / ``tiers`` are allowlists. ``split_file`` is a path to
    a text file with one task_id per line — when supplied, only those
    tasks are kept (used for held-out splits).
    """
    root = root or TASKS_DIR
    fams = set(families) if families else None
    ts = set(tiers) if tiers else None
    split: set[str] | None = None
    if split_file is not None and Path(split_file).exists():
        split = set()
        for line in Path(split_file).read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            split.add(entry)
            # Frozen benchmark splits may contain absolute paths from the
            # machine that materialized the benchmark.  Archives copied to a
            # cluster must still match by stable task directory name.
            split.add(Path(entry).name)
    out: list[TaskInfo] = []
    for child in sorted(Path(root).iterdir()):
        if not child.is_dir() or not (child / "task.toml").exists():
            continue
        meta = _read_metadata(child)
        family = str(meta.get("family") or "")
        tier = str(meta.get("tier") or "")
        if not family or not tier:
            f2, t2 = _read_task_toml_lite(child)
            family = family or f2
            tier = tier or t2
        if fams is not None and family not in fams:
            continue
        if ts is not None and tier not in ts:
            continue
        if split is not None and child.name not in split:
            continue
        prompt = (child / "prompt.md").read_text()
        task_toml = (child / "task.toml").read_text()
        out.append(TaskInfo(
            task_id=child.name,
            family=family or child.name,
            tier=tier or "unknown",
            prompt=prompt,
            task_toml=task_toml,
            task_dir=child,
        ))
    return out


# --------------------------------------------------------------------- #
# Completion extraction                                                  #
# --------------------------------------------------------------------- #


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_design_py(text: str) -> tuple[str, bool]:
    """Pull the first triple-backticked Python block out of *text*.

    Returns (source, parsed_ok). Falls back to the whole completion when
    no fenced block is found.
    """
    m = _CODE_FENCE_RE.search(text)
    source = m.group(1) if m else text
    source = re.sub(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|</s>))*\s*$", "", source, flags=re.IGNORECASE)
    return source.strip() + "\n", bool(m)


# --------------------------------------------------------------------- #
# Scoring                                                                #
# --------------------------------------------------------------------- #


def _failure_codes_of(blob: dict) -> list[str]:
    codes: list[str] = []
    for f in blob.get("feedback") or []:
        c = f.get("code")
        if isinstance(c, str) and c not in codes:
            codes.append(c)
    return codes


def _feedback_of(blob: dict, *, max_items: int = 8) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for f in blob.get("feedback") or []:
        if not isinstance(f, dict):
            continue
        item: dict[str, str] = {}
        for key in ("code", "severity", "message", "where"):
            val = f.get(key)
            if val is not None:
                item[key] = str(val)
        if item:
            out.append(item)
        if len(out) >= max_items:
            break
    return out


def score(
    task: TaskInfo,
    raw_text: str,
    *,
    parse_bonus: float = _PARSE_BONUS_DEFAULT,
    scratch_root: Path | None = None,
    timeout_s: float = 60.0,
) -> EpisodeResult:
    source, parsed = extract_design_py(raw_text)

    cleanup = False
    if scratch_root is None:
        scratch_root = Path(tempfile.mkdtemp(prefix="mech_rl_"))
        cleanup = True
    submission_dir = Path(scratch_root) / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "design.py").write_text(source)
    scratch = Path(scratch_root) / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mech_bench", "evaluate",
        "--task", str(task.task_dir),
        "--submission", str(submission_dir),
        "--scratch", str(scratch),
        "--full",
        "--allow-partial",
    ]
    env = dict(os.environ)
    env.pop("MECH_BENCH_USE_FAKE_ORACLE", None)
    env.pop("MECH_BENCH_TEST_MODE", None)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, check=False, cwd=REPO_ROOT, env=env,
        )
    except subprocess.TimeoutExpired:
        return EpisodeResult(
            task_id=task.task_id, parsed_ok=parsed, passed=False,
            score=0.0, max_score=100.0,
            details=f"timeout after {timeout_s}s",
            raw_text=raw_text, error="timeout",
            failure_codes=["timeout"],
            evaluation_valid=False,
            parse_bonus=parse_bonus if parsed else 0.0,
        )

    try:
        blob = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return EpisodeResult(
            task_id=task.task_id, parsed_ok=parsed, passed=False,
            score=0.0, max_score=100.0,
            details=(proc.stderr or proc.stdout)[-400:],
            raw_text=raw_text, error="runner_json_error",
            failure_codes=["runner_json_error"],
            evaluation_valid=False,
            parse_bonus=parse_bonus if parsed else 0.0,
        )

    codes = _failure_codes_of(blob)
    feedback = _feedback_of(blob)
    raw_score = float(blob.get("score") or 0.0)
    valid = bool(blob.get("evaluation_valid"))
    gate = bool(blob.get("hard_gate_passed"))
    # mech_bench returns score in [0,1]; rescale to [0,100] for parity
    # with engdesign reward magnitudes.
    final_score = (raw_score * 100.0) if (valid and gate) else 0.0
    # Pull per-probe scores. These power the dense_pct reward used by
    # RL — they're emitted on every run, even when the hard gate
    # fails, so the loop has a smooth signal to climb.
    subs: list[float] = []
    for r in blob.get("probe_results") or []:
        try:
            subs.append(float(r.get("score", 0.0)))
        except (TypeError, ValueError):
            pass
    dense_pct = (sum(subs) / len(subs) * 100.0) if subs else 0.0

    if cleanup:
        # Keep design.py for diagnostic; only clean the heavy scratch.
        try:
            import shutil
            shutil.rmtree(scratch, ignore_errors=True)
        except OSError:
            pass

    return EpisodeResult(
        task_id=task.task_id, parsed_ok=parsed,
        passed=(valid and gate and final_score > 0.0),
        score=final_score, max_score=100.0,
        details=f"valid={valid} gate={gate} probes={len(subs)} codes={codes}",
        raw_text=raw_text, error="",
        subscores=subs,
        parse_bonus=parse_bonus if parsed else 0.0,
        failure_codes=codes,
        feedback=feedback,
        evaluation_valid=valid,
        cad_audits=cad_audit_count(blob),
        chrono_audits=chrono_audit_count(blob),
        physical_metrics=extract_physical_metrics(blob),
        no_procedural_fallback=extract_no_procedural_fallback(blob),
        dense_pct=dense_pct,
    )


# --------------------------------------------------------------------- #
# CLI sanity check                                                       #
# --------------------------------------------------------------------- #


def _main() -> int:
    tasks = list_tasks()
    print(f"loaded {len(tasks)} tasks")
    by_tier: dict[str, int] = {}
    for t in tasks:
        by_tier[t.tier] = by_tier.get(t.tier, 0) + 1
    print("by_tier:", by_tier)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
