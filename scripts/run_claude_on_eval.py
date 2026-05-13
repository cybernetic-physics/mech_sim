#!/usr/bin/env python3
"""Run Claude Code (headless) against the mech_bench suite.

Per task:
    1. Materialize a per-task scratch dir with an empty
       `submission/` folder and a sandboxed workdir.
    2. Spawn `claude -p` with --output-format json, a strict
       allowlist of tools (Read/Write/Edit/Glob/Grep + a few
       harmless Bash idioms), --add-dir for the task input + the
       submission output, --max-budget-usd as a cost cap, and a
       wall-clock timeout. The system prompt is appended via
       --append-system-prompt; the per-task user prompt is sent on
       stdin (because --allowedTools is variadic and would consume a
       trailing positional).
    3. Capture the JSON result (which includes total_cost_usd +
       duration_ms).
    4. Run `python -m mech_bench evaluate ...` against the produced
       `design.py` and parse the verified score, hard-gate state,
       and failure codes.

Auth: relies on the user's existing Claude Code session (OAuth or
keychain). Pass ``--claude-arg --bare`` to lock down to
``ANTHROPIC_API_KEY`` only.

Outputs a per-task results JSON and an aggregate scorecard.

Usage::

    python scripts/run_claude_on_eval.py \
        --tasks tasks \
        --report-dir /tmp/claude_eval_reports \
        --model sonnet \
        --max-budget-usd 0.25 \
        --timeout 180 \
        --concurrency 4 \
        --families mounting_plate_hole_pitch,spur_gear_ratio_analytic
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = REPO_ROOT / "scripts" / "agent_system_prompt.md"
USER_PROMPT_TEMPLATE = """Solve task **{task_id}** from the mech_bench benchmark.

## Task directory
Read-only inputs for the agent:
- `{task_dir}/prompt.md`
- `{task_dir}/task.toml`

Do NOT open `reference_solution/`, `negative_solutions/`,
`eval_config*.toml`, `expected_failures.json`, or `metadata.json` —
those are evaluator-private.

## Submission directory
Write your solution to:

    {submission_dir}/design.py

The file must define `build_design(out_dir: Path) -> dict` returning
a DesignIR (schema_version "design_ir.v2"). See your system prompt
for the schema. When the file is written, stop.
"""


# --------------------------------------------------------------------- #
# Data                                                                  #
# --------------------------------------------------------------------- #


@dataclass
class TaskOutcome:
    task_id: str
    family: str
    tier: str
    agent_ok: bool
    agent_cost_usd: float | None
    agent_duration_s: float
    agent_exit_code: int
    agent_error: str
    submission_exists: bool
    eval_valid: bool | None
    eval_hard_gate_passed: bool | None
    eval_score: float | None
    eval_failure_codes: list[str] = field(default_factory=list)
    eval_duration_s: float = 0.0

    def passed(self) -> bool:
        return bool(
            self.eval_valid
            and self.eval_hard_gate_passed
            and (self.eval_score or 0.0) > 0.0
        )


# --------------------------------------------------------------------- #
# Step 1 — run the agent                                                #
# --------------------------------------------------------------------- #


def run_agent(
    *,
    task_dir: Path,
    submission_dir: Path,
    workdir: Path,
    system_prompt: str,
    model: str,
    max_budget_usd: float,
    timeout_s: int,
    extra_args: list[str],
) -> tuple[int, dict[str, Any] | None, str, float]:
    """Invoke `claude -p` and return (exit_code, json_payload, stderr_tail, duration_s)."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        task_id=task_dir.name,
        task_dir=str(task_dir),
        submission_dir=str(submission_dir),
    )
    # NOTE: `--allowedTools` is variadic in Claude Code 2.x, so any
    # trailing positional prompt is consumed as another tool. We
    # therefore pass the user prompt via stdin and keep --allowedTools
    # last on the flag list (with nothing positional after it).
    cmd = [
        "claude", "-p",
        "--no-session-persistence",
        "--model", model,
        "--output-format", "json",
        "--max-budget-usd", str(max_budget_usd),
        "--append-system-prompt", system_prompt,
        "--add-dir", str(task_dir),
        "--add-dir", str(submission_dir),
        *extra_args,
        "--allowedTools",
        "Read,Write,Edit,Glob,Grep,Bash(ls:*),Bash(cat:*),Bash(echo:*),Bash(python3:*)",
    ]
    # Auth: relies on the user's existing Claude Code session
    # (OAuth / keychain). Pass `--claude-arg --bare` to lock down to
    # ANTHROPIC_API_KEY only.

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return 124, None, f"timed out after {timeout_s}s: {e}", time.perf_counter() - t0
    except FileNotFoundError:
        return 127, None, "`claude` not on PATH", time.perf_counter() - t0

    elapsed = time.perf_counter() - t0
    payload: dict[str, Any] | None = None
    if proc.stdout:
        # The CLI's json format prints a single JSON object on stdout.
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload, (proc.stderr or "")[-2000:], elapsed


# --------------------------------------------------------------------- #
# Step 2 — score the submission                                         #
# --------------------------------------------------------------------- #


def score_submission(
    *,
    task_dir: Path,
    submission_dir: Path,
    scratch_dir: Path,
    report_dir: Path | None,
) -> tuple[bool, bool, float, list[str], float]:
    """Run mech-bench evaluate and parse the report.

    Returns (eval_valid, hard_gate_passed, score, failure_codes,
    duration_s).
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mech_bench", "evaluate",
        "--task", str(task_dir),
        "--submission", str(submission_dir),
        "--scratch", str(scratch_dir),
        "--allow-partial",
        "--full",
    ]
    if report_dir is not None:
        cmd += ["--report-dir", str(report_dir)]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        cwd=REPO_ROOT,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0 and not proc.stdout.strip().startswith("{"):
        return False, False, 0.0, ["runner_error"], elapsed

    blob: dict[str, Any]
    try:
        blob = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, False, 0.0, ["runner_json_error"], elapsed

    codes: list[str] = []
    for f in blob.get("feedback", []) or []:
        c = f.get("code")
        if isinstance(c, str) and c not in codes:
            codes.append(c)

    return (
        bool(blob.get("evaluation_valid")),
        bool(blob.get("hard_gate_passed")),
        float(blob.get("score", 0.0) or 0.0),
        codes,
        elapsed,
    )


# --------------------------------------------------------------------- #
# Per-task driver                                                       #
# --------------------------------------------------------------------- #


def _read_task_meta(task_dir: Path) -> tuple[str, str]:
    """Return (family, tier) for *task_dir*, falling back to defaults."""
    meta_path = task_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            return str(meta.get("family", task_dir.name)), str(
                meta.get("tier", "unknown")
            )
        except (OSError, json.JSONDecodeError):
            pass
    return task_dir.name, "unknown"


def drive_one(
    *,
    task_dir: Path,
    out_root: Path,
    system_prompt: str,
    model: str,
    max_budget_usd: float,
    timeout_s: int,
    extra_args: list[str],
    keep_report: bool,
) -> TaskOutcome:
    family, tier = _read_task_meta(task_dir)
    submission_dir = out_root / task_dir.name / "submission"
    workdir = out_root / task_dir.name / "workdir"
    scratch_dir = out_root / task_dir.name / "_scratch"
    report_dir = (out_root / task_dir.name / "report") if keep_report else None

    rc, payload, stderr_tail, duration = run_agent(
        task_dir=task_dir,
        submission_dir=submission_dir,
        workdir=workdir,
        system_prompt=system_prompt,
        model=model,
        max_budget_usd=max_budget_usd,
        timeout_s=timeout_s,
        extra_args=extra_args,
    )

    cost_usd: float | None = None
    if isinstance(payload, dict):
        for key in ("total_cost_usd", "cost_usd", "cost"):
            v = payload.get(key)
            if isinstance(v, (int, float)):
                cost_usd = float(v)
                break

    submission_path = submission_dir / "design.py"
    submission_exists = submission_path.is_file()

    if not submission_exists:
        return TaskOutcome(
            task_id=task_dir.name, family=family, tier=tier,
            agent_ok=False, agent_cost_usd=cost_usd,
            agent_duration_s=duration, agent_exit_code=rc,
            agent_error=(
                stderr_tail
                or ("agent exited without writing design.py "
                    f"(rc={rc})")
            ),
            submission_exists=False,
            eval_valid=None, eval_hard_gate_passed=None,
            eval_score=None, eval_failure_codes=[],
        )

    valid, gate, score, codes, eval_dur = score_submission(
        task_dir=task_dir,
        submission_dir=submission_dir,
        scratch_dir=scratch_dir,
        report_dir=report_dir,
    )
    return TaskOutcome(
        task_id=task_dir.name, family=family, tier=tier,
        agent_ok=(rc == 0),
        agent_cost_usd=cost_usd,
        agent_duration_s=duration,
        agent_exit_code=rc,
        agent_error=stderr_tail if rc != 0 else "",
        submission_exists=True,
        eval_valid=valid,
        eval_hard_gate_passed=gate,
        eval_score=score,
        eval_failure_codes=codes,
        eval_duration_s=eval_dur,
    )


# --------------------------------------------------------------------- #
# Aggregation                                                           #
# --------------------------------------------------------------------- #


def aggregate(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    n = len(outcomes)
    n_passed = sum(1 for o in outcomes if o.passed())
    by_tier: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    total_cost = 0.0
    cost_known = 0
    for o in outcomes:
        tier_bucket = by_tier.setdefault(o.tier, {"n": 0, "passed": 0})
        tier_bucket["n"] += 1
        if o.passed():
            tier_bucket["passed"] += 1
        fam_bucket = by_family.setdefault(o.family, {"n": 0, "passed": 0})
        fam_bucket["n"] += 1
        if o.passed():
            fam_bucket["passed"] += 1
        if o.agent_cost_usd is not None:
            total_cost += o.agent_cost_usd
            cost_known += 1

    def _rate(bucket: dict[str, int]) -> float:
        return bucket["passed"] / bucket["n"] if bucket["n"] else 0.0

    return {
        "version": "mech_bench.claude_eval.v1",
        "n_tasks": n,
        "n_passed": n_passed,
        "pass_rate": n_passed / n if n else 0.0,
        "total_cost_usd": total_cost,
        "tasks_with_cost_known": cost_known,
        "by_tier": {
            t: {**b, "pass_rate": _rate(b)} for t, b in by_tier.items()
        },
        "by_family": {
            f: {**b, "pass_rate": _rate(b)}
            for f, b in by_family.items()
        },
        "tasks": [asdict(o) for o in outcomes],
    }


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def _pick_task_dirs(
    tasks_root: Path,
    families: set[str] | None,
    limit: int | None,
    only: set[str] | None,
) -> list[Path]:
    selected: list[Path] = []
    for child in sorted(tasks_root.iterdir()):
        if not child.is_dir() or not (child / "task.toml").exists():
            continue
        family, _ = _read_task_meta(child)
        if only is not None and child.name not in only:
            continue
        if families is not None and family not in families:
            continue
        selected.append(child)
    if limit is not None:
        selected = selected[:limit]
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_claude_on_eval",
        description=(
            "Run Claude (headless) as the agent on every mech_bench "
            "task and emit an aggregate scorecard."
        ),
    )
    parser.add_argument("--tasks", default="tasks",
                        help="directory of generated/curated tasks")
    parser.add_argument("--report-dir", required=True,
                        help="where per-task workdirs / submissions / "
                             "reports + the aggregate JSON land")
    parser.add_argument("--model", default="sonnet",
                        help="model alias or full id "
                             "(default: 'sonnet')")
    parser.add_argument("--max-budget-usd", type=float, default=0.25,
                        help="per-task USD budget cap")
    parser.add_argument("--timeout", type=int, default=180,
                        help="per-task wall-clock timeout (seconds)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="max tasks run in parallel")
    parser.add_argument("--families", default=None,
                        help="comma-separated family allowlist")
    parser.add_argument("--only", default=None,
                        help="comma-separated task-id allowlist "
                             "(overrides --families)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of tasks run")
    parser.add_argument("--keep-report", action="store_true",
                        help="also write per-task evaluator report bundles")
    parser.add_argument(
        "--claude-arg", action="append", default=[],
        help=("extra arg(s) forwarded to `claude` (can repeat). "
              "Example: --claude-arg --effort=high"),
    )
    args = parser.parse_args(argv)

    tasks_root = (REPO_ROOT / args.tasks).resolve()
    if not tasks_root.is_dir():
        print(f"error: tasks dir {tasks_root} not found", file=sys.stderr)
        return 2

    out_root = Path(args.report_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    families = (
        set(s.strip() for s in args.families.split(",") if s.strip())
        if args.families else None
    )
    only = (
        set(s.strip() for s in args.only.split(",") if s.strip())
        if args.only else None
    )
    task_dirs = _pick_task_dirs(tasks_root, families, args.limit, only)
    if not task_dirs:
        print("error: no tasks matched the filters", file=sys.stderr)
        return 2

    system_prompt = SYSTEM_PROMPT_PATH.read_text()

    # Make sure tests' env vars don't leak into the agent or evaluator.
    for var in ("MECH_BENCH_USE_FAKE_ORACLE", "MECH_BENCH_TEST_MODE"):
        os.environ.pop(var, None)

    started = time.perf_counter()
    outcomes: list[TaskOutcome] = []
    print(
        f"[claude-eval] {len(task_dirs)} tasks; model={args.model}; "
        f"budget=${args.max_budget_usd}/task; timeout="
        f"{args.timeout}s; concurrency={args.concurrency}",
        file=sys.stderr,
    )

    def _go(td: Path) -> TaskOutcome:
        return drive_one(
            task_dir=td,
            out_root=out_root,
            system_prompt=system_prompt,
            model=args.model,
            max_budget_usd=args.max_budget_usd,
            timeout_s=args.timeout,
            extra_args=list(args.claude_arg or []),
            keep_report=args.keep_report,
        )

    if args.concurrency <= 1:
        for td in task_dirs:
            outcome = _go(td)
            outcomes.append(outcome)
            _print_progress(outcome, started)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = {pool.submit(_go, td): td for td in task_dirs}
            for fut in concurrent.futures.as_completed(futures):
                outcome = fut.result()
                outcomes.append(outcome)
                _print_progress(outcome, started)

    outcomes.sort(key=lambda o: o.task_id)
    summary = aggregate(outcomes)
    summary["wall_clock_s"] = time.perf_counter() - started
    summary_path = out_root / "claude_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[claude-eval] wrote summary -> {summary_path}", file=sys.stderr)

    compact = {
        "n_tasks": summary["n_tasks"],
        "n_passed": summary["n_passed"],
        "pass_rate": round(summary["pass_rate"], 3),
        "total_cost_usd": round(summary["total_cost_usd"], 4),
        "wall_clock_s": round(summary["wall_clock_s"], 1),
        "by_tier": {
            t: f"{b['passed']}/{b['n']}"
            for t, b in summary["by_tier"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


def _print_progress(o: TaskOutcome, started: float) -> None:
    score_str = (
        f"{o.eval_score:.2f}" if o.eval_score is not None else "—"
    )
    cost_str = (
        f"${o.agent_cost_usd:.4f}"
        if o.agent_cost_usd is not None else "$?"
    )
    mark = "PASS" if o.passed() else "FAIL"
    print(
        f"[{mark}] {o.task_id:48}  score={score_str}  "
        f"cost={cost_str}  agent={o.agent_duration_s:5.1f}s  "
        f"eval={o.eval_duration_s:4.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
