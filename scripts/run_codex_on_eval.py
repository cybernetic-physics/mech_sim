#!/usr/bin/env python3
"""Run OpenAI Codex CLI against the mech_bench suite (symmetric to
`run_claude_on_eval.py`).

Per task:
    1. Materialize a per-task scratch dir with an empty submission
       folder and a sandboxed workdir.
    2. Spawn ``codex exec --json`` with the shared system prompt
       (see ``scripts/agent_system_prompt.md``) prepended, the per-
       task user prompt fed on stdin, ``--sandbox workspace-write``,
       ``--cd <workdir>``, ``--add-dir <task>`` and ``--add-dir
       <submission>``, ``--skip-git-repo-check``, and ``--ephemeral``.
    3. Parse the JSONL event stream for token usage; estimate cost
       from a per-model rate table.
    4. Run ``python -m mech_bench evaluate`` against the produced
       ``design.py`` and parse the verified score, hard-gate state,
       and failure codes.
    5. Aggregate into a scorecard at
       ``<report-dir>/codex_eval_summary.json`` — the same shape the
       Claude harness emits so the two can be diffed mechanically.

Auth: relies on the user's existing Codex login (``codex login`` or
``OPENAI_API_KEY`` in env).
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
a DesignIR (schema_version "design_ir.v2"). See the system prompt
for the schema. When the file is written, stop.
"""


# Rough per-million-token USD pricing for cost estimation only — the
# real billing is whatever OpenAI charges; this is a useful proxy
# for cross-task / cross-agent comparisons. Update when prices move.
# Keyed by lowercase model id substrings (longest match wins).
_PRICE_TABLE: dict[str, dict[str, float]] = {
    "gpt-5.5":       {"in":  1.25, "cached_in": 0.125, "out": 10.0},
    "gpt-5-codex":   {"in":  1.25, "cached_in": 0.125, "out": 10.0},
    "gpt-5-mini":    {"in":  0.25, "cached_in": 0.025, "out":  2.0},
    "gpt-5":         {"in":  1.25, "cached_in": 0.125, "out": 10.0},
    "gpt-4.1-mini":  {"in":  0.40, "cached_in": 0.10,  "out":  1.60},
    "gpt-4.1":       {"in":  2.50, "cached_in": 0.50,  "out": 10.0},
    "o4-mini":       {"in":  1.10, "cached_in": 0.275, "out":  4.40},
    "o3":            {"in": 10.0,  "cached_in": 2.50,  "out": 40.0},
}


def _estimate_cost_usd(model: str, usage: dict[str, int]) -> float:
    key = (model or "").lower()
    rates: dict[str, float] | None = None
    for k in sorted(_PRICE_TABLE, key=len, reverse=True):
        if k in key:
            rates = _PRICE_TABLE[k]
            break
    if rates is None:
        # Unknown model — fall back to the gpt-5 rate so the number
        # is at least order-of-magnitude correct.
        rates = _PRICE_TABLE["gpt-5"]
    inp = int(usage.get("input_tokens", 0) or 0)
    cached = int(usage.get("cached_input_tokens", 0) or 0)
    out = (
        int(usage.get("output_tokens", 0) or 0)
        + int(usage.get("reasoning_output_tokens", 0) or 0)
    )
    fresh_in = max(inp - cached, 0)
    return (
        fresh_in * rates["in"] / 1e6
        + cached * rates["cached_in"] / 1e6
        + out * rates["out"] / 1e6
    )


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
    usage: dict[str, int] = field(default_factory=dict)

    def passed(self) -> bool:
        return bool(
            self.eval_valid
            and self.eval_hard_gate_passed
            and (self.eval_score or 0.0) > 0.0
        )


# --------------------------------------------------------------------- #
# Step 1 — run the agent                                                #
# --------------------------------------------------------------------- #


def _parse_codex_jsonl(stdout: str) -> tuple[dict[str, int], list[str]]:
    """Walk the JSONL event stream and return (usage, errors)."""
    usage_total: dict[str, int] = {}
    errors: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "turn.completed":
            u = obj.get("usage") or {}
            for k, v in u.items():
                if isinstance(v, (int, float)):
                    usage_total[k] = usage_total.get(k, 0) + int(v)
        elif t == "turn.failed" or t == "error":
            err = obj.get("error") or obj.get("message") or str(obj)
            errors.append(str(err)[:400])
    return usage_total, errors


def run_agent(
    *,
    task_dir: Path,
    submission_dir: Path,
    workdir: Path,
    system_prompt: str,
    model: str,
    timeout_s: int,
    sandbox_mode: str,
    extra_args: list[str],
) -> tuple[int, str, str, float]:
    """Invoke `codex exec --json` and return
    (exit_code, stdout_jsonl, stderr_tail, duration_s)."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        task_id=task_dir.name,
        task_dir=str(task_dir),
        submission_dir=str(submission_dir),
    )
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

    cmd = [
        "codex", "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color", "never",
        "--cd", str(workdir),
        "--add-dir", str(task_dir),
        "--add-dir", str(submission_dir),
        "--sandbox", sandbox_mode,
    ]
    # Empty `--model ""` means "use whatever the account's codex
    # default is" — important because ChatGPT-tier accounts can't
    # always select arbitrary OpenAI model ids.
    if model:
        cmd += ["--model", model]
    cmd += list(extra_args)
    # Prompt comes from stdin so it survives multi-line content
    # without shell-quoting headaches.
    cmd += ["-"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timed out after {timeout_s}s: {e}", time.perf_counter() - t0
    except FileNotFoundError:
        return 127, "", "`codex` not on PATH", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    return proc.returncode, proc.stdout or "", (proc.stderr or "")[-2000:], elapsed


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
    meta_path = task_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            return str(meta.get("family", task_dir.name)), str(
                meta.get("tier", "unknown"))
        except (OSError, json.JSONDecodeError):
            pass
    return task_dir.name, "unknown"


def drive_one(
    *,
    task_dir: Path,
    out_root: Path,
    system_prompt: str,
    model: str,
    timeout_s: int,
    sandbox_mode: str,
    extra_args: list[str],
    keep_report: bool,
) -> TaskOutcome:
    family, tier = _read_task_meta(task_dir)
    submission_dir = out_root / task_dir.name / "submission"
    workdir = out_root / task_dir.name / "workdir"
    scratch_dir = out_root / task_dir.name / "_scratch"
    report_dir = (out_root / task_dir.name / "report") if keep_report else None

    rc, stdout, stderr_tail, duration = run_agent(
        task_dir=task_dir,
        submission_dir=submission_dir,
        workdir=workdir,
        system_prompt=system_prompt,
        model=model,
        timeout_s=timeout_s,
        sandbox_mode=sandbox_mode,
        extra_args=extra_args,
    )
    usage, jsonl_errors = _parse_codex_jsonl(stdout)
    cost_usd = _estimate_cost_usd(model, usage) if usage else None
    if jsonl_errors and not stderr_tail:
        stderr_tail = "; ".join(jsonl_errors)[-2000:]

    # Persist raw stdout for debugging.
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "codex.jsonl").write_text(stdout)

    submission_path = submission_dir / "design.py"
    submission_exists = submission_path.is_file()
    if not submission_exists:
        return TaskOutcome(
            task_id=task_dir.name, family=family, tier=tier,
            agent_ok=False, agent_cost_usd=cost_usd,
            agent_duration_s=duration, agent_exit_code=rc,
            agent_error=(
                stderr_tail
                or f"agent exited without writing design.py (rc={rc})"
            ),
            submission_exists=False,
            eval_valid=None, eval_hard_gate_passed=None,
            eval_score=None, eval_failure_codes=[],
            usage=usage,
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
        usage=usage,
    )


# --------------------------------------------------------------------- #
# Aggregation                                                           #
# --------------------------------------------------------------------- #


def aggregate(outcomes: list[TaskOutcome], model: str) -> dict[str, Any]:
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

    def _rate(b: dict[str, int]) -> float:
        return b["passed"] / b["n"] if b["n"] else 0.0

    return {
        "version": "mech_bench.codex_eval.v1",
        "agent": "codex",
        "model": model,
        "n_tasks": n,
        "n_passed": n_passed,
        "pass_rate": n_passed / n if n else 0.0,
        "total_cost_usd_estimate": total_cost,
        "tasks_with_cost_estimate": cost_known,
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
        prog="run_codex_on_eval",
        description=(
            "Run OpenAI Codex (`codex exec --json`) as the agent on "
            "every mech_bench task and emit an aggregate scorecard."
        ),
    )
    parser.add_argument("--tasks", default="tasks")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--model", default="gpt-5.5",
        help=("model passed to codex (default: gpt-5.5). Pass an "
              "empty string to omit the flag and let codex pick its "
              "account default."),
    )
    parser.add_argument("--timeout", type=int, default=240,
                        help="per-task wall-clock timeout (seconds)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--families", default=None)
    parser.add_argument("--only", default=None,
                        help="comma-separated task-id allowlist")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sandbox", default="workspace-write",
                        choices=["read-only", "workspace-write",
                                 "danger-full-access"])
    parser.add_argument("--keep-report", action="store_true")
    parser.add_argument(
        "--codex-arg", action="append", default=[],
        help="extra arg(s) forwarded to `codex exec` (repeatable)")
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

    for var in ("MECH_BENCH_USE_FAKE_ORACLE", "MECH_BENCH_TEST_MODE"):
        os.environ.pop(var, None)

    started = time.perf_counter()
    outcomes: list[TaskOutcome] = []
    print(
        f"[codex-eval] {len(task_dirs)} tasks; model={args.model}; "
        f"timeout={args.timeout}s; concurrency={args.concurrency}; "
        f"sandbox={args.sandbox}",
        file=sys.stderr,
    )

    def _go(td: Path) -> TaskOutcome:
        return drive_one(
            task_dir=td,
            out_root=out_root,
            system_prompt=system_prompt,
            model=args.model,
            timeout_s=args.timeout,
            sandbox_mode=args.sandbox,
            extra_args=list(args.codex_arg or []),
            keep_report=args.keep_report,
        )

    if args.concurrency <= 1:
        for td in task_dirs:
            outcome = _go(td)
            outcomes.append(outcome)
            _print_progress(outcome)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = {pool.submit(_go, td): td for td in task_dirs}
            for fut in concurrent.futures.as_completed(futures):
                outcome = fut.result()
                outcomes.append(outcome)
                _print_progress(outcome)

    outcomes.sort(key=lambda o: o.task_id)
    summary = aggregate(outcomes, args.model)
    summary["wall_clock_s"] = time.perf_counter() - started
    summary_path = out_root / "codex_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[codex-eval] wrote summary -> {summary_path}", file=sys.stderr)

    compact = {
        "agent": "codex",
        "model": args.model,
        "n_tasks": summary["n_tasks"],
        "n_passed": summary["n_passed"],
        "pass_rate": round(summary["pass_rate"], 3),
        "total_cost_usd_estimate": round(
            summary["total_cost_usd_estimate"], 4),
        "wall_clock_s": round(summary["wall_clock_s"], 1),
        "by_tier": {
            t: f"{b['passed']}/{b['n']}"
            for t, b in summary["by_tier"].items()
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


def _print_progress(o: TaskOutcome) -> None:
    score_str = f"{o.eval_score:.2f}" if o.eval_score is not None else "—"
    cost_str = (
        f"${o.agent_cost_usd:.4f}"
        if o.agent_cost_usd is not None else "$?"
    )
    mark = "PASS" if o.passed() else "FAIL"
    print(
        f"[{mark}] {o.task_id:48}  score={score_str}  "
        f"cost~={cost_str}  agent={o.agent_duration_s:5.1f}s  "
        f"eval={o.eval_duration_s:4.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
