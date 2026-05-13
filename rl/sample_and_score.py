"""Zero-shot smoke test: sample one completion per task from a
Worldlines (Tinker-API) backend running Qwen3-0.6B, then score
each via `mech_bench.evaluate`. Writes a scorecard in the same
shape `run_claude_on_eval.py` emits so it can be diff'd against
the agent runs.

Usage::

    python rl/sample_and_score.py \\
        --base-url http://127.0.0.1:8000 \\
        --api-key wld-local \\
        --base-model Qwen/Qwen3-0.6B \\
        --tasks tasks \\
        --report-dir /tmp/qwen3_smoke \\
        --samples-per-task 1 \\
        --concurrency 4

This is the **baseline** number — the zero-shot pass rate any
training run needs to beat.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from rl.mech_bench_reward import RewardResult, score_completion  # noqa: E402


SYSTEM_PROMPT_PATH = REPO_ROOT / "scripts" / "agent_system_prompt.md"
USER_PROMPT_TEMPLATE = """Solve task **{task_id}** from the mech_bench benchmark.

## prompt.md
{prompt_md}

## task.toml
```toml
{task_toml}
```

Emit one Python file named `design.py` that defines
`build_design(out_dir: Path) -> dict`. Wrap the full file in a
single fenced ```python ... ``` block. Do not include any other
prose outside the block.
"""


# --------------------------------------------------------------------- #
# Worldlines / Tinker client                                            #
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


def _build_user_prompt(task_dir: Path) -> str:
    prompt_md = (task_dir / "prompt.md").read_text()
    task_toml = (task_dir / "task.toml").read_text()
    return USER_PROMPT_TEMPLATE.format(
        task_id=task_dir.name,
        prompt_md=prompt_md,
        task_toml=task_toml,
    )


def sample_from_worldlines(
    *,
    base_url: str,
    api_key: str,
    base_model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1536,
    temperature: float = 0.7,
    timeout_s: float = 180.0,
) -> tuple[str, dict[str, int]]:
    """Sample a single completion via the Tinker-shaped SamplingClient.

    The Worldlines client speaks the same wire protocol as Tinker —
    we use its public ``sample`` entrypoint so the trainer can
    later swap backends transparently.
    """
    try:
        from worldlines.lib.public_interfaces import (
            sampling_client as sc_mod,
        )
    except ImportError as e:
        raise RuntimeError(
            "`worldlines` SDK is not importable. Activate the "
            "worldlines venv (`/dev/shm/wld-venv`) before running."
        ) from e

    os.environ.setdefault("WORLDLINES_BASE_URL", base_url)
    os.environ.setdefault("WORLDLINES_API_KEY", api_key)

    client = sc_mod.SamplingClient(  # type: ignore[attr-defined]
        base_model=base_model,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rsp = client.sample(  # type: ignore[attr-defined]
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )
    text = getattr(rsp, "text", None) or getattr(rsp, "completion", "")
    usage = getattr(rsp, "usage", {}) or {}
    return str(text), dict(usage)


# --------------------------------------------------------------------- #
# Per-task driver                                                       #
# --------------------------------------------------------------------- #


@dataclass
class SampleOutcome:
    task_id: str
    family: str
    tier: str
    sample_duration_s: float
    sample_tokens_in: int
    sample_tokens_out: int
    completion_chars: int
    reward: RewardResult | None
    error: str = ""

    def passed(self) -> bool:
        return bool(self.reward and self.reward.verified_score > 0.0)

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "family": self.family,
            "tier": self.tier,
            "sample_duration_s": self.sample_duration_s,
            "sample_tokens_in": self.sample_tokens_in,
            "sample_tokens_out": self.sample_tokens_out,
            "completion_chars": self.completion_chars,
            "error": self.error,
        }
        if self.reward is not None:
            d.update(self.reward.to_dict())
        return d


def run_one(
    task_dir: Path,
    *,
    base_url: str,
    api_key: str,
    base_model: str,
    system_prompt: str,
    out_root: Path,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> SampleOutcome:
    family, tier = _read_task_meta(task_dir)
    user_prompt = _build_user_prompt(task_dir)
    t0 = time.perf_counter()
    try:
        text, usage = sample_from_worldlines(
            base_url=base_url, api_key=api_key,
            base_model=base_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
    except Exception as e:  # noqa: BLE001 — driver firewall
        return SampleOutcome(
            task_id=task_dir.name, family=family, tier=tier,
            sample_duration_s=time.perf_counter() - t0,
            sample_tokens_in=0, sample_tokens_out=0,
            completion_chars=0, reward=None,
            error=f"{type(e).__name__}: {e}"[:400],
        )
    dur = time.perf_counter() - t0

    per_task = out_root / task_dir.name
    per_task.mkdir(parents=True, exist_ok=True)
    (per_task / "completion.txt").write_text(text)
    reward = score_completion(
        text, task_dir, scratch_root=per_task)
    return SampleOutcome(
        task_id=task_dir.name, family=family, tier=tier,
        sample_duration_s=dur,
        sample_tokens_in=int(usage.get("input_tokens", 0) or 0),
        sample_tokens_out=int(usage.get("output_tokens", 0) or 0),
        completion_chars=len(text),
        reward=reward,
    )


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sample_and_score")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default="wld-local")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--tasks", default="tasks")
    p.add_argument("--report-dir", required=True)
    p.add_argument("--samples-per-task", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--families", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args(argv)

    tasks_root = (REPO_ROOT / args.tasks).resolve()
    out_root = Path(args.report_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    system_prompt = SYSTEM_PROMPT_PATH.read_text()

    only = (
        set(s.strip() for s in args.only.split(",") if s.strip())
        if args.only else None
    )
    families = (
        set(s.strip() for s in args.families.split(",") if s.strip())
        if args.families else None
    )

    task_dirs: list[Path] = []
    for child in sorted(tasks_root.iterdir()):
        if not child.is_dir() or not (child / "task.toml").exists():
            continue
        family, _ = _read_task_meta(child)
        if only and child.name not in only:
            continue
        if families and family not in families:
            continue
        task_dirs.append(child)
    if args.limit:
        task_dirs = task_dirs[: args.limit]
    if not task_dirs:
        print("no tasks matched", file=sys.stderr)
        return 2

    print(
        f"[smoke] {len(task_dirs)} tasks × {args.samples_per_task} samples "
        f"each; model={args.base_model}; concurrency={args.concurrency}",
        file=sys.stderr,
    )

    def _go(td: Path) -> list[SampleOutcome]:
        outs: list[SampleOutcome] = []
        for k in range(args.samples_per_task):
            out_root_k = out_root / f"sample_{k}"
            out_root_k.mkdir(parents=True, exist_ok=True)
            o = run_one(
                td,
                base_url=args.base_url,
                api_key=args.api_key,
                base_model=args.base_model,
                system_prompt=system_prompt,
                out_root=out_root_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout,
            )
            outs.append(o)
            mark = "PASS" if o.passed() else "FAIL"
            score = (o.reward.verified_score
                     if o.reward is not None else 0.0)
            print(
                f"[{mark}] {o.task_id:48} k={k} score={score:.2f} "
                f"tok={o.sample_tokens_out:>4}  "
                f"sample={o.sample_duration_s:5.1f}s  "
                f"err={o.error[:60]}",
                file=sys.stderr,
            )
        return outs

    started = time.perf_counter()
    all_outcomes: list[SampleOutcome] = []
    if args.concurrency <= 1:
        for td in task_dirs:
            all_outcomes.extend(_go(td))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [pool.submit(_go, td) for td in task_dirs]
            for fut in concurrent.futures.as_completed(futures):
                all_outcomes.extend(fut.result())

    # Per-task: best-of-K reward (max verified_score across samples).
    by_task: dict[str, list[SampleOutcome]] = {}
    for o in all_outcomes:
        by_task.setdefault(o.task_id, []).append(o)
    best: list[dict] = []
    n_passed = 0
    for tid, lst in by_task.items():
        winner = max(
            lst,
            key=lambda o: (
                o.reward.verified_score if o.reward else 0.0
            ),
        )
        best.append(winner.to_dict())
        if winner.passed():
            n_passed += 1

    summary = {
        "version": "mech_bench.local_rl_smoke.v1",
        "agent": "worldlines",
        "model": args.base_model,
        "n_tasks": len(by_task),
        "samples_per_task": args.samples_per_task,
        "n_passed_best_of_k": n_passed,
        "pass_rate_best_of_k": (
            n_passed / len(by_task) if by_task else 0.0
        ),
        "wall_clock_s": time.perf_counter() - started,
        "tasks": best,
        "all_samples": [o.to_dict() for o in all_outcomes],
    }
    out_path = out_root / "smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_path}", file=sys.stderr)
    print(json.dumps({
        "model": summary["model"],
        "n_tasks": summary["n_tasks"],
        "samples_per_task": summary["samples_per_task"],
        "n_passed_best_of_k": summary["n_passed_best_of_k"],
        "pass_rate_best_of_k": round(
            summary["pass_rate_best_of_k"], 3),
        "wall_clock_s": round(summary["wall_clock_s"], 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
