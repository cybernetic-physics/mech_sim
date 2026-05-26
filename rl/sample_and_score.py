"""Zero-shot or adapter smoke test: sample one completion per task from a
Worldlines (Tinker-API) backend, then score
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

The base-model mode is the **baseline** number. Supplying
``--model-path worldlines://...`` evaluates saved sampler weights.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from rl.mech_bench_reward import RewardResult, score_completion  # noqa: E402
from rl import chat_rollout as cr  # noqa: E402


SYSTEM_PROMPT_PATH = REPO_ROOT / "rl" / "agent_prompt_rl.md"
USER_PROMPT_TEMPLATE = """Solve task **{task_id}** from the mech_bench benchmark.

## verifier contract
{contract}

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

Use the exact required port ids and exact `params.*` keys shown in
the verifier contract or prompt.md. Do not add a `declared_` prefix
unless the contract or prompt itself uses that exact key. Legal port
kinds are only `frame`,
`revolute_joint`, and `prismatic_joint`. The task's explicit
`requirements.expected_mobility`, prompt mobility statement, and
required port kinds override the task title and tier name.
"""


_PARAM_RE = re.compile(r"params\.([A-Za-z_][A-Za-z0-9_]*)")


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


def _read_split_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _collect_param_paths(raw: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(raw, dict):
        for value in raw.values():
            found.update(_collect_param_paths(value))
    elif isinstance(raw, list):
        for value in raw:
            found.update(_collect_param_paths(value))
    elif isinstance(raw, str) and raw.startswith("params."):
        found.add(raw)
    return found


def _format_param_constraint(probe: dict[str, Any]) -> str | None:
    path = probe.get("path")
    if not isinstance(path, str) or not path.startswith("params."):
        return None
    if "expected" not in probe:
        return None
    comparator = str(probe.get("comparator", "eq"))
    op = {
        "eq": "==",
        "le": "<=",
        "lt": "<",
        "ge": ">=",
        "gt": ">",
    }.get(comparator, comparator)
    expected = probe.get("expected")
    bits = [f"{path} {op} {expected}"]
    if "tolerance_abs" in probe:
        bits.append(f"abs_tol={probe['tolerance_abs']}")
    if "tolerance_pct" in probe:
        bits.append(f"pct_tol={probe['tolerance_pct']}")
    if len(bits) > 1:
        return f"{bits[0]} ({', '.join(bits[1:])})"
    return bits[0]


def _collect_public_param_constraints(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    out: list[str] = []
    for probe in raw.get("probes", []) or []:
        if not isinstance(probe, dict):
            continue
        if probe.get("type") != "analytic_param_check":
            continue
        constraint = _format_param_constraint(probe)
        if constraint:
            out.append(constraint)
    return out


def _contract_from_task(
    prompt_md: str,
    task_toml: str,
    eval_config_toml: str = "",
) -> str:
    ports: list[str] = []
    mobility: int | None = None
    try:
        import tomllib
        blob = tomllib.loads(task_toml)
        req = blob.get("requirements", {})
        raw_ports = req.get("required_ports") or []
        if isinstance(raw_ports, list):
            ports = [str(p) for p in raw_ports]
        raw_mobility = req.get("expected_mobility")
        if raw_mobility is not None:
            mobility = int(raw_mobility)
    except Exception:  # noqa: BLE001 - prompt helper must stay best effort
        pass

    params = {f"params.{m}" for m in _PARAM_RE.findall(prompt_md)}
    constraints: list[str] = []
    if eval_config_toml:
        try:
            import tomllib
            eval_blob = tomllib.loads(eval_config_toml)
            params.update(_collect_param_paths(eval_blob))
            constraints = _collect_public_param_constraints(eval_blob)
        except Exception:  # noqa: BLE001
            pass
    params = sorted(params)
    lines: list[str] = []
    if mobility is not None:
        lines.append(f"- expected_mobility: {mobility}")
    if ports:
        lines.append("- required_ports: " + ", ".join(f"`{p}`" for p in ports))
    if params:
        lines.append("- required_params: " + ", ".join(f"`{p}`" for p in params))
    for constraint in constraints:
        lines.append(f"- public_param_constraint: `{constraint}`")
    if mobility == 0:
        lines.append(
            "- static topology: use one fixed carrier part or fixed joints "
            "only; do not add revolute/prismatic joints unless the prompt "
            "explicitly requires them"
        )
    if mobility == 1:
        lines.append(
            "- mobility-1 topology: create exactly the required moving joint "
            "and point joint-kind ports at the joint id"
        )
    return "\n".join(lines) if lines else "- no extracted contract"


def _build_user_prompt(task_dir: Path) -> str:
    prompt_md = (task_dir / "prompt.md").read_text()
    task_toml = (task_dir / "task.toml").read_text()
    eval_config = ""
    eval_config_path = task_dir / "eval_config.public.toml"
    if not eval_config_path.exists():
        eval_config_path = task_dir / "eval_config.toml"
    if eval_config_path.exists():
        eval_config = eval_config_path.read_text()
    return USER_PROMPT_TEMPLATE.format(
        task_id=task_dir.name,
        contract=_contract_from_task(prompt_md, task_toml, eval_config),
        prompt_md=prompt_md,
        task_toml=task_toml,
    )


_WLD_CACHE: dict[tuple[str, str, str | None, str], tuple[Any, Any, Any]] = {}
_WLD_CACHE_LOCK = threading.Lock()


def _get_clients(
    base_url: str,
    api_key: str,
    base_model: str,
    model_path: str | None = None,
):
    """Cache (ServiceClient, SamplingClient, tokenizer) per base_model."""
    cache_key = (base_url, api_key, model_path, base_model)
    with _WLD_CACHE_LOCK:
        if cache_key in _WLD_CACHE:
            return _WLD_CACHE[cache_key]
        try:
            from worldlines import ServiceClient  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "`worldlines` SDK not importable. Run inside the "
                "worldlines venv (/dev/shm/wld-venv/bin/python)."
            ) from e
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        os.environ["WORLDLINES_BASE_URL"] = base_url
        os.environ["WORLDLINES_API_KEY"] = api_key
        service = ServiceClient()
        if model_path:
            sampling = service.create_sampling_client(model_path=model_path)
        else:
            sampling = service.create_sampling_client(base_model=base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        _WLD_CACHE[cache_key] = (service, sampling, tokenizer)
        return _WLD_CACHE[cache_key]


def sample_from_worldlines(
    *,
    base_url: str,
    api_key: str,
    base_model: str,
    model_path: str | None,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1536,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int | None = None,
    timeout_s: float = 180.0,
) -> tuple[str, dict[str, int]]:
    """Sample a single completion via the Tinker-shaped SamplingClient.

    Builds a chat-templated prompt (Qwen3 chat template), tokenizes,
    calls ``sampling.sample(num_samples=1, max_tokens, temperature)``,
    and decodes the first sample.
    """
    from worldlines import types as wld_types  # type: ignore[import-not-found]

    _, sampling, tokenizer = _get_clients(
        base_url, api_key, base_model, model_path,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt = wld_types.ModelInput.from_ints(prompt_ids)
    params = wld_types.SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=seed,
    )
    future = sampling.sample(
        prompt=prompt, sampling_params=params, num_samples=1,
    )
    rsp = future.result(timeout=timeout_s)
    sample = rsp.sequences[0]
    completion_ids = list(sample.tokens)
    text = tokenizer.decode(
        completion_ids, skip_special_tokens=True)
    usage = {
        "input_tokens": len(prompt_ids),
        "output_tokens": len(completion_ids),
    }
    return str(text), usage


# --------------------------------------------------------------------- #
# Per-task driver                                                       #
# --------------------------------------------------------------------- #


@dataclass
class SampleOutcome:
    task_id: str
    family: str
    tier: str
    sample_idx: int
    sample_duration_s: float
    sample_tokens_in: int
    sample_tokens_out: int
    completion_chars: int
    reward: RewardResult | None
    pass_threshold: float = 1.0
    error: str = ""

    def passed(self) -> bool:
        return bool(
            self.reward
            and self.reward.verified_score >= self.pass_threshold
        )

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "family": self.family,
            "tier": self.tier,
            "sample_idx": self.sample_idx,
            "sample_duration_s": self.sample_duration_s,
            "sample_tokens_in": self.sample_tokens_in,
            "sample_tokens_out": self.sample_tokens_out,
            "completion_chars": self.completion_chars,
            "strict_pass_threshold": self.pass_threshold,
            "strict_passed": self.passed(),
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
    model_path: str | None,
    sglang_lora_path: str | None,
    rollout_backend: str,
    system_prompt: str,
    out_root: Path,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout_s: float,
    pass_threshold: float,
    max_turns: int = 1,
    sample_idx: int = 0,
) -> SampleOutcome:
    family, tier = _read_task_meta(task_dir)
    user_prompt = _build_user_prompt(task_dir)
    t0 = time.perf_counter()
    try:
        if rollout_backend == "sglang_chat":
            if model_path and not sglang_lora_path:
                raise ValueError(
                    "--model-path requires --rollout-backend "
                    "worldlines_sampling unless --sglang-lora-path "
                    "names a loaded SGLang LoRA adapter"
                )
            task = SimpleNamespace(
                task_id=task_dir.name,
                prompt=(task_dir / "prompt.md").read_text(),
                task_toml=(task_dir / "task.toml").read_text(),
                task_dir=task_dir,
            )
            rollout = cr.run_rollout(
                base_url=base_url,
                model=base_model,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_turns=max_turns,
                max_tokens_per_turn=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout_s=timeout_s,
                parse_bonus=0.0,
                seed=seed,
                lora_path=sglang_lora_path,
            )
            text = (
                rollout.turns[-1].assistant_text
                if rollout.turns else ""
            )
            usage = {
                "input_tokens": rollout.total_tokens_in,
                "output_tokens": rollout.total_tokens_out,
            }
        elif max_turns <= 1:
            text, usage = sample_from_worldlines(
                base_url=base_url, api_key=api_key,
                base_model=base_model,
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                timeout_s=timeout_s,
            )
        else:
            _, sampling, tokenizer = _get_clients(
                base_url, api_key, base_model, model_path,
            )
            task = SimpleNamespace(
                task_id=task_dir.name,
                prompt=(task_dir / "prompt.md").read_text(),
                task_toml=(task_dir / "task.toml").read_text(),
                task_dir=task_dir,
            )
            rollout = cr.run_rollout_with_sampling_client(
                sampling_client=sampling,
                tokenizer=tokenizer,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_turns=max_turns,
                max_tokens_per_turn=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout_s=timeout_s,
                parse_bonus=0.0,
                seed=seed,
            )
            text = (
                rollout.turns[-1].assistant_text
                if rollout.turns else ""
            )
            usage = {
                "input_tokens": rollout.total_tokens_in,
                "output_tokens": rollout.total_tokens_out,
            }
    except Exception as e:  # noqa: BLE001 — driver firewall
        return SampleOutcome(
            task_id=task_dir.name, family=family, tier=tier,
            sample_idx=sample_idx,
            sample_duration_s=time.perf_counter() - t0,
            sample_tokens_in=0, sample_tokens_out=0,
            completion_chars=0, reward=None,
            pass_threshold=pass_threshold,
            error=f"{type(e).__name__}: {e}"[:400],
        )
    dur = time.perf_counter() - t0

    if text.startswith("[sampler_error:"):
        return SampleOutcome(
            task_id=task_dir.name, family=family, tier=tier,
            sample_idx=sample_idx,
            sample_duration_s=dur,
            sample_tokens_in=int(usage.get("input_tokens", 0) or 0),
            sample_tokens_out=int(usage.get("output_tokens", 0) or 0),
            completion_chars=len(text),
            reward=None,
            pass_threshold=pass_threshold,
            error=text[:400],
        )

    per_task = out_root / task_dir.name
    per_task.mkdir(parents=True, exist_ok=True)
    (per_task / "completion.txt").write_text(text)
    reward = score_completion(
        text, task_dir, scratch_root=per_task)
    return SampleOutcome(
        task_id=task_dir.name, family=family, tier=tier,
        sample_idx=sample_idx,
        sample_duration_s=dur,
        sample_tokens_in=int(usage.get("input_tokens", 0) or 0),
        sample_tokens_out=int(usage.get("output_tokens", 0) or 0),
        completion_chars=len(text),
        reward=reward,
        pass_threshold=pass_threshold,
    )


def _retry_max_tokens_after_context_error(
    error: str,
    current_max_tokens: int,
) -> int | None:
    m = re.search(
        r"maximum context length of (\\d+).*?"
        r"(\\d+) tokens from the input messages and "
        r"(\\d+) tokens for the completion",
        error,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    context_limit = int(m.group(1))
    input_tokens = int(m.group(2))
    requested_completion = int(m.group(3))
    allowed = max(256, context_limit - input_tokens - 64)
    return min(current_max_tokens - 128, requested_completion - 128, allowed)


def _is_retryable_sampler_error(o: SampleOutcome) -> bool:
    if not o.error:
        return False
    retryable_bits = (
        "[sampler_error:",
        "RequestFailedError",
        "timed out",
        "maximum context length",
    )
    return any(bit in o.error for bit in retryable_bits)


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sample_and_score")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default="wld-local")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-path", default=None,
                   help="optional worldlines:// sampler weights path")
    p.add_argument("--sglang-lora-path", default=None,
                   help="optional loaded SGLang LoRA adapter name/path for "
                        "adapter-aware sglang_chat rollouts")
    p.add_argument("--rollout-backend", default="worldlines_sampling",
                   choices=["worldlines_sampling", "sglang_chat"],
                   help="sampling path. sglang_chat uses an "
                        "OpenAI-compatible /v1/chat/completions endpoint "
                        "and cannot load worldlines:// adapter checkpoints")
    p.add_argument("--tasks", default="tasks")
    p.add_argument("--report-dir", required=True)
    p.add_argument("--system-prompt-file", default=str(SYSTEM_PROMPT_PATH))
    p.add_argument("--samples-per-task", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--families", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--split-file", default=None,
                   help="optional newline-delimited task_id allowlist")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--max-turns", type=int, default=1,
                   help="assistant turns per sample; >1 enables verifier "
                        "feedback between turns")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--sampler-retries", type=int, default=2,
                   help="retry transport/context sampler errors before "
                        "recording a failed sample")
    p.add_argument("--pass-threshold", type=float, default=1.0,
                   help="verified_score threshold for PASS and best-of-K; "
                        "1.0 requires all scored probes to pass")
    args = p.parse_args(argv)

    tasks_root = (REPO_ROOT / args.tasks).resolve()
    out_root = Path(args.report_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    system_prompt = Path(args.system_prompt_file).read_text()

    only = (
        set(s.strip() for s in args.only.split(",") if s.strip())
        if args.only else None
    )
    split = _read_split_file(
        Path(args.split_file).resolve() if args.split_file else None
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
        if split and child.name not in split:
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
        f"each; turns={args.max_turns}; model={args.model_path or args.base_model}; "
        f"concurrency={args.concurrency}",
        file=sys.stderr,
    )

    def _go(td: Path, task_idx: int) -> list[SampleOutcome]:
        outs: list[SampleOutcome] = []
        for k in range(args.samples_per_task):
            out_root_k = out_root / f"sample_{k}"
            out_root_k.mkdir(parents=True, exist_ok=True)
            run_max_tokens = args.max_tokens
            o = None
            for attempt in range(args.sampler_retries + 1):
                o = run_one(
                    td,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    base_model=args.base_model,
                    model_path=args.model_path,
                    sglang_lora_path=args.sglang_lora_path,
                    rollout_backend=args.rollout_backend,
                    system_prompt=system_prompt,
                    out_root=out_root_k,
                    max_tokens=run_max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed + task_idx * 1000 + k,
                    timeout_s=args.timeout,
                    pass_threshold=args.pass_threshold,
                    max_turns=args.max_turns,
                    sample_idx=k,
                )
                if not _is_retryable_sampler_error(o):
                    break
                if attempt >= args.sampler_retries:
                    break
                retry_max = _retry_max_tokens_after_context_error(
                    o.error,
                    run_max_tokens,
                )
                if retry_max is not None and retry_max >= 256:
                    run_max_tokens = retry_max
                print(
                    f"[RETRY] {td.name:48} k={k} attempt={attempt + 1} "
                    f"max_tokens={run_max_tokens} err={o.error[:80]}",
                    file=sys.stderr,
                )
            assert o is not None
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
        for i, td in enumerate(task_dirs):
            all_outcomes.extend(_go(td, i))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [
                pool.submit(_go, td, i)
                for i, td in enumerate(task_dirs)
            ]
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
        "model_path": args.model_path,
        "sglang_lora_path": args.sglang_lora_path,
        "rollout_backend": args.rollout_backend,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "sampler_retries": args.sampler_retries,
        "tasks_root": str(args.tasks),
        "split_file": str(args.split_file) if args.split_file else None,
        "n_tasks": len(by_task),
        "samples_per_task": args.samples_per_task,
        "max_turns": args.max_turns,
        "pass_threshold": args.pass_threshold,
        "n_passed_best_of_k": n_passed,
        "pass_rate_best_of_k": (
            n_passed / len(by_task) if by_task else 0.0
        ),
        "n_samples": len(all_outcomes),
        "n_passed_raw": sum(1 for o in all_outcomes if o.passed()),
        "pass_rate_raw": (
            sum(1 for o in all_outcomes if o.passed()) / len(all_outcomes)
            if all_outcomes else 0.0
        ),
        "wall_clock_s": time.perf_counter() - started,
        "tasks": best,
        "all_samples": [o.to_dict() for o in all_outcomes],
    }
    out_path = out_root / "smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out_path}", file=sys.stderr)
    print(json.dumps({
        "model": summary["model_path"] or summary["model"],
        "n_tasks": summary["n_tasks"],
        "samples_per_task": summary["samples_per_task"],
        "n_passed_best_of_k": summary["n_passed_best_of_k"],
        "pass_rate_best_of_k": round(
            summary["pass_rate_best_of_k"], 3),
        "n_passed_raw": summary["n_passed_raw"],
        "pass_rate_raw": round(summary["pass_rate_raw"], 3),
        "wall_clock_s": round(summary["wall_clock_s"], 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
