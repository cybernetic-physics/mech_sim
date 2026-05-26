#!/usr/bin/env python3
"""Sample MechanicalEvolve actuator candidates with MLX-LM.

The script loads an open MLX model once, optionally applies a LoRA adapter, and
writes JSONL proposals consumable by ``scripts/mechanical_evolve.py``. Raw model
generations are saved separately so invalid JSON is auditable instead of hidden.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mechanical_evolve as mech  # noqa: E402


SCHEMA = "mech_bench.mechanical_evolve.mlx_sampler.v1"


@dataclass(frozen=True)
class SampleConfig:
    model: str
    adapter_path: str | None
    count: int
    batch_size: int
    max_tokens: int
    temp: float
    top_p: float
    seed: int
    method: str
    proposer: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--raw-json", default=None)
    parser.add_argument("--archive", default=None)
    parser.add_argument("--model", default=mech.DEFAULT_LORA_MODEL)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--method", default=None)
    parser.add_argument("--proposer", default=None)
    args = parser.parse_args()

    adapter_path = (
        str(Path(args.adapter_path).expanduser().resolve())
        if args.adapter_path else None
    )
    method = args.method or (
        "mechanical_evolve_ttrl" if adapter_path else "llm_zero_shot"
    )
    proposer = args.proposer or (
        "mlx_lora_policy" if adapter_path else "mlx_base_policy"
    )
    config = SampleConfig(
        model=str(args.model),
        adapter_path=adapter_path,
        count=max(1, int(args.count)),
        batch_size=max(1, int(args.batch_size)),
        max_tokens=max(32, int(args.max_tokens)),
        temp=max(0.0, float(args.temp)),
        top_p=max(0.0, min(1.0, float(args.top_p))),
        seed=int(args.seed),
        method=method,
        proposer=proposer,
    )
    archive = read_json(Path(args.archive).expanduser().resolve()) if args.archive else {}
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    raw_json = (
        Path(args.raw_json).expanduser().resolve()
        if args.raw_json else out_jsonl.with_suffix(".raw.json")
    )
    result = sample_candidates(config, archive=archive)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for proposal in result["proposals"]:
            f.write(json.dumps(proposal, sort_keys=True) + "\n")
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": SCHEMA,
        "out_jsonl": str(out_jsonl),
        "raw_json": str(raw_json),
        "model": config.model,
        "adapter_path": config.adapter_path,
        "requested": config.count,
        "valid_proposals": len(result["proposals"]),
        "invalid_generations": len(result["invalid_generations"]),
    }, indent=2, sort_keys=True))
    return 0 if result["proposals"] else 2


def sample_candidates(
    config: SampleConfig,
    *,
    archive: dict[str, Any],
) -> dict[str, Any]:
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.generate import make_sampler
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency
        raise SystemExit(f"mlx-lm is required for sampling: {exc}") from exc

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    mx.random.seed(config.seed)
    model, tokenizer = load(
        config.model,
        tokenizer_config={"trust_remote_code": True},
        adapter_path=config.adapter_path,
    )
    proposals: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    raw_generations: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    call_idx = 0
    max_calls = max(4, math.ceil(config.count / config.batch_size) * 4)
    while len(proposals) < config.count and call_idx < max_calls:
        want = min(config.batch_size, config.count - len(proposals))
        prompt = build_prompt(
            archive=archive,
            count=want,
            seed=config.seed + call_idx,
            adapter_active=config.adapter_path is not None,
            already_tried=sorted(seen),
        )
        tokens = chat_tokens(tokenizer, prompt)
        text = generate(
            model,
            tokenizer,
            tokens,
            verbose=False,
            max_tokens=config.max_tokens,
            sampler=make_sampler(temp=config.temp, top_p=config.top_p),
        )
        parsed = parse_generation(
            text,
            method=config.method,
            proposer=config.proposer,
            id_prefix=f"mlx_{call_idx:03d}",
            prompt=prompt,
        )
        accepted = []
        for proposal in parsed:
            key = candidate_key(proposal.get("params", {}))
            if key in seen:
                continue
            seen.add(key)
            proposal["id"] = f"{config.method}_{len(proposals):03d}"
            proposal["sample_call"] = call_idx
            proposals.append(proposal)
            accepted.append(proposal["id"])
            if len(proposals) >= config.count:
                break
        if not accepted:
            invalid.append({
                "call": call_idx,
                "raw_text": text,
                "reason": "no_new_valid_candidate_json",
            })
        raw_generations.append({
            "call": call_idx,
            "prompt": prompt,
            "raw_text": text,
            "accepted_ids": accepted,
        })
        call_idx += 1
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.__dict__,
        "proposals": proposals,
        "invalid_generations": invalid,
        "raw_generations": raw_generations,
    }


def build_prompt(
    *,
    archive: dict[str, Any],
    count: int,
    seed: int,
    adapter_active: bool,
    already_tried: list[tuple[Any, ...]] | None = None,
) -> str:
    elites = compact_elites(archive)
    defects = defect_tags(archive)
    mode = "adapted LoRA policy" if adapter_active else "base zero-shot policy"
    return "\n".join([
        "You are proposing cycloidal/QDD actuator designs for CAD+Chrono verification.",
        f"Policy mode: {mode}.",
        f"Return exactly {count} candidates as JSON only.",
        "Schema: {\"candidates\":[{\"params\":{\"pins\":11,\"eccentricity\":1.982,\"clearance\":0.336,\"driver_circle_diameter\":49.5,\"driver_pin_collision_shrink_mm\":0.129},\"notes\":\"short rationale\"}]}",
        "Hard bounds: pins integer 8..14; eccentricity 1.5..3.0; clearance 0.25..1.15; driver_circle_diameter 36..58; driver_pin_collision_shrink_mm 0..0.82.",
        (
            "Verifier gate: fallback=false, Chrono SMC, out_omega_med>=0.5, "
            "finite ratio, ratio_error_pct<=25, max_penetration_mm<1.0, "
            "contact_force_rms_N<=3000, n_contacts_max<=128, lockup=false; "
            "report power_balance_error_pct and torque_ripple_pct as "
            "physical quality metrics."
        ),
        "Prefer variants near verified elites, but do not copy any elite exactly.",
        "Each candidate must differ from every elite and already-tried tuple in at least one numeric variable.",
        "Verified elites: " + json.dumps(elites, sort_keys=True, separators=(",", ":")),
        "Already tried tuples [pins,eccentricity,clearance,driver_circle_diameter,shrink]: "
        + json.dumps(already_tried or [], sort_keys=True, separators=(",", ":")),
        "Defects to avoid: " + json.dumps(defects, sort_keys=True, separators=(",", ":")),
        f"Sampling seed: {seed}.",
        "Output JSON only, no markdown, no explanation, no <think> block.",
    ])


def chat_tokens(tokenizer: Any, prompt: str) -> Any:
    messages = [
        {
            "role": "system",
            "content": (
                "You output strict JSON for mechanical actuator search. "
                "Do not include hidden reasoning, markdown, or prose."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
    except Exception:  # noqa: BLE001 - tokenizer compatibility fallback
        return prompt


def parse_generation(
    text: str,
    *,
    method: str,
    proposer: str,
    id_prefix: str,
    prompt: str,
) -> list[dict[str, Any]]:
    for raw in json_candidates(text):
        proposals = mech.parse_model_payload(
            raw,
            method=method,
            id_prefix=id_prefix,
            proposer=proposer,
        )
        if proposals:
            out = []
            for proposal in proposals:
                row = proposal.to_dict()
                row["prompt"] = prompt
                out.append(row)
            return out
    return []


def json_candidates(text: str) -> list[Any]:
    chunks: list[str] = []
    chunks.append(text.strip())
    chunks.extend(match.group(1).strip() for match in re.finditer(
        r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE))
    decoder = json.JSONDecoder()
    parsed: list[Any] = []
    for chunk in chunks:
        try:
            parsed.append(json.loads(chunk))
            continue
        except json.JSONDecodeError:
            pass
        for idx, char in enumerate(chunk):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(chunk[idx:])
            except json.JSONDecodeError:
                continue
            parsed.append(value)
            break
    return parsed


def compact_elites(archive: dict[str, Any]) -> list[dict[str, Any]]:
    cells = archive.get("cells", {}) if isinstance(archive, dict) else {}
    rows = [row for row in cells.values() if isinstance(row, dict)]
    rows.sort(
        key=lambda row: (
            float(row.get("verified_reward", 0.0) or 0.0),
            float(row.get("fast_reward", 0.0) or 0.0),
        ),
        reverse=True,
    )
    out = []
    for row in rows[:5]:
        metrics = row.get("metrics") or {}
        out.append({
            "params": compact_params(row.get("params", {})),
            "verified_reward": row.get("verified_reward"),
            "defects": row.get("defects", []),
            "ratio_error_pct": metrics.get("ratio_error_pct"),
            "out_omega_med": metrics.get("out_omega_med"),
        })
    return out


def compact_params(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in mech.DESIGN_VARIABLES:
        if key in params:
            out[key] = params[key]
    return out


def defect_tags(archive: dict[str, Any]) -> list[str]:
    cells = archive.get("cells", {}) if isinstance(archive, dict) else {}
    counts: dict[str, int] = {}
    for row in cells.values():
        if not isinstance(row, dict):
            continue
        for defect in row.get("defects", []):
            counts[str(defect)] = counts.get(str(defect), 0) + 1
    return [
        key for key, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ][:10]


def candidate_key(params: dict[str, Any]) -> tuple[Any, ...]:
    clean = mech.normalize_params(params)
    return (
        clean["pins"],
        clean["eccentricity"],
        clean["clearance"],
        clean["driver_circle_diameter"],
        clean["driver_pin_collision_shrink_mm"],
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    raise SystemExit(main())
