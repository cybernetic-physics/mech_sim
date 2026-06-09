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
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from rl.mech_bench_reward import RewardResult, score_completion  # noqa: E402
from rl import chat_rollout as cr  # noqa: E402


SYSTEM_PROMPT_PATH = REPO_ROOT / "rl" / "agent_prompt_rl.md"
STRICT_FENCED_OUTPUT_INSTRUCTION = (
    "/no_think\n"
    "You must reply with exactly one fenced ```python code block containing "
    "the complete design.py implementation. Use compact valid Python only: "
    "no comments, no analysis, and no explanations. Return a complete "
    "build_design(out_dir: Path) implementation with all dictionaries, lists, "
    "braces, parentheses, and the final code fence closed. Do not include "
    "additional file writes, additional write_text calls, STEP contents, "
    "markdown outside that one code block, or any text before or after it. "
    "If an assistant code prefill is present, continue the already-open "
    "`parts` list after `frame(out_dir),`; do not repeat imports, helper "
    "functions, schema_version, materials, parts, or def build_design. "
    "After closing the parts list, add joints, ports, and params keys. "
    "For timing-belt and chain-sprocket analytic tasks, never create a "
    "part, joint, role, or id containing `belt` or `chain`; represent the "
    "belt/chain only by top-level params and use exactly two moving rotating "
    "parts plus the prefilled frame. "
    "Start your response with exactly ```python followed by the design.py "
    "source."
)
ASSISTANT_CODE_PREFILL = (
    "```python\n"
    "from pathlib import Path\n\n"
    "STEP='ISO-10303-21;END-ISO-10303-21;\\n'\n"
    "def cad(out_dir,n): (out_dir/n).write_text(STEP); return n\n"
    "I=((1e-4,0.0,0.0),(0.0,1e-4,0.0),(0.0,0.0,1e-4))\n"
    "def mp(m,c,s=None):\n"
    "    p={'cad_mass_properties':{'mass_kg':m,'com_local_mm':c,'inertia_kg_m2':I}}\n"
    "    if s is not None: p['chrono_collision']=s\n"
    "    return p\n\n"
    "def cyl(r,l): return {'shape':'cylinder','radius_mm':r,'length_mm':l}\n"
    "def box(x,y,z): return {'shape':'box','size_mm':(x,y,z)}\n"
    "def cm(m,c,s): return mp(m,c,s)\n\n"
    "M={'steel':{'density_kg_m3':7850.0,'provenance':'datasheet'},'aluminum':{'density_kg_m3':2700.0,'provenance':'datasheet'},'rubber':{'density_kg_m3':1100.0,'provenance':'datasheet'}}\n\n"
    "def frame(out_dir): return {'id':'frame','role':'ground','mass_kg':0.001,'fixed':True,'com_local_mm':(0.0,0.0,0.0),'geometry':{'cad':cad(out_dir,'frame.step')},'material':'steel','params':cm(0.001,(0.0,0.0,0.0),box(20,20,5))}\n\n"
    "def build_design(out_dir: Path) -> dict:\n"
    "    return {'schema_version':'design_ir.v2','units':'mm','materials':M,\n"
    "        'parts':[frame(out_dir),\n"
)
USER_PROMPT_TEMPLATE = """Solve task **{task_id}** from the mech_bench benchmark.

## verifier contract
{contract}

## prompt.md
{prompt_md}

## task.toml
```toml
{task_toml}
```

## final task-specific overrides
These override all examples above:
{contract}

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
required port kinds override the task title, tier name, and any
conflicting example in the system prompt.
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
    required_kinds: dict[str, str] = {}
    require_grounded: list[str] = []
    has_trusted_cad_preflight = False
    requires_trusted_mass = False
    uses_chrono_contact = False
    chrono_fallback_disabled = False
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
            adapters = eval_blob.get("adapters") or {}
            if isinstance(adapters, dict):
                chrono_cfg = adapters.get("chrono_contact") or {}
                if isinstance(chrono_cfg, dict):
                    uses_chrono_contact = True
                    chrono_fallback_disabled = (
                        chrono_cfg.get("procedural_cycloidal_fallback") is False
                    )
            for probe in eval_blob.get("probes", []) or []:
                if not isinstance(probe, dict):
                    continue
                if probe.get("type") == "trusted_asset_preflight":
                    has_trusted_cad_preflight = True
                    requires_trusted_mass = (
                        requires_trusted_mass
                        or probe.get("require_trusted_mass_properties") is True
                    )
                if probe.get("adapter") == "chrono_contact":
                    uses_chrono_contact = True
                if probe.get("type") != "required_ports":
                    continue
                raw_grounded = probe.get("require_grounded") or []
                if isinstance(raw_grounded, list):
                    require_grounded.extend(str(p) for p in raw_grounded)
                raw_kinds = probe.get("require_kinds") or {}
                if isinstance(raw_kinds, dict):
                    required_kinds.update(
                        {str(k): str(v) for k, v in raw_kinds.items()}
                    )
        except Exception:  # noqa: BLE001
            pass
    params = sorted(params)
    lines: list[str] = []
    if mobility is not None:
        lines.append(f"- expected_mobility: {mobility}")
    if ports:
        lines.append("- required_ports: " + ", ".join(f"`{p}`" for p in ports))
    if required_kinds:
        lines.append(
            "- required_port_kinds: "
            + ", ".join(
                f"`{port}` must be `{kind}`"
                for port, kind in sorted(required_kinds.items())
            )
        )
    if require_grounded:
        lines.append(
            "- grounded_ports: "
            + ", ".join(f"`{p}`" for p in sorted(set(require_grounded)))
        )
    if has_trusted_cad_preflight:
        lines.append(
            "- trusted_cad_preflight: include CAD geometry roles, material "
            "records, material provenance, and trusted CAD-derived mass/COM/"
            "inertia evidence for checked positive-mass parts"
        )
    if requires_trusted_mass:
        lines.append(
            "- trusted_mass_properties_required: positive-mass parts must "
            "provide `params.cad_mass_properties` evidence recomputed from "
            "trusted CAD; declared mass alone is not enough"
        )
    if uses_chrono_contact:
        lines.append(
            "- chrono_contact_required: design must support real Chrono SMC "
            "contact/dynamics verification; do not rely on fake or procedural "
            "contact outputs"
        )
    if chrono_fallback_disabled:
        lines.append(
            "- procedural_fallback_disabled: "
            "`procedural_cycloidal_fallback` is false"
        )
    if params:
        lines.append("- required_params: " + ", ".join(f"`{p}`" for p in params))
        lines.append(
            "- params rule: the top-level `params` dict must contain the "
            "required key names exactly; do not substitute similarly named "
            "keys copied from examples"
        )
    if "params.declared_travel_per_rev_mm" in params:
        lines.append(
            "- lead_screw_param_alias_warning: use "
            "`params.declared_travel_per_rev_mm`; do not use "
            "`params.declared_linear_per_rev_mm`"
        )
    slider_crank_params = {
        "params.declared_quick_return_ratio",
        "params.declared_stroke_mm",
    }
    if required_kinds.get("output_port") == "prismatic_joint" and (
        slider_crank_params & set(params)
    ):
        lines.append(
            "- slider_crank_topology_warning: this is a "
            "slider-crank, not a four-bar. Use parts `ground`, `crank`, "
            "`coupler`, `slider`; use joints `joint_input`, `joint_bc`, "
            "`joint_cs`, and `joint_slide`; `joint_slide` must have "
            "`type == \"prismatic\"` and x-axis `(1.0, 0.0, 0.0)`; "
            "`ports[\"output_port\"][\"part\"]` must be `joint_slide`; do "
            "not create `rocker`, four-bar `joint_output`, or a `G` ground "
            "length; do not use undefined four-bar variables `A`, `B`, "
            "`C`, or `G`"
        )
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


def _append_strict_fenced_instruction(content: str) -> str:
    if STRICT_FENCED_OUTPUT_INSTRUCTION in content:
        return content
    return content.rstrip() + "\n\n" + STRICT_FENCED_OUTPUT_INSTRUCTION


def _sglang_one_turn_messages(
    *,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _append_strict_fenced_instruction(user_prompt),
        },
        {"role": "assistant", "content": ASSISTANT_CODE_PREFILL},
    ]


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
        if not model_path:
            # The local PEFT runtime samples from registered model sessions.
            # A fresh managed backend has no session yet, so register a
            # zero-step LoRA session and the patched sampler disables the
            # adapter while step == 0. This gives true base-model sampling
            # without depending on a previous training phase staying alive.
            service.create_lora_training_client(
                base_model=base_model,
                rank=1,
                train_unembed=False,
            )
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
    verifier_calls: int = 0
    cad_audits: int = 0
    chrono_audits: int = 0
    audit_retry_count: int = 0
    pass_threshold: float = 1.0
    error: str = ""
    turn_traces: list[dict[str, Any]] = field(default_factory=list)

    def passed(self) -> bool:
        return bool(
            self.reward
            and self.reward.verified_score >= self.pass_threshold
        )

    def verifier_valid_passed(self) -> bool:
        return bool(
            self.reward
            and self.reward.evaluation_valid
            and self.reward.hard_gate_passed
            and not self.reward.failure_codes
        )

    def repair_attempted(self) -> bool:
        return int(self.verifier_calls or 0) > 1

    def repair_succeeded(self) -> bool:
        return self.repair_attempted() and self.verifier_valid_passed()

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
            "verifier_calls": self.verifier_calls,
            "cad_audits": self.cad_audits,
            "chrono_audits": self.chrono_audits,
            "audit_retry_count": self.audit_retry_count,
            "strict_pass_threshold": self.pass_threshold,
            "strict_passed": self.passed(),
            "verifier_valid_passed": self.verifier_valid_passed(),
            "repair_attempted": self.repair_attempted(),
            "repair_succeeded": self.repair_succeeded(),
            "error": self.error,
        }
        if self.turn_traces:
            d["turn_traces"] = self.turn_traces
        if self.reward is not None:
            reward_dict = self.reward.to_dict()
            reward_dict["cad_audits"] = self.cad_audits
            reward_dict["chrono_audits"] = self.chrono_audits
            d.update(reward_dict)
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
    verifier_calls_before_final = 0
    rollout_reward: RewardResult | None = None
    turn_traces: list[dict[str, Any]] = []
    try:
        if rollout_backend == "sglang_chat":
            if model_path and not sglang_lora_path:
                raise ValueError(
                    "--model-path requires --rollout-backend "
                    "worldlines_sampling unless --sglang-lora-path "
                    "names a loaded SGLang LoRA adapter"
                )
            if max_turns <= 1:
                resp = cr._chat_completion(
                    base_url=base_url,
                    model=base_model,
                    messages=_sglang_one_turn_messages(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    ),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_s=timeout_s,
                    seed=seed,
                    lora_path=sglang_lora_path,
                    continue_final_message=True,
                    extra_body={
                        "separate_reasoning": False,
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                            "thinking": False,
                        },
                    },
                )
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                continuation = str(msg.get("content") or "")
                text = (
                    continuation
                    if continuation.startswith(ASSISTANT_CODE_PREFILL)
                    else ASSISTANT_CODE_PREFILL + continuation
                )
                usage_raw = resp.get("usage") or {}
                usage = {
                    "input_tokens": int(usage_raw.get("prompt_tokens") or 0),
                    "output_tokens": int(
                        usage_raw.get("completion_tokens") or 0
                    ),
                }
            else:
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
                    stop_on_pass=False,
                    assistant_prefill=ASSISTANT_CODE_PREFILL,
                    strict_user_suffix=STRICT_FENCED_OUTPUT_INSTRUCTION,
                )
                text = (
                    rollout.turns[-1].assistant_text
                    if rollout.turns else ""
                )
                verifier_calls_before_final = _rollout_verifier_calls(rollout)
                rollout_reward = _reward_from_rollout_best_turn(rollout)
                turn_traces = _rollout_turn_trace_dicts(rollout)
                if rollout_reward is not None:
                    _apply_rollout_audit_totals(rollout_reward, rollout)
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
                stop_on_pass=False,
            )
            text = (
                rollout.turns[-1].assistant_text
                if rollout.turns else ""
            )
            verifier_calls_before_final = _rollout_verifier_calls(rollout)
            rollout_reward = _reward_from_rollout_best_turn(rollout)
            turn_traces = _rollout_turn_trace_dicts(rollout)
            if rollout_reward is not None:
                _apply_rollout_audit_totals(rollout_reward, rollout)
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
            verifier_calls=0,
            cad_audits=0,
            chrono_audits=0,
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
            verifier_calls=verifier_calls_before_final,
            cad_audits=0,
            chrono_audits=0,
            pass_threshold=pass_threshold,
            error=text[:400],
            turn_traces=turn_traces,
        )

    per_task = out_root / task_dir.name
    per_task.mkdir(parents=True, exist_ok=True)
    (per_task / "completion.txt").write_text(text)
    if rollout_reward is None:
        reward = score_completion(
            text, task_dir, scratch_root=per_task)
        verifier_calls = verifier_calls_before_final + 1
    else:
        reward = rollout_reward
        verifier_calls = verifier_calls_before_final
    cad_audits = int(reward.cad_audits or 0)
    chrono_audits = int(reward.chrono_audits or 0)
    return SampleOutcome(
        task_id=task_dir.name, family=family, tier=tier,
        sample_idx=sample_idx,
        sample_duration_s=dur,
        sample_tokens_in=int(usage.get("input_tokens", 0) or 0),
        sample_tokens_out=int(usage.get("output_tokens", 0) or 0),
        completion_chars=len(text),
        reward=reward,
        verifier_calls=verifier_calls,
        cad_audits=cad_audits,
        chrono_audits=chrono_audits,
        pass_threshold=pass_threshold,
        turn_traces=turn_traces,
    )


def _rollout_verifier_calls(rollout: Any) -> int:
    calls = 0
    for turn in getattr(rollout, "turns", []) or []:
        failure_codes = set(getattr(turn, "failure_codes", []) or [])
        if "sampler_error" in failure_codes:
            continue
        calls += 1
    return calls


def _rollout_scored_turns(rollout: Any) -> list[Any]:
    turns = getattr(rollout, "turns", []) or []
    return [
        turn for turn in turns
        if "sampler_error" not in (getattr(turn, "failure_codes", []) or [])
    ]


def _rollout_turn_trace_dicts(rollout: Any) -> list[dict[str, Any]]:
    return [asdict(turn) for turn in _rollout_scored_turns(rollout)]


def _append_attempt_turn_traces(
    target: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    sample_idx: int,
    audit_attempt: int,
    sampler_attempt: int,
    trace_kind: str,
) -> None:
    for retry_trace_idx, raw_trace in enumerate(traces):
        trace = dict(raw_trace)
        trace["sample_idx"] = sample_idx
        trace["audit_attempt"] = audit_attempt
        trace["sampler_attempt"] = sampler_attempt
        trace["retry_trace_idx"] = retry_trace_idx
        trace["trace_kind"] = trace_kind
        trace["verifier_call_idx_within_sample"] = len(target)
        target.append(trace)


def _append_missing_outcome_turn_traces(
    outcome: SampleOutcome,
    per_task_root: Path,
) -> None:
    expected_calls = int(outcome.verifier_calls or 0)
    traces = outcome.turn_traces
    missing = expected_calls - len(traces)
    if missing <= 0:
        return
    completion_path = per_task_root / "completion.txt"
    assistant_text = (
        completion_path.read_text()
        if completion_path.is_file()
        else str(outcome.error or "")
    )
    reward = outcome.reward
    for _ in range(missing):
        traces.append({
            "assistant_text": assistant_text,
            "turn_idx": len(traces),
            "dense_pct": (
                float(reward.score) * 100.0 if reward is not None else 0.0
            ),
            "score": (
                float(reward.verified_score) * 100.0
                if reward is not None else 0.0
            ),
            "passed": (
                bool(reward.hard_gate_passed) if reward is not None else False
            ),
            "parsed_ok": (
                bool(reward.design_py_extracted)
                if reward is not None else False
            ),
            "evaluation_valid": (
                bool(reward.evaluation_valid) if reward is not None else False
            ),
            "failure_codes": (
                list(reward.failure_codes) if reward is not None else []
            ),
            "feedback": (
                list(reward.feedback) if reward is not None else []
            ),
            "cad_audits": int(outcome.cad_audits or 0),
            "chrono_audits": int(outcome.chrono_audits or 0),
            "physical_metrics": (
                dict(reward.physical_metrics) if reward is not None else {}
            ),
            "no_procedural_fallback": (
                reward.no_procedural_fallback
                if reward is not None else None
            ),
            "completion_tokens": int(outcome.sample_tokens_out or 0),
            "stop_reason": "terminal_sample_evidence",
        })


def _rollout_audit_totals(rollout: Any) -> tuple[int, int]:
    cad = 0
    chrono = 0
    for turn in _rollout_scored_turns(rollout):
        cad += int(getattr(turn, "cad_audits", 0) or 0)
        chrono += int(getattr(turn, "chrono_audits", 0) or 0)
    return cad, chrono


def _apply_rollout_audit_totals(reward: RewardResult, rollout: Any) -> None:
    cad, chrono = _rollout_audit_totals(rollout)
    reward.cad_audits = cad
    reward.chrono_audits = chrono


def _reward_from_rollout_best_turn(rollout: Any) -> RewardResult | None:
    scored_turns = _rollout_scored_turns(rollout)
    if not scored_turns:
        return None
    turn = max(
        scored_turns,
        key=lambda item: (
            bool(getattr(item, "evaluation_valid", False))
            and bool(getattr(item, "passed", False))
            and not (getattr(item, "failure_codes", []) or []),
            float(getattr(item, "score", 0.0) or 0.0),
            float(getattr(item, "dense_pct", 0.0) or 0.0),
        ),
    )
    failure_codes = list(getattr(turn, "failure_codes", []) or [])
    score = float(getattr(turn, "dense_pct", 0.0) or 0.0) / 100.0
    verified_score = float(getattr(turn, "score", 0.0) or 0.0) / 100.0
    return RewardResult(
        score=score,
        verified_score=verified_score,
        hard_gate_passed=bool(getattr(turn, "passed", False)),
        evaluation_valid=bool(getattr(turn, "evaluation_valid", False)),
        failure_codes=failure_codes,
        feedback=list(getattr(turn, "feedback", []) or []),
        design_py_extracted=bool(getattr(turn, "parsed_ok", False)),
        cad_audits=int(getattr(turn, "cad_audits", 0) or 0),
        chrono_audits=int(getattr(turn, "chrono_audits", 0) or 0),
        physical_metrics=dict(getattr(turn, "physical_metrics", {}) or {}),
        no_procedural_fallback=getattr(turn, "no_procedural_fallback", None),
    )


def _reward_from_rollout_final(rollout: Any) -> RewardResult | None:
    """Backward-compatible alias for tests and older imports."""
    return _reward_from_rollout_best_turn(rollout)


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


def _needs_audit_retry(
    o: SampleOutcome,
    *,
    required_cad_audits: int,
    required_chrono_audits: int,
) -> bool:
    if o.error:
        return False
    return (
        int(o.cad_audits or 0) < required_cad_audits
        or int(o.chrono_audits or 0) < required_chrono_audits
    )


def _task_required_audits(task_dir: Path, *, max_turns: int) -> tuple[int, int]:
    """Return per-sample CAD/Chrono audit targets implied by eval_config.toml."""
    required = max(1, int(max_turns or 1))
    cfg_path = task_dir / "eval_config.toml"
    if not cfg_path.is_file():
        return 0, 0
    try:
        cfg = tomllib.loads(cfg_path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return 0, 0

    probes = cfg.get("probes") or []
    adapters = cfg.get("adapters") or {}
    needs_cad = any(
        isinstance(probe, dict)
        and str(probe.get("type") or "") == "trusted_asset_preflight"
        for probe in probes
    )
    needs_chrono = (
        isinstance(adapters, dict)
        and isinstance(adapters.get("chrono_contact"), dict)
    ) or any(
        isinstance(probe, dict)
        and str(probe.get("adapter") or "") == "chrono_contact"
        for probe in probes
    )
    return (required if needs_cad else 0, required if needs_chrono else 0)


def actual_verifier_calls_total(outcomes: list[SampleOutcome]) -> int:
    return sum(int(item.verifier_calls or 0) for item in outcomes)


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
    p.add_argument("--audit-retries", type=int, default=0,
                   help="diagnostic-only replacement attempts for samples "
                        "that fail before spending their planned CAD/Chrono "
                        "audit budget; retries are counted in actual budget")
    p.add_argument("--max-verifier-calls-per-task", type=int, default=0,
                   help="hard cap on actual verifier calls per task. When "
                        "set, sampling stops once this budget is spent and "
                        "the final sample's max turns are clipped to the "
                        "remaining budget.")
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
        verifier_call_cap = max(0, int(args.max_verifier_calls_per_task or 0))
        required_cad_audits, required_chrono_audits = _task_required_audits(
            td,
            max_turns=int(args.max_turns or 1),
        )
        for k in range(args.samples_per_task):
            if verifier_call_cap and actual_verifier_calls_total(outs) >= verifier_call_cap:
                break
            o = None
            audit_retry_count = 0
            actual_verifier_calls = 0
            actual_cad_audits = 0
            actual_chrono_audits = 0
            actual_turn_traces: list[dict[str, Any]] = []
            for audit_attempt in range(args.audit_retries + 1):
                if audit_attempt == 0:
                    out_root_k = out_root / f"sample_{k}"
                else:
                    out_root_k = out_root / (
                        f"sample_{k}_audit_retry_{audit_attempt}"
                    )
                out_root_k.mkdir(parents=True, exist_ok=True)
                run_max_tokens = args.max_tokens
                for attempt in range(args.sampler_retries + 1):
                    spent_so_far = actual_verifier_calls_total(outs) + actual_verifier_calls
                    remaining_calls = (
                        verifier_call_cap - spent_so_far
                        if verifier_call_cap
                        else int(args.max_turns)
                    )
                    if verifier_call_cap and remaining_calls <= 0:
                        break
                    run_max_turns = (
                        min(int(args.max_turns), int(remaining_calls))
                        if verifier_call_cap
                        else int(args.max_turns)
                    )
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
                        seed=(
                            args.seed
                            + task_idx * 1000
                            + k
                            + audit_attempt * 100000
                        ),
                        timeout_s=args.timeout,
                        pass_threshold=args.pass_threshold,
                        max_turns=run_max_turns,
                        sample_idx=k,
                    )
                    _append_missing_outcome_turn_traces(
                        o,
                        out_root_k / td.name,
                    )
                    actual_verifier_calls += int(o.verifier_calls or 0)
                    actual_cad_audits += int(o.cad_audits or 0)
                    actual_chrono_audits += int(o.chrono_audits or 0)
                    retryable_sampler_error = _is_retryable_sampler_error(o)
                    _append_attempt_turn_traces(
                        actual_turn_traces,
                        o.turn_traces,
                        sample_idx=k,
                        audit_attempt=audit_attempt,
                        sampler_attempt=attempt,
                        trace_kind=(
                            "sampler_error_retry"
                            if retryable_sampler_error
                            else "scored_attempt"
                        ),
                    )
                    if (
                        verifier_call_cap
                        and actual_verifier_calls_total(outs) + actual_verifier_calls
                        >= verifier_call_cap
                    ):
                        break
                    if not retryable_sampler_error:
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
                        f"[RETRY] {td.name:48} k={k} "
                        f"attempt={attempt + 1} "
                        f"max_tokens={run_max_tokens} err={o.error[:80]}",
                        file=sys.stderr,
                    )
                assert o is not None
                if (
                    verifier_call_cap
                    and actual_verifier_calls_total(outs) + actual_verifier_calls
                    >= verifier_call_cap
                ):
                    break
                if not _needs_audit_retry(
                    o,
                    required_cad_audits=required_cad_audits,
                    required_chrono_audits=required_chrono_audits,
                ):
                    break
                if audit_attempt >= args.audit_retries:
                    break
                audit_retry_count += 1
                print(
                    f"[AUDIT_RETRY] {td.name:48} k={k} "
                    f"attempt={audit_attempt + 1} "
                    f"cad={o.cad_audits}/{required_cad_audits} "
                    f"chrono={o.chrono_audits}/{required_chrono_audits} "
                    f"err={o.error[:60]}",
                    file=sys.stderr,
                )
            assert o is not None
            o.verifier_calls = actual_verifier_calls
            o.cad_audits = actual_cad_audits
            o.chrono_audits = actual_chrono_audits
            o.turn_traces = actual_turn_traces
            if audit_retry_count:
                o.audit_retry_count = audit_retry_count
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
    n_strict_passed = 0
    n_verifier_valid_passed = 0
    for tid, lst in by_task.items():
        winner = max(
            lst,
            key=lambda o: (
                o.reward.verified_score if o.reward else 0.0
            ),
        )
        best.append(winner.to_dict())
        if winner.passed():
            n_strict_passed += 1
        if winner.verifier_valid_passed():
            n_verifier_valid_passed += 1

    n_strict_passed_raw = sum(1 for o in all_outcomes if o.passed())
    n_verifier_valid_passed_raw = sum(
        1 for o in all_outcomes if o.verifier_valid_passed()
    )
    n_verifier_calls = sum(int(o.verifier_calls or 0) for o in all_outcomes)
    n_cad_audits = sum(int(o.cad_audits or 0) for o in all_outcomes)
    n_chrono_audits = sum(int(o.chrono_audits or 0) for o in all_outcomes)

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
        "audit_retries": args.audit_retries,
        "n_audit_retries": sum(
            int(getattr(o, "audit_retry_count", 0) or 0)
            for o in all_outcomes
        ),
        "tasks_root": str(args.tasks),
        "split_file": str(args.split_file) if args.split_file else None,
        "n_tasks": len(by_task),
        "samples_per_task": args.samples_per_task,
        "max_turns": args.max_turns,
        "pass_threshold": args.pass_threshold,
        "n_passed_best_of_k": n_strict_passed,
        "pass_rate_best_of_k": (
            n_strict_passed / len(by_task) if by_task else 0.0
        ),
        "n_verifier_valid_best_of_k": n_verifier_valid_passed,
        "verifier_valid_pass_rate_best_of_k": (
            n_verifier_valid_passed / len(by_task) if by_task else 0.0
        ),
        "n_samples": len(all_outcomes),
        "n_verifier_calls": n_verifier_calls,
        "n_cad_audits": n_cad_audits,
        "n_chrono_audits": n_chrono_audits,
        "n_passed_raw": n_strict_passed_raw,
        "pass_rate_raw": (
            n_strict_passed_raw / len(all_outcomes)
            if all_outcomes else 0.0
        ),
        "n_verifier_valid_raw": n_verifier_valid_passed_raw,
        "verifier_valid_pass_rate_raw": (
            n_verifier_valid_passed_raw / len(all_outcomes)
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
        "n_verifier_valid_best_of_k": summary[
            "n_verifier_valid_best_of_k"
        ],
        "verifier_valid_pass_rate_best_of_k": round(
            summary["verifier_valid_pass_rate_best_of_k"], 3),
        "n_passed_raw": summary["n_passed_raw"],
        "pass_rate_raw": round(summary["pass_rate_raw"], 3),
        "wall_clock_s": round(summary["wall_clock_s"], 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
