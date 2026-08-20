# mech_bench — agent evaluation results + RL roadmap

> **Historical snapshot (2026-05-13).** This document preserves the first
> agent-evaluation and training-cook analysis. It is not the current project
> status. The comparison has a contaminated evaluation arm, the legacy trainer
> is not exact GRPO, the native Chrono runner now exists, and later Level-1
> held-out-family results supersede the learning conclusions here. Start with
> [`project-status.md`](project-status.md).
>
> Written 2026-05-13 after the first full RL cooks. Cites real run
> artifacts under `evals/` and `runs/`. Honest, including the bits
> that didn't work.

## Contents

1. [TL;DR](#tldr)
2. [Agent-as-eval results — Claude Code and Codex](#agent-as-eval-results)
3. [RL training results — DeepHermes-3 on worldlines](#rl-training-results)
4. [What works (the stack is real)](#what-works)
5. [Honest critique of the RL environment](#honest-critique)
6. [Improvement roadmap, ranked](#improvement-roadmap)
7. [Open questions](#open-questions)

---

## TL;DR

| Track | Result |
| --- | --- |
| **Eval / inference** | OpenAI Codex (gpt-5.5) **34/51 = 66.7 %** vs Claude Code (sonnet) **13/25 = 52 %**\* on the 51-task suite. Both at ~$2.50, ~10 min wall-clock at concurrency 4. |
| **RL training**     | DeepHermes-3-Llama-3-3B-Preview, **6 % → 9 % pass rate** over 60 rounds of multi-turn GRPO on worldlines (480 rollouts, 41 optim steps, 88 min). Modest, within-noise gain, but the full SGLang + worldlines + GRPO loop is verified end-to-end. |
| **Bottom line**     | Eval infrastructure works and is reproducible. RL infrastructure works. The **reward landscape is too sparse + the base model is too small** for meaningful learning in 40 optim steps. Ranked improvements at the bottom of this doc. |

\* Claude run was contaminated by an `nvm`-PATH bug under concurrency 6
(26 of 51 tasks failed before the API fired). The 25 that actually ran
are the honest sample.

---

## Agent-as-eval results

Setup (commits `9277c08`, `7031361`, `555bf44`, `b7bd237`):

- Per task: spawn the CLI (`claude -p` / `codex exec --json`) with a
  shared system prompt (`scripts/agent_system_prompt.md`).
- Read-only inputs: `prompt.md` + `task.toml` only.
- Strict tool allowlist; per-task wall-clock + budget caps.
- Score with `python -m mech_bench evaluate --full --allow-partial`.
- Aggregate to `<report>/claude_eval_summary.json` /
  `codex_eval_summary.json`.

### Headline numbers (2026-05-13)

| Agent | Model | n_tasks | Passed | Rate | Cost USD | Wall (min) |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Code | claude-sonnet-4-6 | 51 | 13 | 25.5 %* | $2.48 | 8.5 |
| Codex CLI   | gpt-5.5            | 51 | **34** | **66.7 %** | $2.44 | 10.4 |

\* 26 of those Claude tasks errored out with `claude not on PATH`
**before any API call** because the `nvm`-shimmed binary lookup
breaks at `--concurrency 6`. The honest Claude denominator is 25;
on those it's 13/25 ≈ **52 %**. See `evals/README.md` for the
caveat. Rerunning at `--concurrency ≤ 3` would clean this up.

### Head-to-head (51 common tasks)

```
both pass            : 13
neither pass         : 17
only codex passes    : 21
only claude passes   :  0
agreement rate       : 58.8 %
```

Codex strictly dominates the head-to-head on the 51 overlap (no task
where Claude passed and Codex did not). With the Claude infra failures
excluded, it would be roughly 13/25 vs 34/51 — Codex still ahead, but
not 0-21.

### Failure-code histograms

| Code | Claude | Codex |
| --- | --- | --- |
| `invalid_artifact`      | 5 | 8 |
| `wrong_topology`        | 2 | 5 |
| `path_error`            | 1 | 3 |
| `capability_unavailable`| 1 | 2 |
| `wrong_ratio`           | 0 | 1 |

Most failures are **schema or grounding errors** (forgot
`"fixed": True`, port references a missing joint), not deep
mechanical-engineering mistakes. The benchmark is doing its job —
hard-gate probes catch the agent's most common mistake (un-grounded
plate) before any dense reward is awarded.

### Reproducer

```bash
python scripts/run_claude_on_eval.py --tasks tasks \
    --report-dir /tmp/eval_claude --model sonnet \
    --max-budget-usd 0.30 --timeout 300 --concurrency 3

python scripts/run_codex_on_eval.py --tasks tasks \
    --report-dir /tmp/eval_codex --model gpt-5.5 \
    --timeout 300 --concurrency 4

python scripts/compare_agent_evals.py \
    --left  /tmp/eval_claude/claude_eval_summary.json \
    --right /tmp/eval_codex/codex_eval_summary.json
```

---

## RL training results

### Stack (commits `0682b89`, `0089dff`, `5b6295f`, `b7bd237`)

```
GPU 1  SGLang OpenAI server (:30000)         16k ctx, bf16
GPU 0  Worldlines backend  (:18100)          PEFT LoRA rank 32, bf16
tmpfs  /dev/shm/wld-venv                     uv-managed Py 3.12 venv
tmpfs  /dev/shm/hf-cache                     HF model cache
tmpfs  /dev/shm/wld-artifacts                LoRA checkpoints
```

Per round:

1. Pick K tasks. For each, run N multi-turn rollouts via
   SGLang `/v1/chat/completions` with the
   `scripts/agent_system_prompt.md` system + a per-task user prompt.
2. After each turn, run `mech-bench evaluate` and inject a structured
   verifier-feedback user message; loop up to `max_turns`.
3. Score each rollout with `dense_pct = mean(per-probe score) × 100`.
4. Group-relative advantages over the K rollouts, clipped to ±2.
5. Per-token weight = `advantage / completion_len` (this
   normalisation matters — without it gradients explode, see
   cook05 below).
6. Worldlines `forward_backward(loss_fn="cross_entropy")` +
   `optim_step(AdamParams(lr))`.
7. `save_state` every N optim steps.

### Cook timeline

| Run | Outcome | Notes |
| --- | --- | --- |
| smoke01      | 0 optim steps | Qwen3-1.7B base — model not chat-tuned, produced noise tokens. |
| smoke02-04   | 0 optim steps | Tried DeepHermes-3-3B but worldlines stub-sampler returned random tokens (no SGLang). |
| smoke05-08   | 0 optim steps | Added SGLang serving, real inference, multi-turn rollouts. Hit a PEFT trainer `cpu vs cuda:0` gather bug. |
| smoke09-10   | **3 optim steps** | Fixed via in-process monkey-patch (`rl/launch_trainer_patched.py`) that (a) re-orders the `target_tokens` device cast and (b) forces the whole PEFT-wrapped model onto cuda:0 (PEFT was leaving adapters on CPU). |
| cook04       | 19 optim steps | First clean cook. 30 rounds, 100 % parse, **loss displayed as 0.0** because train_grpo was reading `fb.loss` but worldlines returns `loss_fn_outputs[i]["loss"]`. Gradients were flowing, we just couldn't see it. |
| smoke11      | 3 optim steps | Fixed loss display: real losses came out at -4.9, -15.1, -199.4. |
| cook05       | 19 optim steps **aborted** | LR 5e-4 + un-normalised per-token weights produced loss = -25 000 by step 16. Killed and added per-token weight normalisation. |
| smoke12      | 2 optim steps | Confirmed losses are now in single-digit range (+0.022, -0.015). |
| **cook06**   | **41 optim steps**, **4 checkpoints** | The headline run below. |

### cook06 — first real training curve

`runs/cook06_stable/` (commit `b7bd237`):

```
DeepHermes-3-Llama-3-3B-Preview
60 rounds × 2 tasks × 4 samples × 2 multi-turns = 480 rollouts
41 optim steps, checkpoints at 10/20/30/40
LR 1e-4, LoRA rank 32, advantage clip ±2
87.8 min wall-clock on 2× RTX 3090
```

| Window | n | Passed | Parsed | Dense avg |
| --- | --- | --- | --- | --- |
| r 0–14 | 120 | 7 (6 %)   | 100 % | 10.7 |
| r15–29 | 120 | 8 (7 %)   |  98 % | 10.3 |
| r30–44 | 120 | 9 (8 %)   | 100 % |  8.9 |
| r45–59 | 120 | **11 (9 %)** | 100 % | **14.2** |

**Loss curve** (every 5th of 41 steps):

```
step  1:  +0.0021
step  4:  -0.0033
step  6:  -0.098
step 11:  -5.81
step 16:  -18.12
step 21:  -21.35
step 26:  -32.63
step 31:  -35.49
step 36:  -18.14
step 41:  -15.71
```

Monotonic drift more-negative through step 31, then partial unwind.
That magnitude is the model rapidly shifting its token probability
mass; **no clipping or KL penalty exists yet**.

**Per-task winners** (clearly above zero-shot):

| Task | Passes |
| --- | --- |
| `mounting_plate_hole_pitch_s0001`   | **4 / 4** |
| `flange_bolt_circle_s0001`          | **5 / 8** |
| `spacer_stack_height_s0001`         | **2 / 4** |
| `standoff_pattern_square_s0001`     | **5 / 12** |
| `gear_pair_load_trial_stub_s0001`   | 4 / 16 |
| `cam_follower_contact_stub_s0001`   | 4 / 20 |
| `ratchet_pawl_engagement_stub_s0001`| 5 / 24 |

All Tier-0 declared-value or Tier-3 synthetic-contact tasks where the
verifier mostly cares about "did the agent write the right number in
`params`."

**Per-task zeros** (still 0/N):

Every four-bar variant, every two-stage gear ratio, every transmission
that requires computed link geometry or per-probe pose. The model
cannot make the leap from "valid IR shape" to "geometrically correct
mechanism."

---

## What works

The stack is real. End-to-end, the following code paths verified:

- **Eval (Claude Code / Codex)**: per-task isolation, tool allowlist,
  budget cap, JSON scorecard, comparison report.
  - `scripts/run_claude_on_eval.py`, `scripts/run_codex_on_eval.py`,
    `scripts/compare_agent_evals.py`, `scripts/Dockerfile.eval`.
- **Inference**: SGLang 0.5.11 with `--tool-call-parser llama3` serving
  DeepHermes-3-3B at 16k context, on a single RTX 3090.
- **Training**: Worldlines (Tinker-API-compatible) on a separate 3090,
  PEFT LoRA rank 32 in bf16, bf16 forward / fp32 LoRA grads, in-process
  monkey-patch keeps the worldlines submodule unmodified.
- **Multi-turn rollouts** with verifier-feedback user messages
  (`rl/chat_rollout.py`).
- **Group-relative advantages** with proper per-token weight
  normalisation (`rl/train_grpo.py`).
- **Per-run artifacts** in the rl-spark dashboard shape under
  `runs/<run>/{heartbeat.json,history.jsonl,task_scores.jsonl}`.
- **Checkpoints** saved by name to `worldlines://model_X/weights/...`
  on `/dev/shm/wld-artifacts`.
- **Dashboard** at `:8002` (the rl-spark FastAPI app pointed at our
  `runs/<run>/`).

The full one-line bring-up:

```bash
bash rl/run.sh up        # both servers
bash rl/run.sh train     # the GRPO cook
```

---

## Honest critique

Where the RL env is structurally weak — ranked from "biggest blocker"
to "annoyance":

### 1. Reward is binary at the probe level

`dense_pct = mean(probe.score)*100`. For most probes
(`dof_grubler`, `required_ports`, `analytic_param_check` at default
config), `probe.score` is 0.0 or 1.0. A 3-probe task therefore
returns dense_pct ∈ {0, 33, 67, 100}. That's a 4-bin discretisation —
the policy gradient has nowhere to "climb" between bins, so once the
model is on a plateau it stays there.

`analytic_param_check` already computes `error_pct` internally — we
just never expose it as the score. **Trivial fix.**

### 2. Zero-variance groups are common

GRPO advantages are `(reward − mean) / std`. When all K rollouts pass
or all K fail, `std = 0` → advantage = 0 → no gradient. In cook06,
~30 % of rounds were skipped this way. Two cures:

- **Curriculum**: only sample from "warm" tasks (1–3 of K pass).
  Tasks where the model nails everything *or* fails everything get
  dropped from the queue.
- **Cross-group baseline**: subtract a running global mean instead of
  per-group mean, so a single-passer-in-failing-group still gets
  positive advantage relative to other tasks.

### 3. Path-tracing tasks are end-to-end hard gates

A four-bar coupler-path task only awards score after the IR validates,
mobility = 1, all three required ports exist, AND the coupler trace
fits a target CSV within chamfer 0.05. Failing any earlier step zeros
the chamfer probe and the model gets no learning signal about the
geometric placement. **Decompose**: emit a partial score for each
earlier gate so the model learns "first get mobility right, then
ports, then coupler offset" as a curriculum the agent itself sees.

### 4. Advantage-weighted CE isn't proper PPO

Our `forward_backward(loss_fn="cross_entropy")` with signed weights
is a hack. No KL penalty against the base policy, no value head,
no clipped ratio. Cook05's -25 000 loss spike is exactly the failure
mode this paper-over makes available. Worldlines's
`forward_backward_custom_v2` supports a user-defined loss; the
upstream rl-spark engdesign_train.py notes proper PPO is "future
work" with that hook. **Mid-priority infra work.**

### 5. 3B parameters is too small for this scope

The DeepHermes-3-3B base did not converge in 40 optim steps from a
~10 % baseline because it has not seen this schema before AND has
weak compositional reasoning. The rl-spark equivalent runs Qwen3.6-35B
NVFP4 on a Spark. Here, **Qwen2.5-Coder-7B-Instruct** at LoRA rank 32
fits in 24 GB and is the right scale for "small enough to iterate,
big enough to learn." 3B's pass rate floor is the structural ceiling
of this experiment.

### 6. No tool use during rollout

The agent writes one fenced block. It cannot inspect mech_bench's
output between turns — only the verifier-feedback user message we
synthesise. A real coding-agent rollout would let the model call
`mech-bench evaluate` itself as a tool, see the JSON, and write the
next attempt. SGLang supports `--tool-call-parser llama3`; the agent
loop's `chat_rollout.py` doesn't use that path yet.

### 7. Synthetic Tier-3 contact stubs aren't physical

`fake_contact_oracle` is deterministic synth; reports tag
`oracle_is_synthetic = true`. Cook06's Tier-3 wins are mostly the
agent learning to declare `contact_pair` joints with the right names.
That's a parse / topology lesson, not a contact-dynamics lesson. Not
a *bug*, but the suite's Tier-3 doesn't validate physics until a real
Chrono backend lands.

### 8. Run-level state is not durable across crashes

`forward_backward` requests in flight when the trainer process dies
leave dangling futures the client retries forever. Two cooks (cook01,
cook03) burned 5 minutes each on this before we identified the
pattern. **Fix**: add a hard deadline on client retries, and on the
backend side `expire` stale futures more aggressively.

---

## Improvement roadmap

Ranked by expected pass-rate impact divided by engineering hours.

### Tier-1 — half-day or less, big expected lift

1. **`error_pct`-based smooth reward on analytic probes.** Have
   `analytic_param_check` return `max(0, 1 − error_pct / tolerance_pct)`
   when the gate fails instead of a hard 0. ~30 lines in
   `mech_bench/probes/analytic_param_check.py`. Expected: Tier-0 +
   Tier-2 pass rate doubles before training.

2. **Warm-task curriculum buffer.** Track EMA per-task pass rate;
   sample preferentially from tasks where the EMA is in `[0.2, 0.8]`
   (the model is making *some* progress). Avoid the always-fail
   four-bars and the always-pass mounting plate. ~50 lines in
   `rl/train_grpo.py`. Pattern is already in
   `rl-spark/engdesign_train.py::CurriculumBuffer`.

3. **Cross-group baseline.** Replace per-group `(r - mean(K)) / std(K)`
   with `(r - global_running_mean) / global_running_std`. Eliminates
   the zero-variance-skip problem. ~15 lines.

4. **Move to Qwen2.5-Coder-7B-Instruct.** Already in HF cache for
   most users; fits at LoRA rank 32 in 24 GB. Single env-var swap to
   `rl/launch_worldlines.sh` and `rl/run.sh`. Expected 2–3× pass-rate
   floor at zero-shot.

### Tier-2 — 1–2 days each, meaningful infra

5. **KL penalty / clipped-ratio PPO via `forward_backward_custom_v2`.**
   Compute log-prob ratio between the live policy and the base sampler
   on the kept tokens. Clip to PPO range. Bounds the drift that
   produced cook05's -25 000 loss.

6. **Tool-use rollouts.** Use SGLang's `tool_call_parser=llama3` to
   let the agent `mech_bench_evaluate(submission_str)` as a tool
   between turns. Replaces our hand-synthesised verifier-feedback
   user message with structured JSON the model can attend to.

7. **Per-probe progress reward shaping.** Split path-tracing tasks
   so the agent gets credit for clearing mobility, then for declaring
   ports, then for coupler-point placement, then for path chamfer.
   Each step a measurable score component.

### Tier-3 — multi-day projects

8. **Real Chrono contact oracle.** Vendor in the `_chrono_impl.py`
   shim from phys-sim's `_chrono_mesh_runner.py`. Turn the Tier-3
   synthetic stubs into actual contact-dynamics tasks. Notes in
   `docs/chrono-backend.md`.

9. **`mech-bench-eval` as an Atropos-shape env.** Mirror
   `hermes-agent/environments/hermes_base_env.py` so a Hermes-Agent
   orchestrator can drive our verifier without code edits. Then we
   inherit hermes-agent's tool-call infrastructure and reward
   shaping for free.

10. **Run a 1000-step cook** with all of (1) + (2) + (3) + (4) +
    (5) on, in parallel across both 3090s with the trainer on a
    separate one from inference. This is what an actual research
    answer looks like for "can we train a small model to solve
    mech_bench."

---

## Open questions

- **Is the synthetic contact oracle teaching anything physical, or
  just topology?** Cook06's `cam_follower_contact_stub` 4/20 and
  `ratchet_pawl_engagement_stub` 5/24 results suggest the latter.
  Worth running an ablation where the same task is scored with the
  fake oracle off; if the model still passes by emitting the right
  contact pair name, the task is degenerate.

- **Why did `mounting_plate_hole_pitch` go 4/4 from round 1 but
  `bearing_seat_clearance` stayed at 0/8?** Both are Tier-0 declared-
  value tasks. The dense_pct curves should be similar but they
  aren't. Suggests the verifier-feedback messages are clearer for
  one than the other.

- **Does decoding temperature 0.85 help or hurt?** rl-spark uses 0.7.
  Higher temperature should give more variance for GRPO. Worth a
  small ablation (0.6 / 0.85 / 1.0 over 100 rounds each).

- **Multi-turn vs single-turn budget split.** Cook06 capped at 2
  turns. With 4 turns the model has more chances to repair from a
  bad first attempt, but doubles the rollout cost. Is the per-token
  cost / pass-rate trade-off favorable?

---

## File map for future readers

```
scripts/run_claude_on_eval.py     Claude Code eval harness
scripts/run_codex_on_eval.py      OpenAI Codex eval harness
scripts/compare_agent_evals.py    side-by-side diff
scripts/agent_system_prompt.md    shared DesignIR contract
scripts/Dockerfile.eval           reproducible image

evals/                            frozen eval scorecards (this doc cites)

rl/run.sh                         single-file orchestrator
rl/launch_worldlines.sh           Worldlines backend wrapper (patched entrypoint)
rl/launch_trainer_patched.py      in-process monkey-patch (PEFT cuda hygiene)
rl/chat_rollout.py                multi-turn SGLang chat-completions rollout
rl/mech_env.py                    mech_bench → EpisodeResult wrapper
rl/train_grpo.py                  the GRPO loop
rl/agent_prompt_rl.md             tight system prompt with canonical example

runs/                             per-run heartbeat / history / task scores
```
