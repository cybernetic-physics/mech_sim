# mech_sim vs. ttt_discover/codex_runtime — architectural comparison

> **Historical design note (2026-05-13).** This file records a point-in-time
> comparison with an external runtime. Estimates and recommendations below are
> not an implementation-status checklist. Use [`project-status.md`](project-status.md)
> and [`../rl/README.md`](../rl/README.md) for current guidance.

> Written 2026-05-13 after walking through
> `github.com/cybernetic-physics/ttt_discover`. Honest read: this is
> the architecture I described in the abstract ("the harness IS the
> rollout") **actually built**. We have the same conceptual loop but
> wired through a much weaker harness. Below is what they do, what
> we do, where they win, and the concrete upgrade path.

## What ttt_discover's codex_runtime is

A complete RL training stack where **the real Codex CLI binary is the
rollout harness**. Per training iteration:

```
PUCTSampler                    ← picks start states from the buffer
   │
   ▼
GitStateManager                ← git-snapshots the workspace, restores
   │                             a clean copy for each rollout
   ▼
CodexRuntime                   ← docker run @openai/codex
                                 "codex exec --json --full-auto"
                                 with OPENAI_BASE_URL=LoggingProxy
   │                             (one container per rollout)
   ▼
LoggingProxy (aiohttp)         ← intercepts the OpenAI Responses API,
   │                             translates messages ↔ harmony tokens,
   │                             samples via tinker SamplingClient,
   │                             captures per-token tokens+logprobs
   ▼
CommandVerifier                ← `pytest -q` (or any shell cmd)
   │                             on the workspace post-rollout
   ▼
trace_converter                ← APICallRecord → tinker Trajectory
   │
   ▼
prepare_minibatch              ← advantage estimation, KL penalty,
                                 GAE / group-relative / etc.
   │
   ▼
tinker train_step              ← forward_backward + optim_step
   │
   ▼
save_weights_and_get_sampling_client
                               ← swap proxy's sampling client → loop
```

Headline files:

| File | Role |
| --- | --- |
| `codex_runtime/runtime.py`        | `CodexRuntime.run()` — spawns Docker, runs `codex exec`, captures stdout JSONL |
| `codex_runtime/logging_proxy.py`  | `LoggingProxy` — aiohttp server, per-rollout `/rollout/{id}/v1/responses` routes, harmony translation, logprob capture |
| `codex_runtime/state_manager.py`  | `GitStateManager` / `TarballStateManager` — workspace snapshots |
| `codex_runtime/verifier.py`       | `CommandVerifier` — runs the shell command, computes reward |
| `codex_runtime/trace_converter.py`| API call records → `Trajectory` (one transition per assistant turn) |
| `codex_runtime/rollout.py`        | `run_codex_rollout()` — single-rollout orchestrator |
| `codex_runtime/codex_collector.py`| `do_codex_group_rollout()` — K-parallel rollouts per start state |
| `codex_runtime/codex_training.py` | top-level `run_codex_ttt(config)` — the actual training loop |
| `codex_runtime/Dockerfile`        | `node:20-slim` + `npm i -g @openai/codex` + `pytest` |

## Side-by-side architectural comparison

| Dimension | mech_sim (cook06) | ttt_discover codex_runtime |
| --- | --- | --- |
| **Harness inside rollout** | minimal multi-turn chat, verifier feedback synthesised as a `role=user` message | **the actual Codex CLI binary in Docker** — same code shipped to users |
| **Workspace** | none — model emits one fenced ```python block | full git repo; agent can edit files, run shells, anything Codex does |
| **State** | stateless per task | `GitStateManager` snapshots before/after each rollout — full restore |
| **Concurrency-safe logging** | n/a (single completion per call) | per-rollout URL `/rollout/{uuid}/v1/responses` — multiple containers can log in parallel |
| **Logprob capture** | re-tokenise the final assistant text locally | **captured live at sample-time** by `LoggingProxy` — exact tokens the agent emitted |
| **Token format** | tokenizer.apply_chat_template (Llama-3 chat) | openai-harmony Conversation ↔ token roundtrip (what gpt-oss uses) |
| **Rollout multi-turn** | yes, but turns separated by verifier-feedback user messages | yes; turns are tool calls and responses inside one Codex CLI invocation |
| **Sampler over tasks** | uniform random | **PUCT** over a buffer of (state, action, reward) tuples — proper exploration |
| **K rollouts per task** | yes (samples_per_task) | yes (group_size, run in parallel asyncio) |
| **Reward** | mech_bench probe scores → dense_pct | `CommandVerifier` exit + shaped score |
| **Advantage** | group-relative, std-normalised, clipped ±2, drop-zero-var | `prepare_minibatch` — proper estimators (group_relative, GAE), KL penalty against base, kl_discount |
| **Trainer** | worldlines (Tinker-wire-compatible) | **tinker** (managed Tinker cloud) |
| **Weight publication** | `save_state` to `worldlines://…` | `save_weights_and_get_sampling_client` → proxy hot-swaps |
| **Lines of glue code** | ~600 in `rl/` | ~1500 in `codex_runtime/`, much more featureful |

## Where ttt_discover wins (and why it matters)

### 1. The production binary IS the rollout

This is the architectural insight from yesterday's discussion made
concrete. They don't mock a "multi-turn chat with verifier feedback"
— they `docker run @openai/codex` per rollout, with the exact CLI
flags and sandbox the user would have. The trained policy is
guaranteed to behave the same when deployed because there's no
distribution gap between training and inference harnesses.

We do not have this. Our rollout fakes an agent loop by stitching
together SGLang `/v1/chat/completions` calls with verifier-feedback
user messages. That works for "model writes one design.py with two
chances to fix it," but it isn't how Claude Code or Codex actually
operate when deployed.

### 2. Per-token logprob capture via API interception

The LoggingProxy is clever: it sits at `OPENAI_BASE_URL` for the
Docker'd Codex, translates the Responses API request → Harmony
Conversation, samples through tinker, captures tokens + logprobs,
streams back as SSE. **Every token the agent emits is logged with
its sampling-time logprob.** No re-tokenisation; no drift between
"what the policy actually produced" and "what we train on."

We re-tokenise the final assistant text in train_grpo. That's lossy
for BPE merges and ignores all intermediate tool-call tokens. The
proxy approach is the right way.

### 3. Per-rollout URL paths for concurrency

Every Codex container hits a unique URL like
`/rollout/{uuid}/v1/responses`. The proxy keys all logging by that
rollout_id. So K parallel Codex containers can run concurrently and
each one's trajectory stays cleanly isolated. Trivial to implement,
load-bearing for scalable concurrency.

### 4. State management

`GitStateManager` snapshots the workspace before each rollout and
restores it for the next one. State IDs are git commit SHAs. This
makes the start state of each rollout exactly defined — no leakage
from previous rollouts polluting the workspace.

For us this doesn't apply (our "workspace" is a single design.py).
But the moment we want tool-use rollouts where the agent edits and
re-runs, we'll need this.

### 5. PUCT over state buffer

The discovery loop they're running is test-time-training: given a
hard scientific problem, search over candidate solutions, RL-train
the policy on the discovered ones. PUCT (Polynomial Upper Confidence
Trees) picks which buffer state to expand next. Smart curriculum
that learns over runs.

For mech_bench tasks we don't need full PUCT — the task set is
fixed — but the underlying idea of "track per-task EMA pass rate,
sample preferentially from warm tasks" is exactly the curriculum
improvement we identified in our roadmap.

### 6. `prepare_minibatch` from tinker_cookbook

This does proper advantage estimation (multiple estimators including
GAE, group-relative), KL penalty against a reference policy with
configurable discount, and importance-sampling for off-policy
updates. We hand-wrote `_group_advantages` + naive CE.

This is the single highest-impact upgrade we could make and it's
already factored as a library function.

### 7. tinker_cookbook + tinker integration

ttt_discover is built on the tinker_cookbook's RL recipes — math_rl,
code_rl, ttt — and uses tinker (managed cloud) for training. We use
worldlines (Tinker-wire-compatible, self-hosted). The cookbook code
should work essentially as-is against worldlines because the wire
protocol is the same; we just need to point at the worldlines URL.

## Where we have something they don't

Honest list, this is short:

1. **Locally-hosted training backend.** Worldlines runs on our own
   GPUs. ttt_discover uses managed Tinker (billable cloud API).
   For experimentation / privacy / cost iteration, ours is better.
2. **Verifier is a Python library, not a shell command.** We import
   `mech_bench.evaluate` directly. They `subprocess.run("pytest")`.
   Ours is faster (~0.2 s/rollout vs 5–30 s for pytest), but theirs
   is more flexible (any test framework, any reward command).

That's it for genuine wins. Most of our other "wins" are actually
"we built less."

## Concrete upgrade path

If we wanted to match ttt_discover's architecture for mech_bench,
ranked by engineering hours:

### Step 1: Use tinker_cookbook's `prepare_minibatch` (2-4 hours)

Replace `rl/train_grpo.py:_group_advantages` and the per-token weight
loop with a call to `tinker_cookbook.rl.train.prepare_minibatch`.
We get GAE, KL penalty, importance sampling, and group-relative
advantages all factored out. Their cookbook ships against tinker but
the trainer-protocol calls (`forward_backward`, `optim_step`) are
identical for worldlines.

### Step 2: Build a `mech_bench_runtime` analog of codex_runtime (1-2 days)

Mirror their structure:

```
rl/mech_bench_runtime/
    runtime.py          → spawn a per-rollout python subprocess that runs
                           `python -m mech_bench evaluate ...`
    state_manager.py    → tarball workspace (or just per-task scratch dir;
                           we don't really need git here)
    verifier.py         → wrap mech_bench evaluate as a CommandVerifier
    rollout.py          → orchestrate one rollout
    codex_collector.py  → K-parallel rollouts per task
    logging_proxy.py    → aiohttp /v1/chat/completions proxy (we don't
                           need full Responses API since we're not using
                           the actual Codex CLI), captures logprobs
    codex_training.py   → top-level loop
```

Most of this is straight port; the proxy is the biggest piece.

### Step 3: The HARDER architectural win — use real Codex CLI as our harness (3-5 days)

If we want to train a *Codex-style* coding agent on mech_bench, the
right move is to run **the actual Codex CLI in Docker per rollout**,
with the LoggingProxy sitting at `OPENAI_BASE_URL` pointing at SGLang
or worldlines, and let the agent edit files freely. Verifier becomes:

```bash
python -m mech_bench evaluate --task /workspace/task \
    --submission /workspace --full --allow-partial
```

This is the **real version** of "RL-train an agent that lives in a
harness." It's exactly ttt_discover's codex_runtime adapted for our
verifier. The big work:

1. Codex CLI doesn't natively talk to a local model; it talks to
   OpenAI. The LoggingProxy needs to translate Responses API ↔
   our model's chat format (we're using DeepHermes-3 / Hermes-3,
   not gpt-oss/Harmony). So we'd need our own
   `messages_to_chat_template` instead of their
   `messages_to_conversation` Harmony pipeline.
2. Each rollout is a docker container; we need a workspace template
   for each task (just unpack the task dir into /workspace).
3. Reward is the verifier output, but Codex doesn't get the
   verifier output as a tool call by default — we'd have to register
   `mech-bench evaluate` as a Codex tool, or post-hoc score after
   Codex exits.

This is the most aligned with how the user actually deploys agents.
It's also the most work.

### Step 4 (later): Migrate to tinker_cookbook entirely

ttt_discover is built on top of tinker_cookbook. If we did, our
`train_grpo.py` would become a 50-line config that imports from
`tinker_cookbook.recipes.code_rl` or `recipes.ttt` and points at
worldlines.

The cookbook (Apache-2.0, public) is at
`github.com/thinking-machines-lab/tinker-cookbook`. It already
supports Qwen3 family, has a working `math_rl` recipe, and the
`code_rl` recipe is the closest analog to what we want.

## Bottom line

ttt_discover answers the question "**how do you actually RL-train an
LLM agent that runs in a coding-agent harness?**" with a complete
implementation. The architectural pattern is exactly what I sketched
yesterday: the production binary IS the rollout, log every token
the agent emits via an API-intercepting proxy, snapshot workspace
state, train with proper advantage / KL infrastructure.

Our mech_sim RL stack has the same conceptual loop but with a
hand-rolled, weaker harness (multi-turn chat + verifier feedback
injected as a user message). It works as a smoke test — cook06
proved gradients flow end-to-end — but it's not the production-style
agent-RL setup.

Concrete next move (recommended): **Step 1** above (swap in
`prepare_minibatch`) is half a day's work and gives us proper PPO
math. Everything else is bigger investments that should be staged
behind a clear "do we want a Codex-style agent or a smart
one-shot model?" decision.
