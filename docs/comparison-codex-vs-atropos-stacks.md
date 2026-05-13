# Codex runtime vs Atropos / Hermes-Agent — RL stack comparison

> Written 2026-05-13 after walking through `ttt_discover/codex_runtime`,
> `hermes-agent/environments/hermes_base_env.py`, and
> `hermes-agent/tinker-atropos`. Both stacks solve the same problem:
> *RL-train an LLM that runs inside an agent harness*. They make
> different architectural bets. This doc is the honest comparison.

## Same goal, two architectures

| Stack | "What the rollout is" |
| --- | --- |
| **ttt_discover / codex_runtime** | spawn the *real production binary* (`@openai/codex` in Docker) per rollout; intercept its OpenAI Responses API with a proxy; trainer runs in the same Python process |
| **Atropos + Hermes-Agent** | each rollout is a method (`collect_trajectory`) on an env subclass that *embeds* the harness logic; env runs in a server, trainer runs in a different server, they talk over an HTTP API |

Both achieve "the harness IS the rollout" but **codex_runtime embeds the
harness at the binary level** while **Atropos/Hermes embeds it at the
class level**.

## ttt_discover / codex_runtime stack

```
PUCTSampler
   │
   ▼
GitStateManager           snapshot workspace
   │
   ▼
CodexRuntime              docker run @openai/codex
                          codex exec --json --full-auto
                          OPENAI_BASE_URL=LoggingProxy/rollout/{uuid}/v1
   │
   ▼
LoggingProxy              aiohttp server
                          Responses API ⇄ Harmony tokens
                          tinker SamplingClient
                          capture per-token logprobs
   │
   ▼
CommandVerifier           subprocess("pytest -q")
   │
   ▼
trace_converter           APICallRecord → Trajectory
   │
   ▼
prepare_minibatch         advantages, KL penalty, GAE
   │
   ▼
tinker train_step         forward_backward + optim_step
   │
   ▼
save_weights_and_get_sampling_client
                          swap proxy's sampling client → loop
```

Single Python process. One docker container per rollout. The training
loop and the rollout orchestrator share state in-process.

## Atropos + Hermes-Agent stack

```
Atropos API server           run-api      (Trajectory API)
   │
   │   (HTTP, async)
   ▼
HermesAgentBaseEnv           your env subclass
                             ├─ setup()
                             ├─ get_next_item()
                             ├─ format_prompt()
                             ├─ collect_trajectory()   ← AGENT LOOP
                             │   uses HermesAgentLoop, tools, MCP,
                             │   OpenAIServer or VLLM ManagedServer
                             └─ compute_reward()
   │
   │   POSTs ScoredDataGroup
   ▼
Atropos API server (same)    holds in-memory trajectory queue
   │
   │   GET batches
   ▼
TinkerAtroposTrainer         pulls batches from Atropos
                             tinker forward_backward + optim_step
                             new sampling client → atropos
                             ┌─────────────────────────────────┐
                             │ ALSO runs a FastAPI server with  │
                             │ /chat/completions /completions   │
                             │ /generate — so envs can sample   │
                             │ from the live model              │
                             └─────────────────────────────────┘
   │
   │   weight update event
   ▼
Atropos API server           tells envs "new weights ready,
                             use new sampling URL"
```

Three separate processes that talk over HTTP. Each env can be its own
subclass — `terminal_test_env`, `web_research_env`, `swe_env`,
`gsm8k_tinker`. The trainer is one process, the API is another, the
envs are N more (potentially across machines).

## Architectural diff

| Dimension | codex_runtime | Atropos / Hermes-Agent |
| --- | --- | --- |
| **# processes** | 1 (plus K transient Docker containers) | 3+ persistent (API + trainer + N envs) |
| **Distribution** | single host | designed for many-host env workers |
| **Rollout harness embedding** | spawn the production CLI binary in Docker per rollout | implement harness as a method in an env class; same in-process loop runs every rollout |
| **Logprob capture** | LoggingProxy intercepts Responses API live during sampling | env builds a `ScoredDataGroup` with sampled tokens + logprobs from the trainer's serving endpoint |
| **State / workspace** | full git-snapshotted workspace, freely-edited filesystem | per-`ToolContext` workspace (terminal_backend = modal, docker, local) — also full filesystem |
| **K rollouts** | `do_codex_group_rollout` in asyncio.gather, K parallel Docker containers | env-side `group_size` config; multiple workers run in parallel asyncio loops |
| **Token format** | openai-harmony (gpt-oss native) | whatever tokenizer the env's serving endpoint emits |
| **Sampler over tasks** | PUCT over state buffer | env decides — uniform, weighted, dataset iteration |
| **Advantage / loss math** | `tinker_cookbook.rl.train.prepare_minibatch` | `tinker_atropos/trainer.py` — atropos POSTs raw rewards, trainer normalises |
| **Trainer backend** | tinker | tinker (atropos is backend-agnostic; one could swap worldlines in) |
| **Concurrency-safe rollout isolation** | per-rollout URL `/rollout/{uuid}/v1/responses` | per-env-worker HTTP client, env state pinned to a worker_id |
| **Tool / agent loop richness** | whatever Codex CLI ships with (apply_patch, shell, network sandboxing) | configurable per env — `enabled_toolsets`, `tool_call_parser`, MCP servers, ACP adapters; full HermesAgentLoop with retry/critic |
| **Lines of glue code in their repo** | ~1500 (codex_runtime/) | atroposlib core ~5000, hermes_base_env ~750, hermes-agent harness ~30k |
| **Where weights live** | tinker cloud | tinker cloud (or any wire-compatible backend) |
| **License** | MIT (ttt_discover) | Apache-2.0 (atropos), Apache-2.0 (hermes-agent), Apache-2.0 (tinker-cookbook) |

## Where each stack wins

### codex_runtime wins

1. **Production-binary fidelity.** The literal `@openai/codex` Node
   binary is what runs. Zero behavioural drift between training and
   deployment. If your goal is "make Codex CLI better at task X",
   this is the lowest-overhead path: no env subclassing, no
   reimplementing tool semantics. Use the real CLI in Docker.

2. **Single-host operational simplicity.** One Python process, one
   Docker daemon. Spin up, smoke test, tear down. No API server
   to keep alive, no env workers to babysit.

3. **PUCT + state buffer for discovery problems.** Built for
   test-time-training search over a buffer of candidate solutions.
   The whole point of ttt_discover is "find a state s such that
   R(s) beats SOTA" — Atropos doesn't have a notion of starting
   from non-initial states; codex_runtime does.

4. **Per-token logprob capture is exact.** The proxy captures tokens
   as the sampler emits them. No re-tokenisation, no drift between
   "what the policy did" and "what we train on." Atropos envs
   roughly do the same but the boundary is the trainer's serving
   endpoint, not the agent's input prompt.

5. **Harmony format support out of the box.** gpt-oss native. If
   you want to train gpt-oss models inside the Codex CLI workflow,
   you're done. Atropos would need its own Harmony tokenizer plumbing.

### Atropos / Hermes-Agent wins

1. **Built for many parallel env workers across machines.** The
   `run-api` server can have N env workers connected, each
   producing trajectories asynchronously. Scales to fleets. codex_
   runtime is single-host (could be parallelised but isn't designed
   for it).

2. **Pluggable harness logic per env.** The same trainer can drive
   `gsm8k_tinker`, `terminal_test_env`, `web_research_env`,
   `agentic_opd_env`, `swe_env` — different agent loops, different
   tools, different reward functions, all behind one `BaseEnv`
   interface. codex_runtime is purpose-built for one harness
   (Codex CLI).

3. **HermesAgentLoop is a much richer agent than Codex CLI.**
   - Configurable toolsets (`enabled_toolsets` per env)
   - Tool-call parser swappable (`hermes`, `llama3`, `qwen`)
   - MCP server support (Model Context Protocol)
   - ACP (Agent-to-Agent Communication Protocol) adapters
   - Retry / critic / planner sub-agents
   - Two-mode operation (OpenAI server or VLLM ManagedServer for
     Phase 2 RL where the model's own logprobs are computed
     in-process)

4. **Env library reusability.** Any Atropos env from
   `github.com/NousResearch/atropos/environments` plugs into
   tinker_atropos with `python /path/to/env.py serve --config
   your_config.yaml`. Hundreds of community envs.

5. **WandB-first.** The trainer logs to WandB by default; the docs
   assume it. codex_runtime is more "files on disk." Both
   workable, but for shared-team-run visibility WandB wins.

6. **Distributed sampling.** When the env wants the live model's
   completions, it hits the trainer's `/v1/chat/completions`
   endpoint (the trainer doubles as an OpenAI-compatible server).
   So you can run dozens of env workers across machines all
   sampling from the same live policy.

7. **Async-native.** Atropos uses asyncio end-to-end. codex_
   runtime is also async-aware but the per-rollout Docker spawn
   is the bottleneck.

## What we'd build for each path on mech_sim

If we wanted to **adopt codex_runtime style** for mech_sim:

```
rl/mech_runtime/
    runtime.py        spawn python -m mech_bench evaluate per rollout
                      (or actually spawn @openai/codex with our SGLang
                      proxy if we want a real coding agent)
    state_manager.py  tarball workspace per rollout
    verifier.py       mech_bench.evaluate as a CommandVerifier
    rollout.py        single rollout orchestrator
    logging_proxy.py  intercept SGLang's chat-completions, log tokens
    training.py       tinker_cookbook prepare_minibatch + train_step
```

Estimate: 2 days port. Best for "smarter mech_bench one-shot" + the
specific TTT-Discover discovery pattern.

If we wanted to **adopt Atropos style**:

```
environments/mech_bench_env/
    mech_bench_env.py     subclass HermesAgentBaseEnv (or BaseEnv direct)
                          setup() = env.list_tasks(...)
                          get_next_item() = pop a task
                          format_prompt() = prompt.md + task.toml
                          collect_trajectory() = harness loop here
                          compute_reward() = mech_bench.evaluate
    default.yaml          env config (tokenizer, group_size, etc.)
```

Plus stand up:
- `run-api` (Atropos API server)
- `launch_training.py --config configs/mech.yaml` (the trainer)
- `python environments/mech_bench_env/mech_bench_env.py serve --config default.yaml` (the env worker)

Estimate: 3-4 days port (more infrastructure to stand up, but you
inherit the entire HermesAgentLoop / tool stack / community envs).

## Which is right for us — honest take

For the specific goal of "**RL-train a small model to be good at
mech_bench**", these are the two reasonable paths:

### Recommended: **codex_runtime style**, adapted

Reasons:
- Our verifier is a Python function call (~0.2 s), not a shell
  command needing pytest (~5 s). The single-process architecture
  is fast for that.
- Our base model is DeepHermes-3 / Qwen3, not gpt-oss. The
  Harmony piece is replaceable but we don't need its specific
  benefits.
- We already have most of it built. `rl/train_grpo.py` is the
  trainer; `rl/chat_rollout.py` is a primitive `LoggingProxy`
  analog; `rl/mech_env.py` is the `CommandVerifier`+state manager.
  Two days of work to factor it cleanly into the codex_runtime
  shape gives us all the benefits.
- `tinker_cookbook.rl.train.prepare_minibatch` works against
  worldlines (same Tinker wire protocol) — half a day for proper
  PPO + KL math.

### Better for "train a real coding agent": **Atropos style**

Reasons:
- If we want to RL-train a model that USES tools (write_file,
  run_pytest, etc.) the way Codex or Claude Code actually do
  in production, we need a proper agent loop. HermesAgentLoop
  is that loop, with tool-use already plumbed.
- We can pick up community envs (swe_env, terminal_test_env)
  for free and use them as a curriculum AROUND mech_bench.
- For research-quality RL we'd eventually want WandB, distributed
  env workers, and config-driven experiments. Atropos has them.

### Worst path (don't do this): keep our hand-rolled stack as-is

Our current `rl/train_grpo.py` is fine for "verify the loop works"
but it's missing proper advantage math, KL penalty, importance
sampling, a real tool-use harness, and dashboard support. Spending
more weeks adding those features one-at-a-time is reimplementing
either codex_runtime or atroposlib badly.

## One-paragraph TL;DR

**codex_runtime is a small, sharp, single-host tool: spawn the real
production binary per rollout, intercept its API, train. Atropos +
Hermes-Agent is a bigger, distributed, env-pluggable platform: write
a Python class with an agent-loop method, plug it into a trainer that
can drive many env workers across many machines.** For mech_sim the
right move depends on whether the goal is "make our verifier-scored
benchmark slightly better-trained" (codex_runtime style, half a week)
or "build a real coding agent that uses tools on mech_bench" (Atropos
style, 1-2 weeks).
