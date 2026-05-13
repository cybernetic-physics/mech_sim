# RL training on mech_bench

Train a small open-weights model (Qwen3 0.6B or 1.7B) on the
mech_bench verified-reward signal, using the user's
**Worldlines** backend for inference + LoRA training (Tinker-wire-
compatible) and a single-shot GRPO rollout that calls
`mech_bench.evaluate` as the reward function.

## Why this stack

| Concern | Choice | Why |
|---|---|---|
| Inference + LoRA train backend | **Worldlines** (`cybernetic-physics/worldlines`, private) | Tinker-API-compatible single-node backend; serves LoRA adapters via SGLang; the user owns this. |
| RL algorithm | **GRPO** (group-relative, no value head) | Native fit for verifiable rewards. Works without a separate value model — cheaper to run on a 0.6B base. |
| Reward source | `python -m mech_bench evaluate` | Probe pipeline already deterministic and JSON-serialized; we read `score` + `hard_gate_passed` + `evaluation_valid`. |
| Base model | `Qwen/Qwen3-0.6B` (then 1.7B) | Best small-model coder per the 2026 leaderboards; available with permissive licensing; runs on 1× 24 GB GPU. |
| Rollout style | Single-shot generate→score (Tier 0/2 first), then multi-turn agent loop for Tier 1/3 later | Tier 0/2 (analytic) tasks need no tool use — emit one Python block, parse, score. Tier 1/3 with paths and contacts benefit from tool-using rollouts. |

We considered TRL GRPOTrainer (cleanest "single-shot, verifier-reward"
fit) and SkyRL-Agent (built for multi-turn coding agents). We're
going Worldlines-first because (a) the user already runs that stack
and (b) it gives us LoRA serving + training in one process, which
collapses the GRPO rollout/update loop to one HTTP boundary.

Both TRL and SkyRL-Agent remain easy fallbacks — see
`docs/rl_alternatives.md` (TBD) for the swap.

## Pieces in this directory

- `train_grpo.py` — top-level GRPO loop. Pulls a batch of tasks,
  samples K completions per task from Worldlines, scores each with
  `mech_bench.evaluate`, computes group-relative advantages, and
  pushes a `forward_backward` + `optim_step` per minibatch.
- `mech_bench_reward.py` — extracts a `design.py` from a model
  completion (parses the first `\`\`\`python ... \`\`\`` block),
  writes it to a scratch dir, invokes `python -m mech_bench
  evaluate --allow-partial --full`, and returns
  `{score, hard_gate_passed, evaluation_valid, failure_codes}`.
- `prompt_format.py` — shared prompt builder: takes a task dir
  and returns a chat-template-ready message list (system prompt
  in `scripts/agent_system_prompt.md`, user prompt = `prompt.md`
  + a fenced `task.toml`).
- `configs/qwen3_0p6b.yaml` — base model, LoRA hyperparameters,
  rollout K, learning rate, optimizer, KL coefficient.
- `configs/qwen3_1p7b.yaml` — scaled-up variant.

## Status

| Step | State |
|---|---|
| Worldlines clone + dep install | done (`/home/freiza/worldlines`, venv at `/dev/shm/wld-venv`) |
| Worldlines backend launch w/ Qwen3-0.6B base | in progress |
| `mech_bench_reward.py` parser + scorer | TBD |
| `train_grpo.py` rollout/update loop | TBD |
| First training run (Tier 0 only, 100 steps) | TBD |
| Eval improvement vs zero-shot Qwen3-0.6B | TBD |

## Run plan

1. **Smoke**: 50 tasks × K=1 sample × no learning, just to confirm
   Qwen3-0.6B can connect, sample, parse, and score. Establishes
   the zero-shot baseline.
2. **Tier 0 GRPO**: 13 tasks × K=8 × 200 steps, LoRA rank 16.
   Expected: pass rate climbs from <10 % to >50 % on the
   declared-ratio tasks, demonstrating the loop works.
3. **All tiers GRPO**: 50 tasks × K=4 × longer.
4. **Move to Qwen3-1.7B**, same configs, retrain.

## Reward shaping

`mech_bench.evaluate` returns `score ∈ [0, 1]` with a hard-gate
binary. The reward we pass into GRPO is:

```
reward = score if (evaluation_valid and hard_gate_passed) else 0.0
```

i.e. the **verified_score** general metric mech_bench already
emits. Failure-code information goes into the prompt history (so
the model sees its own mistakes), not into the reward.
