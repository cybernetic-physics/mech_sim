# Verifier-Guided Learning

The learning layer turns a generated completion into `design.py`, evaluates it
through the same trusted benchmark path used for offline scoring, and returns a
gated reward plus structured repair feedback.

There are two trainer paths. They are not the same algorithm and should not be
reported under one label.

## Reward contract

```text
completion
→ extract design.py
→ isolated build and validation
→ configured probes and adapters
→ hard gate
→ verified score, public feedback, and scalar channels
```

The compact API lives in `mech_bench.rlvr.evaluate_for_rlvr`. Its reward is
zero when evaluation is invalid or a hard gate fails. The payload also records
whether any selected adapter was synthetic, so an experiment can exclude or
separately analyze that signal.

This contract prevents a model from receiving positive reward for output the
verifier could not safely interpret. It does not by itself prevent every form
of reward hacking; held-out configurations and anti-shortcut variants remain
necessary.

## Exact GRPO path

Use this path when a result is described as GRPO:

```bash
uv sync --extra training-grpo

uv run --extra training-grpo python rl/train_true_grpo_trl.py \
  --model <hf-causal-lm-or-local-path> \
  --output-dir runs/true_grpo/<run-name> \
  --split-file <train-split.txt> \
  --num-generations 4 \
  --max-steps 100
```

Implementation:

- Hugging Face TRL `GRPOTrainer`;
- PEFT LoRA;
- deterministic verifier reward from
  `rl.mech_bench_reward.score_completion`;
- reward equal to `verified_score * reward_scale`;
- no learned value head; and
- policy-ratio clipping and KL handling as implemented by the pinned TRL
  version.

Every reward call is appended to `reward_log.jsonl` in the run directory.
Model, dependency, task split, sampling configuration, and verifier commit
should be frozen with any reported result.

## Multi-turn repair rollouts

`rl/chat_rollout.py` supports repeated completion–evaluation–feedback turns.
The model receives public failures and suggestions, not hidden thresholds or
private traces. A run should record:

- maximum turns and samples per task;
- actual verifier, CAD, and native-solver calls;
- sampler errors, timeouts, and replacement retries;
- the best and final verified outcome; and
- whether a successful design appeared only after feedback.

Matched-budget comparisons must count actual expensive calls, including failed
attempts, rather than only planned calls.

## Legacy Worldlines path

`rl/train_grpo.py` is retained to reproduce early experiments. It performs:

- multiple rollouts per task;
- deterministic verifier scoring;
- group-relative normalized rewards;
- advantage/reward-weighted cross-entropy on assistant tokens; and
- a Worldlines LoRA optimization step.

It does **not** implement policy-ratio-clipped GRPO, old/current-policy ratio
optimization, or PPO value-function training. Describe it as:

```text
group-relative verifier-weighted CE LoRA
```

The older training-cook analysis is preserved in
[`docs/results-and-rl-roadmap.md`](../docs/results-and-rl-roadmap.md), with a
historical-status warning.

## Evidence and limits

The strongest checked-in learning result is the June 2026 Level-1
held-out-family experiment in
[`runs/mechanism_repair_ttrl_final`](../runs/mechanism_repair_ttrl_final/README.md).
Its claim audit supports improved executable mechanism-program repair under the
recorded matched-budget setup.

It does not establish CAD, contact-physics, manufacturing, or hardware
performance. The prepared Level-2/3 experiment bundle contains benchmark and
audit scaffolding but records zero executed result rows. See the
[project evidence ledger](../docs/project-status.md#evidence-ledger).

## Testing

The portable project check does not install PyTorch and excludes training-only
test modules:

```bash
scripts/check_core.sh
```

For changes to the exact trainer or rollout code, install the training group
and run the relevant modules explicitly:

```bash
uv sync --extra training-grpo
uv run --extra training-grpo pytest -q \
  tests/test_train_true_grpo_trl.py \
  tests/test_train_sft_peft.py \
  tests/test_chat_rollout.py
```

Do not infer native-solver coverage from these tests; physics dependencies and
evidence have their own validation tier.
