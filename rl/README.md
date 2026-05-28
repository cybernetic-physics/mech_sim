# RL Training On mech_bench

There are now two distinct trainer paths. Do not describe them as the same
algorithm.

## Exact GRPO

Use this when the claim says GRPO:

```bash
uv run --extra training-grpo python rl/train_true_grpo_trl.py \
  --model <hf-causal-lm-or-local-path> \
  --output-dir runs/true_grpo/<run_name> \
  --split-file <train_split.txt> \
  --num-generations 4 \
  --max-steps 100
```

Implementation:

- `rl/train_true_grpo_trl.py`
- Hugging Face TRL `GRPOTrainer`
- PEFT LoRA
- verifier reward from `rl.mech_bench_reward.score_completion`
- reward = `verified_score * reward_scale`
- no learned value head
- uses TRL's GRPO objective, including policy-ratio clipping/KL handling as
  implemented by TRL

Install path:

```bash
uv sync --extra training-grpo
```

or invoke directly with:

```bash
uv run --extra training-grpo python rl/train_true_grpo_trl.py ...
```

## Legacy Worldlines Trainer

`rl/train_grpo.py` is retained for reproducibility of old branch results, but
it is not exact GRPO.

It performs:

- K rollouts per task
- deterministic verifier scoring
- group-relative normalized rewards
- advantage/reward-weighted cross-entropy on final assistant tokens
- Worldlines LoRA optimizer step

It does not perform:

- policy-ratio clipped GRPO objective
- old-policy/current-policy log-prob ratio optimization
- PPO value-function training

Refer to it as:

```text
group-relative verifier-weighted CE LoRA
```

not as GRPO.

## Reward Contract

The verifier reward source is the existing deterministic benchmark path:

```text
completion -> design.py -> python -m mech_bench evaluate --full --allow-partial
```

`verified_score` is nonzero only when the generated artifact is evaluation-valid
and passes the hard gate. The exact-GRPO trainer logs every reward call to
`reward_log.jsonl` in the run directory.
