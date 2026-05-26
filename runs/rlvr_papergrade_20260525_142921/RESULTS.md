# RLVR Paper-Grade Run Summary

Date: 2026-05-26

Model: `NousResearch/DeepHermes-3-Llama-3-3B-Preview`

## Benchmark

- Generated hard suite: `tasks_hard_v1`
- Heldout split: `hard_v1_eval_visible_s0038_s0039.txt`
- Heldout size: 42 tasks = 21 validated families x 2 task seeds
- Reference validity: `reference_hard_v1_public` passed 250/250 public references.
- Evaluation setting: `samples_per_task=1`, `max_turns=2`, `max_tokens=3000`, `temperature=0.2`, `top_p=0.95`, `pass_threshold=0.999`.
- Final serving path: SGLang `/v1/chat/completions` for all reported rows. Adapter rows use `sglang_lora_path`; prompted baseline uses the base model without an adapter.

Final aggregate artifact:

`papergrade_final_clean_aggregate.json`

Final RLVR adapter:

`worldlines://model_cc562c22f4dc/sampler_weights/hard_v1_rlvr_visible_residual_from_sft_state_s8_cap2_final`

Reference SFT adapter:

`worldlines://model_bb6988118204/sampler_weights/hard_v1_ref_sft_r8_lr2e5_w1_s250_final`

## Main Result: Prompted Baseline vs RLVR

All rows are clean retry-fixed direct SGLang chat runs with zero `sampler_error` strings in the included summaries.

| Seed | Prompted baseline | RLVR adapter | Delta |
| ---: | ---: | ---: | ---: |
| 910 | 18/42 | 41/42 | +23 |
| 911 | 21/42 | 42/42 | +21 |
| 912 | 23/42 | 42/42 | +19 |

Mean prompted baseline: **20.67/42 = 49.21%**

Mean RLVR adapter: **41.67/42 = 99.21%**

Mean paired gain: **+21.00 tasks = +50.00 percentage points**

Paired seed-task test: RLVR wins 64 discordant seed-task pairs, prompted baseline wins 1, both pass 61, both fail 0. Exact two-sided sign-test `p = 3.58e-18`.

This supports the paper claim that the RLVR-trained adapter strongly improves verified mechanical-design task success over prompted base-model rollouts on this heldout suite.

## SFT Ablation

| Seed | Reference SFT | RLVR adapter | Delta |
| ---: | ---: | ---: | ---: |
| 910 | 38/42 | 41/42 | +3 |
| 911 | 39/42 | 42/42 | +3 |
| 912 | 39/42 | 42/42 | +3 |

Mean reference SFT: **38.67/42 = 92.06%**

Mean RLVR adapter: **41.67/42 = 99.21%**

Mean paired gain over SFT: **+3.00 tasks = +7.14 percentage points**

Paired seed-task test: RLVR wins 10 discordant pairs, SFT wins 1, both pass 115, both fail 0. Exact two-sided sign-test `p = 0.0117`.

This supports a modest but real improvement over the reference SFT adapter on the matched direct-SGLang heldout evaluation. The larger effect is still against prompted base-model rollouts.

## SFT vs Prompted Baseline

Reference SFT also strongly improves over the clean prompted baseline:

- Mean SFT gain: **+18.00 tasks = +42.86 percentage points**
- Paired seed-task counts: SFT wins 58, baseline wins 4, both pass 58, both fail 6.
- Exact two-sided sign-test `p = 2.59e-13`.

## RLVR Remaining Failures

Across the final direct-SGLang seeds 910-912, RLVR has 1 total failure:

- `rack_pinion_conversion_s0038`: fails on seed 910.

## Audit Notes

- `papergrade_final_clean_aggregate.json` reports `all_included_summaries_sampler_error_count = 0`.
- The earlier prompted-baseline table in `papergrade_threeway_aggregate_completed.json` is superseded; those old runs contained transport/context sampler errors that were scored as invalid Python before the retry fix.
- `papergrade_clean_baseline_vs_rlvr_aggregate.json` is also superseded; it used clean prompted baselines but older Worldlines-sampling RLVR summaries.
- `rl/sample_and_score.py` now records backend, seed, task root, split file, sampler retry count, and optional `sglang_lora_path` in each `smoke_summary.json`.
- `rl/sample_and_score.py` now retries transport/context sampler errors and avoids writing `[sampler_error: ...]` into `design.py`.
- `rl/chat_rollout.py` now supports adapter-aware SGLang chat rollouts via `lora_path`; this fixed the blocked Worldlines polling path and made the final SFT/RLVR tables serving-path matched.

## Reproduction Commands

Prompted baseline:

```bash
for seed in 910 911 912; do
  PYTHONUNBUFFERED=1 .venv/bin/python rl/sample_and_score.py \
    --base-url http://127.0.0.1:30000 --api-key sglang-local \
    --base-model NousResearch/DeepHermes-3-Llama-3-3B-Preview \
    --rollout-backend sglang_chat \
    --tasks runs/rlvr_papergrade_20260525_142921/tasks_hard_v1 \
    --split-file runs/rlvr_papergrade_20260525_142921/hard_v1_eval_visible_s0038_s0039.txt \
    --samples-per-task 1 --concurrency 1 \
    --max-turns 2 --max-tokens 3000 --temperature 0.2 --top-p 0.95 \
    --seed "$seed" --timeout 300 --sampler-retries 2 --pass-threshold 0.999 \
    --report-dir "runs/rlvr_papergrade_20260525_142921/baseline_hard_v1_visible_eval_seed${seed}_bof1_turn2_contractv2_retryfix"
done
```

Reference SFT:

```bash
for seed in 910 911 912; do
  PYTHONUNBUFFERED=1 .venv/bin/python rl/sample_and_score.py \
    --base-url http://127.0.0.1:30000 --api-key sglang-local \
    --base-model NousResearch/DeepHermes-3-Llama-3-3B-Preview \
    --model-path worldlines://model_bb6988118204/sampler_weights/hard_v1_ref_sft_r8_lr2e5_w1_s250_final \
    --sglang-lora-path worldlines_6eb64c00c9fa23d7 \
    --rollout-backend sglang_chat \
    --tasks runs/rlvr_papergrade_20260525_142921/tasks_hard_v1 \
    --split-file runs/rlvr_papergrade_20260525_142921/hard_v1_eval_visible_s0038_s0039.txt \
    --samples-per-task 1 --concurrency 1 \
    --max-turns 2 --max-tokens 3000 --temperature 0.2 --top-p 0.95 \
    --seed "$seed" --timeout 300 --sampler-retries 2 --pass-threshold 0.999 \
    --report-dir "runs/rlvr_papergrade_20260525_142921/ref_sft_r8_s250_visible_eval_seed${seed}_bof1_turn2_contractv2_retryfix_sglang_lora"
done
```

RLVR:

```bash
for seed in 910 911 912; do
  PYTHONUNBUFFERED=1 .venv/bin/python rl/sample_and_score.py \
    --base-url http://127.0.0.1:30000 --api-key sglang-local \
    --base-model NousResearch/DeepHermes-3-Llama-3-3B-Preview \
    --model-path worldlines://model_cc562c22f4dc/sampler_weights/hard_v1_rlvr_visible_residual_from_sft_state_s8_cap2_final \
    --sglang-lora-path worldlines_c54db85c4d4079d6 \
    --rollout-backend sglang_chat \
    --tasks runs/rlvr_papergrade_20260525_142921/tasks_hard_v1 \
    --split-file runs/rlvr_papergrade_20260525_142921/hard_v1_eval_visible_s0038_s0039.txt \
    --samples-per-task 1 --concurrency 1 \
    --max-turns 2 --max-tokens 3000 --temperature 0.2 --top-p 0.95 \
    --seed "$seed" --timeout 300 --sampler-retries 2 --pass-threshold 0.999 \
    --report-dir "runs/rlvr_papergrade_20260525_142921/rlvr_s8_cap2_visible_eval_seed${seed}_bof1_turn2_contractv2_retryfix_sglang_lora"
done
```
