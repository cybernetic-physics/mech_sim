#!/usr/bin/env bash
set -euo pipefail

# Portable validation for the evaluator and control plane. Native-solver,
# training-stack, and frozen experiment-replay suites have separate dependency
# and artifact requirements documented in CONTRIBUTING.md.
uv run pytest -q \
  --ignore=tests/test_benchmark.py \
  --ignore=tests/test_chat_rollout.py \
  --ignore=tests/test_family_generalization_results.py \
  --ignore=tests/test_generators.py \
  --ignore=tests/test_mechanism_repair_physics_experiment.py \
  --ignore=tests/test_sample_and_score_pass_metrics.py \
  --ignore=tests/test_train_sft_peft.py \
  --ignore=tests/test_train_true_grpo_trl.py
