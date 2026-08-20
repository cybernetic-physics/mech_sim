# Eval results

> **Frozen debugging snapshot.** One comparison arm suffered 26 pre-inference
> environment failures, so these files must not be used as a clean model
> leaderboard. They remain useful for reproducing the harness and diagnosing
> submission failure modes.

This directory holds frozen agent-vs-benchmark scorecards. Each row
is one sweep of the 50 generated tasks + the hand-written
`fourbar_path_t001` reference (= 51 tasks).

## Method

`scripts/run_claude_on_eval.py` and `scripts/run_codex_on_eval.py`
each spawn the agent (Claude Code `claude -p` or OpenAI Codex
`codex exec`) per task with the same system prompt
(`scripts/agent_system_prompt.md`) and a tight tool allowlist. The
agent reads `prompt.md` + `task.toml` and writes one `design.py`
into a per-task scratch dir; `python -m mech_bench evaluate` then
runs the verifier and the harness records the score, cost, and
failure codes. `scripts/compare_agent_evals.py` diffs two such
runs.

## Runs so far

| Date | Agent | Model | n_tasks | passed | rate | cost | wall |
|---|---|---|---|---|---|---|---|
| 2026-05-13 | Claude Code | sonnet (claude-sonnet-4-6) | 51 | 13 | 25.5%* | $2.48 | 8.5 min |
| 2026-05-13 | Codex CLI | gpt-5.5 | 51 | 34 | 66.7%  | $2.44 | 10.4 min |

\* The Claude run at concurrency=6 was contaminated: 26 of its 51
tasks hit "`claude` not on PATH" before any API call (a
node-version / nvm interaction at high concurrency, not a model
failure). The 25 tasks that actually reached the API earned 13/25
≈ 52% — see the per-task JSON for details. Will rerun at
concurrency ≤ 3 for a clean number.

## Head-to-head (full 51 task overlap)

```
both pass            : 13
neither pass         : 17
only codex passes    : 21
only claude passes   :  0
agreement rate       : 58.8%
```

(With the Claude infra failures excluded, head-to-head would
approximately equal claude≈13/25, codex≈34/51 across the same
tasks.)

## Failure-code histograms

- claude: `invalid_artifact:5`, `wrong_topology:2`,
  `capability_unavailable:1`, `path_error:1`
- codex: `invalid_artifact:8`, `wrong_topology:5`, `path_error:3`,
  `capability_unavailable:2`, `wrong_ratio:1`

`wrong_topology` = the agent forgot to mark a ground part
`fixed=True`. `invalid_artifact` = the IR validation rejected the
shape (often a port pointing at a missing joint, or a non-finite
value). `path_error` = chamfer distance over threshold on a
coupler-path task. `capability_unavailable` = the agent did not
opt into the synthetic fake oracle on a Tier-3 stub.

## Reproducing

```bash
# Claude:
python scripts/run_claude_on_eval.py --tasks tasks \
    --report-dir /tmp/eval_claude --model sonnet \
    --max-budget-usd 0.30 --timeout 300 --concurrency 3

# Codex:
python scripts/run_codex_on_eval.py --tasks tasks \
    --report-dir /tmp/eval_codex --model gpt-5.5 \
    --timeout 300 --concurrency 4

# Compare:
python scripts/compare_agent_evals.py \
    --left  /tmp/eval_claude/claude_eval_summary.json \
    --right /tmp/eval_codex/codex_eval_summary.json \
    --left-label  claude:sonnet \
    --right-label codex:gpt-5.5 \
    --out evals/comparison_$(date -u +%Y%m%d).json
```
