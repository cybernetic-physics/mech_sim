# MechanicalEvolve Experiment Goal

This file is the execution contract for `/goal pursue goals.md`.

The goal is an experiment, not a document-writing exercise. Markdown summaries
may be produced after the experiment, but they do not count as completion. The
completion unit is a frozen benchmark, executed matched-budget runs, raw
verifier evidence, statistical analysis, and a binary answer to the hypothesis
below.

## Experimental Hypothesis

Primary hypothesis:

> On held-out motion-transmission mechanism families, online GRPO/LoRA updates
> from executable mechanical verifier rewards increase verified mechanism
> repair success over an identical no-update verifier-feedback loop under the
> same actual verifier budget.

Primary null hypothesis:

> Under the same actual verifier budget, `mechanical_evolve_ttrl` is no better
> than `llm_evolve_no_update` on held-out mechanism repair.

Operational success threshold:

> At `B = 32` actual verifier calls per task/seed/method cell,
> `mechanical_evolve_ttrl` must improve held-out verified repair success over
> `llm_evolve_no_update` by at least 15 percentage points, and must have a
> positive paired `best_verified_reward` delta with statistical support.

This is the central experiment. Frozen model, SFT, verifier-gated, and
best-of-K search baselines are required controls, but the paper-level claim
fails if TTRL cannot beat `llm_evolve_no_update`.

## What Makes This Non-Toy

The final benchmark must not be a pile of one-parameter analytic puzzles.

Every final task must require a mechanism program with at least two independent
classes of constraints:

- topology or mobility: correct bodies, joints, mobility, and grounded
  input/output ports;
- functional behavior: ratio, stroke, path, transmission direction, timing, or
  other kinematic/mechanical target;
- artifact validity where applicable: generated CAD/trusted assets, materials,
  mass properties, and provenance;
- contact/dynamics where applicable: lockup, penetration, contact force,
  output speed, or power/ratio metrics.

A task that can be solved by setting one scalar in metadata is a diagnostic or
warmup, not a final benchmark task.

The final benchmark must include enough physically meaningful mechanism
structure that the model has to repair mechanical abstractions, not just match
strings.

## Experimental Population

Primary families:

- `belt`
- `chain`
- `rack_pinion`
- `lead_screw`
- `planetary`
- `fourbar`
- `slider_crank`
- `cycloidal`

Conditional contact stress-test families:

- `cam_follower`
- `contact_gear_pair`
- `rack_pinion_contact`

The primary claim is about the eight primary families. Contact stress-test
families can strengthen the paper only if their real verifier path is stable
and reference-valid. If contact families fail, report them as limitations; do
not hide them and do not make them the main claim.

Static fit families such as `box_lid_register_fit`,
`standoff_pattern_square`, `snap_tab_clearance_static`, and
`press_fit_hub_interference` may be used as diagnostics or ablations. They must
not carry the headline claim.

## Benchmark Size

Build and freeze `MechanismRepair-TTRL`.

Minimum final benchmark:

- 8 primary families;
- at least 5 final tasks per primary family;
- at least 40 final tasks total;
- at least 3 stochastic seeds per method;
- 2 family-held-out splits;
- passing reference solution for every final task;
- at least one negative control per final task.

The resulting minimum evaluation size is:

- Split A unseen cells: 4 unseen families x 5 tasks x 3 seeds = 60 cells per
  method.
- Split B unseen cells: 4 unseen families x 5 tasks x 3 seeds = 60 cells per
  method.
- Total primary held-out cells: 120 task/seed cells per method.
- Six required methods means at least 720 method/task/seed cells.
- At `B = 32`, that is at least 23,040 actual verifier-call slots before
  retries.

If the benchmark cannot meet this scale with reference-valid non-toy tasks,
build the benchmark first. Do not run a final paper job on a toy substitute.

## Frozen Splits

Split A:

- seen families: `belt`, `chain`, `rack_pinion`, `fourbar`
- held-out families: `planetary`, `lead_screw`, `slider_crank`, `cycloidal`

Split B:

- seen families: `planetary`, `lead_screw`, `fourbar`, `slider_crank`
- held-out families: `belt`, `chain`, `rack_pinion`, `cycloidal`

These splits may change only before preflight and only for a documented
experimental reason, such as a primary family lacking a reference-valid
non-toy verifier after real engineering effort. Replacements must be
motion-transmission or planar-linkage mechanism families, not static-fit
families.

No held-out family task, near duplicate, reference solution, repair trace, or
verifier-derived exemplar may appear in seen-family SFT data, TTRL warmup data,
prompt examples, model-selection data, or debugging examples.

## Protocol

The experiment is a prequential online repair protocol.

For each split:

1. Train or initialize all seen-family artifacts using seen families only.
2. Freeze model choices, prompts, hyperparameters, task order, verifier
   thresholds, and budget.
3. Evaluate methods on the held-out-family stream.
4. Each method receives the same actual verifier budget for the same
   task/seed cell.
5. `llm_evolve_no_update` receives the same multi-turn verifier feedback as
   TTRL but never changes weights.
6. `mechanical_evolve_ttrl` may update its adapter only from verifier results
   already observed in the current online stream or current task.
7. No method may use future held-out tasks or labels.
8. Report both per-task reset performance and stream-carryover performance if
   stream-carryover updates are used.

The essential causal contrast is:

```text
same base model
same tasks
same task order
same prompts
same verifier feedback
same actual verifier calls
same CAD/Chrono calls where applicable
different only in whether verifier rewards update LoRA weights
```

## Required Methods

Run all six:

1. `frozen_model`
   - no updates;
   - same budget.

2. `verifier_gated`
   - low-temperature or verifier-gated proposal baseline;
   - no updates;
   - same budget.

3. `no_update_search`
   - high-temperature best-of-K or search baseline;
   - no updates;
   - same budget.

4. `llm_evolve_no_update`
   - same multi-turn verifier-feedback loop as TTRL;
   - no LoRA updates;
   - no RL updates;
   - same budget;
   - primary baseline.

5. `sft_model`
   - supervised adapter from seen-family data only;
   - no test-time verifier updates;
   - same budget.

6. `mechanical_evolve_ttrl`
   - same base model as the baselines;
   - exact TRL `GRPOTrainer` path in `rl/train_true_grpo_trl.py`, or a
     predeclared equivalent policy-ratio-clipped GRPO implementation;
   - verifier-derived LoRA updates;
   - nonzero `adapter_updates`, `trained_tokens`, `rl_trained_tokens`, and
     `n_rl_datums`;
   - same task order, prompts, verifier, and actual budget as
     `llm_evolve_no_update`.

## Budget

Primary budget:

- `B = 32` actual verifier calls per task/seed/method cell.

Budget curve if compute permits:

- `B = 8`
- `B = 16`
- `B = 32`
- optional `B = 64`

Actual budget accounting must be per task/seed/method cell:

- `candidate_count`
- `verifier_calls`
- `cad_audits`
- `chrono_audits`
- `sampler_error_count`
- `invalid_artifact_count`
- `timeout_count`
- `audit_retry_count`
- `planned_max_verifier_calls`
- `actual_budget_match_group`

TTRL cannot support the claim if it uses more actual verifier, CAD, or Chrono
calls than the matched baseline in the same cell.

## Verifier Levels

Each task must declare the verifier level it contributes to.

Level 1: mechanism-program verifier.

- DesignIR validity;
- mobility;
- required parts/joints;
- grounded input/output ports;
- ratio, stroke, path, or direction constraints.

Level 2: trusted CAD/artifact verifier.

- nonempty generated geometry where applicable;
- trusted mass properties or explicit trusted preflight;
- material/provenance checks;
- no agent-declared physical quantities counted as truth.

Level 3: contact/dynamics verifier.

- real Chrono/contact execution;
- `contact_model=smc` where contact is required;
- `procedural_cycloidal_fallback=false`;
- lockup, penetration, contact force, output speed, ratio, or power metrics.

The main claim may use Level 1 plus Level 2 if that is the stable final
benchmark. A stronger contact/dynamics claim requires Level 3 evidence. Do not
mix levels in the abstract claim.

## Primary Metrics

Primary outcome:

- `verified_repair_success_at_32`

Secondary primary outcome:

- `best_verified_reward_at_32`

Required secondary metrics:

- `first_valid_verifier_call`
- `strict_score_pass_rate`
- `wrong_mobility_rate`
- `missing_port_rate`
- `ungrounded_port_rate`
- `invalid_topology_rate`
- `invalid_artifact_rate`
- `cad_pass_rate`
- `chrono_real_geometry_rate`
- `no_procedural_fallback_rate`
- `lockup_rate`
- `contact_lockup_rate`
- `best_ratio_error_pct`
- `best_path_trace_error`
- `best_max_penetration_mm`
- `best_contact_force_rms_N`
- `adapter_updates`
- `trained_tokens`
- `rl_trained_tokens`
- `n_rl_datums`

## Statistical Test

Unit of analysis:

- paired task/seed/method cell within held-out families.

Primary test:

- TTRL minus `llm_evolve_no_update` on `verified_repair_success_at_32`.

Success requires:

- absolute improvement >= 15 percentage points;
- paired bootstrap 95 percent confidence interval lower bound > 0, or a
  predeclared paired permutation/sign test with `p <= 0.05`;
- positive paired delta on `best_verified_reward_at_32`;
- TTRL beats every other required baseline on held-out
  `best_verified_reward_at_32`;
- TTRL wins in a majority of held-out primary families;
- removing any one held-out primary family does not flip the sign of the TTRL
  versus `llm_evolve_no_update` reward delta.

If these tests fail, the hypothesis is unsupported. Do not relabel the result
as success by switching metrics after the run.

## Mechanistic Test

The experiment must test why TTRL helped or failed.

Required analyses:

- first-attempt versus final-attempt failure-code changes;
- TTRL versus `llm_evolve_no_update` failure-code changes on identical cells;
- per-family failure-code deltas;
- at least 8 matched repair traces where both methods receive the same
  verifier feedback;
- evidence of whether TTRL improved topology, mobility, port grounding,
  ratio/path parameters, artifact validity, or contact behavior.

The paper-level insight must be about learned repair behavior, not only a
score table.

## Required Output Artifacts

The primary artifacts are run artifacts, not prose docs.

Required machine-readable artifacts:

- `runs/mechanism_repair_ttrl_final/benchmark_manifest.json`
- `runs/mechanism_repair_ttrl_final/split_manifest_A.json`
- `runs/mechanism_repair_ttrl_final/split_manifest_B.json`
- `runs/mechanism_repair_ttrl_final/verifier_manifest.json`
- `runs/mechanism_repair_ttrl_final/method_manifest.json`
- `runs/mechanism_repair_ttrl_final/raw_completions/`
- `runs/mechanism_repair_ttrl_final/verifier_outputs/`
- `runs/mechanism_repair_ttrl_final/training_logs/`
- `runs/mechanism_repair_ttrl_final/adapter_checkpoints/`
- `runs/mechanism_repair_ttrl_final/results.json`
- `runs/mechanism_repair_ttrl_final/results.csv`
- `runs/mechanism_repair_ttrl_final/stats.json`
- `runs/mechanism_repair_ttrl_final/failure_analysis.json`
- `runs/mechanism_repair_ttrl_final/trace_pairs.json`
- `runs/mechanism_repair_ttrl_final/claim_audit.json`

Derived markdown summaries are allowed, but they are not completion criteria.

`claim_audit.json` must contain exactly one of:

- `"claim_status": "supports_primary_hypothesis"`
- `"claim_status": "does_not_support_primary_hypothesis"`

## Completion Criteria

The goal is complete only when:

- the non-toy benchmark exists and passes reference/negative-control audits;
- both family-held-out splits are frozen;
- all six methods ran on both splits;
- all actual verifier budgets are matched per task/seed/method cell;
- TTRL used real verifier-derived GRPO/LoRA updates;
- raw completions and verifier outputs are preserved;
- statistics were computed from paired held-out task/seed cells;
- failure-mode and trace analyses were computed;
- `claim_audit.json` answers the primary hypothesis honestly.

If the hypothesis is unsupported but all the experimental criteria above are
met, the experiment is complete as a negative result. If the goal is to produce
a positive CoRL paper result, keep improving the benchmark/method and rerun
under a new frozen contract instead of pretending a failed hypothesis passed.

## What Not To Do

- Do not count writing docs as progress toward the experimental claim.
- Do not count a seven-task pilot as success.
- Do not count static-fit tasks as headline motion-transmission evidence.
- Do not use held-out tasks for prompt tuning, hyperparameter selection, or
  debugging examples.
- Do not hide failed contact families.
- Do not change the verifier after seeing final results unless all affected
  methods are rerun.
- Do not let TTRL have more actual verifier/CAD/Chrono calls.
- Do not claim hardware readiness.

No corner cutting: the thing to accomplish is the experiment.
