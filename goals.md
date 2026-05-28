# MechanicalEvolve / TTRL Paper Goal

This branch is aiming at a CoRL-grade result for RLVR/TTRL-driven mechanical
actuator discovery, specifically cycloidal/QDD actuator design. The claim is
not generic robot morphology optimization and not merely "RL improves designs."
The claim must be:

> Under equal expensive CAD + Chrono physics-verification budget, iterative
> RLVR/TTRL adaptation discovers stronger verified cycloidal actuator designs
> than non-updating baselines.

## Current Completed Foundation

The CAD-generated cycloidal reducer Chrono path is now validated enough to use
as the expensive verifier.

Evidence:

- Commit: `f18eb0e Validate CAD cycloidal Chrono SMC path`
- Proof command:

```bash
MECH_BENCH_FREECAD_CMD=$PWD/.external/bin/mech-freecadcmd \
MECH_BENCH_CYCLOID_GEARBOX_PATH=$PWD/.external/src/CycloidGearBox \
MECH_BENCH_CHRONO_PYTHON=$PWD/.external/micromamba/envs/mech-chrono/bin/python \
uv run --extra dev python scripts/prove_cycloidal_chrono_real_geometry.py \
  --out-dir runs/cycloidal_real_geometry_validation_passing \
  --proof-json runs/cycloidal_real_geometry_validation_passing/proof.json
```

Validated:

- FreeCAD/OCCT named STEP/STL asset generation.
- CAD datums and static contact audits.
- Trusted CAD mass properties.
- Chrono real-geometry runs with `procedural_cycloidal_fallback=false`.
- Unloaded SMC ratio near declared 9:1 ratio.
- Loaded SMC with bounded penetration and finite power/torque/contact metrics.
- Sample and timestep convergence checks.

This is necessary infrastructure, not the final paper result.

## Broader Paper Result Still Required

The paper-grade result requires a matched-budget experiment across methods.
Every method must use the same task family, same verifier thresholds, same CAD
pipeline, same Chrono SMC configuration, same random seeds, and the same total
Chrono audit budget.

Required methods:

1. `verifier_gated`
   - Search baseline using the current branch setup.
   - No model updates.
   - No LoRA.
   - Same Chrono audit budget as every other method.

2. `llm_evolve_no_update`
   - Uses the same LLM proposal/mutation path as MechanicalEvolve.
   - No adapter updates.
   - No RL updates.
   - No LoRA training.
   - Same Chrono audit budget.

3. `mechanical_evolve_ttrl`
   - Uses `mlx-community/Qwen3.6-35B-A3B-4bit`.
   - Performs actual iterative LoRA/TTRL updates.
   - Uses the same proposal format, CAD generator, DesignIR bridge, and Chrono
     verifier.
   - Same Chrono audit budget.

Optional reported baseline:

- `cma_es_fast_only`, if included, must be reported honestly as an additional
  baseline. It must not silently redefine the required win condition.

## Required Verifier Configuration

Unless a new experiment file explicitly supersedes this, the matched-budget
paper experiment should use:

- `contact_model=smc`
- `procedural_cycloidal_fallback=false`
- FreeCAD/CycloidGearBox CAD generation
- Chrono real-geometry audit
- `samples=41`
- `duration_s=0.15`
- `input_speed_rad_s=10.0`
- `output_load_Nm=0.75`
- same power, torque, contact, penetration, and ratio limits for every method
- target budget: `160` Chrono audits per method

Paper gates:

- finite `ratio_observed`
- `out_omega_med >= 0.5 rad/s` where applicable
- `max_penetration_mm < 1.0`
- `contact_force_rms_N <= 3000`
- `n_contacts_max <= 128`
- `ratio_error_pct <= 25`
- no lockup for verified successes
- no procedural fallback

## Required Output Artifacts

The final matched-budget result must write:

- `docs/cycloidal_mechanical_evolve_equal_budget_results.csv`
- `docs/cycloidal_mechanical_evolve_equal_budget_results.json`
- `docs/cycloidal_mechanical_evolve_equal_budget.md`

The markdown summary must explicitly state:

- all methods used equal Chrono audit budget
- all methods used identical verifier settings
- all methods used `procedural_cycloidal_fallback=false`
- whether TTRL wins under equal budget
- if TTRL loses, which target/metric/regime caused the loss

## Required Metrics

For every method:

- `candidate_count`
- `chrono_audits`
- `best_verified_reward`
- `verified_pass_rate`
- `cad_pass_rate`
- `chrono_real_geometry_rate`
- `lockup_rate`
- `best_out_omega_med`
- `best_ratio_error_pct`
- `best_power_balance_error_pct`
- `best_torque_ripple_pct`
- `best_max_penetration_mm`
- `best_contact_force_rms_N`
- `adapter_updates`
- `trained_tokens`

## Success Condition

The broader paper goal is achieved only if `mechanical_evolve_ttrl`
outperforms both required non-updating baselines under equal Chrono audit budget
on:

- `best_verified_reward`

and at least one of:

- `verified_pass_rate`
- `ratio_error_pct`
- `lockup_rate`

The claim must not be made if TTRL only wins because it used more Chrono audits.

## Current Honest Status

The CAD/Chrono verifier foundation is complete and pushed.

The broader MechanicalEvolve/TTRL paper result is not yet proven. Prior partial
results showed TTRL wins on nominal and mostly on high-load, but high-speed was
a counterexample where TTRL lost to stronger non-updating/search baselines.
Therefore the paper claim is still open until the matched-budget run completes
and the final artifacts above show a clean equal-budget win or honestly document
the failure regime.

## Do Not Drift

Do not redefine success as:

- a smoke test
- a procedural fallback result
- a single successful Chrono run
- a verifier-gated search result without TTRL updates
- an LLM best-of-N result without updates
- a win caused by larger audit budget

The end state is a fair, matched-budget MechanicalEvolve/TTRL comparison using
CAD-generated real-geometry Chrono verification.
