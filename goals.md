# MechanicalEvolve / TTRL Paper Goal And Proof Contract

This branch is aiming at a CoRL-grade paper result for MechanicalEvolve:
test-time RLVR/TTRL-driven mechanical design reasoning under executable CAD
and contact-physics verification.

The broader paper claim is not limited to cycloidal/QDD actuator tuning. The
unit of generalization is the mechanism family. The model should learn repair
and redesign strategies that transfer across families such as cycloidal
reducers, planetary reducers, rack-pinion systems, lead screws, fourbars,
slider-cranks, cam followers, belt drives, and chain drives.

The paper claim must become:

> Under equal expensive CAD + Chrono physics-verification budget, iterative
> RLVR/TTRL adaptation learns reusable mechanical reasoning that improves
> verified designs on unseen mechanism families better than non-updating
> baselines.

The cycloidal/QDD result is still important, but it is now a foundation and
anchor benchmark, not the whole paper. The broader paper result is a
controlled empirical claim that test-time learning from executable mechanical
verifiers improves design reasoning across mechanism families under the same
expensive verification budget available to non-learning methods.

In plain terms: MechanicalEvolve should not win because it memorized one
cycloidal quirk, ran more samples, used a toy procedural simulator, or
optimized a fast proxy reward. It must win because iterative verifier-derived
adaptation makes better design and repair proposals that transfer to held-out
mechanism families under the same expensive verification budget.

## Broader MechanicalEvolve/TTRL Paper Result

The paper result we are pursuing is:

> MechanicalEvolve performs test-time mechanical reasoning: an open reasoning
> model proposes design or repair actions, receives executable CAD +
> Chrono verifier feedback, updates a lightweight adapter during the
> experiment, and under the same expensive physics-audit budget learns repair
> strategies that transfer to unseen mechanism families better than
> non-updating proposal/search baselines.

This is broader than "optimize cycloidal parameters." The intended paper is
about a verifier-grounded learning loop for mechanical design families:

- the design object is a mechanism family instance, not a single tuned
  parameter vector;
- the reward source is executable mechanical validity, not a language-model
  preference or a fast proxy alone;
- candidates must survive FreeCAD/OCCT geometry generation, trusted DesignIR
  checks, Chrono SMC contact simulation, and physical defect gates;
- the learning signal is produced at test time from verifier-labeled candidate
  groups and repair traces;
- the comparison is budget matched on expensive Chrono audits, so search volume
  alone cannot explain the result;
- the generalization split is by mechanism family, so the model cannot win by
  memorizing cycloidal-specific quirks.

The final paper should prove five things, in this order:

1. **Verifier credibility.** CAD-generated real geometry, with procedural
   fallback disabled, can be simulated in Chrono SMC and produces the required
   actuator metrics.
2. **Family-level benchmark structure exists.** The benchmark is grouped by
   mechanism family with seen-family and unseen-family splits.
3. **Search pressure matters.** Fast reward and unrestricted candidate
   generation produce many attractive but invalid designs, so CAD/contact
   verification is not an optional postprocess.
4. **Adaptation matters under equal budget.** With the same number of Chrono
   audits, iterative TTRL/LoRA updates improve held-out-family performance versus
   `verifier_gated`, `llm_evolve_no_update`, frozen-model SFT, and no-update
   search.
5. **The result is not a one-off.** The improvement is visible across multiple
   held-out mechanism families, or any failure regime is reported explicitly and
   becomes the next engineering/scientific target.

The main empirical table must therefore be a family-held-out matched-budget
comparison. Cycloidal should remain the strongest validated anchor family, but
the headline metric is generalization to unseen families.

## Family Generalization Benchmark

The benchmark should be grouped by mechanism family, not by a single
parameter setting. The family list should at minimum include:

- cycloidal reducers
- planetary reducers
- rack-pinion systems
- lead screws
- fourbars
- slider-cranks
- cam followers
- belt drives
- chain drives

The evaluation split should explicitly separate:

- seen families used for training and adapter updates
- unseen families held out for final evaluation

A valid paper run should train on one family subset and test on disjoint
families, for example:

- train: cycloidal, belt, chain, rack-pinion, fourbar
- test: planetary, lead screw, cam follower, slider crank

or the reverse, as long as the split is family-held-out and frozen before the
evaluation run.

The benchmark output should report at minimum:

- best verified reward on seen families
- best verified reward on unseen families
- family-level pass rate
- family-level lockup rate
- family-level repair success rate
- comparison against frozen model, SFT model, and no-update search baselines

The headline claim is only valid if RLVR/TTRL beats the frozen model, SFT
baseline, and no-update search on unseen families under equal verifier budget.

The claim we may make after success is:

> Under equal expensive CAD + Chrono verification budgets, test-time
> verifier-derived adapter updates improve mechanical reasoning on unseen
> mechanism families.

The claims we must not make from this branch unless separately proven are:

- that the learned design policy is ready for hardware fabrication;
- that Chrono SMC is a perfect physical oracle;
- that the method solves arbitrary mechanical CAD design;
- that TTRL is better because it ran more audits, more candidates, or looser
  physical gates;
- that a procedural fallback or fast reward result is paper-grade mechanical
  verification;
- that a single family result is sufficient to claim transferable reasoning.
- that the cycloidal anchor benchmark alone proves family-level transfer.

## Broader Research Thesis

MechanicalEvolve is the mechanical-design analogue of AlphaEvolve/DeepEvolve
and TTRL:

- A proposal policy generates actuator design programs or parameter edits.
- A fast actuator reward screens many candidates cheaply.
- CAD generation converts elite candidates into named mechanical assets.
- Chrono SMC verifies the generated real geometry with compliant contact.
- The verifier returns scalar reward plus structured defects.
- The proposal policy is improved at test time using verifier-derived rewards.
- The archive keeps lineage, defects, metrics, and generated assets.

The intended scientific result is not just "we found one cycloidal design."
The result is:

- under matched Chrono audit budgets, iterative TTRL produces better verified
  design repairs and redesigns than non-updating search/evolution baselines;
- the advantage holds on held-out mechanism families, not just the training
  families;
- fast-only optimization finds candidates that often look good before CAD and
  contact verification but fail more often under the real verifier;
- LLM evolution without weight/adaptation updates can explore, but does not
  receive the same verifier-derived policy improvement signal;
- CAD + Chrono verification is therefore not an implementation detail, but the
  task-defining reward source that makes transferable mechanical reasoning
  credible.

The final paper should report both the best design and the learning/search
behavior that produced it: pass rates, lockup rates, defect regimes, verified
reward, ratio error, contact forces, penetration, torque ripple, power balance,
and verifier calls to first valid design.

## Primary Paper Result

The result we need to publish is:

> MechanicalEvolve with iterative TTRL/LoRA updates, using the same CAD +
> Chrono SMC verification budget as non-updating baselines, achieves higher
> verified cycloidal/QDD actuator reward and at least one stronger physical
> validity metric across the target suite.

That statement has three required parts:

1. **Equal-budget design search.** Every compared method receives the same
   number of real Chrono audits per target/seed trial. Extra fast proxy samples
   are allowed only if they are declared and do not change the expensive
   verifier budget.
2. **Executable mechanical validity.** Reward only counts after FreeCAD/OCCT
   asset generation, trusted DesignIR checks, and Chrono SMC simulation with
   `procedural_cycloidal_fallback=false`.
3. **Actual test-time adaptation.** `mechanical_evolve_ttrl` must perform
   iterative adapter updates from verifier-labeled candidate groups. A
   best-of-N sampler, prompt-only mutation loop, or MAP-Elites archive without
   updates is a baseline, not the target method.

The paper should use the matched-budget suite as the main result table and
treat single-trial examples as qualitative case studies only.

## Experimental Hierarchy

The work should be interpreted in this order:

1. **Verifier acceptance.** Prove CAD-generated cycloidal assets run in Chrono
   SMC with fallback disabled and emit the required physical metrics.
2. **Single-trial equal-budget proof.** Show the required methods can run to
   the same Chrono audit budget on one target/seed and produce the required
   artifacts.
3. **Multi-target, multi-seed proof suite.** Run the full matched-budget suite
   over nominal, high-load, and high-speed regimes with fixed seeds.
4. **Statistical paper claim.** Claim success only if aggregate and paired
   results support TTRL over the required non-updating baselines.

Earlier stages are prerequisites. They are not substitutes for the final
matched-budget result.

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

The paper-grade result requires a matched-budget experiment across mechanism
families. Every method must use the same family split, same verifier
thresholds, same CAD pipeline, same Chrono SMC configuration, same random
seeds, and the same total Chrono audit budget within each family/seed trial.

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

4. `frozen_model`
   - The base reasoning model without test-time updates.
   - Same family split and verifier budget.

5. `sft_model`
   - A supervised fine-tuned model without verifier-driven test-time updates.
   - Same family split and verifier budget.

6. `no_update_search`
   - Search or evolution without weight updates.
   - Same family split and verifier budget.

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
- the configured target variants for high-load and high-speed trials, if the
  proof suite sets them explicitly
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

The single-trial success condition is achieved only if
`mechanical_evolve_ttrl` outperforms both required non-updating baselines under
equal Chrono audit budget on:

- `best_verified_reward`

and at least one of:

- `verified_pass_rate`
- `ratio_error_pct`
- `lockup_rate`

The claim must not be made if TTRL only wins because it used more Chrono audits.

The broader paper success condition is stricter. Across the full family-held-out
proof suite, the final artifacts must show:

- equal real Chrono audit budget within every target/seed trial;
- no procedural fallback for any counted verified result;
- nonzero `adapter_updates` and `trained_tokens` for `mechanical_evolve_ttrl`;
- TTRL has the best aggregate `best_verified_reward_mean` on unseen families;
- paired TTRL-minus-baseline deltas are positive against frozen, SFT, and
  no-update search baselines;
- the suite-level win is not driven solely by the seen families while failing
  the unseen families without explanation.

If an unseen family remains a counterexample, the markdown must say that
directly and the branch must treat it as the next scientific/engineering
failure to solve, not as a result to hide.

## Current Honest Status

The CAD/Chrono verifier foundation is complete and pushed.

The cycloidal anchor benchmark is complete and pushed. That result establishes
the verifier stack and demonstrates verifier-driven updates on one mechanism
family, but it is not yet the headline paper result.

The broader family-held-out MechanicalEvolve/TTRL paper result is still
required. The current cycloidal artifacts are useful anchor evidence, but they
do not by themselves prove transfer across mechanism families.

Current branch work should therefore focus on:

- building the family-grouped mechanical design benchmark;
- freezing family-held-out train/test splits;
- running verifier-matched seen-family and unseen-family evaluations;
- comparing against frozen model, SFT, and no-update search baselines;
- producing final CSV/JSON/markdown artifacts that expose every required metric
  and do not overclaim the result.

## Do Not Drift

Do not redefine success as:

- a smoke test
- a procedural fallback result
- a single successful Chrono run
- a verifier-gated search result without TTRL updates
- an LLM best-of-N result without updates
- a win caused by larger audit budget

The end state is a fair, matched-budget MechanicalEvolve/TTRL comparison using
CAD-generated real-geometry Chrono verification. The publishable result is a
statistical matched-budget actuator-discovery result, not merely a working
simulator or a one-off optimized cycloidal reducer.

## Result State

The requested end state has not yet been achieved for the family-held-out
generalization claim.

The current artifact set supports a narrower statement only:

- the cycloidal anchor benchmark is complete;
- `mechanical_evolve_ttrl` demonstrated real verifier-driven updates in that
  anchor benchmark;
- the family-transfer claim still needs a family-held-out benchmark and
  results table before it can be published.
