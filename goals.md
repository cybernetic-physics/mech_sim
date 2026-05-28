# MechanicalEvolve / TTRL Paper Goal And Proof Contract

This branch is aiming at a CoRL-grade paper result for MechanicalEvolve:
test-time RLVR/TTRL-driven mechanical actuator discovery under executable CAD
and contact-physics verification.

The target domain is cycloidal/QDD actuator design. The paper is not about
generic robot morphology optimization, generic code generation, or merely
"RL improves designs." Recent co-design work already shows that automated
morphology/control search can improve simulated robot reward. The stronger
claim here is that an open reasoning model can improve an internal mechanical
actuator mechanism while being constrained by CAD generation, real collision
assets, Chrono contact dynamics, and physical defect checks.

The paper claim must become:

> Under equal expensive CAD + Chrono physics-verification budget, iterative
> RLVR/TTRL adaptation discovers stronger verified cycloidal actuator designs
> than non-updating baselines.

The broader paper result is not a single good reducer, a better optimizer
trace, or an infrastructure demo. It is a controlled empirical claim that
test-time learning from executable mechanical verifiers improves internal
actuator invention under the same expensive verification budget available to
non-learning methods.

In plain terms: MechanicalEvolve should not win because it ran more samples,
because it used a toy procedural simulator, or because it optimized a fast
proxy reward. It must win because iterative verifier-derived adaptation makes
better design proposals under the same expensive verification budget.

## Broader MechanicalEvolve/TTRL Paper Result

The paper result we are pursuing is:

> MechanicalEvolve performs test-time mechanical invention: an open reasoning
> model proposes cycloidal/QDD actuator designs, receives executable CAD +
> Chrono verifier feedback, updates a lightweight adapter during the experiment,
> and under the same expensive physics-audit budget discovers stronger verified
> actuator designs than non-updating proposal/search baselines.

This is broader than "optimize cycloidal parameters." The intended paper is
about a verifier-grounded learning loop for mechanical design:

- the design object is an internal actuator mechanism, not external robot
  morphology;
- the reward source is executable mechanical validity, not a language-model
  preference or a fast proxy alone;
- candidates must survive FreeCAD/OCCT geometry generation, trusted DesignIR
  checks, Chrono SMC contact simulation, and physical defect gates;
- the learning signal is produced at test time from verifier-labeled candidate
  groups;
- the comparison is budget matched on expensive Chrono audits, so search volume
  alone cannot explain the result.

The final paper should prove four things, in this order:

1. **Verifier credibility.** CAD-generated real geometry, with procedural
   fallback disabled, can be simulated in Chrono SMC and produces the required
   actuator metrics: output speed, ratio error, lockup, contact force,
   penetration, torque ripple, and power balance.
2. **Search pressure matters.** Fast reward and unrestricted candidate
   generation produce many attractive but invalid designs, so CAD/contact
   verification is not an optional postprocess.
3. **Adaptation matters under equal budget.** With the same number of Chrono
   audits, iterative TTRL/LoRA updates improve best verified reward versus
   `verifier_gated` and `llm_evolve_no_update`.
4. **The result is not a one-off.** The improvement is visible across target
   regimes and seeds, or any failure regime is reported explicitly and becomes
   the next engineering/scientific target.

The main empirical table must therefore be a matched-budget comparison over
`verifier_gated`, `llm_evolve_no_update`, and `mechanical_evolve_ttrl`.
Qualitative design renders, single best reducers, and optimizer traces are
supporting evidence only; they do not replace the matched-budget table.

The claim we may make after success is:

> Under equal expensive CAD + Chrono verification budgets, test-time
> verifier-derived adapter updates improve mechanical actuator discovery.

The claims we must not make from this branch unless separately proven are:

- that the learned actuator is ready for hardware fabrication;
- that Chrono SMC is a perfect physical oracle;
- that the method solves arbitrary mechanical CAD design;
- that TTRL is better because it ran more audits, more candidates, or looser
  physical gates;
- that a procedural fallback or fast reward result is paper-grade mechanical
  verification.

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
  actuator designs than non-updating search/evolution baselines;
- the advantage holds across multiple target regimes and seeds, not just one
  lucky nominal trial;
- fast-only optimization finds candidates that often look good before CAD and
  contact verification but fail more often under the real verifier;
- LLM evolution without weight/adaptation updates can explore, but does not
  receive the same verifier-derived policy improvement signal;
- CAD + Chrono verification is therefore not an implementation detail, but the
  task-defining reward source that makes mechanical invention credible.

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

The paper-grade result requires a matched-budget experiment across methods.
Every method must use the same task family, same verifier thresholds, same CAD
pipeline, same Chrono SMC configuration, same random seeds, and the same total
Chrono audit budget within each target/seed trial.

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

The broader paper success condition is stricter. Across the full proof suite,
the final artifacts must show:

- equal real Chrono audit budget within every target/seed trial;
- no procedural fallback for any counted verified result;
- nonzero `adapter_updates` and `trained_tokens` for `mechanical_evolve_ttrl`;
- TTRL has the best aggregate `best_verified_reward_mean`;
- paired TTRL-minus-baseline deltas are positive against both required
  non-updating baselines;
- the suite-level win is not driven solely by one target while failing the
  other regimes without explanation.

If a target regime remains a counterexample, the markdown must say that
directly and the branch must treat it as the next scientific/engineering
failure to solve, not as a result to hide.

## Current Honest Status

The CAD/Chrono verifier foundation is complete and pushed.

The broader MechanicalEvolve/TTRL paper result is not yet proven. Prior partial
results showed TTRL wins on nominal and mostly on high-load, but high-speed was
a counterexample where TTRL lost to stronger non-updating/search baselines.
Therefore the paper claim is still open until the matched-budget run completes
and the final artifacts above show a clean equal-budget win or honestly document
the failure regime.

Current branch work should therefore focus on completing and stabilizing:

- the equal-budget proof suite runner;
- the real TTRL/LoRA update loop;
- full target/seed completion without silent child-process loss;
- final CSV/JSON/markdown artifacts that expose every required metric.

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
