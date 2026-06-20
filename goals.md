# MechanicalEvolve-Physics Goal

This file is the execution contract for `/goal pursue goals.md`.

The goal is an experiment, not a document-writing exercise. Markdown summaries
may be produced after the experiment, but they do not count as completion. The
completion unit is a frozen benchmark, executed matched-budget runs, raw
verifier evidence, anti-shortcut audits, statistical analysis, and a binary
answer to the hypothesis below.

No corner cutting: the thing to accomplish is the experiment.

## Cluster Conduct Constraint

Do not poll the MATX login host or any cluster login node aggressively. In
particular, do not run repeated `ssh sc ...` checks every 30 seconds or at any
similar high frequency for `squeue`, `sacct`, log tails, filesystem scans, or
disk checks. This is forbidden even during early failure triage.

Cluster monitoring must be sparse and respectful of shared infrastructure:

- Prefer Slurm dependencies, job output files, and cluster-side scripts over
  repeated SSH polling from the laptop.
- Use manual or low-frequency status checks only when necessary.
- If active monitoring is required, batch multiple checks into one remote
  command and wait a substantial interval before checking again.
- Never create an automated SSH loop against `sc` unless the user explicitly
  authorizes it and the polling interval is acceptable under cluster policy.

Cluster GPU usage must also be conservative by default:

- Do not launch broad GPU arrays or high-concurrency shard runs by default.
- For MATX physics runs, default to one GPU shard at a time unless the user
  explicitly authorizes higher concurrency after coordinating with the lab.
- If a run needs more than one concurrent GPU shard, record the intended GPU
  count, job IDs, and expected duration before launch.
- If lab mates or cluster staff report pressure, cancel pending GPU work first
  and preserve partial artifacts for later analysis rather than continuing.

## Working Title

MechanicalEvolve-Physics: Test-Time Verifier-Derived Reinforcement Learning
for Transferable Mechanical Repair Under CAD and Contact-Physics Constraints.

## Scientific Motivation

Current 2026 CAD-generation work makes executable CAD alone an insufficient
research target. CADBench evaluates 18,000 samples across six benchmark
families, five input modalities, six metrics, and more than 1.4 million
generated CAD programs, and reports that models still degrade with geometric
complexity and modality shift:

- Anna C. Doris, Jacob Thomas Sony, Ghadi Nehme, Era Syla, Amin Heyrani
  Nobari, and Faez Ahmed. "CADBench: A Multimodal Benchmark for AI-Assisted
  CAD Program Generation." arXiv:2605.10873, 2026.
  https://arxiv.org/abs/2605.10873

BenchCAD evaluates 17,900 execution-verified CadQuery programs across 106
industrial part families and reports that current systems often recover coarse
outer geometry but fail at faithful parametric CAD programs, especially on
unseen families:

- Haozhe Zhang, Kaichen Liu, Miaomiao Chen, Lei Li, Shaojie Yang, Cheng Peng,
  and Hanjie Chen. "BenchCAD: A Comprehensive, Industry-Standard Benchmark for
  Programmatic CAD." arXiv:2605.10865, 2026.
  https://arxiv.org/abs/2605.10865

MUSE argues the same point from a design-quality angle: many models can produce
executable code, fewer produce valid geometry, and still fewer satisfy
functionality, manufacturability, and assemblability:

- Xiaoyu Dong, Zhi Li, and Xiao-Ming Wu. "MUSE: Benchmarking Manufacturable,
  Functional, and Assemblable Text-to-CAD Generation." arXiv:2605.28579, 2026.
  https://arxiv.org/abs/2605.28579

CADFS uses 450,000 real-world CAD models and 15 modeling operations, raising
the bar for realistic CAD generation:

- Vladislav Pyatov, Gleb Bobrovskikh, Saveliy Galochkin, Nikita Boldyrev,
  Oleg Voynov, Alexander Filippov, Gonzalo Ferrer, Peter Wonka, and Evgeny
  Burnaev. "CADFS: A Big CAD Program Dataset and Framework for Computer-Aided
  Design with Large Language Models." arXiv:2605.01925, 2026.
  https://arxiv.org/abs/2605.01925

Text2CAD-Bench evaluates text-to-parametric CAD across geometric complexity
levels and reports degradation on complex topology and advanced features:

- Liang Wang, Heng Meng, Zekai Xiang, Jin Liu, Pingyi Zhou, Litao Chen, and
  Yongqiang Tang. "Text2CAD-Bench: A Benchmark for LLM-based
  Text-to-Parametric CAD Generation." arXiv:2605.18430, 2026.
  https://arxiv.org/abs/2605.18430

Physics-in-the-Loop CAD agents already formulate engineering design as
closed-loop generation, evaluation, and revision with physical verification:

- Elias Berger, Muhammad Usama, Jan Mehlstäubl, Bernhard Saske, and Kristin
  Paetzold-Byhain. "Physics-in-the-Loop: A Hybrid Agentic Architecture for
  Validated CAD Engineering Design." arXiv:2605.19717, 2026.
  https://arxiv.org/abs/2605.19717

TTRL-CoCoV and T^3RL show that test-time reinforcement learning needs stronger
verification and confidence/tool-conditioned rewards:

- Jiahui Li et al. "Exploiting Verification-Generation Gap: Test-Time
  Reinforcement Learning with Confidence-Conditioned Verification."
  arXiv:2606.03608, 2026. https://arxiv.org/abs/2606.03608
- Ruotong Liao, Nikolai Roehrich, Xiaohan Wang, Yuhui Zhang, Yasaman
  Samadzadeh, Volker Tresp, and Serena Yeung-Levy. "Tool Verification for
  Test-Time Reinforcement Learning." arXiv:2603.02203, 2026.
  https://arxiv.org/abs/2603.02203

RLVR reward-hacking work shows that models can learn verifier-passing
shortcuts instead of intended abstractions:

- Lukas Helff, Quentin Delfosse, David Steinmann, Ruben Haerle, Hikaru Shindo,
  Patrick Schramowski, Wolfgang Stammer, Kristian Kersting, and Felix
  Friedrich. "LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking."
  arXiv:2604.15149, 2026. https://arxiv.org/abs/2604.15149

Robot co-design work also sets a higher physical bar: morphology/control
co-design papers optimize robot bodies with controllers and simulation, not
only symbolic mechanism programs:

- Luke Strgar and Sam Kriegman. "Accelerated co-design of robots through
  morphological pretraining." ICLR 2026. https://openreview.net/forum?id=WVliGyFwZv
- Jiawei Fang, Yuxuan Sun, Chengtian Ma, Qiuyu Lu, and Lining Yao.
  "RoboMoRe: LLM-based Robot Co-design via Joint Optimization of Morphology
  and Reward." arXiv:2506.00276, 2025. https://arxiv.org/abs/2506.00276

The current pushed result is a useful pilot: online TTRL improved held-out
mechanism-program repair under matched verifier budget. It does not yet prove
engineering-grade mechanical design because most of the benchmark is still
Level-1 symbolic/analytic repair. The next paper-level result must target
test-time learning from trusted mechanical verifiers that include CAD,
topology, interface, mass/material, hidden semantic, and contact-physics
constraints.

## Core Hypothesis

Primary hypothesis:

> Under a fixed expensive verification budget, online GRPO/LoRA adaptation from
> CAD and physics verifier feedback learns reusable mechanical repair operators
> that transfer to unseen mechanism families, improving Level-2/Level-3
> verified repair success over non-updating verifier-feedback agents, adaptive
> search/evolution, SFT adapters, and frozen models.

Operational interpretation:

- The model receives the same task stream, prompts, verifier calls, CAD calls,
  and physics calls as the no-update baselines.
- The only causal difference is whether verifier-derived rewards update the
  LoRA adapter online.
- Success is measured only on held-out mechanism families and hidden
  perturbation variants.
- A pass requires executable code, valid DesignIR, CAD artifacts where
  applicable, and physical checks where applicable.
- A pass cannot be credited from public analytic parameter matching alone.

Primary null hypothesis:

> Under equal CAD/physics verifier budget, online verifier-derived GRPO/LoRA
> updates do not improve held-out Level-2/Level-3 mechanical repair success
> over the same model with verifier feedback but no weight updates.

## Distinct Contribution

The paper must not claim generic "LLM generates CAD" or generic "TTRL improves
reasoning." Those are already active 2026 research areas.

The distinct contribution must be:

> MechanicalEvolve-Physics studies whether verifier-derived online learning can
> acquire reusable mechanical repair operators across mechanism families when
> the verifier includes CAD, topology, interface, mass/material, hidden
> semantic, and contact-physics constraints.

The scientific unit is not a single generated part, a one-shot CAD program, or
a scalar parameter optimization. The scientific unit is a repair behavior that
transfers across mechanism families under expensive verifier budget.

## Benchmark: MechanismRepair-Physics

Build and freeze `MechanismRepair-Physics`.

Minimum final benchmark:

- 12 mechanism families.
- At least 10 final tasks per family.
- At least 120 final tasks total.
- At least 3 stochastic seeds per method.
- At least 6 required methods, preferably 8.
- At least 2 family-held-out splits.
- 1 hidden perturbation split.
- 1 external-style CAD/design split if feasible.
- At least 40 percent of final tasks must be Level 2.
- At least 25 percent of final tasks must be Level 3.
- Level-1-only tasks may remain as diagnostics but must not carry the headline
  result.

Required mechanism families:

1. `cycloidal_reducer`
2. `planetary_reducer`
3. `spur_compound_gear_train`
4. `belt_drive`
5. `chain_drive`
6. `rack_pinion`
7. `lead_screw`
8. `slider_crank`
9. `fourbar_linkage`
10. `cam_follower`
11. `geneva_indexer`
12. `shaft_bearing_coupling`

Optional strengthening families:

- compliant snap mechanism
- rolling bearing preload assembly
- timing cam or dwell mechanism
- simple gripper linkage
- pulley block or cable transmission

## Anti-Toy Task Rule

A task counts toward the headline benchmark only if it requires at least three
independent constraint classes.

Constraint classes:

- topology/mobility: correct parts, joints, grounded bodies, and degrees of
  freedom.
- interface: correct input/output ports and joint kinds.
- functional behavior: ratio, stroke, dwell, path, indexing, timing, torque
  direction, or transmission direction.
- CAD artifact: STEP/B-Rep generation, watertightness, mass/COM/inertia
  evidence, material provenance, and trusted geometry roles.
- physics/contact: Chrono contact, penetration, lockup, output speed, contact
  force, power/torque consistency, stroke under load, or path under actuation.
- manufacturability/assembly: clearances, overlaps, fastener accessibility,
  tolerance windows, material/process constraints, and assembly feasibility.

Reject a task if any of these are true:

- A reference solution can pass by setting only one scalar such as
  `params.declared_ratio`.
- Public examples contain enough literal structure to copy the answer.
- The hidden verifier checks no semantic property beyond a public key/value.
- A negative control can accidentally pass all hard gates.
- The solution does not construct a meaningful multi-body mechanism or
  assembly.
- The task is static fit only and is being counted toward the headline result.

Static-fit tasks may be retained for diagnostics but must not count toward the
primary paper claim.

## Verifier Levels

Every task must be assigned one verifier level.

### Level 1: Executable Mechanism Program

Required checks:

- parse and execute generated Python safely;
- produce valid DesignIR;
- pass topology checks;
- pass required port checks;
- pass mobility checks;
- pass analytic functional checks.

Level 1 examples:

- gear ratio from tooth counts;
- lead screw travel per revolution;
- rack-pinion linear travel;
- fourbar approximate path from provided fixtures.

Level 1 tasks are useful for diagnosis but cannot carry the headline result.

### Level 2: CAD/Mechanical Artifact

Required checks:

- generate STEP/B-Rep or trusted CAD artifacts;
- verify watertight/manifold/non-self-intersecting geometry where applicable;
- verify material records and provenance;
- verify mass, COM, and inertia consistency;
- verify no invalid overlaps or collisions;
- verify basic assembly feasibility;
- preserve correct ports and mobility after CAD artifact generation.

Level 2 examples:

- shaft/bearing/coupling with fit and clearance constraints;
- gear train with generated wheel bodies and valid shaft axes;
- slider-crank with CAD links and joint placements;
- cam/follower layout with generated contact surfaces.

### Level 3: Physical/Contact Behavior

Required checks:

- run Chrono SMC or an equivalent trusted physics simulator;
- verify required contact engagement;
- verify no lockup;
- bound maximum penetration;
- record contact force RMS and maximum;
- measure output speed, ratio, stroke, path, or indexing under actuation;
- check torque/power consistency where meaningful;
- preserve Level-1 and Level-2 validity.

Level 3 examples:

- cam follower maintaining contact without lockup;
- cycloidal reducer with ring-pin/disc engagement;
- rack-pinion contact under load;
- chain/belt surrogate contact or tension dynamics if implemented credibly;
- Geneva indexer with dwell/indexing contact.

The headline result must be computed on Level-2 and Level-3 tasks only.

## Required Methods

Run all required methods under matched actual verifier budget.

1. `frozen_model`
   - Base model only.
   - No feedback updates.
   - Same budget.

2. `sft_seen_family`
   - LoRA/SFT on seen-family reference and repair traces only.
   - No held-out tasks or labels.
   - No test-time updates.
   - Same budget.

3. `llm_evolve_no_update`
   - Same verifier-feedback loop as TTRL.
   - No LoRA updates.
   - No RL updates.
   - Same task order, prompts, and verifier feedback.
   - Same budget.
   - Primary causal baseline.

4. `verifier_gated_search`
   - Best-of-K, beam, or verifier-gated proposal search.
   - No weight updates.
   - Same budget.

5. `adaptive_evolution`
   - AlphaEvolve/AdaEvolve-style population memory or search baseline.
   - May use archive memory and mutation/reflection.
   - No gradient or LoRA updates.
   - Same expensive verifier budget.

6. `mechanical_evolve_ttrl`
   - Online GRPO/LoRA updates from verifier rewards.
   - Same budget and task order as `llm_evolve_no_update`.
   - Must report nonzero adapter updates, RL datums, and trained tokens.

7. `mechanical_evolve_ttrl_tool_verified`
   - TTRL with tool-verification-style reward filtering motivated by T^3RL.
   - Verifier evidence upweights or filters reward estimates.
   - Same budget.
   - This is the preferred primary method if implemented.

8. `mechanical_evolve_ttrl_confidence`
   - Confidence-conditioned exploration/reward ablation motivated by
     TTRL-CoCoV.
   - Same budget.

The final comparison must include a current TTRL-strengthening variant because
plain TTRL is no longer enough for a 2026 claim.

## Budget

Primary budget:

- `B = 32` expensive verifier calls per task/seed/method cell.

If compute permits, report a budget curve:

- `B = 8`
- `B = 16`
- `B = 32`

Budget accounting must distinguish:

- actual Python execution calls;
- actual DesignIR verifier calls;
- actual CAD/OCCT calls;
- actual Chrono/contact simulation calls;
- failed sampler calls;
- replacement retry calls;
- hidden perturbation calls.

The primary comparison is invalid if TTRL receives more actual expensive CAD or
physics calls than the no-update baselines on the same task/seed cell.

## Anti-Shortcut Design

Every final task must have public, hidden, and isomorphic variants.

Hidden perturbations:

- rename parameters while preserving semantics;
- change dimensions and target values;
- rotate or translate coordinate frames;
- swap equivalent joint and part naming;
- add distractor params that should not be used;
- hide exact tolerance thresholds;
- evaluate generated solutions under multiple verifier configs.

Isomorphic mechanism tests:

- same kinematic graph with renamed nodes;
- same physical objective with changed coordinate convention;
- same assembly with equivalent but different reference construction;
- same function with a different but mechanically equivalent parameterization.

Reward-hacking failure criteria:

- A method passes public tasks but fails hidden semantic variants.
- A method emits verifier-specific aliases that do not satisfy isomorphic
  checks.
- A method enumerates public key/value patterns without preserving physical
  relations.
- A method succeeds only when public field names match training examples.

The anti-shortcut audit is mandatory and must be reported beside the primary
metric.

## Primary Metrics

Primary metric:

- `level23_verified_repair_success_at_B`

Primary result table must include:

- `level23_verified_repair_success_at_32`
- `hidden_variant_success_at_32`
- `anti_shortcut_pass_rate_at_32`
- `best_verified_reward_at_32`
- `actual_verifier_calls`
- `actual_cad_calls`
- `actual_chrono_calls`

Secondary metrics:

- `cad_valid_rate`
- `chrono_valid_rate`
- `first_valid_verifier_call`
- `mobility_repair_success`
- `port_grounding_repair_success`
- `artifact_validity_repair_success`
- `contact_repair_success`
- `max_penetration_mm`
- `contact_force_rms_N`
- `ratio_error_pct`
- `stroke_error_mm`
- `path_chamfer_error`
- `lockup_rate`
- `invalid_topology_rate`
- `invalid_artifact_rate`
- `missing_port_rate`
- `ungrounded_port_rate`
- `wrong_mobility_rate`
- `adapter_updates`
- `rl_datums`
- `trained_tokens`
- `rl_trained_tokens`

## Statistical Test

Unit of analysis:

- paired task/seed/method cell on held-out Level-2/Level-3 tasks.

Primary comparison:

- `mechanical_evolve_ttrl_tool_verified` versus `llm_evolve_no_update`.

Fallback primary comparison if the tool-verified variant is not implemented:

- `mechanical_evolve_ttrl` versus `llm_evolve_no_update`.

Success requires all of:

- absolute Level-2/3 success improvement >= 15 percentage points;
- paired bootstrap 95 percent confidence interval lower bound > 0;
- paired sign or permutation test `p <= 0.05`;
- positive `best_verified_reward_at_32` delta;
- positive `hidden_variant_success_at_32` delta;
- positive `anti_shortcut_pass_rate_at_32` delta;
- TTRL beats `adaptive_evolution` under equal verifier budget;
- TTRL beats `verifier_gated_search` under equal verifier budget;
- positive delta in at least 8 of 12 families;
- leave-one-family-out delta remains positive.

If these fail, the result is negative. Do not switch metrics after the run.

## Mechanistic Analysis

The paper must show what was learned, not only that a score improved.

Required analyses:

- failure-code transition matrix from first attempt to final attempt;
- TTRL versus no-update on identical verifier traces;
- per-family repair deltas;
- matched trace pairs for at least 24 cases;
- adapter update timeline versus repair success;
- failure modes before and after online learning;
- hidden perturbation failure analysis.

Repair taxonomy:

- topology repair;
- port repair;
- mobility repair;
- ratio/stroke/path repair;
- CAD artifact repair;
- material/mass property repair;
- collision/clearance repair;
- contact/lockup repair;
- manufacturability/assembly repair.

The key figure should show that online updates increase the probability of
specific repair operators after observing verifier feedback.

## External Positioning

Do not claim to beat CADBench, BenchCAD, CADFS, Text2CAD-Bench, or MUSE unless
those tasks are actually evaluated.

Use those papers as motivation:

- CADBench and BenchCAD show current CAD systems fail under complexity,
  modality shift, and unseen families.
- MUSE shows executable geometry is not enough; engineering criteria matter.
- CADFS shows large-scale CAD-program generation is moving toward real design
  histories and richer operation sets.
- Physics-in-the-Loop shows closed-loop physical verification is now expected.
- TTRL-CoCoV and T^3RL show vanilla TTRL needs stronger verifier-conditioned
  reward design.
- RLVR reward-hacking work motivates hidden and isomorphic verifier audits.

The contribution is not a bigger CAD benchmark. The contribution is a
mechanical repair learning benchmark plus an online adaptation result under
expensive CAD/physics verifiers.

## Required Output Artifacts

Required benchmark artifacts:

- `runs/mechanism_repair_physics_final/benchmark_manifest.json`
- `runs/mechanism_repair_physics_final/split_manifest_*.json`
- `runs/mechanism_repair_physics_final/verifier_manifest.json`
- `runs/mechanism_repair_physics_final/method_manifest.json`
- `runs/mechanism_repair_physics_final/level_manifest.json`
- `runs/mechanism_repair_physics_final/hidden_variant_manifest.json`
- `runs/mechanism_repair_physics_final/tasks/`

Required run artifacts:

- `runs/mechanism_repair_physics_final/raw_completions/`
- `runs/mechanism_repair_physics_final/verifier_outputs/`
- `runs/mechanism_repair_physics_final/cad_artifacts/`
- `runs/mechanism_repair_physics_final/chrono_outputs/`
- `runs/mechanism_repair_physics_final/training_logs/`
- `runs/mechanism_repair_physics_final/adapter_checkpoints/`
- `runs/mechanism_repair_physics_final/results.json`
- `runs/mechanism_repair_physics_final/results.csv`
- `runs/mechanism_repair_physics_final/cell_results.jsonl`

Required analysis artifacts:

- `runs/mechanism_repair_physics_final/stats.json`
- `runs/mechanism_repair_physics_final/failure_analysis.json`
- `runs/mechanism_repair_physics_final/trace_pairs.json`
- `runs/mechanism_repair_physics_final/repair_taxonomy.json`
- `runs/mechanism_repair_physics_final/anti_shortcut_audit.json`
- `runs/mechanism_repair_physics_final/budget_audit.json`
- `runs/mechanism_repair_physics_final/claim_audit.json`

`claim_audit.json` must contain exactly one of:

- `"claim_status": "supports_primary_hypothesis"`
- `"claim_status": "does_not_support_primary_hypothesis"`

Derived markdown summaries are allowed, but they are not completion criteria.

## Completion Criteria

The goal is complete only when all of these are true:

- `MechanismRepair-Physics` exists and is frozen.
- The benchmark has at least 120 final tasks.
- At least 40 percent of final tasks are Level 2.
- At least 25 percent of final tasks are Level 3.
- Every final task has a passing reference solution.
- Every final task has at least two negative controls.
- Every final task has hidden perturbation variants.
- Every final task passes the anti-toy rule.
- Both family-held-out splits are frozen.
- All required methods ran under matched actual verifier budget.
- CAD and Chrono call counts are audited.
- Raw completions and verifier outputs are preserved.
- CAD and Chrono artifacts are preserved where applicable.
- Training logs and adapter checkpoints are preserved for all learning methods.
- Main statistics are computed on held-out Level-2/Level-3 tasks.
- Anti-shortcut audit is computed.
- Failure transition and repair-operator analyses are computed.
- `claim_audit.json` answers the primary hypothesis honestly.

If the hypothesis is unsupported but all experimental criteria above are met,
the experiment is complete as a negative result. If the goal is to produce a
positive CoRL paper result, keep improving the benchmark, verifier, or method
and rerun under a new frozen contract instead of pretending a failed hypothesis
passed.

## What Not To Do

- Do not count writing docs as progress toward the experimental claim.
- Do not count a Level-1-heavy benchmark as a paper-level result.
- Do not count one-parameter analytic tasks toward the headline metric.
- Do not count static-fit tasks toward the headline metric.
- Do not use held-out tasks for prompt tuning, hyperparameter selection, or
  debugging examples.
- Do not hide failed contact families.
- Do not change the verifier after seeing final results unless all affected
  methods are rerun.
- Do not let TTRL have more actual verifier, CAD, or Chrono calls.
- Do not claim hardware readiness.
- Do not claim industrial CAD readiness unless external CAD/design benchmarks
  are actually evaluated.
- Do not claim reward-hacking robustness without hidden and isomorphic audits.
