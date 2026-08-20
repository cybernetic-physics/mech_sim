# Project Status

**Reviewed:** 2026-08-19  
**Status:** alpha research platform

`mech-sim` already has a substantial and useful core: it can execute an
agent-generated mechanical submission in isolation, validate a structured
mechanism contract, select checks by required capability, score the result,
and preserve replayable evidence. Its strongest contribution today is this
**verification and learning control plane**, not a claim that every simulated
mechanism predicts hardware accurately.

## What is valuable here

Most mechanical-generation systems stop at code execution, geometric
similarity, or a rendered shape. `mech-sim` is organized around the harder
question: *what evidence would justify accepting an agent-designed part?*

Four design decisions make the repository unusually valuable:

1. **The policy and verifier are separate.** Agent code, paths, geometry,
   physical properties, and performance claims enter through an explicit
   trust boundary.
2. **Checks are capability-driven.** Cheap analytic checks can reject an
   invalid design before an expensive solver is invoked, while contact tasks
   can require a real physics capability.
3. **Failures are training data.** A closed failure-code grammar, public hints,
   scalar channels, and retry suggestions turn evaluation into a repair loop
   instead of a single pass/fail label.
4. **Evidence type is explicit.** Analytic, synthetic, simulated, and validated
   results are not treated as interchangeable.

This is the foundation needed to train agents that improve mechanical
artifacts through repeated design–evaluate–repair trajectories.

## Implemented today

### Evaluator and trust boundary

- `DesignIR`, `TaskSpec`, and `EvalConfig` parsing and validation.
- Subprocess-isolated execution of `design.py`, with timeouts and rejection of
  malformed or non-serializable output.
- Path traversal and symlink-escape checks for submitted geometry.
- Validation of topology, identifiers, units-bearing physical fields, mass,
  center of mass, inertia, materials, joints, and ports.
- Public/hidden metric separation and redaction of private traces.
- Hard-gate scoring plus bounded dense scores.
- Twelve registered probe types and a closed, machine-readable failure-code
  vocabulary.

### Mechanical evaluation

- Always-available analytic evaluation for static artifact checks,
  transmission relationships, mobility, and planar kinematics.
- A generator registry with **58 mechanism families** across four tiers:
  13 static, 16 planar-kinematic, 14 analytic-transmission, and 15
  contact-dynamics families.
- A checked-in **51-task** snapshot from an earlier registry state. The
  difference is intentional evidence versioning, but the snapshot should be
  regenerated before a new benchmark release.
- Reference solutions, public/hidden configurations, negative controls, and
  expected failure records.

### Native geometry and physics

- An optional Project Chrono adapter with bodies, revolute/prismatic/fixed
  constraints, motors, loads, primitive and mesh collision geometry, NSC and
  SMC contact, pose and force traces, and solver diagnostics.
- A trusted CAD bridge for STEP/mesh fixtures, geometry digests, material
  provenance, and CAD-derived mass properties.
- A real-geometry cycloidal fixture demonstrating FreeCAD/OCCT export into
  Chrono without procedural collision fallback.
- An explicitly opt-in deterministic contact adapter for testing pipeline
  behavior. Its outputs are marked synthetic in code and reports.

### Learning and experiment infrastructure

- A compact RLVR reward API that returns reward, dense score, public feedback,
  scalar channels, retry suggestions, and the synthetic-evidence flag.
- Multi-turn verifier-feedback rollouts.
- An exact GRPO path using TRL and PEFT LoRA.
- A legacy group-relative verifier-weighted cross-entropy path retained for
  reproducibility and labeled separately from exact GRPO.
- SFT, online adaptation, matched-budget experiment, family-split, resume,
  merge, statistics, and audit tooling.

### Reproducibility

- JSON reports, HDF5 trace support, dashboards, media manifests, run bundles,
  task and geometry digests, and frozen result summaries.
- Audits for evaluation coverage, matched verifier budgets, family overlap,
  missing evidence, anti-shortcut variants, and incomplete experiment cells.

## Evidence ledger

The table below distinguishes implemented capability from executed evidence.

| Evidence | Result | What it supports | What it does not support |
|---|---|---|---|
| [Portable core check](../CONTRIBUTING.md#validation-tiers), current `main` audit | 370 passed, 12 skipped | Evaluator, schemas, isolation, probes, reports, and portable tooling | Native solver or GPU training availability |
| [May 2026 benchmark snapshot](paper-results-current.md) | 50/51 reference controls passed; 104/104 expected negative failures detected | Benchmark wiring and failure discrimination on that frozen suite | Current 58-family coverage; physical validation |
| [CAD-to-Chrono cycloidal packet](cycloidal-real-geometry-proof.md) | Named STEP/STL assets, trusted mass properties, real mesh collision, NSC/SMC runs | The native CAD-to-contact path executes and records diagnostics | A calibrated reducer model or acceptable loaded behavior |
| [June 2026 Level-1 family-heldout experiment](../runs/mechanism_repair_ttrl_final/README.md) | 85.8% vs 23.3% verified success for online adaptation vs the matched no-update baseline; +62.5 percentage points across 120 paired cells | Online verifier-derived learning improved executable mechanism-program repair on held-out families under the recorded setup | CAD validity, contact physics, manufacturing, or hardware transfer |
| [Level-2/3 experiment package](../runs/mechanism_repair_physics_final/claim_audit.json) | 120 tasks prepared: 90 Level 2 and 30 Level 3 | Benchmark construction and audit infrastructure | No outcome claim: the committed claim audit records 0 observed rows of 9,360 planned cells |

The Level-1 result is the strongest learning result in the repository. It
supersedes the earlier family-transfer audit, which correctly found that an
older run held out seeds but not mechanism families.

## What is not yet working

### Physics credibility is narrower than solver coverage

The native runner is implemented, but broad predictive credibility is not.
The cycloidal proof is valuable because it exercises real CAD and collision
geometry; it also records exactly why the result should not be overstated. The
loaded SMC case still reports a power-balance failure, contact and material
parameters are not broadly calibrated, and representative mechanisms do not
yet have systematic timestep/mesh refinement plus independent or physical
reference comparisons.

Using a solver does not make a model validated. Project Chrono itself publishes
validation studies for selected components and exposes comparison utilities;
credibility remains specific to a model, configuration, and validation case.
ASME V&V 10 likewise treats verification, validation, and uncertainty
quantification as distinct activities.

### Structural and manufacturing evaluation is incomplete

The probe contract anticipates static FEA, safety factor, printability, and
manufacturing constraints, but there is no general solver-backed structural
adapter. Several tasks still evaluate declared parameters or analytic proxies.
There is no versioned manufacturing-process model for tolerances, operation
ordering, inspection, machine availability, time, or cost.

### The high-value physics learning experiment is prepared, not run

The repository contains a strong Level-2/3 benchmark manifest and extensive
execution/audit machinery. The committed final bundle is nevertheless a
pre-execution artifact: its budget audit reports every planned cell missing.
It must not be presented as a learning or physics result.

### Developer setup has rough edges

- The native Chrono/CAD stack is optional and is not available in a default
  portable installation.
- The `dev` dependency group does not include PyTorch, so three training test
  modules fail during default collection unless the training dependencies are
  installed. The portable script excludes those modules deliberately.
- With the training dependencies installed, the three training modules pass
  56/56 tests. The all-surfaces run currently reports 549 passed, 12 skipped,
  and 7 failed: contact-task expectations disagree with unavailable native
  capability behavior, one family-task materialization helper is absent, and a
  frozen physics fixture still resolves an original machine path.
- Some frozen run indices contain machine-specific absolute paths. The
  lightweight summaries are useful, but the heaviest raw artifacts are not
  independently replayable from Git alone.
- The old two-agent comparison contains a major infrastructure failure in one
  arm, so it is a debugging record rather than a clean model comparison.

## Highest-value next work

1. **Make the repository self-consistent.** Regenerate a versioned task
   snapshot from all 58 registered families, record the generator commit, and
   publish a fresh reference/negative-control summary.
2. **Make native validation routine.** Pin a supported solver environment, run
   native regression jobs, and publish deterministic diagnostic bundles for a
   small set of representative mechanisms.
3. **Calibrate before expanding.** For each representative contact family,
   add material/contact provenance, refinement studies, and comparison with an
   independent or physical reference.
4. **Complete one small Level-2/3 learning study.** Run a deliberately bounded
   subset of the prepared benchmark with matched actual CAD/solver calls and
   complete evidence before scaling to thousands of cells.
5. **Add a real manufacturing contract.** Model process capability, tolerance,
   material stock, operation sequence, inspection, time, and cost so reward
   measures a producible robotics part rather than only a valid mechanism.
6. **Close the virtual/physical loop.** Fabricate selected parts, measure them,
   compare predicted and observed behavior, and feed discrepancies back into
   both calibration and training.

## Research context

- [Project Chrono collision and contact documentation](https://api.projectchrono.org/collisions.html)
  distinguishes NSC constraint-based contact from SMC penalty-based contact.
- [Project Chrono validation studies](https://projectchrono.org/validation/)
  show the case-specific comparisons expected for credible solver use.
- [ASME V&V 10](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics/2019)
  provides the vocabulary for verification, validation, and uncertainty
  quantification used in this project.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO as a
  memory-efficient policy-optimization approach; this repository distinguishes
  its exact TRL implementation from the older weighted-CE trainer.
- [Text2CAD](https://arxiv.org/abs/2409.17106) demonstrates progress in
  text-to-parametric CAD generation. `mech-sim` addresses a complementary gap:
  verifying functional, physical, and eventually manufacturing outcomes after
  an artifact is generated.

## Claim policy

Use the narrowest label supported by the evidence:

| Label | Required evidence |
|---|---|
| Analytic | Deterministic topology, geometry, or closed-form computation |
| Synthetic | Test data explicitly labeled as nonphysical |
| Simulated | An identified solver, inputs, settings, and diagnostics |
| Verified | Numerical or implementation checks against a defined reference |
| Validated | Comparison with trusted independent or physical observations, with stated limits |

No result inherits validation merely because it uses validated software. The
model, parameters, operating regime, numerical settings, and comparison data
must support the specific claim.
