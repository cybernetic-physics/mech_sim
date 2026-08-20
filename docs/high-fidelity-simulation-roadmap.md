# Mechanical Simulation Credibility Roadmap

The goal is to make `mech-sim` a credible verifier for agent-designed
mechanical assemblies. The runtime already executes analytic checks and native
rigid-body/contact simulations. The remaining work is to make predictive
claims traceable, calibrated, converged, and validated for defined operating
regimes.

This roadmap is ordered by evidence dependency. Breadth should not outrun
credibility.

## Current baseline

Implemented:

- generic evaluator, task contracts, capability dispatch, hard gates, dense
  scoring, feedback, and evidence bundles;
- analytic static, transmission, and planar-kinematic checks;
- trusted geometry manifests and CAD-derived mass-property paths;
- a native Project Chrono runner for rigid bodies, constraints, motors, loads,
  collision, NSC/SMC contact, and traces;
- one checked-in CAD-to-Chrono cycloidal acceptance packet; and
- learning infrastructure from frozen inference through exact GRPO and online
  verifier feedback.

Not yet established:

- broad calibration of contact and material parameters;
- general solver-backed structural analysis;
- systematic numerical convergence across representative mechanism families;
- independent-solver and physical-reference agreement;
- uncertainty bounds tied to intended operating regimes; and
- completed Level-2/3 agent-learning results.

## Credibility ladder

Every task and reported result should state its highest completed level.

| Level | Evidence | Current example |
|---|---|---|
| 0. Contract | Valid artifact, schema, interfaces, and topology | Static and analytic tasks |
| 1. Analysis | Deterministic kinematic or closed-form mechanical result | Planar paths and transmission checks |
| 2. Solver execution | Real solver, trusted inputs, settings, diagnostics, and replay | Cycloidal CAD-to-Chrono packet |
| 3. Numerical verification | Refinement, convergence, conservation, and regression evidence | Partial diagnostics only |
| 4. Model validation | Agreement with independent or physical reference data in a stated regime | Not yet established broadly |
| 5. Operational qualification | Validated uncertainty is acceptable for a defined decision | Future work |

Solver execution is necessary but is not validation. The label attached to a
report must reflect the completed level, not the ambition of the task.

## Workstream 1: self-consistent benchmark release

Goal: make task counts, documentation, and evidence point to one versioned
benchmark snapshot.

Required work:

- materialize all 58 registered families at a fixed seed;
- record generator commit, dependency versions, and task hashes;
- run reference solutions under public and hidden configurations;
- run all negative controls and record expected/observed failure codes;
- classify every task as analytic, synthetic, simulated, or validated; and
- publish the aggregate summary beside the frozen task snapshot.

Exit criteria:

- registry, checked-in tasks, and result summary agree;
- every unavailable capability is reported explicitly;
- synthetic contact tasks are separated from real-solver tasks; and
- the suite can be replayed without machine-specific source paths.

## Workstream 2: trusted assembly ingestion

Goal: derive physical inputs from trusted geometry rather than agent claims.

Required work:

- normalize units, frames, materials, tolerances, contact surfaces, actuators,
  and load cases;
- recompute volume, center of mass, inertia, envelope, and density;
- generate role-specific render, collision, and analysis meshes with quality
  reports;
- preserve source geometry, mesh, material, and transformation digests; and
- reject ambiguous units, inconsistent frames, invalid solids, impossible
  density, and unsupported material/process declarations.

Exit criteria:

- representative assemblies enter the solver without trusting declared mass
  or inertia;
- reports identify every recomputed and declared quantity; and
- geometry conversion is deterministic or records the source of variation.

## Workstream 3: native contact verification

Goal: make the current Chrono backend numerically defensible for a small set of
representative mechanisms.

Required work:

- choose representative cam, gear, indexer, and reducer fixtures;
- pin solver version, integrator, contact formulation, timestep, solver
  tolerances, and collision settings;
- record constraint violation, penetration, impulses/forces, energy and power
  balance, iteration state, and failure diagnostics;
- run timestep and collision-mesh refinement studies;
- define convergence and conservation acceptance criteria before running; and
- add regression thresholds that detect solver or geometry drift.

Exit criteria:

- reference mechanisms execute without synthetic fallback;
- reference and negative designs fail for mechanical rather than pipeline
  reasons;
- reported metrics are stable within predeclared refinement bounds; and
- every contact report identifies its applicability limits.

Project Chrono’s official
[contact documentation](https://api.projectchrono.org/collisions.html)
explains the different NSC and SMC formulations. Choosing between them is a
modeling decision that must be justified and recorded.

## Workstream 4: structural and manufacturing evidence

Goal: evaluate whether a valid mechanism can be made and survive its loads.

Required work:

- add a solver-backed static structural adapter for stress and displacement;
- derive boundary conditions from trusted ports, joints, fixtures, and load
  cases;
- version material properties with source, temperature, process, orientation,
  and uncertainty;
- emit mesh quality, residual/convergence, extrema, and safety-factor data;
- add tolerance-stack, clearance, tool-access, and assembly checks; and
- model process capability, material stock, operation order, inspection, time,
  and cost.

Exit criteria:

- structural failures come from solver-backed fields rather than declared
  scalar values;
- manufacturing constraints affect both acceptance and optimization reward;
- negative controls isolate boundary-condition, material, geometry, and
  process errors; and
- reports state the model form and uncertainty assumptions.

## Workstream 5: calibration and validation

Goal: establish when simulated metrics predict independent or physical
observations.

Required work:

- build simple calibration fixtures before complex assemblies: sliding block,
  impact/contact pair, shaft fit, gear mesh, cam follower, and gripper pad;
- measure geometry, material, friction, compliance, damping, backlash, load,
  and response with uncertainty;
- fit parameters on calibration data and evaluate on held-out trials;
- cross-check selected cases with an independent solver where its assumptions
  match; and
- version calibration data and include its digest in every dependent report.

Exit criteria:

- prediction error and uncertainty are reported against held-out references;
- the accepted operating range is explicit;
- calibration data is distinct from validation data; and
- tasks outside the validated range do not inherit the validated label.

[Project Chrono’s validation studies](https://projectchrono.org/validation/)
illustrate case-specific comparison against analytical, independent-software,
or experimental references. [ASME V&V 10](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics/2019)
provides the verification, validation, and uncertainty-quantification
vocabulary used here.

## Workstream 6: Level-2/3 learning result

Goal: determine whether verifier-derived learning improves CAD- and
physics-constrained repair, not merely executable program repair.

The repository already contains a 120-task Level-2/3 benchmark manifest and
matched-budget experiment tooling. Its committed final audit records zero
observed result rows, so execution remains outstanding.

Use a staged experiment:

1. select two or three credible families with completed native verification;
2. freeze public, hidden, and isomorphic variants;
3. compare frozen, no-update feedback, search, SFT, and online-learning methods;
4. match actual CAD and solver calls, including retries and failures;
5. require complete raw evidence and adapter-update logs; and
6. expand only after the small study passes its audit.

Exit criteria:

- all compared methods receive matched actual expensive-verifier budgets;
- held-out families and anti-shortcut variants are genuinely disjoint;
- learning improvement is supported by paired statistics and complete
  evidence; and
- the conclusion is scoped separately for CAD validity, contact behavior,
  manufacturability, and physical transfer.

## Workstream 7: durable operations

Goal: make expensive evidence reproducible across machines and over time.

Required work:

- pin and publish native solver environments;
- cache geometry and simulation artifacts by digest;
- use a queue for expensive simulations with bounded retries;
- remove machine-specific absolute paths from portable manifests;
- add fast, native-nightly, and periodic validation regression tiers; and
- replay a sample of old bundles whenever solver or geometry dependencies
  change.

Exit criteria:

- a reviewer can reconstruct a selected result from committed or retrievable
  content-addressed inputs;
- missing heavy artifacts are reported as missing, not implied to exist;
- solver regressions are detected before benchmark numbers change; and
- evidence labels remain stable across dashboards, summaries, and training
  rewards.

## Near-term sequence

The shortest credible path is:

```text
versioned 58-family snapshot
→ representative native contact verification
→ trusted structural/manufacturing checks
→ calibration and held-out validation
→ bounded Level-2/3 learning experiment
→ broader curriculum and physical feedback
```

The [project status](project-status.md) should be updated whenever an exit
criterion becomes supported by checked-in evidence.
