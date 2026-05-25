# High-Fidelity Mechanical Simulation Roadmap

This is the canonical long-horizon plan for turning `mech_sim` from a
mechanical-design RLVR benchmark runtime into a full high-fidelity
mechanical simulation stack. Future planning, refactors, benchmark
expansion, and agent work should preserve this direction unless this
document is intentionally revised.

## Mission

Build a credible high-fidelity mechanical simulation stack, not a narrow
demo or MVP. The desired end state is a trusted oracle that can evaluate
AI-generated mechanical assemblies from CAD and structured metadata,
simulate rigid and flexible behavior, produce calibrated contact and
structural evidence, and emit validation-grade reports.

`mech_bench` remains the verifier and control plane. The missing layer is
the physical oracle underneath it: trusted CAD ingestion, real
multibody/contact dynamics, structural analysis, material/contact
calibration, convergence evidence, and validation datasets.

## Current Baseline

- `mech_bench` has a working evaluator, `DesignIR`, `TaskSpec`,
  `EvalConfig`, probe registry, adapter registry, hard-gate scoring,
  failure grammar, report bundles, RLVR reward API, and task suite.
- `planar_kinematics` is an analytic adapter for limited planar
  mechanisms. It is useful, but it is not a general mechanical simulator.
- `fake_contact_oracle` is synthetic. It is for tests and demos only.
  It must never be described as high fidelity or used as physical
  validation evidence.
- `chrono_contact` is a skeleton. A real `_chrono_impl` runner must be
  added before contact-dynamics tasks become physically credible.
- Many current tasks are analytic, topology, or declared-parameter
  checks. They are valid benchmark tasks, but they do not prove
  high-fidelity simulation credibility.

## Non-Negotiable Principles

- Trusted physical truth is recomputed from CAD/geometry on the trusted
  side. Agent-declared mass, inertia, envelope, contact area, and
  strength values are hints at most, never the source of truth.
- Synthetic oracle output is never validation evidence. It may exercise
  pipeline wiring, but reports must keep `oracle_is_synthetic=true` and
  reward policy must treat unavailable real physics honestly.
- High-fidelity claims require solver diagnostics, calibration, and
  verification/validation evidence. A passing probe is not enough.
- Breadth follows depth. Do not add large families of nominally
  "physics" tasks before representative mechanisms have real CAD,
  real solver execution, convergence checks, and validation packets.
- RL progress and physics credibility are separate claims. A model
  improving on proxy tasks does not make the simulator high fidelity.
- All simulation artifacts must be replayable from content-addressed
  inputs: task, submission, geometry, material database, solver version,
  solver config, and random seeds.

## Target Architecture

### Trusted Asset Layer

Inputs:

- `design.py` submission.
- STEP or constrained parametric geometry.
- Explicit units, frames, joints, ports, actuators, contact surfaces,
  materials, tolerances, manufacturing process, and load cases.

Trusted outputs:

- `DesignIR v2`.
- Canonical scene graph.
- OpenUSD physics scene.
- Chrono scene package.
- FEA model package.
- Mesh and material manifests.
- Hash-addressed evidence bundle.

Responsibilities:

- Import CAD and normalize units/frames.
- Recompute volume, center of mass, inertia tensor, envelope, density,
  wall thickness, clearance, and collision geometry.
- Generate render meshes, collision meshes, convex proxies, and FEA
  meshes with quality reports.
- Reject nonphysical geometry, bad mass properties, path escapes,
  malformed frames, invalid units, and inconsistent material metadata.

### Physics Oracle Layer

Primary backend:

- Project Chrono for multibody dynamics, contact dynamics, motors,
  constraints, loads, and eventually flexible-body coupling.

Cross-check backends:

- Drake for selected contact and hydroelastic sanity checks.
- MuJoCo for fast robotics-style dynamics comparisons where its contact
  assumptions are appropriate.
- Isaac Sim/PhysX/OpenUSD for scene interchange, robotics workflow, and
  selected USD physics compatibility tests.

The adapter contract remains the existing `SimOutput` shape, extended as
needed for torques, impulses, residuals, solver status, energy channels,
mesh provenance, and uncertainty metadata.

### Structural and Material Layer

- Add solver-backed static FEA probes for stress, displacement, and
  safety factor.
- Add fatigue, modal, thermal, and flexible-body analyses only after
  static load cases are validated.
- Maintain a material database with provenance, units, temperature
  assumptions, print/process orientation where relevant, and uncertainty
  ranges.
- Maintain a contact database for friction, restitution, stiffness,
  damping, compliance, surface roughness, backlash, bearing losses, and
  tolerance assumptions.

### V&V Layer

Every high-fidelity report must include:

- Solver backend and exact version.
- Solver settings, timestep, tolerances, integrator, contact method,
  collision representation, and mesh resolution.
- Constraint violation, penetration, contact impulse/force statistics,
  residuals, energy/power balance, and convergence status.
- Timestep refinement and mesh refinement evidence for benchmark
  validation runs.
- Material/contact database versions and uncertainty bands.
- Link to validation fixture or trusted offline simulation dataset.

Use ASME V&V 10 style language: verification, validation, uncertainty
quantification, and stated credibility limits.

## Phase Gates

### Phase 0: Preserve the Control Plane

Goal: Keep the current evaluator reliable while the physics stack is
built.

Exit criteria:

- Existing tests and negative controls pass.
- The fake oracle remains explicitly synthetic and opt-in.
- Public docs point to this roadmap.
- No new task claims high-fidelity dynamics without real solver evidence.

### Phase 1: DesignIR v2 and Trusted CAD Ingestion

Goal: represent real assemblies and recompute physical properties from
trusted geometry.

Required work:

- Add `DesignIR v2` fields for units, frames, materials, tolerances,
  contact surfaces, actuators, joint losses, load cases, and mesh roles.
- Implement schema migration from current `DesignIR`.
- Add STEP import and mass-property recomputation through a trusted CAD
  kernel.
- Add geometry artifact generation: render mesh, collision mesh, convex
  proxy, FEA mesh, and mesh-quality report.
- Add validation failures for unit ambiguity, frame inconsistency,
  non-watertight geometry, bad inertia, impossible density, and
  unsupported material/process declarations.

Exit criteria:

- A submitted CAD assembly can be converted into trusted geometry
  artifacts without trusting agent-declared mass or inertia.
- Existing non-CAD tasks still run through compatibility paths.
- Reports identify exactly which physical quantities were recomputed.

### Phase 2: Real Chrono Multibody and Contact Backend

Goal: replace synthetic contact for Tier-3 physics tasks with real
Chrono execution.

Required work:

- Implement `mech_bench/adapters/_chrono_impl.py`.
- Map trusted scene graph bodies, joints, motors, springs, dampers,
  loads, contact materials, and collision geometry into Chrono.
- Support both NSC and SMC contact modes where appropriate.
- Emit time series for body poses, joint states, velocities, motor
  torques, applied loads, contact forces, penetration, constraint
  violation, and energy/power channels.
- Add solver diagnostics and capability-unavailable failure modes that
  distinguish missing dependencies from failed simulations.

Exit criteria:

- Representative contact tasks run without `fake_contact_oracle`.
- Solver diagnostics are present in every contact report.
- Reference designs and negative controls separate physical failures
  from pipeline failures.

### Phase 3: Solver-Backed Structural Analysis

Goal: make structural probes physical rather than declared-metric checks.

Required work:

- Add an FEA adapter for static stress, displacement, and safety factor.
- Generate boundary conditions from ports, joints, fixtures, and load
  cases.
- Add material model support with provenance and process assumptions.
- Emit mesh quality, boundary condition summary, stress extrema,
  displacement extrema, FOS, and solver convergence data.

Exit criteria:

- `safety_factor` and related structural probes are solver-backed for
  CAD-backed tasks.
- Negative controls fail for real stress/deformation reasons.
- Reports state credibility limits for each structural result.

### Phase 4: Contact and Material Calibration

Goal: replace magic constants with calibrated, versioned physical
parameters.

Required work:

- Build calibration fixtures: sliding block, gear pair, cam follower,
  brake pad, gripper pad, snap fit, bearing seat, latch/ratchet, and
  belt/chain proxy.
- Record physical measurements or trusted offline simulation data.
- Fit contact/material parameters with uncertainty ranges.
- Version the calibration database and include it in report provenance.

Exit criteria:

- Contact tasks cite calibration records.
- Reports include uncertainty bands and validation residuals.
- Contact parameters are not ad hoc task-local constants.

### Phase 5: Benchmark Migration and Expansion

Goal: migrate from proxy tasks to broad CAD-backed physics tasks.

Required work:

- Replace synthetic contact stubs with real CAD, real load cases, and
  real solver-backed expected failures.
- Expand families for gears, belts, chains, cams, clutches, brakes,
  bearings, linkages, flexures, snap fits, fasteners, tolerance stacks,
  and actuator transmissions.
- For each family, include reference design, negative controls,
  validation packet, public/hidden eval configs, and V&V assumptions.

Exit criteria:

- Each tier has physically meaningful solver-backed tasks where
  appropriate.
- New task families are blocked unless they include validation artifacts
  or are explicitly marked analytic/proxy.
- Benchmark dashboards distinguish analytic, synthetic, simulated, and
  validated evidence.

### Phase 6: Operations, Scale, and Cross-Engine Validation

Goal: make the stack reproducible and durable for long-running agent
training and evaluation.

Required work:

- Containerize solver workers with pinned versions.
- Add a job queue for expensive simulation and FEA runs.
- Cache geometry and simulation artifacts by digest.
- Add deterministic replay from report bundles.
- Add CI tiers: fast unit tests, nightly solver regression, weekly V&V
  regression.
- Add cross-engine checks for selected cases using Drake, MuJoCo, and
  Isaac/OpenUSD where their assumptions match the task.

Exit criteria:

- Any high-fidelity result can be replayed from its bundle.
- Solver regressions are caught before benchmark results change.
- Reports clearly separate verified, validated, cross-checked, and
  unsupported claims.

## Drift Guards

Future agents and maintainers should check this list before making
architecture changes:

- Do not rename synthetic contact output into oracle output.
- Do not let `fake_contact_oracle` silently satisfy production
  high-fidelity tasks.
- Do not accept non-CAD declared values as trusted physical evidence.
- Do not bury solver settings or material parameters in task-local
  scripts.
- Do not add broad task families before at least one representative
  family has passed the full CAD -> Chrono/FEA -> V&V path.
- Do not make `DesignIR v2` incompatible with existing generated tasks
  without a migration path.
- Do not report a single pass/fail score without preserving diagnostic
  evidence.
- Do not treat RL reward improvement as simulator validation.

## Minimum Staffing Assumption

The full stack needs sustained specialist ownership:

- Multibody/contact simulation engineer.
- CAD/geometry/meshing engineer.
- FEA/materials engineer.
- Infra/backend engineer.
- Benchmark/V&V engineer.
- Mechanical testing or calibration support.

The critical path is:

`DesignIR v2` -> trusted CAD ingestion -> real Chrono backend ->
solver-backed contact tasks -> FEA backend -> calibration database ->
V&V reporting -> broad benchmark migration.

## External Reference Points

These are not dependencies by themselves; they define the level of
capability this project should be measured against.

- Project Chrono: https://projectchrono.org/
- Chrono repository: https://github.com/projectchrono/chrono
- Chrono collision/contact docs: https://api.chrono.projectchrono.org/collisions.html
- Drake hydroelastic/contact docs: https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html
- MuJoCo computation/contact docs: https://mujoco.readthedocs.io/en/2.1.5/computation.html
- Isaac Sim physics docs: https://docs.isaacsim.omniverse.nvidia.com/latest/physics/index.html
- OpenUSD physics schema: https://openusd.org/release/api/usd_physics_page_front.html
- ASME V&V 10 overview: https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics
