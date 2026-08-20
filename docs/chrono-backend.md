# Chrono Backend

`chrono_contact` is the optional native rigid-body and contact adapter. It is a
real Project Chrono execution path, but its results are described as
**simulated**, not automatically **validated**.

## Availability

The adapter registers only when both the local runner and a compatible Project
Chrono Python runtime are available. The runtime can be in the active Python
environment or supplied through `MECH_BENCH_CHRONO_PYTHON`.

Inspect a machine before relying on native physics:

```bash
uv run mech-bench chrono-diagnostic
uv run mech-bench oracle-smoke --require-real
```

The first command explains import and registration state. The second exercises
the configured native dependency stack and basic kernel operations. For the
pinned solver environment, use:

```bash
scripts/solver_smoke.sh
```

Missing native dependencies produce `capability_unavailable`; they do not
silently route a physical task through synthetic output.

## Implemented capabilities

The runner in `mech_bench/adapters/_chrono_impl.py` supports:

- rigid bodies with mass, center of mass, and inertia;
- revolute, prismatic, and fixed constraints;
- rotational motors, prescribed loads, and time-varying load profiles;
- primitive, triangle-mesh, and convex-decomposition collision geometry;
- non-smooth-contact (NSC) and smooth-contact (SMC) systems;
- body pose and velocity, contact force, penetration, motor torque,
  constraint, and energy/power traces;
- solver, timestep, contact, and preflight diagnostics; and
- a trusted-asset bridge used by the checked-in cycloidal fixtures.

Project Chrono documents NSC as a constraint-based formulation and SMC as a
penalty formulation. They can produce materially different behavior for the
same imperfectly conditioned model, which is why the repository records the
contact method and treats comparison as a diagnostic rather than selecting the
more favorable result after the fact. See the official
[contact documentation](https://api.projectchrono.org/collisions.html).

## Geometry and trust boundary

For CAD-backed tasks, the trusted side can:

1. import or generate the named CAD assets;
2. record geometry digests and feature frames;
3. recompute mass, center of mass, and inertia through the CAD path;
4. construct Chrono collision bodies from trusted geometry roles; and
5. record whether any procedural geometry fallback was used.

An agent-declared mass or performance number is not equivalent to recomputed
evidence. Tasks can require trusted mass properties and can reject procedural
fallback explicitly.

## Synthetic adapter policy

`fake_contact_oracle` is deterministic pipeline-test infrastructure. It is
registered for a run only when task configuration or an explicit test-mode
setting opts in. Reports consuming it carry:

```json
{
  "oracle_is_synthetic": true,
  "is_physical_oracle": false,
  "trust_level": "synthetic_test_or_demo"
}
```

Synthetic contact output can test dispatch, scoring, feedback, and learning
plumbing. It cannot establish contact mechanics or hardware behavior.

## Current evidence

The strongest checked-in native packet is the
[cycloidal real-geometry proof](cycloidal-real-geometry-proof.md). It shows
that named CAD assets, CAD-derived physical properties, mesh collision, and
both contact formulations reach the real runner without procedural fallback.

That packet is a solver-path acceptance test. It is not a calibrated gearbox
model: the loaded SMC result still reports a power-balance failure, and the NSC
case enters a poor numerical/physical regime. The later
[acceptance run](cycloidal_real_geometry_chrono_validation.md) adds refinement
diagnostics but retains the same loaded-case limitation.

## What is required for stronger claims

A simulated report should record geometry, material and contact parameters,
solver version, contact formulation, timestep, tolerances, collision
representation, and relevant convergence diagnostics. A validated result also
needs comparison with a trusted independent or physical dataset in the stated
operating regime.

Priority credibility work:

- calibrate representative friction, compliance, damping, and material data;
- run systematic timestep and collision-mesh refinement studies;
- compare representative mechanisms with an independent solver or reference
  solution;
- test selected mechanisms against controlled physical measurements; and
- preserve uncertainty and applicability limits with every result.

Project Chrono publishes [validation studies](https://projectchrono.org/validation/)
for selected components and provides utilities for comparing simulation and
reference traces. [ASME V&V 10](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics/2019)
provides the broader verification, validation, and uncertainty-quantification
framework used for the project’s credibility language.

The [high-fidelity simulation roadmap](high-fidelity-simulation-roadmap.md)
turns these requirements into ordered implementation gates.
