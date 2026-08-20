# Chrono backend status and policy

This document records the implementation state and trust policy for contact
dynamics in this repository.

## Current implementation

`chrono_contact` is an optional real-solver adapter. It registers when a
Project Chrono Python runtime is available, either in-process or through the
interpreter configured by `MECH_BENCH_CHRONO_PYTHON`.

The runner in `mech_bench/adapters/_chrono_impl.py` currently supports:

- rigid bodies with mass, center-of-mass, and inertia properties;
- revolute, prismatic, and fixed constraints;
- motors, loads, and time-varying load profiles;
- primitive, triangle-mesh, and convex-decomposition collision geometry;
- NSC and SMC contact configuration;
- pose, velocity, contact, penetration, torque, constraint, and energy traces;
- solver metadata and structured capability/preflight/error reporting; and
- a procedural cycloidal path used by the checked-in geometry proof.

Run these commands to inspect a machine's actual capability:

```bash
uv run mech-bench chrono-diagnostic
uv run mech-bench oracle-smoke
```

The containerized native dependency gate is `scripts/solver_smoke.sh`.

## Evidence policy

The presence of a real solver is necessary but not sufficient for a
high-fidelity claim. A simulated result must record its geometry, materials,
solver version, contact model, timestep, tolerances, and relevant diagnostics.
A validated result additionally needs comparison against a trusted physical or
independent reference dataset.

`fake_contact_oracle` remains synthetic test infrastructure. It must be enabled
explicitly through task configuration or a test-mode environment variable.
Reports that consume it are tagged with:

```json
{
  "oracle_is_synthetic": true,
  "is_physical_oracle": false,
  "trust_level": "synthetic_test_or_demo"
}
```

Synthetic output must never be presented as physical validation evidence.

## Remaining credibility work

- Calibrate representative contact and material parameters against trusted
  reference data.
- Add systematic timestep, mesh, and solver-convergence studies.
- Broaden trusted CAD ingestion beyond the checked-in proof fixtures.
- Cross-check representative mechanisms with an independent solver.
- Pair virtual tasks with controlled physical trials.

The [high-fidelity simulation roadmap](high-fidelity-simulation-roadmap.md)
defines the gates for that work.
