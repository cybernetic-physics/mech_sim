# Future Chrono oracle

This document records the policy for the contact-dynamics oracle in
this repository.

## Current state (this branch)

* `chrono_contact` is **skeleton-only**. It declares the contact /
  rigid-body / motor / load / pose capabilities but registers itself
  only when both:
    1. `pychrono` is importable, AND
    2. `mech_bench.adapters._chrono_impl` is importable.
  Until both conditions hold, the dispatcher surfaces
  `capability_unavailable` for any probe that needs contact forces.
* `fake_contact_oracle` is **synthetic test/demo infrastructure**.
  It is not a physical oracle. It must be enabled explicitly:
    * env: `MECH_BENCH_USE_FAKE_ORACLE=1` / `MECH_BENCH_TEST_MODE=1`;
    * eval config: `[adapters.fake_contact_oracle] enabled = true`;
    * mode-level: `forced_adapter = "fake_contact_oracle"`.
  Reports that consume its output are tagged with
  `oracle_is_synthetic = true`, `is_physical_oracle = false`, and
  `trust_level = "synthetic_test_or_demo"`.
* Tier 3 generated contact tasks are **test/demo tasks**, not
  physical proof. They drive the dispatcher and fake oracle through
  realistic-looking probe configs; they do not validate physics.

## Future state

Real Chrono integration is a separate phase:

* Port a phys-sim-style `_chrono_impl.py` shim into
  `mech_bench/adapters/`. It must expose `run(ir, config) -> dict`
  in the canonical SimOutput shape and respect the
  `MECH_BENCH_CHRONO_PYTHON` alt-interpreter contract.
* Add STEP / mesh ingestion (likely OpenCascade or OCP) so generated
  contact tasks can carry geometry instead of analytic placeholders.
* Promote `mass_properties` validation from heuristic to authoritative
  by reading directly from the geometry kernel.

Do **not** implement any of the above in this branch.
