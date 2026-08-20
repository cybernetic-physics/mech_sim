# Cycloidal Real-Geometry Chrono Acceptance Run

> **Scope:** solver-path and fixture acceptance, including refinement
> diagnostics. This is not hardware or predictive-model validation. The loaded
> SMC case still reports `power_balance_error`.

Executed on branch `mech-sim-natalia` with FreeCAD/OCCT-generated
CycloidGearBox assets and Project Chrono SMC contact.

Command:

```bash
MECH_BENCH_FREECAD_CMD=$PWD/.external/bin/mech-freecadcmd \
MECH_BENCH_CYCLOID_GEARBOX_PATH=$PWD/.external/src/CycloidGearBox \
MECH_BENCH_CHRONO_PYTHON=$PWD/.external/micromamba/envs/mech-chrono/bin/python \
uv run --extra dev python scripts/prove_cycloidal_chrono_real_geometry.py \
  --out-dir runs/cycloidal_real_geometry_validation_passing \
  --proof-json runs/cycloidal_real_geometry_validation_passing/proof.json
```

Result: `ok=true`.

Required gates passed:

- CAD export, manifest, named bodies, nonempty STEP/STL assets.
- CAD datums and static contact audit.
- Trusted CAD mass properties.
- Chrono NSC and SMC run on real geometry with `procedural_cycloidal_fallback=false`.
- Loaded SMC has bounded penetration and finite power/torque/contact metrics.
- Unloaded SMC ratio is near the declared 9:1 reducer ratio.
- Sample and timestep convergence checks are stable.

Key proof metrics:

| run | ratio_observed | ratio_error_pct | out_omega_med | lockup | max_penetration_mm | n_contacts_max | contact_force_rms_N | failure_mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smc_unloaded | 10.2298 | 13.6644 | 0.9775 | 0 | 0.2431 | 23 | 24.2514 | none |
| smc_loaded | 3.7474 | 58.3623 | -2.6685 | 0 | 0.4008 | 33 | 148.1739 | power_balance_error |
| nsc_loaded | 55.1972 | 513.3021 | -0.1812 | 1 | 0.2434 | 101 | 417360.3208 | lockup_mechanism_jammed |

Convergence:

- Sample refinement ratios: `10.2298`, `10.1523`, `9.4349`.
- Max sample-refinement ratio error: `13.6644%`.
- Timestep ratios: `9.8061`, `9.4139`, `10.2298`.
- Max timestep ratio error: `13.6644%`.

Diagnostic note: `nsc_bad_regime_observed=false` is retained as a non-gating
diagnostic in the proof JSON. The NSC run still locks up and reports much
larger contact forces than SMC, but it does not satisfy the older hard-coded
bad-regime predicate.
