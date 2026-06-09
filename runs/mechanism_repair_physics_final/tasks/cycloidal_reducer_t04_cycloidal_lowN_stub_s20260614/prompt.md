# Low-N cycloidal stub

Single-stage cycloidal reducer with 12 ring pins (target ratio 11).

Requires torque-load and contact-force capabilities; evaluated by the Chrono contact adapter when available.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `cycloidal_reducer`. Source task `cycloidal_lowN_stub_s20260614` from generator `cycloidal_lowN_stub` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Cycloidal speed reducer: an eccentric input shaft drives a cycloidal disc inside a fixed housing/ring-pin set, and an output pin carrier takes reduced rotation from the disc.
- Required mechanism roles/parts:
  - fixed housing or ring-pin ground part named housing
  - eccentric input shaft/crank
  - cycloidal disc named disc
  - output pin carrier or output hub
- Preferred stable part ids: `housing`, `input_shaft`, `disc`, `output_carrier`
- Do not submit:
  - plain spur gear pair
  - planetary gearbox
  - belt or chain drive
  - generic pulley transmission

Task objective:
- Cycloidal reducer with 12 ring pins; declared ratio 11.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 1
- `max_envelope_mm`: [200, 200, 80]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: no.
- `output_port`: kind `revolute_joint`, grounded required: no.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `ratio` requires `params.declared_ratio` eq 11.0 (tolerance_pct=1.0).

Hard-gated verifier probes:
- `ports` must pass.
- `trusted_asset_preflight` must pass.
- `chrono_contact_smoke` must pass.

DesignIR deliverable:
- Return only Python code defining `build_design(out_dir: Path) -> dict`.
- The returned dict must use `schema_version="design_ir.v2"`, `units="mm"`, and include `parts`, `joints`, `ports`, `params`, `materials`, and `provenance`.
- Include a fixed ground/frame part and positive, finite mass for moving physical parts.
- In `ports`, `revolute_joint` and `prismatic_joint` entries must reference joint ids; `frame` entries must reference part ids.
- Write or reference CAD artifacts through `geometry["cad"]` for checked parts; artifact paths should be relative to `out_dir`.
- Define material records with density, elastic modulus, Poisson ratio, yield strength, process, and provenance.
- Every positive-mass checked part must include trusted `params["cad_mass_properties"]` with mass, COM, and inertia consistent with the part mass.
- Do not use `fake_contact_oracle`, synthetic oracle outputs, or placeholder mechanisms for headline success.

Level-3 Chrono contact requirements:
- Use real `chrono_contact`; fake contact oracle outputs are rejected.
- Include `contact_pair` joints and `params["chrono"]` metadata for contact simulation.
- Contact bodies must provide `params["chrono_collision"]` primitive or trusted collision geometry.
- Required contact pairs: `housing:disc`
  - Pair `housing:disc` means part `housing` contacts part `disc`.

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
