# Planetary (ring fixed) ratio (analytic)

Ratio = 1 + ring/sun = 1 + 40/20 = 3.0.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `planetary_reducer`. Source task `planetary_fixed_ring_ratio_analytic_s20261618` from generator `planetary_fixed_ring_ratio_analytic` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Planetary reducer: coaxial sun, planet gears on a carrier, and ring gear; the selected fixed member and output member must match the task's ratio semantics.
- Required mechanism roles/parts:
  - fixed frame/housing
  - sun gear
  - planet gears
  - planet carrier
  - ring gear
- Preferred stable part ids: `frame`, `sun`, `planet_0`, `carrier`, `ring`
- Do not submit:
  - single external spur pair
  - belt drive
  - chain drive
  - lead screw

Task objective:
- Planetary fixed-ring ratio = 3.0.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 2
- `max_envelope_mm`: [200, 200, 80]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `revolute_joint`, grounded required: yes.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `ratio` requires `params.declared_ratio` eq 3.0 (tolerance_pct=2.0).

Hard-gated verifier probes:
- `mobility` must pass.
- `ports` must pass.
- `trusted_asset_preflight` must pass.

DesignIR deliverable:
- Return only Python code defining `build_design(out_dir: Path) -> dict`.
- The returned dict must use `schema_version="design_ir.v2"`, `units="mm"`, and include `parts`, `joints`, `ports`, `params`, `materials`, and `provenance`.
- Include a fixed ground/frame part and positive, finite mass for moving physical parts.
- In `ports`, `revolute_joint` and `prismatic_joint` entries must reference joint ids; `frame` entries must reference part ids.
- Write or reference CAD artifacts through `geometry["cad"]` for checked parts; artifact paths should be relative to `out_dir`.
- Define material records with density, elastic modulus, Poisson ratio, yield strength, process, and provenance.
- Every positive-mass checked part must include trusted `params["cad_mass_properties"]` with mass, COM, and inertia consistent with the part mass.
- Do not use `fake_contact_oracle`, synthetic oracle outputs, or placeholder mechanisms for headline success.

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
