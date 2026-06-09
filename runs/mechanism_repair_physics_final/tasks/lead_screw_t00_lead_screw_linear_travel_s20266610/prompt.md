# Lead screw linear travel

Declare `params.declared_travel_per_rev_mm` = lead_mm = 5.734.
* Input revolute, output prismatic.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `lead_screw`. Source task `lead_screw_linear_travel_s20266610` from generator `lead_screw_linear_travel` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Lead screw actuator: a rotating screw/nut pair converts input rotation into linear travel according to the specified lead.
- Required mechanism roles/parts:
  - fixed frame
  - rotating screw or driven nut
  - translating nut or carriage
  - helical screw constraint
- Preferred stable part ids: `frame`, `screw`, `nut`, `carriage`
- Do not submit:
  - rack and pinion
  - belt drive
  - gear train only
  - slider crank

Task objective:
- Lead screw travel/rev = 5.734 mm.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 2
- `max_envelope_mm`: [200, 80, 50]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `prismatic_joint`, grounded required: no.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `travel` requires `params.declared_travel_per_rev_mm` eq 5.734 (tolerance_pct=2.0).

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
