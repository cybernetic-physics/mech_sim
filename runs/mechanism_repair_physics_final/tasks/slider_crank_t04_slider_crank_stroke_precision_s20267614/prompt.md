# Slider-crank stroke precision

Design a slider-crank with crank 24.89, coupler 61.4.
* Declare `params.declared_stroke_mm` = 49.78 mm.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `slider_crank`. Source task `slider_crank_stroke_precision_s20267614` from generator `slider_crank_stroke_precision` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Slider-crank mechanism: a crank, connecting rod, and prismatic slider convert rotary motion into reciprocating linear motion.
- Required mechanism roles/parts:
  - fixed frame
  - rotating crank
  - connecting rod
  - prismatic slider
- Preferred stable part ids: `frame`, `crank`, `connecting_rod`, `slider`
- Do not submit:
  - four-bar without a slider
  - rack and pinion
  - lead screw
  - gear train

Task objective:
- Slider-crank stroke = 49.78 mm.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 1
- `max_envelope_mm`: [220, 80, 50]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `prismatic_joint`, grounded required: no.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `stroke` requires `params.declared_stroke_mm` eq 49.78 (tolerance_pct=2.0).

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
