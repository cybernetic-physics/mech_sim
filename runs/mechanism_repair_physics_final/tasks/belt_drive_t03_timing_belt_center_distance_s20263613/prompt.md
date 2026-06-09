# Timing belt center distance

Belt length = pitch × tooth_count = 5.0×120 = 600.0 mm.
Center distance = 280.0 mm.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `belt_drive`. Source task `timing_belt_center_distance_s20263613` from generator `timing_belt_center_distance` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Belt drive: two pulleys connected by a belt, with the pulley radii/diameters and center distance satisfying the task.
- Required mechanism roles/parts:
  - fixed frame
  - input pulley/shaft
  - output pulley/shaft
  - belt span or belt loop representation
- Preferred stable part ids: `frame`, `input_pulley`, `output_pulley`, `belt`
- Do not submit:
  - gear mesh
  - chain and sprocket
  - rack and pinion
  - lead screw

Task objective:
- Timing belt geometry.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 2
- `max_envelope_mm`: [400, 200, 80]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `revolute_joint`, grounded required: yes.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `belt_length` requires `params.declared_belt_length_mm` eq 600.0 (tolerance_pct=2.0).
- `center` requires `params.declared_center_distance_mm` eq 280.0 (tolerance_pct=2.0).

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
