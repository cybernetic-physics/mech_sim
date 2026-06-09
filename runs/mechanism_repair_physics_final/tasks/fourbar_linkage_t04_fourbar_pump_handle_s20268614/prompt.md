# Four-bar pump handle

Output rocker should sweep within the declared range.
* Declare `params.output_rocker_min_rad` = -0.785.
* Declare `params.output_rocker_max_rad` = 0.785.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `fourbar_linkage`. Source task `fourbar_pump_handle_s20268614` from generator `fourbar_pump_handle` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Planar four-bar linkage with ground, input crank, coupler, and output rocker/crank; coupler geometry must satisfy the path or arc objective.
- Required mechanism roles/parts:
  - fixed ground link
  - input crank
  - coupler link
  - output rocker or crank
  - coupler point when required
- Preferred stable part ids: `ground`, `input_crank`, `coupler`, `output_rocker`
- Do not submit:
  - slider-crank with prismatic output
  - gear train
  - belt drive
  - cam follower

Task objective:
- Pump-handle four-bar; bounded output.

Task-level requirements:
- `required_ports`: ["input_port", "output_port", "coupler_point"]
- `expected_mobility`: 1
- `max_envelope_mm`: [220, 220, 50]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `revolute_joint`, grounded required: yes.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `min_range` requires `params.output_rocker_min_rad` le -0.785 (tolerance_abs=0.1).
- `max_range` requires `params.output_rocker_max_rad` ge 0.785 (tolerance_abs=0.1).

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
