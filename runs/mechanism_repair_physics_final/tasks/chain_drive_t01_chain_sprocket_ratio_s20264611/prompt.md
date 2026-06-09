# Chain sprocket ratio

Ratio = driven/driver = 56/14 = 4.0.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `chain_drive`. Source task `chain_sprocket_ratio_s20264611` from generator `chain_sprocket_ratio` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Chain drive: two sprockets coupled by a chain loop; tooth counts and sprocket radii must encode the requested ratio.
- Required mechanism roles/parts:
  - fixed frame
  - input sprocket/shaft
  - output sprocket/shaft
  - chain loop or chain span representation
- Preferred stable part ids: `frame`, `input_sprocket`, `output_sprocket`, `chain`
- Do not submit:
  - belt drive without chain/sprocket semantics
  - spur gear mesh
  - rack and pinion
  - lead screw

Task objective:
- Chain ratio = 4.0.

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
- `ratio` requires `params.declared_ratio` eq 4.0 (tolerance_pct=2.0).

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
