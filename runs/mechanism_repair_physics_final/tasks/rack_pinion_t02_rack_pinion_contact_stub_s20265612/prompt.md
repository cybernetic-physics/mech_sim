# Rack–pinion contact stub

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `rack_pinion`. Source task `rack_pinion_contact_stub_s20265612` from generator `rack_pinion_contact_stub` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Rack and pinion: a rotating pinion meshes with a translating rack to convert rotary input into linear output.
- Required mechanism roles/parts:
  - fixed frame
  - rotating pinion named pinion
  - translating rack named rack
  - contact or mesh relation between pinion and rack
- Preferred stable part ids: `frame`, `pinion`, `rack`
- Do not submit:
  - two rotating gears only
  - belt drive
  - chain drive
  - lead screw without rack teeth

Task objective:
- rack_pinion_contact_stub synthetic contact stub.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 2
- `max_envelope_mm`: [200, 200, 80]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `prismatic_joint`, grounded required: no.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `travel` requires `params.declared_travel_per_rev_mm` eq 68.1977 (tolerance_pct=2.0).

Hard-gated verifier probes:
- `ports` must pass.
- `contact` must pass.
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

Level-3 Chrono contact requirements:
- Use real `chrono_contact`; fake contact oracle outputs are rejected.
- Include `contact_pair` joints and `params["chrono"]` metadata for contact simulation.
- Contact bodies must provide `params["chrono_collision"]` primitive or trusted collision geometry.
- Required contact pairs: `pinion:rack`
  - Pair `pinion:rack` means part `pinion` contacts part `rack`.

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
