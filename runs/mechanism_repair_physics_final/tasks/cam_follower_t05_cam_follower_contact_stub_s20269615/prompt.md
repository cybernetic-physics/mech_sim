# Cam–follower contact (synthetic stub)

Synthetic Tier-3 stub: fake oracle reports cam:follower contact force and penetration.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `cam_follower`. Source task `cam_follower_contact_stub_s20269615` from generator `cam_follower_contact_stub` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Cam-follower mechanism: a rotating cam drives a follower through physical contact, producing the requested output motion.
- Required mechanism roles/parts:
  - fixed frame
  - rotating cam named cam
  - follower named follower
  - contact pair between cam and follower
- Preferred stable part ids: `frame`, `cam`, `follower`
- Do not submit:
  - gear pair
  - belt or chain drive
  - Geneva indexer
  - plain four-bar linkage

Task objective:
- Cam-follower contact stub.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 2
- `max_envelope_mm`: [120, 120, 80]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `revolute_joint`, grounded required: no.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

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
- Required contact pairs: `cam:follower`
  - Pair `cam:follower` means part `cam` contacts part `follower`.

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
