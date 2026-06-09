# Bearing seat clearance

Design a fixed bearing seat with bore Ø21.18 mm and bearing OD Ø21.13 mm.

* Declare `params.clearance_mm` ≥ 0.02 mm.
* Mobility = 0 (bearing fixed in housing for this analytic task).

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `shaft_bearing_coupling`. Source task `bearing_seat_clearance_s20271617` from generator `bearing_seat_clearance` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Shaft/bearing/coupling assembly: a shaft, hub or bearing seat, and keyed/clearance/interference interfaces satisfy the fit and alignment constraints.
- Required mechanism roles/parts:
  - shaft
  - hub, bearing, or housing seat
  - key or retaining feature when required
  - fixed reference frame or housing
- Preferred stable part ids: `shaft`, `hub`, `bearing`, `housing`, `key`
- Do not submit:
  - gear train
  - belt drive
  - linkage
  - contact-only cam task

Task objective:
- Bearing-seat clearance ≥ 0.02 mm.

Task-level requirements:
- `required_ports`: ["bore_face", "bearing_seat"]
- `expected_mobility`: 0
- `max_envelope_mm`: [120, 120, 50]

Required interface ports:
- `bore_face`: kind `any`, grounded required: yes.
- `bearing_seat`: kind `any`, grounded required: no.
- Port ids must match exactly.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `clearance` requires `params.clearance_mm` ge 0.02 (tolerance_abs=0.001).

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
