# Chain sprocket ratio

Ratio = driven/driver = 28/14 = 2.0.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `chain_drive`. Source task `chain_sprocket_ratio_s20264618` from generator `chain_sprocket_ratio` is being evaluated under this canonical family.
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
- Chain ratio = 2.0.

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
- `ratio` requires `params.declared_ratio` eq 2.0 (tolerance_pct=2.0).

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

Minimal trusted CAD/material evidence pattern:
- Use this as a schema pattern, not as the mechanism answer; adapt part ids, joint ids, dimensions, masses, and params to the task.
- Do not return placeholder strings such as `replace_with_task_part_id`; replace them with concrete task part ids and CAD filenames.
- Define every helper you call; submissions fail if they call undefined helpers such as `cad(...)`.
- `materials` must be a dict keyed by material id, not a list.
- Every checked part needs `material`, `geometry["cad"]`, and positive-mass parts need `params["cad_mass_properties"]`.
```python
from pathlib import Path

def _write_step(out_dir: Path, name: str) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rel = name if name.endswith('.step') else f'{name}.step'
    (out_path / rel).write_text(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('submitted CAD artifact'),'2;1');\n"
        "ENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
    )
    return rel

def _mass_props(mass_kg: float, com=(0.0, 0.0, 0.0)) -> dict:
    scale = max(float(mass_kg), 1.0e-6)
    return {
        'mass_kg': float(mass_kg),
        'com_local_mm': tuple(com),
        'inertia_kg_m2': (
            (scale * 1.0e-5, 0.0, 0.0),
            (0.0, scale * 1.2e-5, 0.0),
            (0.0, 0.0, scale * 1.5e-5),
        ),
    }

materials = {
    'steel_1045': {
        'name': 'AISI 1045 steel',
        'density_kg_m3': 7850.0,
        'elastic_modulus_pa': 205000000000.0,
        'poisson_ratio': 0.29,
        'yield_strength_pa': 530000000.0,
        'process': 'machined',
        'provenance': 'standard engineering material table',
    }
}
part = {
    'id': 'replace_with_task_part_id',
    'role': 'replace_with_mechanism_role',
    'mass_kg': 0.05,
    'fixed': False,
    'com_local_mm': (0.0, 0.0, 0.0),
    'material': 'steel_1045',
    'geometry': {'cad': _write_step(out_dir, 'replace_with_task_part_id.step')},
    'params': {'cad_mass_properties': _mass_props(0.05)},
}
params = {
    'cad_source': {
        'kernel': 'submitted CAD artifact',
        'source': 'build_design',
        'family': 'chain_drive',
        'verifier_level': 2,
    }
}
provenance = {'submission': {'created_by': 'build_design'}}
```

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
