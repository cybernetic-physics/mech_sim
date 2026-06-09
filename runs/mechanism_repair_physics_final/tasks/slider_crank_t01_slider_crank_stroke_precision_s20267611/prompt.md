# Slider-crank stroke precision

Design a slider-crank with crank 24.37, coupler 77.32.
* Declare `params.declared_stroke_mm` = 48.74 mm.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `slider_crank`. Source task `slider_crank_stroke_precision_s20267611` from generator `slider_crank_stroke_precision` is being evaluated under this canonical family.
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
- Slider-crank stroke = 48.74 mm.

Task-level requirements:
- `required_ports`: ["input_port", "output_port"]
- `expected_mobility`: 1
- `max_envelope_mm`: [220, 80, 50]

Required interface ports:
- `input_port`: kind `revolute_joint`, grounded required: yes.
- `output_port`: kind `prismatic_joint`, grounded required: no.
- Port ids must match exactly.
- `ports` must be a dict keyed by port id, not a list. Each value must include the same `id` field as its key.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Functional/numeric checks:
- `stroke` requires `params.declared_stroke_mm` eq 48.74 (tolerance_pct=2.0).

Hard-gated verifier probes:
- `mobility` must pass.
- `ports` must pass.
- `trusted_asset_preflight` must pass.

DesignIR deliverable:
- Return only Python code defining `build_design(out_dir: Path) -> dict`.
- The returned dict must use `schema_version="design_ir.v2"`, `units="mm"`, and include `parts`, `joints`, `ports`, `params`, `materials`, and `provenance`.
- `parts` and `joints` must be lists. `ports` must be a dict keyed by port id, not a list; for example `{'input_port': {'id': 'input_port', ...}}`.
- Include a fixed ground/frame part and positive, finite mass for moving physical parts.
- In the `ports` dict, `revolute_joint` and `prismatic_joint` values must reference joint ids; `frame` values must reference part ids.
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
        'family': 'slider_crank',
        'verifier_level': 2,
    }
}
provenance = {'submission': {'created_by': 'build_design'}}
```

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
