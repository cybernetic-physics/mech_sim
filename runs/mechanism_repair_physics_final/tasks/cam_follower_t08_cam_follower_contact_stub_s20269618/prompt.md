# Cam–follower contact (synthetic stub)

Synthetic Tier-3 stub: fake oracle reports cam:follower contact force and penetration.

## MechanismRepair-Physics canonical contract

Canonical mechanism family: `cam_follower`. Source task `cam_follower_contact_stub_s20269618` from generator `cam_follower_contact_stub` is being evaluated under this canonical family.
Build this canonical mechanism family, not an analogous transmission.

Family mechanism:
- Cam-follower mechanism: a rotating cam drives a follower through physical contact, producing the requested output motion.
- Required mechanism roles/parts:
  - fixed frame
  - rotating cam named cam
  - follower named follower
  - contact pair between cam and follower
- Preferred stable part ids: `frame`, `cam`, `follower`
- Physical bodies only go in `parts`; joints, contact pairs, and port records go in `joints`/`ports`, not in `parts`.
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
- `ports` must be a dict keyed by port id, not a list. Each value must include the same `id` field as its key.
- For `revolute_joint` or `prismatic_joint` ports, `port.part` must reference the id of the corresponding joint in `joints`, not the moving part id.
- Create explicit revolute/prismatic joints for these ports; do not set a revolute/prismatic port to a physical part id such as `cam`, `follower`, `pinion`, `rack`, `screw`, or `slider`.
- For `frame` ports, `port.part` must reference a part id. Grounded port checks pass only when the referenced joint touches a fixed ground/frame part.

Hard-gated verifier probes:
- `ports` must pass.
- `contact` must pass.
- `trusted_asset_preflight` must pass.

DesignIR deliverable:
- Return only Python code defining `build_design(out_dir: Path) -> dict`.
- The returned dict must use `schema_version="design_ir.v2"`, `units="mm"`, and include `parts`, `joints`, `ports`, `params`, `materials`, and `provenance`.
- `parts` and `joints` must be lists. `ports` must be a dict keyed by port id, not a list; for example `{'input_port': {'id': 'input_port', ...}}`.
- `parts` entries must be physical bodies only. Do not put `contact_pair`, revolute/prismatic joint, or port records inside `parts`.
- Revolute/prismatic `ports` values must reference explicit joint ids from `joints`, not physical part ids.
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
        'family': 'cam_follower',
        'verifier_level': 3,
    }
}
provenance = {'submission': {'created_by': 'build_design'}}
```

Level-3 Chrono contact requirements:
- Use real `chrono_contact`; fake contact oracle outputs are rejected.
- Include `contact_pair` joints and `params["chrono"]` metadata for contact simulation.
- Contact bodies must provide `params["chrono_collision"]` primitive or trusted collision geometry.
- Required contact pairs: `cam:follower`
  - Pair `cam:follower` means part `cam` contacts part `follower`.

Minimal Level-3 contact evidence pattern:
- Add this kind of metadata to the actual contacting parts named by the required contact pair.
- Do not append contact pairs to `parts`; `contact_pair` is a joint record only.
```python
joints.append({
    'id': 'cam_follower_contact',
    'type': 'contact_pair',
    'parent': 'cam',
    'child': 'follower',
    'axis_world': (0.0, 0.0, 1.0),
    'anchor_world_mm': (0.0, 0.0, 0.0),
})
# On part 'cam' and part 'follower', include a primitive collision shape:
part['params']['chrono_collision'] = {
    'shape': 'cylinder',
    'radius_mm': 20.0,
    'height_mm': 8.0,
    'center_mm': tuple(part.get('com_local_mm', (0.0, 0.0, 0.0))),
    'axis': (0.0, 0.0, 1.0),
}
params['chrono'] = {
    'collision_filter_named_pairs': True,
    'contact_margin_m': 2.0e-5,
    'contact_envelope_m': 2.0e-5,
    'normal_stiffness_N_m': 25000.0,
    'normal_damping_N_s_m': 250.0,
    'friction_mu': 0.05,
}
```

A submission only counts if it preserves topology, exact interfaces, functional behavior, trusted CAD/material/mass evidence, and the hidden variant semantics.
