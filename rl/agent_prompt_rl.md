# mech_bench RL agent

You solve one mech_bench task. Output ONE Python file as a single
fenced ```python ... ``` block. No prose outside the block.

## Canonical static example (copy this shape exactly for mobility=0 tasks)

Below is a working solution for a simple mounting-plate task with
two grounded frame ports 40 mm apart. Every field shown here is
required — DO NOT drop any.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    PITCH = 40.0
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {
                "id": "plate",
                "role": "ground",
                "mass_kg": 0.05,
                "fixed": True,
                "com_local_mm": (PITCH / 2, 0.0, 0.0),
            },
        ],
        "joints": [],
        "ports": {
            "mount_a": {
                "id": "mount_a",
                "part": "plate",
                "kind": "frame",
                "pose_local_mm": (0.0, 0.0, 0.0),
            },
            "mount_b": {
                "id": "mount_b",
                "part": "plate",
                "kind": "frame",
                "pose_local_mm": (PITCH, 0.0, 0.0),
            },
        },
        "params": {"pitch_mm": PITCH},
    }
```

## Canonical revolute example (copy this shape for mobility=1 tasks)

For a required port whose prompt says `kind = revolute_joint`, the
port's `part` field must be the revolute joint id, not a part id.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    SHAFT_DIAMETER_MM = 11.071
    KEYWAY_WIDTH_MM = 5.467
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {
                "id": "hub",
                "role": "ground",
                "mass_kg": 0.08,
                "fixed": True,
                "com_local_mm": (0.0, 0.0, 0.0),
            },
            {
                "id": "shaft",
                "role": "output",
                "mass_kg": 0.04,
                "fixed": False,
                "com_local_mm": (0.0, 0.0, 0.0),
            },
        ],
        "joints": [
            {
                "id": "shaft_spin",
                "type": "revolute",
                "parent": "hub",
                "child": "shaft",
                "axis_world": (0.0, 0.0, 1.0),
                "anchor_world_mm": (0.0, 0.0, 0.0),
            },
        ],
        "ports": {
            "hub_face": {
                "id": "hub_face",
                "part": "hub",
                "kind": "frame",
                "pose_local_mm": (0.0, 0.0, 0.0),
            },
            "output_port": {
                "id": "output_port",
                "part": "shaft_spin",
                "kind": "revolute_joint",
                "pose_local_mm": (0.0, 0.0, 0.0),
            },
        },
        "params": {
            "shaft_diameter_mm": SHAFT_DIAMETER_MM,
            "keyway_width_mm": KEYWAY_WIDTH_MM,
        },
    }
```

## Canonical two-axis transmission example (mobility=2)

For ratio tasks with one rotating input and one rotating output, use
two moving parts and two revolute joints to ground. The input/output
ports point at the joint ids.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    TEETH_IN = 20
    TEETH_OUT = 60
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "frame", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "input_gear", "role": "gear_input", "mass_kg": 0.02,
             "com_local_mm": (0.0, 0.0, 0.0),
             "params": {"teeth": TEETH_IN}},
            {"id": "output_gear", "role": "gear_output", "mass_kg": 0.05,
             "com_local_mm": (40.0, 0.0, 0.0),
             "params": {"teeth": TEETH_OUT}},
        ],
        "joints": [
            {"id": "input_axis", "type": "revolute", "parent": "frame",
             "child": "input_gear", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "output_axis", "type": "revolute", "parent": "frame",
             "child": "output_gear", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (40.0, 0.0, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "input_axis",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "output_axis",
                            "kind": "revolute_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
        },
        "params": {
            "teeth_in": TEETH_IN,
            "teeth_out": TEETH_OUT,
            "declared_ratio": TEETH_OUT / TEETH_IN,
        },
    }
```

## Canonical revolute-to-prismatic example (mobility=2)

For rack-pinion tasks, the output port is a `prismatic_joint` and
must point at the slider joint id.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    PITCH_RADIUS_MM = 20.0
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "frame", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "pinion", "role": "input", "mass_kg": 0.02,
             "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "rack", "role": "output", "mass_kg": 0.04,
             "com_local_mm": (0.0, -PITCH_RADIUS_MM, 0.0)},
        ],
        "joints": [
            {"id": "input_axis", "type": "revolute", "parent": "frame",
             "child": "pinion", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "output_slide", "type": "prismatic",
             "parent": "frame", "child": "rack",
             "axis_world": (1.0, 0.0, 0.0),
             "anchor_world_mm": (0.0, -PITCH_RADIUS_MM, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "input_axis",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "output_slide",
                            "kind": "prismatic_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
        },
        "params": {
            "pitch_radius_mm": PITCH_RADIUS_MM,
            "declared_linear_per_rev_mm": 2.0 * 3.141592653589793
                                         * PITCH_RADIUS_MM,
        },
    }
```

## Canonical lead-screw example (mobility=2)

For lead-screw tasks, use one revolute input joint and one prismatic
output joint. The verifier checks `params.declared_travel_per_rev_mm`.
Do not use the rack-pinion key `declared_linear_per_rev_mm`.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    LEAD_MM = 6.933
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "frame", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "screw", "role": "input", "mass_kg": 0.02,
             "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "nut", "role": "slider", "mass_kg": 0.08,
             "com_local_mm": (0.0, 0.0, 0.0)},
        ],
        "joints": [
            {"id": "input_axis", "type": "revolute", "parent": "frame",
             "child": "screw", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "output_axis", "type": "prismatic",
             "parent": "frame", "child": "nut",
             "axis_world": (1.0, 0.0, 0.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "input_axis",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "output_axis",
                            "kind": "prismatic_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
        },
        "params": {
            "lead_mm": LEAD_MM,
            "declared_travel_per_rev_mm": LEAD_MM,
        },
    }
```

## Canonical planetary examples (mobility=2)

Planetary analytic tasks are scored from grounded revolute ports and
`params.declared_ratio`, not from a full gear mesh. Use two revolute
joints to the fixed frame.

For fixed-ring ratio tasks, input is the sun axis and output is the
carrier axis:

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    SUN, RING = 16, 36
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "frame", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "sun", "role": "input", "mass_kg": 0.02,
             "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "carrier", "role": "output", "mass_kg": 0.05,
             "com_local_mm": (60.0, 0.0, 0.0)},
        ],
        "joints": [
            {"id": "input_axis", "type": "revolute", "parent": "frame",
             "child": "sun", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "output_axis", "type": "revolute", "parent": "frame",
             "child": "carrier", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (60.0, 0.0, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "input_axis",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "output_axis",
                            "kind": "revolute_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
        },
        "params": {
            "sun_teeth": SUN,
            "ring_teeth": RING,
            "declared_ratio": 1.0 + RING / SUN,
        },
    }
```

For fixed-sun ratio tasks, input is the carrier axis, output is the
ring axis, and `carrier_port` is a frame port on the carrier part.
Set `params["declared_ratio"]` to the exact prompt value.

## Canonical slider-crank example (mobility=1)

Slider-crank tasks need three revolute joints plus one prismatic
slider joint. The input port points at the crank revolute joint and
the output port points at the prismatic slider joint.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    CRANK, COUPLER = 27.09, 69.17
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "ground", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "crank", "role": "crank", "mass_kg": 0.02,
             "com_local_mm": (CRANK / 2, 0.0, 0.0)},
            {"id": "coupler", "role": "coupler", "mass_kg": 0.05,
             "com_local_mm": (COUPLER / 2, 0.0, 0.0)},
            {"id": "slider", "role": "slider", "mass_kg": 0.08,
             "com_local_mm": (0.0, 0.0, 0.0)},
        ],
        "joints": [
            {"id": "joint_input", "type": "revolute",
             "parent": "ground", "child": "crank",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "joint_bc", "type": "revolute",
             "parent": "crank", "child": "coupler",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (CRANK, 0.0, 0.0)},
            {"id": "joint_cs", "type": "revolute",
             "parent": "coupler", "child": "slider",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (CRANK + COUPLER, 0.0, 0.0)},
            {"id": "joint_slide", "type": "prismatic",
             "parent": "ground", "child": "slider",
             "axis_world": (1.0, 0.0, 0.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "joint_input",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "joint_slide",
                            "kind": "prismatic_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
        },
        "params": {
            "crank_mm": CRANK,
            "coupler_mm": COUPLER,
            "declared_stroke_mm": 2.0 * CRANK,
        },
    }
```

For quick-return proxy tasks, keep the same topology and replace or
add the exact key `params["declared_quick_return_ratio"]` with the
prompt value. Do not switch to the four-bar output pattern: the
output joint id remains `joint_slide`, its type remains `prismatic`,
and `ports["output_port"]["part"]` remains `"joint_slide"`. Do not
create `joint_output` or use a ground length variable such as `G` for
slider-crank tasks.

## Canonical cam-follower contact example (mobility=2)

For synthetic cam-follower contact tasks, create `cam` and
`follower` moving parts, two grounded revolute joints, and a
`contact_pair` joint whose parent is `cam` and child is `follower`.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "frame", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "cam", "role": "input", "mass_kg": 0.04,
             "com_local_mm": (0.0, 0.0, 0.0)},
            {"id": "follower", "role": "output", "mass_kg": 0.05,
             "com_local_mm": (40.0, 0.0, 0.0)},
        ],
        "joints": [
            {"id": "input_axis", "type": "revolute", "parent": "frame",
             "child": "cam", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "output_axis", "type": "revolute", "parent": "frame",
             "child": "follower", "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (40.0, 0.0, 0.0)},
            {"id": "cam_follower", "type": "contact_pair",
             "parent": "cam", "child": "follower",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "input_axis",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "output_axis",
                            "kind": "revolute_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
        },
        "params": {"declared_pair": "cam:follower"},
    }
```

## Canonical four-bar example (mobility=1)

For four-bar tasks, use four parts and four revolute joints. Only
the input and output ports are joint ports; a requested coupler point
is a frame port on the coupler part.

```python
from pathlib import Path

def build_design(out_dir: Path) -> dict:
    G, A, B, C = 100.0, 30.0, 70.0, 80.0
    return {
        "schema_version": "design_ir.v2",
        "parts": [
            {"id": "ground", "role": "ground", "mass_kg": 0.0,
             "fixed": True, "com_local_mm": (G / 2, 0.0, 0.0)},
            {"id": "crank", "role": "crank", "mass_kg": 0.02,
             "com_local_mm": (A / 2, 0.0, 0.0)},
            {"id": "coupler", "role": "coupler", "mass_kg": 0.06,
             "com_local_mm": (B / 2, 0.0, 0.0)},
            {"id": "rocker", "role": "rocker", "mass_kg": 0.05,
             "com_local_mm": (C / 2, 0.0, 0.0)},
        ],
        "joints": [
            {"id": "joint_input", "type": "revolute",
             "parent": "ground", "child": "crank",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (0.0, 0.0, 0.0)},
            {"id": "joint_bc", "type": "revolute",
             "parent": "crank", "child": "coupler",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (A, 0.0, 0.0)},
            {"id": "joint_cd", "type": "revolute",
             "parent": "coupler", "child": "rocker",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (A + B, 0.0, 0.0)},
            {"id": "joint_output", "type": "revolute",
             "parent": "ground", "child": "rocker",
             "axis_world": (0.0, 0.0, 1.0),
             "anchor_world_mm": (G, 0.0, 0.0)},
        ],
        "ports": {
            "input_port": {"id": "input_port", "part": "joint_input",
                           "kind": "revolute_joint",
                           "pose_local_mm": (0.0, 0.0, 0.0)},
            "output_port": {"id": "output_port", "part": "joint_output",
                            "kind": "revolute_joint",
                            "pose_local_mm": (0.0, 0.0, 0.0)},
            "coupler_point": {"id": "coupler_point", "part": "coupler",
                              "kind": "frame",
                              "pose_local_mm": (B / 2, 0.0, 0.0)},
        },
        "params": {
            "link_lengths_mm": {
                "ground": G, "crank": A, "coupler": B, "rocker": C,
            },
        },
    }
```

## Schema rules

- `schema_version` is literally `"design_ir.v2"`.
- If the task asks for slider-crank or quick-return behavior, use a
  slider-crank topology only: parts `ground`, `crank`, `coupler`,
  `slider`; joints `joint_input`, `joint_bc`, `joint_cs`,
  `joint_slide`; output port part `joint_slide`; no `rocker`, no
  `joint_output`, and no four-bar variables `A`, `B`, `C`, or `G`.
- **Each port dict MUST include all four keys**: `id` (matches the
  outer dict key), `part` (a PART id when `kind="frame"`, or a
  JOINT id when `kind="revolute_joint"` / `kind="prismatic_joint"`),
  `kind`, and `pose_local_mm` (a 3-tuple of finite floats).
- A `revolute_joint` or `prismatic_joint` port NEVER points at the
  moving part. It points at the joint id from `joints`. If you create
  `{"id": "output_port", "type": "revolute", "parent": "hub",
  "child": "shaft"}`, then `ports["output_port"]["part"]` must be
  `"output_port"`, not `"shaft"`.
- Every joint `parent` and `child` must be an existing part id. If a
  joint connects `"hub"` to `"shaft"`, both parts must be listed in
  `parts`.
- **Each part dict MUST include**: `id` (matches the grammar
  `[A-Za-z_][A-Za-z0-9_.:-]{0,127}`), `mass_kg`, `com_local_mm`.
  `fixed` (bool), `role` (str), and `params` (dict) are optional.
- **A part that anchors a grounded frame port MUST set
  `"fixed": True`.** This is the single most common mistake.
- **Port `kind` has only three legal values**: `"frame"`,
  `"revolute_joint"`, and `"prismatic_joint"`. Do not invent
  feature-specific kinds such as `"face"`, `"standoff"`, `"hole"`,
  `"shaft"`, or `"mount"`.
- **DO NOT create extra parts per port.** One plate with multiple
  frame ports on it is correct; one part per port is wrong (it
  inflates the Grübler mobility count).
- `joints` is a list. Each entry: `id`, `type` ∈ {revolute,
  prismatic, fixed, contact_pair, spherical}, `parent` (part id),
  `child` (part id), `axis_world` (3-tuple), `anchor_world_mm`
  (3-tuple).
- If the task prompt says `params.foo`, the top-level `params` dict
  key must be exactly `"foo"`. Do not rename it to
  `"declared_foo"`, `"foo_mm_declared"`, or any other alias.
- Mobility (Grübler, planar): `M = 3(n-1) - 2(j_revolute +
  j_prismatic) - 3(j_fixed)`. Match
  `requirements.expected_mobility` exactly. For "static" tasks
  with mobility=0, you usually want ONE fixed part and NO joints.
- The task's explicit `requirements.expected_mobility`, prompt
  mobility statement, and required port kinds override the task title
  and tier name. For example, a title containing "static" can still
  require two `revolute_joint` ports and mobility 2; in that case build
  the revolute joints.
- For `keyed_shaft_hub_fit`, the canonical topology is exactly:
  fixed `hub` part, moving `shaft` part, one revolute joint connecting
  `hub` to `shaft`, `hub_face` as a frame port on `hub`, and
  `output_port` as a revolute-joint port whose `part` is the joint id.
- For static fit, clearance, interference, wall, pitch, bolt pattern,
  standoff, bracket, plate, collar, seat, snap-tab, register, and
  press-fit tasks, do not model the feature as a moving mechanism.
  Use one fixed carrier part, no joints, and put all required frame
  ports on that fixed part unless the prompt explicitly asks for a
  revolute or prismatic joint.
- If the prompt asks for ports like `so_1`..`so_4`, create exactly
  those port ids. For static artifact ports, use `kind="frame"`.
- All numeric values must be finite. No NaN, no Inf.
- Imports limited to the stdlib (pathlib, math).
- Do not read files inside the task directory's
  `reference_solution/`, `negative_solutions/`,
  `eval_config*.toml`, or `expected_failures.json`.
- For Tier-3 contact tasks: include a `contact_pair` joint
  between the named parts.

Reply with the fenced block only. Stop after the closing ``` .
