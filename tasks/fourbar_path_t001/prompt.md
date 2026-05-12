# Task: four-bar coupler path tracing

You are designing a planar four-bar linkage. The mechanism has four
revolute joints, all axes parallel to the world Z axis. One link is
grounded; one of the ground-pivot joints is the driven input.

**Objective.** Produce a coupler-point trajectory that, when
normalized to its centroid and RMS radius, matches the target path
in `fixtures/target_path.csv` within a symmetric Chamfer distance of
**0.05** (units: dimensionless after RMS normalization).

**Required ports** (must be declared in DesignIR):

- `input_port` — the driven crank-side ground revolute joint.
- `output_port` — the rocker-side ground revolute joint.
- `coupler_point` — a frame on the coupler link whose trajectory is
  scored against the target.

**Required mobility.** Planar Grübler mobility must equal **1**.

**Submission contract.** Provide `design.py` exposing

```python
def build_design(out_dir: Path) -> dict:
    """Return a DesignIR (schema_version='design_ir.v2') describing
    the four-bar."""
```

The submission can author its DesignIR directly from parametric link
lengths and pivot positions; no CAD files are required for this
task.

**What gets scored.**

1. Hard gate — `dof_grubler` must return mobility 1.
2. Dense — `path_trace_chamfer` against the target path. Lower
   Chamfer → higher score; score = 1 − chamfer / max_chamfer,
   clipped to [0, 1].
