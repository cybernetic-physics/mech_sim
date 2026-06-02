"""Project Chrono integration plumbing that does not require PyChrono."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def _minimal_ir():
    from mech_bench.schema import DesignIR, Joint, Part, Port

    return DesignIR(
        schema_version="design_ir.v2",
        parts=[
            Part(id="frame", role="ground", fixed=True, mass_kg=0.0),
            Part(id="input", role="input", mass_kg=0.1),
            Part(id="output", role="output", mass_kg=0.2),
        ],
        joints=[
            Joint(id="j_in", type="revolute", parent="frame",
                  child="input", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="j_out", type="revolute", parent="frame",
                  child="output", axis_world=(0, 0, 1),
                  anchor_world_mm=(10, 0, 0)),
        ],
        ports={
            "input_port": Port(id="input_port", part="j_in",
                               kind="revolute_joint"),
            "output_port": Port(id="output_port", part="j_out",
                                kind="revolute_joint"),
        },
    )


def _cycloidal_ir():
    from mech_bench.schema import DesignIR, Joint, Part, Port

    return DesignIR(
        schema_version="design_ir.v2",
        parts=[
            Part(id="housing", role="ground", fixed=True, mass_kg=0.0),
            Part(id="eccentric", role="eccentric", mass_kg=0.05),
            Part(id="disc", role="cycloidal_disc", mass_kg=0.08,
                 params={"pins": 10}),
            Part(id="carrier", role="carrier", mass_kg=0.04),
        ],
        joints=[
            Joint(id="input_revolute", type="revolute", parent="housing",
                  child="eccentric", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="eccentric_disc", type="revolute", parent="eccentric",
                  child="disc", axis_world=(0, 0, 1),
                  anchor_world_mm=(1, 0, 0)),
            Joint(id="output_revolute", type="revolute", parent="housing",
                  child="carrier", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
            Joint(id="ring_contact", type="contact_pair", parent="housing",
                  child="disc", axis_world=(0, 0, 1),
                  anchor_world_mm=(0, 0, 0)),
        ],
        ports={
            "input_port": Port(id="input_port", part="input_revolute",
                               kind="revolute_joint"),
            "output_port": Port(id="output_port", part="output_revolute",
                                kind="revolute_joint"),
        },
        params={"pins": 10, "declared_ratio": 9.0},
    )


def _cycloidal_cfg(contact_model: str) -> dict:
    return {
        "samples": 120,
        "duration_s": 0.6,
        "contact_model": contact_model,
        "friction": 0.35,
        "restitution": 0.02,
        "young_modulus": 2.1e9,
        "normal_stiffness": 5.0e6,
        "damping": 800.0,
        "contact_margin": 0.0005,
        "contact_envelope": 0.001,
        "timestep": 1.0e-3,
        "solver_iterations": 80,
        "_mech_bench": {
            "task": {
                "id": "cycloidal_lowN_stub_s0001",
                "family": "cycloidal_lowN_stub",
                "difficulty": 3,
                "units": "mm",
            },
            "probe_specs": [
                {
                    "id": "torque",
                    "type": "torque_load_trial",
                    "config": {
                        "input_port": "input_port",
                        "output_port": "output_port",
                        "input_speed_rad_s": 10.0,
                        "output_load_Nm": 0.05,
                        "min_output_speed_rad_s": 0.001,
                        "max_power_error_pct": 25.0,
                        "max_torque_ripple_pct": 30.0,
                    },
                },
            ],
        },
    }


def test_evaluator_runtime_context_is_serializable_and_adapter_scoped(tmp_path):
    from mech_bench.evaluator import (
        ExecutionPlan,
        ProbePlan,
        _adapter_runtime_context,
    )
    from mech_bench.probes import Capability
    from mech_bench.schema import EvalConfig, ProbeSpec, TaskSpec

    task = TaskSpec(id="t1", family="contact", difficulty=1, units="mm",
                    prompt="")
    cfg = EvalConfig(probes=[
        ProbeSpec(
            id="contact",
            type="contact_engagement",
            config={"required_pairs": ["input:output"]},
        ),
        ProbeSpec(id="ports", type="required_ports", config={}),
    ])
    plan = ExecutionPlan(probes=[
        ProbePlan(
            probe_id="contact",
            probe_type="contact_engagement",
            capabilities=frozenset({Capability.CONTACT_FORCES}),
            adapter_type="chrono_contact",
        ),
        ProbePlan(
            probe_id="ports",
            probe_type="required_ports",
            capabilities=frozenset({Capability.NONE}),
            adapter_type=None,
        ),
    ])

    ctx = _adapter_runtime_context(
        task=task,
        cfg=cfg,
        plan=plan,
        adapter_name="chrono_contact",
        build_root=tmp_path,
    )

    assert ctx["task"]["id"] == "t1"
    assert ctx["build_root"] == str(tmp_path.resolve())
    assert [p["id"] for p in ctx["probe_specs"]] == ["contact"]
    assert ctx["probe_specs"][0]["config"]["required_pairs"] == ["input:output"]


def test_chrono_runtime_spec_extracts_contacts_drive_and_load(tmp_path):
    from mech_bench.adapters import _chrono_impl

    ir = _minimal_ir()
    cfg = {
        "_mech_bench": {
            "build_root": str(tmp_path),
            "probe_specs": [
                {
                    "id": "contact",
                    "type": "contact_engagement",
                    "config": {"required_pairs": ["output:input"]},
                },
                {
                    "id": "torque",
                    "type": "torque_load_trial",
                    "config": {
                        "input_port": "input_port",
                        "output_port": "output_port",
                        "input_speed_rad_s": 12.5,
                        "output_load_Nm": 0.75,
                        "output_load_start_s": 0.05,
                        "output_load_ramp_s": 0.15,
                    },
                },
            ],
        },
        "contact_pairs": ["input:output"],
    }

    spec = _chrono_impl._runtime_spec(ir, cfg)

    assert spec.contact_pairs == ["input:output"]
    assert spec.build_root == tmp_path.resolve()
    assert spec.motors == [{
        "id": "drive_torque",
        "joint_id": "j_in",
        "port_id": "input_port",
        "mode": "speed",
        "value": 12.5,
        "ramp_s": 0.0,
    }]
    assert spec.loads == [{
        "id": "load_torque",
        "joint_id": "j_out",
        "port_id": "output_port",
        "mode": "torque",
        "value": 0.75,
        "ramp_s": 0.15,
        "start_s": 0.05,
        "brake_smoothing_rad_s": 0.05,
    }]
    resolved_loads, issues = _chrono_impl._resolve_loads(
        spec.loads, ir, {"output": object()})
    assert issues == []
    assert resolved_loads[0]["start_s"] == 0.05
    assert resolved_loads[0]["ramp_s"] == 0.15


def test_chrono_runtime_spec_aliases_port_contact_pairs_to_joint_bodies(tmp_path):
    from mech_bench.adapters import _chrono_impl

    ir = _minimal_ir()
    cfg = {
        "_mech_bench": {
            "build_root": str(tmp_path),
            "probe_specs": [{
                "id": "contact",
                "type": "contact_engagement",
                "config": {"required_pairs": ["input_port:output_port"]},
            }],
        },
    }

    spec = _chrono_impl._runtime_spec(ir, cfg)

    assert "input_port:output_port" in spec.contact_pairs
    assert "input:output" in spec.contact_pairs
    assert spec.contact_pair_aliases == {
        "input_port:output_port": ["input:output"],
    }


def test_chrono_recorder_fills_required_port_pair_from_body_alias(monkeypatch):
    from mech_bench.adapters import _chrono_impl

    ir = _minimal_ir()
    recorder = _chrono_impl._Recorder(
        ir,
        ["input_port:output_port"],
        2,
        {"input_port:output_port": ["input:output"]},
    )
    spec = _chrono_impl.RuntimeSpec(
        contact_pairs=["input_port:output_port"],
        contact_pair_aliases={"input_port:output_port": ["input:output"]},
        motors=[],
        loads=[],
        probe_specs=[],
        build_root=None,
    )

    monkeypatch.setattr(
        _chrono_impl,
        "_report_contacts",
        lambda chrono, system, bodies: {"input:output": (2.5, 0.125)},
    )
    monkeypatch.setattr(_chrono_impl, "_safe_num_contacts", lambda system: 1)

    recorder.sample(
        chrono=object(),
        system=object(),
        i=0,
        t=0.0,
        bodies={},
        links={},
        motors={},
        loads=[],
        spec=spec,
    )

    assert recorder.contact_forces["input_port:output_port"][0] == 2.5
    assert recorder.penetration["input_port:output_port"][0] == 0.125


def test_chrono_load_modulation_function_ramps_after_start():
    from mech_bench.adapters import _chrono_impl

    class FakeInterp:
        def __init__(self):
            self.points = []

        def AddPoint(self, t, value):  # noqa: N802 - Chrono API shape
            self.points.append((t, value))

    class FakeChrono:
        ChFunctionInterp = FakeInterp

    fun = _chrono_impl._load_modulation_function(FakeChrono, 0.05, 0.15)

    assert fun.points == [(0.0, 0.0), (0.05, 0.0), (0.2, 1.0)]


def test_chrono_passive_brake_load_uses_velocity_opposing_torque():
    from mech_bench.adapters import _chrono_impl

    class FakeVec:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakeBody:
        def __init__(self, omega_z):
            self.omega_z = omega_z

        def GetAngVelParent(self):  # noqa: N802 - Chrono API shape
            return FakeVec(0.0, 0.0, self.omega_z)

    load = {
        "mode": "brake_torque",
        "body": FakeBody(2.0),
        "axis": (0.0, 0.0, 1.0),
        "value": -0.75,
        "start_s": 0.1,
        "ramp_s": 0.2,
        "brake_smoothing_rad_s": 0.05,
    }

    assert _chrono_impl._load_torque_value(load, 0.05) == 0.0
    assert _chrono_impl._load_torque_value(load, 0.2) == pytest.approx(-0.375)
    load["body"] = FakeBody(-2.0)
    assert _chrono_impl._load_torque_value(load, 0.3) == pytest.approx(0.75)


def test_chrono_mesh_shape_uses_collision_frame_center(tmp_path):
    from mech_bench.adapters import _chrono_impl

    class FakeVec:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakeQuat:
        def __init__(self, e0, e1, e2, e3):
            self.e0 = e0
            self.e1 = e1
            self.e2 = e2
            self.e3 = e3

    class FakeFrame:
        def __init__(self, pos, quat):
            self.pos = pos
            self.quat = quat

    class FakeMesh:
        def LoadSTLMesh(self, path):  # noqa: N802 - Chrono API shape
            self.path = path
            return True

    class FakeCollisionShape:
        def __init__(self, *args):
            self.args = args

    class FakeChrono:
        ChVector3d = FakeVec
        ChQuaterniond = FakeQuat
        ChFramed = FakeFrame
        ChTriangleMeshConnected = FakeMesh
        ChCollisionShapeTriangleMesh = FakeCollisionShape

    class FakeBody:
        def __init__(self):
            self.calls = []

        def AddCollisionShape(self, shape, frame):  # noqa: N802
            self.calls.append((shape, frame))

    mesh = tmp_path / "disk.stl"
    mesh.write_text("solid disk\nendsolid disk\n", encoding="utf-8")
    body = FakeBody()

    ok, msg = _chrono_impl._add_mesh_shape(
        FakeChrono,
        body,
        {"shape": "mesh", "mesh": str(mesh), "center_mm": (1.0, 2.0, 3.0)},
        object(),
        tmp_path,
    )

    assert ok is True
    assert msg == ""
    assert len(body.calls) == 1
    frame = body.calls[0][1]
    assert frame.pos.x == 0.001
    assert frame.pos.y == 0.002
    assert frame.pos.z == 0.003


def test_freecad_cycloidal_cad_eccentric_body_frames(tmp_path):
    import struct

    from mech_bench.geometry.cycloidal_freecad import (
        BODY_NAMES,
        CycloidalReducerAssets,
        _offset_collision_shape,
        build_chrono_design_ir_from_assets,
    )

    def write_binary_stl(path, triangles):
        data = bytearray(b"test".ljust(80, b" "))
        data.extend(struct.pack("<I", len(triangles)))
        for tri in triangles:
            data.extend(struct.pack("<3f", 0.0, 0.0, 1.0))
            for point in tri:
                data.extend(struct.pack(
                    "<3f",
                    point[0] / 1000.0,
                    point[1] / 1000.0,
                    point[2] / 1000.0,
                ))
            data.extend(struct.pack("<H", 0))
        path.write_bytes(data)

    bodies = {
        name: {
            "stl": f"{name}.stl",
            "step": f"{name}.step",
            "bbox_mm": (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        }
        for name in BODY_NAMES
    }
    bodies["cycloidalDisk1"]["bbox_mm"] = (
        -36.0, -36.0, 10.0, 36.0, 36.0, 15.0)
    write_binary_stl(
        tmp_path / "cycloidalDisk1.stl",
        [
            ((35.0, 0.0, 10.0), (36.0, 0.0, 15.0), (35.0, 1.0, 10.0)),
            ((25.0, 0.0, 10.0), (26.0, 0.0, 15.0), (25.0, 1.0, 10.0)),
        ],
    )
    assets = CycloidalReducerAssets(
        root=tmp_path,
        manifest_path=tmp_path / "cycloidal_assets_manifest.json",
        bodies=bodies,
        collision_meshes={},
        collision_primitives={},
        feature_frames={
            "axes": {
                "cycloidalDisk1_eccentric_axis": {
                    "body": "cycloidalDisk1",
                    "center_mm": [2.0, 0.0, 12.5],
                },
            },
        },
        static_audit={},
        parameters={
            "ring_pin_count": 10,
            "declared_ratio": 9.0,
            "driver_circle_diameter": 50.0,
            "driver_hole_diameter": 10.0,
            "clearance": 0.6,
        },
        source={},
    )

    ir = build_chrono_design_ir_from_assets(
        assets,
        include_secondary_disc=False,
        collidable_body_names={"cycloidalDisk1"},
        use_primitive_pin_collision=False,
        use_cad_collision_primitives=False,
        use_cad_eccentric_body_frames=True,
    )

    parts = {part.id: part for part in ir.parts}
    disk = parts["cycloidalDisk1"]
    assert disk.params["initial_pose_mm"] == (2.0, 0.0, 12.5)
    assert disk.params["chrono_collision"]["shape"] == "mesh"
    assert disk.params["chrono_collision"]["center_mm"] == (-2.0, -0.0, -12.5)
    eccentric = {joint.id: joint for joint in ir.joints}["eccentric_disc"]
    assert eccentric.anchor_world_mm == (2.0, 0.0, 12.5)

    original = {
        "shape": "compound",
        "children": [
            {"shape": "cylinder", "center_mm": (5.0, 1.0, 0.0)},
            {
                "shape": "convex_hull",
                "points_mm": [
                    (2.0, 0.0, 12.5),
                    (3.0, 0.0, 12.5),
                    (2.0, 1.0, 12.5),
                    (2.0, 0.0, 13.5),
                ],
            },
        ],
    }
    shifted = _offset_collision_shape(original, (-2.0, 0.0, -12.5))
    assert original["children"][0]["center_mm"] == (5.0, 1.0, 0.0)
    assert shifted["children"][0]["center_mm"] == (3.0, 1.0, -12.5)
    assert shifted["children"][1]["points_mm"][0] == (0.0, 0.0, 0.0)

    ir = build_chrono_design_ir_from_assets(
        assets,
        include_secondary_disc=False,
        collidable_body_names={"cycloidalDisk1"},
        use_primitive_pin_collision=False,
        use_cad_collision_primitives=False,
        use_cad_eccentric_body_frames=True,
        use_cad_outer_sidewall_collision=True,
        cad_outer_sidewall_thickness_mm=0.75,
        cad_outer_sidewall_max_hulls=16,
    )
    disk_shape = {part.id: part for part in ir.parts}[
        "cycloidalDisk1"
    ].params["chrono_collision"]
    assert disk_shape["shape"] == "compound"
    assert disk_shape["children"][0]["shape"] == "mesh"
    sidewall_children = [
        child for child in disk_shape["children"]
        if child.get("shape") == "convex_hull"
    ]
    assert len(sidewall_children) == 1
    assert sidewall_children[0]["points_mm"][0] == pytest.approx(
        (33.0, 0.0, -2.5))


def test_chrono_impl_direct_run_reports_missing_pychrono():
    if importlib.util.find_spec("pychrono") is not None:
        pytest.skip("host has PyChrono; direct missing-dependency path not active")

    from mech_bench.adapters import _chrono_impl

    out = _chrono_impl.run(_minimal_ir(), {})

    assert out["__capability_unavailable__"] is True
    assert out["metadata"]["simulator"] == "project_chrono"
    assert "pychrono not importable" in out["metadata"]["preflight_issues"][0]


def test_chrono_mesh_convex_decomposition_hulls_are_cached(tmp_path):
    if importlib.util.find_spec("pychrono") is None:
        pytest.skip("requires PyChrono")

    import pychrono.core as chrono

    from mech_bench.adapters import _chrono_impl

    mesh = tmp_path / "tetra.obj"
    mesh.write_text(
        """v 0 0 0
v 0.01 0 0
v 0 0.01 0
v 0 0 0.01
f 1 2 3
f 1 4 2
f 2 4 3
f 1 3 4
""",
        encoding="utf-8",
    )
    shape = {
        "convex_decomposition_max_hulls": 8,
        "convex_decomposition_max_hull_vertices": 16,
    }

    hulls, msg = _chrono_impl._convex_decomposition_hulls(
        chrono, mesh, shape)
    assert msg == ""
    assert hulls
    assert all(len(hull) >= 4 for hull in hulls)

    cached, msg = _chrono_impl._convex_decomposition_hulls(
        chrono, mesh, shape)
    assert msg == ""
    assert cached is hulls


def test_chrono_cycloidal_nsc_vs_smc_thresholds():
    if importlib.util.find_spec("pychrono") is None:
        pytest.skip("requires PyChrono")

    from mech_bench.adapters import _chrono_impl

    ir = _cycloidal_ir()
    nsc = _chrono_impl.run(ir, _cycloidal_cfg("nsc"))
    smc = _chrono_impl.run(ir, _cycloidal_cfg("smc"))

    nsc_m = nsc["scalar_metrics"]
    smc_m = smc["scalar_metrics"]
    assert nsc_m["lockup_detected"] == 1.0
    assert abs(nsc_m["out_omega_med"]) < 1e-6
    assert nsc_m["ratio_observed"] == float("inf")
    assert nsc["passed"] is False

    assert smc["metadata"]["contact_model"] == "smc"
    assert smc["metadata"]["config"]["friction"] == 0.35
    assert smc["metadata"]["config"]["solver_iterations"] == 80.0
    assert smc_m["lockup_detected"] == 0.0
    assert abs(smc_m["out_omega_med"]) > 0.5
    assert np.isfinite(smc_m["ratio_observed"])
    assert smc_m["max_penetration_mm"] < 1.0
    assert smc_m["n_contacts_max"] < nsc_m["n_contacts_max"]
    assert smc_m["power_balance_error_pct"] <= 25.0
    assert smc_m["torque_ripple_pct"] <= 30.0
    assert smc["passed"] is True
    for key in (
        "lockup_detected",
        "ratio_observed",
        "in_omega_med",
        "out_omega_med",
        "max_penetration_mm",
        "max_constraint_error_mm",
        "n_contacts_max",
        "top_contact_pairs",
        "contact_force_rms_N",
        "power_balance_error_pct",
        "torque_ripple_pct",
    ):
        assert key in smc_m


def test_freecad_cycloidal_assets_run_chrono_without_fallback(tmp_path):
    if importlib.util.find_spec("pychrono") is None:
        pytest.skip("requires PyChrono")

    from mech_bench.adapters import _chrono_impl
    from mech_bench.geometry.cycloidal_freecad import (
        CycloidalCadExportError,
        audit_cycloidal_static_geometry,
        build_chrono_design_ir_from_assets,
        find_cycloid_gearbox_path,
        find_freecad_command,
        generate_cycloidal_reducer_assets,
    )

    if find_freecad_command() is None:
        pytest.skip("requires headless FreeCAD")
    if find_cycloid_gearbox_path() is None:
        pytest.skip("requires CycloidGearBox source checkout")

    try:
        assets = generate_cycloidal_reducer_assets(
            tmp_path / "cad",
            {
                "pins": 10,
                "line_segment_count": 42,
                "clearance": 0.6,
                "driver_pin_collision_shrink_mm": 0.68,
            },
            timeout_s=300.0,
        )
    except CycloidalCadExportError as exc:
        pytest.fail(f"missing bridge {exc.stage}: {exc}")

    assert set(assets.bodies) == {
        "pinDisk",
        "driverDisk",
        "inputShaft",
        "cycloidalDisk1",
        "cycloidalDisk2",
        "eccentricKey",
        "outputShaft",
    }
    for body in assets.bodies.values():
        assert (assets.root / body["step"]).is_file()
        assert (assets.root / body["stl"]).is_file()
    for mesh_name in ("ringPins", "driverPins"):
        collision_mesh = assets.collision_meshes[mesh_name]
        assert (assets.root / collision_mesh["step"]).is_file()
        assert (assets.root / collision_mesh["stl"]).is_file()

    audit = audit_cycloidal_static_geometry(assets)
    assert audit["feature_frame_counts"]["ring_pins"] == 10
    assert audit["feature_frame_counts"]["driver_pins"] == 6
    assert audit["feature_frame_counts"]["cycloidalDisk1_output_holes"] == 6
    assert audit["feature_frame_counts"]["cycloidalDisk2_output_holes"] == 6
    axes = assets.feature_frames["axes"]
    assert axes["cycloidalDisk1_eccentric_axis"]["body"] == "cycloidalDisk1"
    assert axes["cycloidalDisk2_eccentric_axis"]["body"] == "cycloidalDisk2"
    assert axes["cycloidalDisk1_eccentric_axis"]["center_mm"][2] == 12.5
    assert axes["cycloidalDisk2_eccentric_axis"]["center_mm"][2] == 17.5
    assert axes["cycloidalDisk1_eccentric_axis"]["center_mm"] != axes[
        "cycloidalDisk2_eccentric_axis"]["center_mm"]
    for disk_name in ("cycloidalDisk1", "cycloidalDisk2"):
        pin_hole = audit[f"driver_pins_to_{disk_name}_output_holes"]
        assert pin_hole["status"] == "ok"
        assert pin_hole["min_radial_clearance_mm"] > 0.0
        assert audit[f"ring_pins_to_{disk_name}_distance_mm"] >= 0.0

    ir = build_chrono_design_ir_from_assets(
        assets,
        include_secondary_disc=False,
        collision_sweep_radius_m=2.0e-5,
        use_cad_collision_primitives=False,
        use_cad_eccentric_body_frames=True,
        use_cad_outer_sidewall_collision=True,
        cad_outer_sidewall_thickness_mm=0.75,
        cad_outer_sidewall_max_hulls=128,
    )
    cfg_base = _cycloidal_cfg("nsc")
    cfg_base["_mech_bench"]["probe_specs"][0]["config"][
        "min_output_speed_rad_s"
    ] = 0.5
    cfg_base["_mech_bench"]["probe_specs"][0]["config"][
        "input_speed_rad_s"
    ] = 10.0
    cfg_base["_mech_bench"]["probe_specs"][0]["config"][
        "output_load_Nm"
    ] = 0.75
    cfg_base.update(
        samples=61,
        duration_s=0.2,
        timestep=5.0e-5,
        procedural_cycloidal_fallback=False,
        contact_margin=2.0e-5,
        contact_envelope=5.0e-5,
        friction=0.02,
        restitution=0.0,
        young_modulus=3.0e7,
        normal_stiffness=5.0e7,
        damping=2500.0,
        solver_iterations=800,
    )
    cfg_base["_mech_bench"]["build_root"] = str(assets.root)

    collision_shapes = {
        part.id: (part.params or {}).get("chrono_collision", {})
        for part in ir.parts
    }
    parts_by_id = {part.id: part for part in ir.parts}
    for body_name in ("pinDisk", "driverDisk", "cycloidalDisk1"):
        shape = collision_shapes[body_name]
        assert shape["shape"] == "compound"
        assert shape["children"]
    joints_by_id = {joint.id: joint for joint in ir.joints}
    assert "disc_stack_fixed" not in joints_by_id
    assert "eccentric_disc_2" not in joints_by_id
    assert "ring_contact_2" not in joints_by_id
    assert "output_pin_contact_2" not in joints_by_id
    for body_name in ("pinDisk", "driverDisk", "cycloidalDisk1"):
        if body_name in {"pinDisk", "driverDisk"}:
            assert (assets.root / parts_by_id[body_name].params[
                "chrono_collision_asset"
            ]["mesh"]).is_file()
    assert ir.params["cad_static_audit"] == assets.static_audit
    assert "ring_pins" in ir.params["cad_feature_frames"]

    unloaded_cfg = dict(cfg_base)
    unloaded_cfg["contact_model"] = "smc"
    unloaded_cfg["_mech_bench"] = dict(cfg_base["_mech_bench"])
    unloaded_cfg["_mech_bench"]["probe_specs"] = [
        dict(cfg_base["_mech_bench"]["probe_specs"][0])
    ]
    unloaded_cfg["_mech_bench"]["probe_specs"][0]["config"] = dict(
        cfg_base["_mech_bench"]["probe_specs"][0]["config"])
    unloaded_cfg["_mech_bench"]["probe_specs"][0]["config"][
        "output_load_Nm"
    ] = 0.0
    unloaded_cfg["_mech_bench"]["probe_specs"][0]["config"][
        "max_power_error_pct"
    ] = 1.0e12
    unloaded_cfg["_mech_bench"]["probe_specs"][0]["config"][
        "max_torque_ripple_pct"
    ] = 1.0e12
    unloaded = _chrono_impl.run(ir, unloaded_cfg)
    unloaded_metrics = unloaded["scalar_metrics"]
    assert unloaded["metadata"]["config"][
        "use_visual_geometry_as_collision"] is False
    assert unloaded_metrics["unmonitored_contact_pair_count"] == 0.0

    legacy_visual_cfg = dict(unloaded_cfg)
    legacy_visual_cfg["samples"] = 21
    legacy_visual_cfg["duration_s"] = 0.05
    legacy_visual_cfg["use_visual_geometry_as_collision"] = True
    legacy_visual = _chrono_impl.run(ir, legacy_visual_cfg)
    legacy_visual_metrics = legacy_visual["scalar_metrics"]
    assert legacy_visual_metrics["unmonitored_contact_pair_count"] > 0.0
    assert legacy_visual_metrics["unmonitored_top_contact_pairs"]

    assert unloaded_metrics["lockup_detected"] == 0.0
    assert unloaded_metrics["ratio_error_pct"] < 15.0
    assert unloaded_metrics["max_penetration_mm"] < 1.0
    assert unloaded_metrics["out_omega_med"] == unloaded_metrics[
        "out_omega_med_raw"
    ]
    assert "out_omega_med_raw" in unloaded_metrics
    assert "out_omega_fit_rad_s" in unloaded_metrics

    results = {}
    for contact_model in ("nsc", "smc"):
        cfg = dict(cfg_base)
        cfg["contact_model"] = contact_model
        cfg["_mech_bench"] = dict(cfg_base["_mech_bench"])
        out = _chrono_impl.run(ir, cfg)
        if out.get("__capability_unavailable__"):
            issues = out["metadata"].get("preflight_issues", [])
            pytest.fail(
                f"missing bridge collision mesh import/contact setup: {issues}")
        assert out["metadata"].get("execution_mode") != (
            "procedural_cycloidal_contact_fallback"
        )
        assert out["metadata"]["contact_model"] == contact_model
        assert "smc_use_material_properties" in out["metadata"]["config"]
        assert "cad_reference_frames" in out["metadata"]["config"]
        assert out["metadata"]["build_meta"]["n_bodies"] == len(ir.parts)
        assert out["scalar_metrics"]["n_contacts_max"] > 0.0
        assert out["scalar_metrics"]["contact_force_rms_N"] > 0.0
        assert "kinetic_J" in out["energies"]
        assert set(ir.params["cad_source"]) >= {"generator", "commit", "kernel"}
        assert "cycloidalDisk1" in out["body_poses"]
        for key in (
            "lockup_detected",
            "ratio_observed",
            "in_omega_med",
            "out_omega_med",
            "max_penetration_mm",
            "max_constraint_error_mm",
            "n_contacts_max",
            "top_contact_pairs",
            "contact_force_rms_N",
            "output_torque_Nm_mean",
            "output_torque_Nm_signed_mean",
            "power_balance_error_pct",
            "power_balance_residual_pct",
            "mechanical_efficiency_pct",
            "unaccounted_power_W_mean",
            "kinetic_energy_rate_W_mean",
            "torque_ripple_pct",
            "out_omega_med_raw",
            "out_omega_fit_rad_s",
        ):
            assert key in out["scalar_metrics"]
        results[contact_model] = out

    smc_metrics = results["smc"]["scalar_metrics"]
    nsc_metrics = results["nsc"]["scalar_metrics"]

    assert nsc_metrics["n_contacts_max"] > smc_metrics["n_contacts_max"]
    assert nsc_metrics["contact_force_rms_N"] > smc_metrics["contact_force_rms_N"]

    assert smc_metrics["lockup_detected"] == 0.0
    assert abs(smc_metrics["out_omega_med"]) > 0.5
    assert np.isfinite(smc_metrics["ratio_observed"])
    assert smc_metrics["max_penetration_mm"] < 1.0
    assert smc_metrics["failure_mode"] in {"power_balance_error", "torque_ripple"}
    assert smc_metrics["failure_mode"] != "lockup_mechanism_jammed"
    assert smc_metrics["contact_force_rms_N"] < nsc_metrics["contact_force_rms_N"]
    assert smc_metrics["power_balance_error_pct"] > 25.0

    for metrics in (nsc_metrics, smc_metrics):
        top_pairs = {p["pair"]: p for p in metrics["top_contact_pairs"]}
        assert "cycloidalDisk1:driverDisk" in top_pairs
        assert "cycloidalDisk1:pinDisk" in top_pairs
        for pair in top_pairs.values():
            assert "max_penetration_mm" in pair
            assert "rms_penetration_mm" in pair
            assert pair["active_sample_count"] > 0.0
