"""SceneGraph builder tests."""

from __future__ import annotations

from mech_bench.schema import DesignIR, EvalConfig, Joint, Part, Port, TaskSpec
from mech_bench.scene_graph import build_scene_graph_from_design_ir


def _ir() -> DesignIR:
    return DesignIR(
        schema_version="design_ir.v2",
        parts=[
            Part(id="frame", role="ground", fixed=True),
            Part(id="pinion"),
            Part(id="gear"),
        ],
        joints=[
            Joint(id="j_in", type="revolute", parent="frame",
                  child="pinion", axis_world=(0, 0, 1)),
            Joint(id="j_out", type="revolute", parent="frame",
                  child="gear", axis_world=(0, 0, 1)),
        ],
        ports={
            "input_port": Port(id="input_port", part="j_in",
                                kind="revolute_joint"),
            "output_port": Port(id="output_port", part="j_out",
                                kind="revolute_joint"),
        },
    )


def _task() -> TaskSpec:
    return TaskSpec(
        id="t1", family="gear_pair", difficulty=2, units="mm",
        prompt="", required_ports=["input_port", "output_port"],
    )


def _cfg_with_contact() -> EvalConfig:
    return EvalConfig.from_dict({
        "probes": [
            {"id": "engagement", "type": "contact_engagement",
             "required_pairs": ["pinion:gear"],
             "min_rms_force_N": 0.5},
            {"id": "load", "type": "torque_load_trial",
             "input_port": "input_port", "output_port": "output_port",
             "input_speed_rad_s": 2.0, "output_load_Nm": 0.5},
        ],
    })


def test_scene_graph_maps_parts_and_joints():
    res = build_scene_graph_from_design_ir(_ir(), _task(),
                                              _cfg_with_contact())
    scene = res.scene
    assert scene.body_ids() == {"frame", "pinion", "gear"}
    assert scene.joint_ids() == {"j_in", "j_out"}
    assert {p.id for p in scene.ports.values()} == {
        "input_port", "output_port"}


def test_scene_graph_extracts_contact_pair_from_eval():
    res = build_scene_graph_from_design_ir(_ir(), _task(),
                                              _cfg_with_contact())
    pairs = [(c.body_a, c.body_b) for c in res.scene.contact_pairs]
    assert ("pinion", "gear") in pairs or ("gear", "pinion") in pairs


def test_scene_graph_extracts_motor_and_load():
    res = build_scene_graph_from_design_ir(_ir(), _task(),
                                              _cfg_with_contact())
    assert any(m.mode == "speed" for m in res.scene.motors)
    assert any(l.mode == "torque" for l in res.scene.loads)


def test_scene_graph_flags_missing_port():
    task = TaskSpec(id="t1", family="g", difficulty=1, units="mm",
                     prompt="",
                     required_ports=["input_port", "phantom_port"])
    res = build_scene_graph_from_design_ir(_ir(), task,
                                              EvalConfig(probes=[]))
    codes = [f.code.value for f in res.preflight_failures]
    assert "missing_port" in codes
    assert not res.ok


def test_scene_graph_flags_contact_pair_missing_body():
    cfg = EvalConfig.from_dict({
        "probes": [
            {"id": "engagement", "type": "contact_engagement",
             "required_pairs": ["pinion:phantom_body"],
             "min_rms_force_N": 0.5},
        ],
    })
    res = build_scene_graph_from_design_ir(_ir(), _task(), cfg)
    codes = [f.code.value for f in res.preflight_failures]
    assert "wrong_topology" in codes
    assert not res.ok
