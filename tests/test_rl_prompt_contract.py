from __future__ import annotations

from rl.sample_and_score import _contract_from_task
from rl.train_grpo import _contract_from_task as _train_contract_from_task


def test_contract_includes_port_kinds_and_grounding_from_eval_config() -> None:
    task_toml = """
[task]
id = "lead_screw_linear_travel_s0001"
family = "lead_screw_linear_travel"

[requirements]
required_ports = ["input_port", "output_port"]
expected_mobility = 2
"""
    prompt_md = """
# Lead screw linear travel

Declare `params.declared_travel_per_rev_mm` = lead_mm = 6.933.
"""
    eval_config_toml = """
[[probes]]
id = "ports"
type = "required_ports"
ports = ["input_port", "output_port"]
require_grounded = ["input_port"]

[probes.require_kinds]
input_port = "revolute_joint"
output_port = "prismatic_joint"

[[probes]]
id = "travel"
type = "analytic_param_check"
path = "params.declared_travel_per_rev_mm"
expected = 6.933
comparator = "eq"
"""

    contract = _contract_from_task(prompt_md, task_toml, eval_config_toml)

    assert "- expected_mobility: 2" in contract
    assert "`input_port` must be `revolute_joint`" in contract
    assert "`output_port` must be `prismatic_joint`" in contract
    assert "- grounded_ports: `input_port`" in contract
    assert "`params.declared_travel_per_rev_mm`" in contract
    assert "do not use `params.declared_linear_per_rev_mm`" in contract
    assert "`params.declared_travel_per_rev_mm == 6.933" in contract


def test_contract_warns_quick_return_slider_not_fourbar() -> None:
    task_toml = """
[task]
id = "slider_crank_quick_return_proxy_s0001"
family = "slider_crank_quick_return_proxy"

[requirements]
required_ports = ["input_port", "output_port"]
expected_mobility = 1
"""
    prompt_md = """
# Slider-crank quick-return proxy

Declare `params.declared_quick_return_ratio` = 1.6884.
"""
    eval_config_toml = """
[[probes]]
id = "ports"
type = "required_ports"
ports = ["input_port", "output_port"]
require_grounded = ["input_port"]

[probes.require_kinds]
input_port = "revolute_joint"
output_port = "prismatic_joint"

[[probes]]
id = "ratio"
type = "analytic_param_check"
path = "params.declared_quick_return_ratio"
expected = 1.6884
comparator = "eq"
"""

    contract = _contract_from_task(prompt_md, task_toml, eval_config_toml)

    assert "slider_crank_topology_warning" in contract
    assert "Use parts `ground`, `crank`, `coupler`, `slider`" in contract
    assert "`ports[\"output_port\"][\"part\"]` must be `joint_slide`" in contract
    assert "do not create `rocker`, four-bar `joint_output`" in contract
    assert "do not use undefined four-bar variables `A`, `B`, `C`, or `G`" in contract


def test_contract_warns_stroke_slider_not_fourbar() -> None:
    task_toml = """
[task]
id = "slider_crank_stroke_precision_s0001"
family = "slider_crank_stroke_precision"

[requirements]
required_ports = ["input_port", "output_port"]
expected_mobility = 1
"""
    prompt_md = """
# Slider-crank stroke precision

Declare `params.declared_stroke_mm` = 54.18.
"""
    eval_config_toml = """
[[probes]]
id = "ports"
type = "required_ports"
ports = ["input_port", "output_port"]
require_grounded = ["input_port"]

[probes.require_kinds]
input_port = "revolute_joint"
output_port = "prismatic_joint"

[[probes]]
id = "stroke"
type = "analytic_param_check"
path = "params.declared_stroke_mm"
expected = 54.18
comparator = "eq"
"""

    contract = _contract_from_task(prompt_md, task_toml, eval_config_toml)

    assert "slider_crank_topology_warning" in contract
    assert "`joint_slide` must have `type == \"prismatic\"`" in contract
    assert "do not use undefined four-bar variables `A`, `B`, `C`, or `G`" in contract

    train_contract = _train_contract_from_task(
        prompt_md, task_toml, eval_config_toml
    )
    assert "slider_crank_topology_warning" in train_contract
    assert "`joint_slide` must have `type == \"prismatic\"`" in train_contract
    assert "do not use undefined four-bar variables `A`, `B`, `C`, or `G`" in train_contract


def test_contract_includes_paper_verifier_requirements() -> None:
    task_toml = """
[task]
id = "cam_follower_contact_stub_s0001_paper_verifier"
family = "cam_follower_contact_stub"

[requirements]
required_ports = ["input_port", "output_port"]
expected_mobility = 2
"""
    prompt_md = """
# Cam follower
"""
    eval_config_toml = """
[adapters.chrono_contact]
contact_model = "smc"
procedural_cycloidal_fallback = false

[[probes]]
id = "contact"
type = "contact_engagement"
adapter = "chrono_contact"
required_pairs = ["cam:follower"]

[[probes]]
id = "trusted_asset_preflight"
type = "trusted_asset_preflight"
require_geometry_roles = ["cad"]
require_materials = true
require_provenance = true
require_trusted_mass_properties = true
"""

    contract = _contract_from_task(prompt_md, task_toml, eval_config_toml)

    assert "trusted_cad_preflight" in contract
    assert "trusted_mass_properties_required" in contract
    assert "chrono_contact_required" in contract
    assert "procedural_fallback_disabled" in contract

    train_contract = _train_contract_from_task(
        prompt_md, task_toml, eval_config_toml
    )
    assert "trusted_cad_preflight" in train_contract
    assert "trusted_mass_properties_required" in train_contract
    assert "chrono_contact_required" in train_contract
    assert "procedural_fallback_disabled" in train_contract
