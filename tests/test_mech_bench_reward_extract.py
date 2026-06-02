from __future__ import annotations

from rl.mech_bench_reward import extract_design_py


def test_extract_design_py_accepts_unterminated_python_fence() -> None:
    source, extracted = extract_design_py(
        "```python\nfrom pathlib import Path\n\n"
        "def build_design(out_dir: Path) -> dict:\n"
        "    return {'schema_version': 'design_ir.v2'}"
    )

    assert extracted is True
    assert source.startswith("from pathlib import Path")
    assert "```python" not in source


def test_extract_design_py_prefers_closed_python_fence() -> None:
    source, extracted = extract_design_py(
        "before\n```python\nx = 1\n```\nafter\n```python\nx = 2\n```"
    )

    assert extracted is True
    assert source == "x = 1\n"


def test_extract_design_py_prefers_later_build_design_fence() -> None:
    source, extracted = extract_design_py(
        "First, add this contact snippet:\n"
        "```python\n"
        "{\"id\": \"contact\", \"type\": \"contact_pair\"}\n"
        "```\n"
        "Final answer:\n"
        "```python\n"
        "from pathlib import Path\n\n"
        "def build_design(out_dir: Path) -> dict:\n"
        "    return {\"schema_version\": \"design_ir.v2\"}\n"
        "```\n"
    )

    assert extracted is True
    assert source.startswith("from pathlib import Path")
    assert "def build_design" in source
    assert "contact_pair" not in source
