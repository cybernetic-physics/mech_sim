from __future__ import annotations

import importlib.util
from pathlib import Path


def load_bootstrap_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_chrono_env.py"
    spec = importlib.util.spec_from_file_location("bootstrap_chrono_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_conda_frontend_prefers_explicit_executable(tmp_path):
    module = load_bootstrap_module()
    conda = tmp_path / "conda"
    conda.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    conda.chmod(0o755)

    assert module.find_conda_frontend(tmp_path, str(conda)) == conda


def test_site_packages_for_uses_requested_python_abi(tmp_path):
    module = load_bootstrap_module()
    site_packages = tmp_path / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)

    assert module.site_packages_for(tmp_path, "3.12") == site_packages
