"""Unit tests for environment setup, version locking, and package smoke checks."""

import json
import os
import sys
import pytest

from source.utils.env_check import inspect_environment, write_environment_lock


def test_python_version():
    """Assert Python version is at least 3.10."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    assert (major, minor) >= (3, 10), f"Python version must be >= 3.10, found {sys.version}"


def test_core_dependencies_installed():
    """Assert essential packages are installed and report valid non-empty versions."""
    import numpy
    import scipy
    import networkx
    import yaml

    assert hasattr(numpy, "__version__") and len(numpy.__version__) > 0
    assert hasattr(scipy, "__version__") and len(scipy.__version__) > 0
    assert hasattr(networkx, "__version__") and len(networkx.__version__) > 0
    assert hasattr(yaml, "__version__") and len(yaml.__version__) > 0


def test_environment_inspector_structure():
    """Assert inspect_environment returns required fields and valid schema."""
    data = inspect_environment()
    assert "timestamp" in data
    assert "platform" in data
    assert "packages" in data
    assert "hardware" in data

    assert "python" in data["packages"]
    assert "numpy" in data["packages"]
    assert "scipy" in data["packages"]
    assert "networkx" in data["packages"]
    assert "yaml" in data["packages"]

    assert data["packages"]["numpy"]["installed"] is True
    assert data["packages"]["scipy"]["installed"] is True
    assert data["packages"]["networkx"]["installed"] is True
    assert data["packages"]["yaml"]["installed"] is True

    # Hardware structure check
    assert "device" in data["hardware"]
    assert data["hardware"]["device"] in {"cpu", "cuda", "mps"}


def test_environment_lock_json_creation(tmp_path):
    """Assert write_environment_lock creates a valid, parseable JSON file."""
    lock_file = tmp_path / "test_env_lock.json"
    data = write_environment_lock(str(lock_file))
    assert lock_file.exists()

    with open(lock_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["platform"]["system"] == data["platform"]["system"]
    assert loaded["packages"]["python"] == data["packages"]["python"]


def test_project_environment_lock_exists():
    """Assert project/environment_lock.json exists and is valid."""
    path = "project/environment_lock.json"
    assert os.path.exists(path), "project/environment_lock.json must exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "packages" in data
    assert data["packages"]["numpy"]["installed"] is True
