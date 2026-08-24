"""Tests for ReleasePackager (C09-04)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.release.packager import ReleasePackager


def test_release_packager_manifest_assembly():
    """Run packager; assert release/README.md exists and contains reproduction instructions."""
    packager = ReleasePackager(project_root=".")
    result = packager.package_release(output_dir="release")

    assert result["success"] is True, f"Packaging failed with missing: {result['missing_prerequisites']}"
    assert os.path.exists("release/README.md")
    assert os.path.exists("release/environment_lock.json")
    assert os.path.exists("release/replicate.py")

    with open("release/README.md", "r") as f:
        readme = f.read()
    assert "Quick Start" in readme
    assert "Reproduction" in readme or "reproduction" in readme
    assert "TopoRingNet" in readme

    assert result["file_count"] >= 3
