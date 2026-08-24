"""Oracle tests for Gate F-KUSET Certification (Contract C13-05)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.release.kuset_release_packager import KUSETReleasePackager


def test_gate_f_kuset_report_exists_and_certified():
    """Verify project/gate_f_kuset_report.json exists and certifies GATE_F_KUSET_PASS."""
    rep_path = "project/gate_f_kuset_report.json"
    assert os.path.exists(rep_path), "Missing gate_f_kuset_report.json"

    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate"] == "Gate F-KUSET"
    assert data["terminal_status"] == "GATE_F_KUSET_PASS"
    assert data["passed"] is True
    assert data["total_chunks_certified"] == 13
    assert data["total_checkpoints_trained"] == 35

    invs = data["invariant_verifications"]
    for inv_id in ("INV-001", "INV-002", "INV-003", "INV-004", "INV-005", "INV-006", "INV-007", "INV-008"):
        assert inv_id in invs
        assert invs[inv_id]["passed"] is True

    bundle = data["release_bundle"]
    for fname, ok in bundle.items():
        assert ok is True, f"Release file missing: {fname}"


def test_gate_f_kuset_packager_verification_execution():
    """Verify KUSETReleasePackager execution returns GATE_F_KUSET_PASS."""
    packager = KUSETReleasePackager()
    res = packager.verify_and_build_terminal_release()
    assert res["terminal_status"] == "GATE_F_KUSET_PASS"
    assert res["passed"] is True
