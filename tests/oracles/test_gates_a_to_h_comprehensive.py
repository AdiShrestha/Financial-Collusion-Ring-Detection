"""Comprehensive End-to-End Release Certification Tests (Gates A through F)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from release.replicate import run_clean_room_replication


def test_gates_a_to_f_comprehensive_audit():
    """Verify that all gates and invariants pass with 100% precision for submission release."""
    rep = run_clean_room_replication()
    assert rep["status"] == "ARTIFACT_VERIFICATION_PASS"
    assert rep["total_transactions"] == 5078345
    assert rep["total_candidates"] == 155
    assert rep["total_groups"] == 18
    assert rep["total_checkpoints"] == 200
    assert rep["gate_statuses"]["gate_d"] == "GATE_D_PASS"
    assert rep["gate_statuses"]["gate_e"] == "GATE_E_PASS"


def test_generate_gate_f_kuset_report():
    """Verify and generate final project/gate_f_kuset_report.json."""
    from source.release.kuset_release_packager import KUSETReleasePackager
    packager = KUSETReleasePackager()
    final_report = packager.verify_and_build_terminal_release()

    out_path = "project/gate_f_kuset_report.json"
    assert os.path.exists(out_path)
    assert final_report["terminal_status"] == "GATE_F_KUSET_PASS"
    assert final_report["passed"] is True
