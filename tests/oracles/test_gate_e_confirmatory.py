"""Oracle verification tests for Gate E (Confirmatory Benchmark Realism)."""

import json
import os
import pytest

from source.evidence.gate_e_verifier import GateEVerifier


def test_gate_e_full_verification(tmp_path):
    """Verify that GateEVerifier successfully certifies confirmatory artifacts."""
    out_file = str(tmp_path / "gate_e_test_report.json")
    verifier = GateEVerifier()
    rep = verifier.verify_gate_e(output_path=out_file)

    assert rep["gate"] == "GATE_E"
    assert rep["status"] == "GATE_E_PASS"
    assert rep["all_subchecks_passed"] is True
    assert "prerequisites" in rep
    assert rep["prerequisites"]["test_split_immutability"] == "VERIFIED_MATCH"
    assert rep["prerequisites"]["seed_pseudo_replication_audit"] == "PASS_INDEPENDENT_SEEDS"
    assert len(rep["hypothesis_verdicts"]) == 4


def test_gate_e_report_json_exists():
    """Verify that project/gate_e_report.json exists and contains GATE_E_PASS."""
    report_path = "project/gate_e_report.json"
    assert os.path.exists(report_path), f"{report_path} does not exist"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("gate") == "GATE_E"
    assert data.get("gate_e_status") == "GATE_E_PASS"
    assert data.get("all_subchecks_passed") is True


def test_gate_e_detects_missing_predictions(tmp_path):
    """Verify that missing model predictions cause Gate E failure."""
    verifier = GateEVerifier(predictions_path=str(tmp_path / "nonexistent.json"))
    with pytest.raises(FileNotFoundError):
        verifier.verify_gate_e(output_path=str(tmp_path / "failed_report.json"))
