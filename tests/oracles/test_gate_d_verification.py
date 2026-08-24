"""Oracle tests for Gate D (Pre-Registration & Protocol Lock) Verification Suite."""

import json
import os
import pytest
from source.training.gate_d_verifier import GateDVerifier


def test_gate_d_full_verification():
    """Execute complete Gate D verification protocol and assert GATE_D_PASS."""
    verifier = GateDVerifier()
    report = verifier.verify_gate_d()

    assert report["gate_d_status"] == "GATE_D_PASS"
    assert report["all_subchecks_passed"] is True
    assert report["subchecks"]["pre_registration_document"]["passed"] is True
    assert report["subchecks"]["test_partition_immutability"]["passed"] is True
    assert report["subchecks"]["hyperparameter_freeze"]["passed"] is True
    assert report["subchecks"]["statistics_engine"]["passed"] is True


def test_gate_d_report_json_exists():
    """Verify that project/gate_d_report.json exists, is valid JSON, and records GATE_D_PASS."""
    report_path = "project/gate_d_report.json"
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate_d_status"] == "GATE_D_PASS"
    assert "protocol_specifications" in data
    assert data["protocol_specifications"]["evaluation_seeds"] == [42, 43, 44, 45, 46]


def test_gate_d_detects_missing_preregistration(tmp_path):
    """Verify that GateDVerifier returns GATE_D_FAIL if pre-registration document is missing."""
    fake_pre_reg = str(tmp_path / "NONEXISTENT_PRE_REG.md")
    report_file = str(tmp_path / "fail_gate_d_report.json")

    verifier = GateDVerifier(
        output_report_path=report_file,
        pre_reg_path=fake_pre_reg,
    )
    report = verifier.verify_gate_d()

    assert report["gate_d_status"] == "GATE_D_FAIL"
    assert report["all_subchecks_passed"] is False
    assert report["subchecks"]["pre_registration_document"]["passed"] is False
