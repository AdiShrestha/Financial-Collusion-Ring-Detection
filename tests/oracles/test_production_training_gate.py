"""Oracle tests for Production Training Gate (Contract C12-05)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.training.production_gate_verifier import ProductionTrainingGateVerifier


def test_production_training_gate_verification():
    """Verify ProductionTrainingGateVerifier certifies PRODUCTION_TRAINING_PASS."""
    verifier = ProductionTrainingGateVerifier(project_root=".")
    result = verifier.verify_production_checkpoints()

    assert result["status"] == "PRODUCTION_TRAINING_PASS"
    assert result["production_training_status"] == "PRODUCTION_TRAINING_PASS"
    assert result["all_subchecks_passed"] is True
    assert result["total_checkpoints_audited"] == 35
    assert len(result["missing_checkpoints"]) == 0
    assert len(result["corrupt_checkpoints"]) == 0


def test_production_training_report_json_exists():
    """Verify project/production_training_report.json exists and is valid."""
    verifier = ProductionTrainingGateVerifier(project_root=".")
    verifier.verify_production_checkpoints()

    report_path = "project/production_training_report.json"
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate_id"] == "PRODUCTION_TRAINING_GATE"
    assert data["status"] == "PRODUCTION_TRAINING_PASS"
    assert "INV-001" in data["invariants_verified"]
    assert "INV-006" in data["invariants_verified"]
