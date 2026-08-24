"""Oracle test suite for Phase-0 Feasibility Gate (Contract C10-05)."""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.phase0_gate_verifier import Phase0GateVerifier


def test_phase0_gate_full_system_verification():
    """Verify Phase0GateVerifier runs full audit and produces PHASE0_GATE_PASS."""
    verifier = Phase0GateVerifier(project_root=".")
    result = verifier.verify_phase0_feasibility()

    assert result["status"] == "PHASE0_GATE_PASS"
    assert result["phase0_gate_status"] == "PHASE0_GATE_PASS"
    assert result["all_subchecks_passed"] is True

    checks = result["feasibility_checks"]
    assert checks["sha256_hashes_valid"] is True
    assert checks["non_empty_records"] is True
    assert checks["sufficient_independent_groups"] is True
    assert checks["cycle_length_distribution_covered"] is True
    assert checks["transaction_join_rate_valid"] is True


def test_phase0_gate_report_json_exists():
    """Verify project/phase0_gate_report.json is generated, valid, and immutable."""
    verifier = Phase0GateVerifier(project_root=".")
    verifier.verify_phase0_feasibility()

    report_path = "project/phase0_gate_report.json"
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate_id"] == "PHASE0_FEASIBILITY_GATE"
    assert data["status"] == "PHASE0_GATE_PASS"
    assert len(data["raw_data_hashes"]["HI-Small_Trans.csv"]) == 64
    assert len(data["raw_data_hashes"]["HI-Small_Patterns.txt"]) == 64


def test_phase0_gate_detects_insufficient_cycle_lengths():
    """Verify Phase-0 Gate fails if cycle length distribution is incomplete."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_audit = {
            "files": {
                "transactions_csv": {"sha256": "a" * 64, "row_count": 100},
                "patterns_txt": {"sha256": "b" * 64, "pattern_block_count": 10},
            },
            "cycle_typology_breakdown": {
                "length_distribution": {"3": 5, "4": 0, "5": 0, "6": 0}  # Missing 4, 5, 6
            },
            "cluster_independence": {"independent_groups_count": 20},
            "data_quality_summary": {"join_rate": 1.0},
        }
        audit_file = os.path.join(tmpdir, "observed_audit.json")
        report_file = os.path.join(tmpdir, "gate_report.json")
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(mock_audit, f)

        verifier = Phase0GateVerifier(project_root=tmpdir)
        result = verifier.verify_phase0_feasibility(
            audit_path=audit_file,
            output_report_path=report_file,
        )

        assert result["status"] == "PHASE0_GATE_FAIL"
        assert result["feasibility_checks"]["cycle_length_distribution_covered"] is False
