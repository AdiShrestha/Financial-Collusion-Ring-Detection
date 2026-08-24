"""Oracle tests for Gate A Data Provenance & Lineage Verification (Contract C15-04)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.independent_gate_a_verifier import GateAVerifier


def test_gate_a_verifier_execution_and_certification():
    """Verify Gate A independent verifier executes cleanly and passes all provenance checks."""
    verifier = GateAVerifier()
    report = verifier.verify_gate_a()

    assert report["passed"] is True
    assert report["status"] == "GATE_A_PASS"
    assert report["checks"]["trans_csv_hash_matches"] is True
    assert report["checks"]["patterns_txt_hash_matches"] is True
    assert report["checks"]["total_transactions_matches"] is True
    assert report["checks"]["unique_accounts_matches"] is True
    assert report["checks"]["positive_candidate_count_matches"] is True
    assert report["checks"]["disjoint_components_count_matches"] is True
    assert report["checks"]["directed_chaining_valid"] is True
    assert report["checks"]["zero_synthetic_candidate_tokens"] is True


def test_gate_a_report_json_exists():
    """Verify project/gate_a_report.json exists and is populated."""
    assert os.path.exists("project/gate_a_report.json")
    with open("project/gate_a_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "GATE_A_PASS"
