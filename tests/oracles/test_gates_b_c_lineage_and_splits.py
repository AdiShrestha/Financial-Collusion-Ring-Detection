"""Oracle tests for independent Gate B and Gate C verification (Contract C16-04)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.independent_gate_b_c_verifier import GateBCVerifier


def test_gate_b_candidate_lineage_verification():
    """Verify Gate B verifier passes with zero synthetic tokens and 100% master transaction joins."""
    verifier = GateBCVerifier()
    report = verifier.verify_gate_b()

    assert report["passed"] is True
    assert report["status"] == "GATE_B_PASS"
    assert report["checks"]["candidate_count_matches"] is True
    assert report["checks"]["all_candidate_transactions_in_master"] is True
    assert report["checks"]["negative_controls_zero_pattern_overlap"] is True
    assert report["checks"]["zero_synthetic_tokens"] is True


def test_gate_c_group_splits_and_protocol_lock_verification():
    """Verify Gate C verifier passes with zero outer-fold account leakage and locked manifest SHA-256."""
    verifier = GateBCVerifier()
    report = verifier.verify_gate_c()

    assert report["passed"] is True
    assert report["status"] == "GATE_C_PASS"
    assert report["checks"]["n_outer_folds_is_5"] is True
    assert report["checks"]["n_inner_folds_is_3"] is True
    assert report["checks"]["manifest_sha_matches_protocol"] is True
    assert report["checks"]["outer_fold_zero_account_leakage"] is True


def test_gate_reports_exist_and_certified():
    """Verify project/gate_b_report.json and project/gate_c_report.json exist and certify pass."""
    assert os.path.exists("project/gate_b_report.json")
    assert os.path.exists("project/gate_c_report.json")

    with open("project/gate_b_report.json", "r", encoding="utf-8") as f:
        b = json.load(f)
    assert b["status"] == "GATE_B_PASS"

    with open("project/gate_c_report.json", "r", encoding="utf-8") as f:
        c = json.load(f)
    assert c["status"] == "GATE_C_PASS"
