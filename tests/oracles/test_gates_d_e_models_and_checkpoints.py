"""Oracle tests for Gate D (Topological Nilpotency) and Gate E (Model & Checkpoint Verification)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.training.independent_gate_d_e_verifier import IndependentGateDEVerifier


def test_independent_gate_d_verification():
    """Verify Gate D passes with boundary nilpotency B1 @ B2 == 0 across all candidates."""
    verifier = IndependentGateDEVerifier()
    rep = verifier.verify_gate_d(
        features_dir="artifacts/features",
        output_report_path="project/gate_d_report.json",
    )
    assert rep["status"] == "GATE_D_PASS"
    assert rep["nilpotency_verified_count"] > 0


def test_independent_gate_e_verification():
    """Verify Gate E passes with all 200 checkpoints and complete OOF predictions."""
    verifier = IndependentGateDEVerifier()
    rep = verifier.verify_gate_e(
        checkpoints_dir="artifacts/checkpoints",
        training_history_path="artifacts/training/training_history.json",
        output_report_path="project/gate_e_report.json",
    )
    assert rep["status"] == "GATE_E_PASS"
    assert rep["total_checkpoints_verified"] == 200
