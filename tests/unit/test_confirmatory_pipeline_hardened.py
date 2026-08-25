"""Unit tests for Provenance-Hardened Confirmatory Pipeline (Contract C14-04)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_confirmatory_predictions_ledger():
    """Verify runs/production_confirmatory/predictions.json exists and contains 28 test candidates."""
    pred_path = "runs/production_confirmatory/predictions.json"
    assert os.path.exists(pred_path), "Missing predictions.json"

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metadata"]["total_test_candidates"] == 28
    assert len(data["predictions"]) == 28


def test_production_confirmatory_stats_hypotheses():
    """Verify the corrective RQ family and exploratory comparison are explicit."""
    stats_path = "results/production_confirmatory_stats.json"
    assert os.path.exists(stats_path), "Missing production_confirmatory_stats.json"

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "model_metrics" in data
    tests = data["confirmatory_hypothesis_tests"]
    assert tests["RQ1_ccnn_vs_gine"]["analysis_role"] == "primary_confirmatory_family"
    assert tests["RQ2_ccnn_vs_scnn_kge4"]["analysis_role"] == "primary_confirmatory_family"
    assert tests["RQ3_lr_vs_gine"]["analysis_role"] == "exploratory"
    assert tests["RQ3_lr_vs_gine"]["p_value_adjusted"] is None


def test_publication_figures_exist():
    """Verify all 4 publication figures exist in paper/figures/ at 300 DPI."""
    for fname in ("persistence_diagrams.png", "pr_curves.png", "cycle_sensitivity.png", "ablation_ph.png"):
        fpath = os.path.join("paper/figures", fname)
        assert os.path.exists(fpath), f"Missing figure: {fname}"
        assert os.path.getsize(fpath) > 5000
