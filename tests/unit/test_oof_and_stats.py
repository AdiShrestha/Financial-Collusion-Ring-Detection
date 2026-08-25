"""Unit tests for OOF prediction ledger and KUSET hypothesis tester (Contract C18-01)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.experiments.oof_evaluator import export_oof_prediction_ledger
from source.evidence.kuset_hypothesis_tester import KUSETHypothesisTester


def test_export_oof_prediction_ledger():
    """Verify OOF prediction ledger exports all candidate rows across all models and seeds."""
    res = export_oof_prediction_ledger(
        training_history_path="artifacts/training/training_history.json",
        candidates_parquet_path="artifacts/candidates/candidates.parquet",
        fold_manifest_path="artifacts/splits/fold_manifest.json",
        output_parquet_path="artifacts/predictions/oof_predictions.parquet",
    )

    assert res["status"] == "OOF_LEDGER_EXPORTED"
    assert os.path.exists("artifacts/predictions/oof_predictions.parquet")
    assert res["total_records"] == 155 * 8 * 5
    assert res["candidate_count"] == 155
    assert res["group_count"] == 18
    assert res["seeds"] == [42, 43, 44, 45, 46]


def test_kuset_hypothesis_tester_execution():
    """Verify hypothesis tester computes group bootstrap CIs and group-blocked permutation tests."""
    tester = KUSETHypothesisTester(oof_parquet_path="artifacts/predictions/oof_predictions.parquet")
    stats = tester.execute_full_confirmatory_suite(output_results_path="results/production_confirmatory_stats.json")

    assert os.path.exists("results/production_confirmatory_stats.json")
    assert "benchmark_models" in stats
    assert "ccnn" in stats["benchmark_models"]
    assert "hypothesis_tests" in stats
    assert "RQ1_CCNN_vs_GINE" in stats["hypothesis_tests"]
    assert "raw_p_value" in stats["hypothesis_tests"]["RQ1_CCNN_vs_GINE"]
    assert stats["total_groups"] == 18
    assert stats["seeds"] == [42, 43, 44, 45, 46]
    assert stats["hypothesis_tests"]["RQ1_CCNN_vs_GINE"]["monte_carlo_plus_one"] is True
    assert "RQ3_LR_vs_GINE" in stats["hypothesis_tests"]
    assert stats["hypothesis_tests"]["RQ3_LR_vs_GINE"]["analysis_role"] == "exploratory"
    assert stats["hypothesis_tests"]["RQ3_LR_vs_GINE"]["p_value_adjusted"] is None
