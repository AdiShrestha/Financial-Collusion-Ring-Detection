"""Unit tests for KUSETHypothesisTester (Contract C13-02)."""

import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.evidence.kuset_hypothesis_tester import (
    KUSETHypothesisTester,
    holm_bonferroni_correction,
    paired_bootstrap_delta_pr_auc,
)


def test_holm_bonferroni_monotonicity():
    """Verify Holm-Bonferroni correction is strictly monotonic and bounded by 1.0."""
    raw_p = [0.01, 0.04, 0.03]
    adj_p = holm_bonferroni_correction(raw_p)

    assert len(adj_p) == 3
    assert all(0.0 <= p <= 1.0 for p in adj_p)
    # The smallest raw p-value (0.01) multiplied by 3 gives 0.03
    assert adj_p[0] <= adj_p[2] <= adj_p[1]


def test_production_confirmatory_stats_artifact():
    """Verify results/production_confirmatory_stats.json exists and has valid hypotheses."""
    stats_path = "results/production_confirmatory_stats.json"
    assert os.path.exists(stats_path), "Missing production_confirmatory_stats.json"

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "model_metrics" in data
    assert "hypotheses" in data
    assert "H1" in data["hypotheses"]
    assert "H2" in data["hypotheses"]
    assert "H3" in data["hypotheses"]

    for h_id in ("H1", "H2", "H3"):
        h_data = data["hypotheses"][h_id]
        assert h_data["verdict"] in ("SUPPORTED", "FALSIFIED", "INCONCLUSIVE")
        assert "delta_pr_auc" in h_data
        assert "ci_95_lower" in h_data
        assert "ci_95_upper" in h_data
        assert "wilcoxon_p_adj" in h_data
