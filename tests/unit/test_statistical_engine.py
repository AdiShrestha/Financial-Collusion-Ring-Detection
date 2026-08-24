"""Unit tests for Statistical Testing and Metric Computation Engine."""

import pytest
import numpy as np
from source.evidence.statistics import (
    compute_classification_metrics,
    compute_cliffs_delta,
    evaluate_hypothesis_verdict,
    holm_bonferroni_correction,
    paired_wilcoxon_test,
)


def test_classification_metrics():
    """Verify comprehensive metric computations across classification outputs."""
    y_true = [1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]

    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics["pr_auc"] >= 0.95
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["f1_positive"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["brier_score"] < 0.1


def test_wilcoxon_calculation_known_distribution():
    """Verify paired Wilcoxon signed-rank calculation on deterministic score distributions."""
    scores_a = [0.85, 0.90, 0.88, 0.92, 0.89, 0.95, 0.91, 0.93]
    scores_b = [0.70, 0.75, 0.72, 0.78, 0.71, 0.80, 0.74, 0.76]

    stat, p_val = paired_wilcoxon_test(scores_a, scores_b)

    assert p_val < 0.05
    assert stat >= 0.0

    # Identical arrays yield p-value = 1.0
    stat_same, p_same = paired_wilcoxon_test(scores_a, scores_a)
    assert p_same == 1.0


def test_cliffs_delta_bootstrap_ci():
    """Verify Cliff's delta point estimate, bootstrap CI bounding, and magnitude classification."""
    scores_a = [0.9, 0.85, 0.88, 0.92, 0.95]
    scores_b = [0.6, 0.65, 0.62, 0.68, 0.70]

    res = compute_cliffs_delta(scores_a, scores_b, num_bootstraps=500, seed=42)

    assert res["delta"] == 1.0
    assert res["ci_low"] <= res["delta"] <= res["ci_high"]
    assert res["magnitude"] == "large"

    # Symmetric negative test
    res_neg = compute_cliffs_delta(scores_b, scores_a, num_bootstraps=500, seed=42)
    assert res_neg["delta"] == -1.0
    assert res_neg["magnitude"] == "large"

    # Identical groups test
    res_ident = compute_cliffs_delta(scores_a, scores_a, num_bootstraps=100, seed=42)
    assert res_ident["delta"] == 0.0
    assert res_ident["magnitude"] == "negligible"


def test_holm_bonferroni_step_down_ordering():
    """Verify Holm-Bonferroni step-down correction enforces strict monotonicity and proper rejection."""
    raw_p = [0.005, 0.04, 0.015, 0.12]
    corrected = holm_bonferroni_correction(raw_p, alpha=0.05)

    assert len(corrected) == 4

    # Check adjusted p-values are monotonic with respect to rank
    sorted_corrected = sorted(corrected, key=lambda x: x["raw_p_value"])
    adj_values = [item["adjusted_p_value"] for item in sorted_corrected]

    for i in range(len(adj_values) - 1):
        assert adj_values[i] <= adj_values[i + 1]

    # Verify first two hypotheses are rejected under alpha=0.05
    assert sorted_corrected[0]["is_significant"] is True  # 0.005 * 4 = 0.02 < 0.05
    assert sorted_corrected[1]["is_significant"] is True  # 0.015 * 3 = 0.045 < 0.05


def test_verdict_three_valued_assignment():
    """Verify assignment logic across scientific verdict categories."""
    assert evaluate_hypothesis_verdict(p_adj=0.02, delta=0.25) == "SUPPORTED"
    assert evaluate_hypothesis_verdict(p_adj=0.01, delta=-0.30) == "FALSIFIED"
    assert evaluate_hypothesis_verdict(p_adj=0.10, delta=0.50) == "INCONCLUSIVE"
    assert evaluate_hypothesis_verdict(p_adj=0.01, delta=0.05) == "INCONCLUSIVE"
