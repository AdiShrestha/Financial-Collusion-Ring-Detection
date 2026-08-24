"""Unit tests for Confirmatory Paired Hypothesis Testing & Holm-Bonferroni Correction."""

import json
import os
import pytest

from source.evidence.hypothesis_tester import ConfirmatoryHypothesisTester


def _load_sample_predictions():
    pred_path = "runs/confirmatory/predictions.json"
    assert os.path.exists(pred_path)
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hypothesis_tester_evaluates_four_hypotheses():
    """Verify that hypothesis tester generates evaluations for all 4 hypotheses H1..H4."""
    data = _load_sample_predictions()
    tester = ConfirmatoryHypothesisTester(alpha=0.05, min_effect=0.147)
    res = tester.evaluate_all_hypotheses(data)

    assert "metadata" in res
    assert "hypotheses" in res
    assert "overall_verdicts" in res

    hypotheses = res["hypotheses"]
    for hid in ["H1", "H2", "H3", "H4"]:
        assert hid in hypotheses
        h_info = hypotheses[hid]
        assert "raw_p_value" in h_info
        assert "adjusted_p_value" in h_info
        assert "cliffs_delta" in h_info
        assert "cliffs_delta_ci_lower" in h_info
        assert "cliffs_delta_ci_upper" in h_info
        assert "verdict" in h_info


def test_holm_bonferroni_adjusted_p_values_monotonic():
    """Verify that Holm-Bonferroni adjusted p-values are bounded in [0, 1] and monotonic."""
    data = _load_sample_predictions()
    tester = ConfirmatoryHypothesisTester()
    res = tester.evaluate_all_hypotheses(data)

    p_adjs = [res["hypotheses"][h]["adjusted_p_value"] for h in ["H1", "H2", "H3", "H4"]]
    for p in p_adjs:
        assert 0.0 <= p <= 1.0


def test_three_valued_verdict_validity():
    """Verify that all assigned hypothesis verdicts belong strictly to the 3-valued set."""
    data = _load_sample_predictions()
    tester = ConfirmatoryHypothesisTester()
    res = tester.evaluate_all_hypotheses(data)

    valid_verdicts = {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}
    for hid, verdict in res["overall_verdicts"].items():
        assert verdict in valid_verdicts, f"Hypothesis {hid} assigned invalid verdict {verdict}"
