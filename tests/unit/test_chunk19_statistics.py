"""Semantic and metamorphic tests for Chunk 19 inference."""

import numpy as np

from source.evidence.kuset_hypothesis_tester import (
    KUSETHypothesisTester,
    holm_bonferroni_correction,
)


def test_holm_known_answer_and_monotonicity():
    adjusted = holm_bonferroni_correction([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
    ordered = sorted(zip([0.01, 0.04, 0.03], adjusted))
    assert [a for _, a in ordered] == sorted(a for _, a in ordered)


def test_permutation_plus_one_has_nonzero_floor():
    tester = KUSETHypothesisTester()
    result = tester.run_group_permutation_test(
        "logistic_regression", "gine", n_permutations=99, random_seed=7
    )
    assert result["p_value_raw"] >= 1 / 100
    assert result["p_value_raw"] == (result["extreme_permutations"] + 1) / 100


def test_seed_order_and_row_order_do_not_change_mean_estimate():
    tester = KUSETHypothesisTester()
    original = tester.get_model_mean_probs("ccnn")
    tester.by_model_seed["ccnn"] = dict(reversed(list(tester.by_model_seed["ccnn"].items())))
    reordered = tester.get_model_mean_probs("ccnn")
    np.testing.assert_allclose(original, reordered, rtol=0, atol=0)


def test_group_relabeling_does_not_change_point_delta():
    tester = KUSETHypothesisTester()
    before = tester.run_group_permutation_test("ccnn", "gine", n_permutations=49)["delta_ap"]
    mapping = {g: f"renamed_{i}" for i, g in enumerate(np.unique(tester.group_ids))}
    tester.group_ids = np.array([mapping[g] for g in tester.group_ids])
    after = tester.run_group_permutation_test("ccnn", "gine", n_permutations=49)["delta_ap"]
    assert before == after


def test_authoritative_stats_have_real_delta_intervals():
    tester = KUSETHypothesisTester()
    result = tester.paired_group_bootstrap_delta(
        "ccnn", "gine", n_bootstrap=199, random_seed=42
    )
    lo, hi = result["delta_ap_ci_95"]
    assert lo <= result["delta_ap"] <= hi
    assert not np.isclose(result["delta_ap"] - lo, 0.1)
    assert result["n_bootstrap_valid"] + result["n_bootstrap_invalid_one_class"] == 199
