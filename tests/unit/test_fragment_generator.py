"""Unit tests for automated LaTeX and figure fragment generator (Contract C18-02)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.paper.kuset_fragment_generator import generate_kuset_fragments


def test_generate_kuset_fragments():
    """Verify all LaTeX table fragments and publication figures generate cleanly without syntax errors."""
    res = generate_kuset_fragments(
        stats_json_path="results/production_confirmatory_stats.json",
        audit_json_path="artifacts/audit/observed_data_audit.json",
        candidates_parquet_path="artifacts/candidates/candidates.parquet",
        oof_parquet_path="artifacts/predictions/oof_predictions.parquet",
        output_tex_dir="paper/generated",
        output_fig_dir="paper/figures",
    )

    assert res["status"] == "FRAGMENTS_GENERATED"

    expected_tables = [
        "paper/generated/tab_cohort_stats.tex",
        "paper/generated/tab_model_benchmark.tex",
        "paper/generated/tab_hypothesis_tests.tex",
        "paper/generated/tab_ablation.tex",
    ]
    for tab_path in expected_tables:
        assert os.path.exists(tab_path)
        with open(tab_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert len(content) > 100

    expected_figures = [
        "paper/figures/fig_pr_curves.png",
        "paper/figures/fig_k_ablation.png",
    ]
    for fig_path in expected_figures:
        assert os.path.exists(fig_path)
        assert os.path.getsize(fig_path) > 1000
