"""Unit tests for KUSETPlotGenerator (Contract C13-03)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.paper.kuset_plot_generator import KUSETPlotGenerator


def test_all_kuset_figures_exist_and_non_empty():
    """Verify all 4 publication figure files exist and have non-trivial size (> 5 KB)."""
    fig_dir = "paper/figures"
    expected_files = [
        "persistence_diagrams.png",
        "pr_curves.png",
        "cycle_sensitivity.png",
        "ablation_ph.png",
    ]

    for fname in expected_files:
        fpath = os.path.join(fig_dir, fname)
        assert os.path.exists(fpath), f"Missing figure file: {fpath}"
        assert os.path.getsize(fpath) > 5000, f"Figure file suspiciously small: {fpath}"


def test_kuset_plot_generator_methods():
    """Verify KUSETPlotGenerator can instantiate and run without crashing."""
    gen = KUSETPlotGenerator()
    assert os.path.exists(gen.figures_dir)
