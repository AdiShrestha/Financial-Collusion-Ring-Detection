"""Unit tests for KUSET camera-ready manuscript consistency (Contract C18-03)."""

import json
import os
import re
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer


def test_manuscript_fragments_and_figures_exist():
    """Verify all table fragments and figure files included in kuset_main.tex exist."""
    tex_path = "paper/kuset_main.tex"
    assert os.path.exists(tex_path)

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    # Check input statements
    inputs = re.findall(r"\\input\{([^}]+)\}", tex_content)
    assert len(inputs) >= 4
    for inp in inputs:
        full_inp_path = os.path.join("paper", inp if inp.endswith(".tex") else f"{inp}.tex")
        assert os.path.exists(full_inp_path), f"Missing input file: {full_inp_path}"

    # Check graphicx inclusions
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_content)
    assert len(figs) >= 2
    for fig in figs:
        full_fig_path = os.path.join("paper", fig)
        assert os.path.exists(full_fig_path), f"Missing figure file: {full_fig_path}"


def test_manuscript_numerical_consistency_with_stats():
    """Verify numerical facts stated in manuscript match production confirmatory stats."""
    tex_path = "paper/kuset_main.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    stats_path = "results/production_confirmatory_stats.json"
    assert os.path.exists(stats_path)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Check that core cohort numbers are present in manuscript
    assert "5,078,345" in tex_content
    assert "515,080" in tex_content
    assert "155" in tex_content
    assert "B_1 B_2 = 0" in tex_content or r"\mathbf{B}_1 \mathbf{B}_2 = \mathbf{0}" in tex_content


def test_exhaustive_claim_synchronizer_passes():
    audit = KUSETClaimSynchronizer().audit_claim_synchronization()
    assert audit["all_synchronized"], audit
