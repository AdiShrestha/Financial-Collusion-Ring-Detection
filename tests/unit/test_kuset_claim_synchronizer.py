"""Unit tests for KUSETClaimSynchronizer (Contract C13-04)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer


def test_kuset_manuscript_exists_and_sections_present():
    """Verify paper/kuset_main.tex exists and contains all required scientific sections."""
    tex_path = "paper/kuset_main.tex"
    assert os.path.exists(tex_path), "Missing paper/kuset_main.tex"

    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    required_sections = [
        "\\begin{abstract}",
        "\\section{INTRODUCTION}",
        "\\section{MATERIALS AND METHODS}",
        "\\section{RESULTS}",
        "\\section{DISCUSSION}",
        "\\section{CONCLUSION}",
        "\\section*{AUTHOR AND DATA-AVAILABILITY NOTE}",
    ]

    for sec in required_sections:
        assert sec in content, f"Missing required section in LaTeX manuscript: {sec}"


def test_kuset_claim_synchronization_passes():
    """Verify all numerical macros in paper/kuset_main.tex exactly match results/production_confirmatory_stats.json."""
    synchronizer = KUSETClaimSynchronizer()
    res = synchronizer.audit_claim_synchronization(tolerance=1e-4)

    assert res["status"] == "SYNCHRONIZED"
    assert res["all_synchronized"] is True
    assert len(res["discrepancies"]) == 0
    assert res["total_checks"] >= 10
