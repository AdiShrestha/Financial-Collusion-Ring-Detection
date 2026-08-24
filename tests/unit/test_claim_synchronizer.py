"""Tests for ClaimSynchronizer (C09-03)."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.paper.claim_synchronizer import ClaimSynchronizer


def test_claim_synchronization_full_match():
    """Run auditor on real paper and stats; assert all_synced == True and 0 discrepancies."""
    sync = ClaimSynchronizer(
        tex_path="paper/main.tex",
        stats_path="results/confirmatory_stats.json",
    )
    result = sync.audit_synchronization()
    assert result["all_synced"] is True, (
        f"Synchronization failed with {len(result['discrepancies'])} discrepancies: "
        f"{result['discrepancies']}"
    )
    assert len(result["discrepancies"]) == 0
    assert result["num_checked"] > 0


def test_claim_synchronizer_detects_mismatch():
    """Inject artificial discrepancy and verify auditor catches it."""
    # Create a temp tex file with a wrong value
    with open("paper/main.tex", "r") as f:
        real_content = f.read()

    # Inject wrong PR-AUC for GCNBaseline: change 0.6333 to 0.9999
    tampered = real_content.replace(
        "GCNBaseline & $0.6333 \\pm 0.1633$",
        "GCNBaseline & $0.9999 \\pm 0.1633$",
    )
    assert tampered != real_content, "Tampering failed — pattern not found"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tex", delete=False
    ) as tmp:
        tmp.write(tampered)
        tmp_path = tmp.name

    try:
        sync = ClaimSynchronizer(
            tex_path=tmp_path,
            stats_path="results/confirmatory_stats.json",
        )
        result = sync.audit_synchronization()
        assert result["all_synced"] is False
        assert len(result["discrepancies"]) > 0
        # Should detect the PR-AUC mismatch for GCNBaseline
        mismatch_found = any(
            d.get("model") == "GCNBaseline" and d.get("metric") == "mean_pr_auc"
            for d in result["discrepancies"]
        )
        assert mismatch_found, f"Expected GCNBaseline PR-AUC mismatch, got: {result['discrepancies']}"
    finally:
        os.unlink(tmp_path)
