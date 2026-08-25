"""Tests for ClaimSynchronizer (C09-03)."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer


def test_claim_synchronization_full_match():
    """Run auditor on real paper and stats; assert all_synced == True and 0 discrepancies."""
    sync = KUSETClaimSynchronizer(
        tex_path="paper/kuset_main.tex",
        stats_path="results/production_confirmatory_stats.json",
    )
    result = sync.audit_claim_synchronization()
    assert result["all_synced"] is True
    assert len(result["discrepancies"]) == 0
    assert result["total_checks"] == 52


def test_claim_synchronizer_detects_mismatch():
    """Inject artificial discrepancy and verify auditor catches it."""
    # Create a temp tex file with a wrong value
    with open("paper/kuset_main.tex", "r") as f:
        real_content = f.read()

    # Inject a wrong canonical GCN AP macro.
    tampered = real_content.replace(
        r"\def\gcnAP{0.2868}",
        r"\def\gcnAP{0.9999}",
    )
    assert tampered != real_content, "Tampering failed — pattern not found"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tex", delete=False
    ) as tmp:
        tmp.write(tampered)
        tmp_path = tmp.name

    try:
        sync = KUSETClaimSynchronizer(
            tex_path=tmp_path,
            stats_path="results/production_confirmatory_stats.json",
        )
        result = sync.audit_claim_synchronization()
        assert result["all_synced"] is False
        assert len(result["discrepancies"]) > 0
        assert any(item.startswith("gcnAP:") for item in result["discrepancies"])
    finally:
        os.unlink(tmp_path)
