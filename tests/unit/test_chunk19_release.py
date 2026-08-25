"""Regression tests for fail-closed release verification and claim synchronization."""

from pathlib import Path

import pytest

from release.replicate import verify_existing_release
from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer


def test_release_verification_derives_canonical_counts():
    result = verify_existing_release()
    assert result["status"] == "ARTIFACT_VERIFICATION_PASS"
    assert result["total_candidates"] == 155
    assert result["total_groups"] == 18
    assert result["seeds"] == [42, 43, 44, 45, 46]
    assert result["total_checkpoints"] == 200
    assert result["ledger_rows"] == 6200
    assert all(result["checks"].values())


def test_release_missing_artifact_fails_closed(monkeypatch, tmp_path):
    import release.replicate as replicate
    monkeypatch.setattr(replicate, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Required artifact missing"):
        replicate.verify_existing_release()


def test_claim_audit_rejects_stale_literal(tmp_path):
    source = Path("paper/kuset_main.tex").read_text()
    stale = tmp_path / "stale.tex"
    stale.write_text(source + "\nPrior p=0.1108\n")
    audit = KUSETClaimSynchronizer(str(stale)).audit_claim_synchronization()
    assert audit["all_synchronized"] is False
    assert "stale_p_1108" in audit["banned_hits"]


def test_claim_audit_rejects_wrong_macro(tmp_path):
    source = Path("paper/kuset_main.tex").read_text().replace(r"\def\groupCount{18}", r"\def\groupCount{19}")
    wrong = tmp_path / "wrong.tex"
    wrong.write_text(source)
    audit = KUSETClaimSynchronizer(str(wrong)).audit_claim_synchronization()
    assert audit["all_synchronized"] is False
    assert any("groupCount" in item for item in audit["discrepancies"])
