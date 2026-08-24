"""Unit tests for Provenance-Hardened Candidate Extraction & Group-Safe Partitioning (Contract C14-02)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.candidate_dataset import CandidateDatasetManager


def test_candidates_jsonl_provenance_and_validity():
    """Verify data/processed/candidates.jsonl contains 200 balanced candidates with diverse formats."""
    cand_path = "data/processed/candidates.jsonl"
    assert os.path.exists(cand_path), "Missing candidates.jsonl"

    mgr = CandidateDatasetManager()
    candidates = mgr.load_candidates(cand_path)
    assert len(candidates) == 200

    pos_cands = [c for c in candidates if c["label"] == 1]
    neg_cands = [c for c in candidates if c["label"] == 0]
    assert len(pos_cands) == 100
    assert len(neg_cands) == 100

    # Verify diverse payment formats
    pos_formats = set()
    neg_formats = set()
    for c in pos_cands:
        for tx in c.get("transactions", []):
            pos_formats.add(tx.get("payment_format", "").lower())
    for c in neg_cands:
        for tx in c.get("transactions", []):
            neg_formats.add(tx.get("payment_format", "").lower())

    assert len(pos_formats) > 1, "Positive candidates must have diverse payment formats"
    assert len(neg_formats) > 1, "Negative candidates must have diverse payment formats"


def test_split_manifest_cryptographic_disjointness():
    """Verify split_manifest.json and candidate_integrity_report.json show 0.0% account overlap."""
    manifest_path = "data/manifests/split_manifest.json"
    rep_path = "project/candidate_integrity_report.json"
    assert os.path.exists(manifest_path), "Missing split_manifest.json"
    assert os.path.exists(rep_path), "Missing candidate_integrity_report.json"

    with open(rep_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)

    assert rep_data["status"] == "CANDIDATE_INTEGRITY_PASS"
    assert rep_data["passed"] is True
    assert rep_data["account_leakage"]["train_val_overlap"] == 0
    assert rep_data["account_leakage"]["train_test_overlap"] == 0
    assert rep_data["account_leakage"]["val_test_overlap"] == 0
