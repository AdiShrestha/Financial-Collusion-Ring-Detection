"""Oracle test suite for Candidate Integrity Gate (Contract C11-05)."""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.candidate_verifier import CandidateIntegrityVerifier


def test_candidate_integrity_full_system_verification():
    """Verify CandidateIntegrityVerifier executes full check and certifies CANDIDATE_INTEGRITY_PASS."""
    verifier = CandidateIntegrityVerifier(project_root=".")
    result = verifier.verify_candidate_integrity()

    assert result["status"] == "CANDIDATE_INTEGRITY_PASS"
    assert result["candidate_integrity_status"] == "CANDIDATE_INTEGRITY_PASS"
    assert result["all_subchecks_passed"] is True

    checks = result["integrity_checks"]
    assert checks["candidates_file_exists"] is True
    assert checks["sufficient_candidates_count"] is True
    assert checks["split_manifest_exists"] is True
    assert checks["candidate_ids_aligned"] is True
    assert checks["zero_cross_split_account_overlap"] is True
    assert checks["cycle_length_distribution_covered"] is True
    assert checks["label_balance_valid"] is True
    assert checks["split_hashes_valid"] is True


def test_candidate_integrity_report_json_exists():
    """Verify project/candidate_integrity_report.json is created and valid."""
    verifier = CandidateIntegrityVerifier(project_root=".")
    verifier.verify_candidate_integrity()

    report_path = "project/candidate_integrity_report.json"
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate_id"] == "CANDIDATE_INTEGRITY_GATE"
    assert data["status"] == "CANDIDATE_INTEGRITY_PASS"
    assert "INV-006" in data["invariants_verified"]


def test_candidate_integrity_detects_account_leakage():
    """Verify CandidateIntegrityVerifier detects and fails on cross-split account overlap."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create candidates where train and val share an account "LEAK_ACC"
        cands = [
            {"candidate_id": "c1", "nodes": ["LEAK_ACC", "A2", "A3"], "participants": ["LEAK_ACC", "A2", "A3"], "label": 1, "cycle_length": 3},
            {"candidate_id": "c2", "nodes": ["LEAK_ACC", "B2", "B3"], "participants": ["LEAK_ACC", "B2", "B3"], "label": 0, "cycle_length": 3},
        ]
        manifest = {
            "splits": {
                "train": [{"candidate_id": "c1"}],
                "val": [{"candidate_id": "c2"}],
                "test": [],
            },
            "split_hashes": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64},
        }
        cand_path = os.path.join(tmpdir, "candidates.jsonl")
        manifest_path = os.path.join(tmpdir, "split_manifest.json")
        out_path = os.path.join(tmpdir, "report.json")

        with open(cand_path, "w", encoding="utf-8") as f:
            for c in cands:
                f.write(json.dumps(c) + "\n")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        verifier = CandidateIntegrityVerifier(project_root=tmpdir)
        result = verifier.verify_candidate_integrity(
            candidates_path=cand_path,
            manifest_path=manifest_path,
            output_path=out_path,
        )

        assert result["status"] == "CANDIDATE_INTEGRITY_FAIL"
        assert result["integrity_checks"]["zero_cross_split_account_overlap"] is False
