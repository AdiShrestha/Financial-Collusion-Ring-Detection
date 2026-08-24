"""Oracle tests for Ablation Evidence Package & Verification Suite."""

import json
import os
import pytest

from source.evidence.protocol_lock import ProtocolLock
from source.experiments.ablation_verifier import AblationEvidenceVerifier


def test_ablation_summary_json_schema():
    """Verify that results/ablation_summary.json exists and conforms to required schema."""
    summary_path = "results/ablation_summary.json"
    assert os.path.exists(summary_path), f"{summary_path} not found"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "evaluation_metadata" in data
    assert "topological_feature_ablations" in data
    assert "boundary_operator_ablations" in data
    assert "cycle_length_sensitivity" in data
    assert "computational_scalability" in data

    meta = data["evaluation_metadata"]
    assert meta["evaluation_split"] == "validation"
    assert meta["test_split_access"] == "ZERO_ACCESS_CONFIRMED"
    assert meta["status"] == "VERIFIED_NO_LEAKAGE"
    assert meta["seeds"] == [42, 43, 44, 45, 46]


def test_ablation_evidence_zero_test_leakage():
    """Verify test split cryptographic isolation across ablation evidence."""
    verifier = AblationEvidenceVerifier()
    test_ids = verifier.get_test_candidate_ids()

    assert len(test_ids) > 0

    # Synthetic validation candidate IDs
    val_ids = [f"cand_val_{i}" for i in range(100)]
    is_clean, leak_msgs = verifier.verify_zero_test_leakage(val_ids)

    assert is_clean is True
    assert len(leak_msgs) == 0

    # Contaminated candidate IDs must be rejected
    contaminated_ids = list(val_ids) + list(test_ids)[:2]
    is_clean_bad, leak_msgs_bad = verifier.verify_zero_test_leakage(contaminated_ids)
    assert is_clean_bad is False
    assert len(leak_msgs_bad) > 0


def test_ablation_verifier_execution(tmp_path):
    """Verify AblationEvidenceVerifier execution and package compilation."""
    out_file = str(tmp_path / "test_ablation_summary.json")
    verifier = AblationEvidenceVerifier()

    res = verifier.generate_and_save_summary(output_path=out_file)

    assert os.path.exists(out_file)
    assert res["evaluation_metadata"]["test_split_sha256"] == ProtocolLock.LOCKED_TEST_HASH
    assert "full" in res["topological_feature_ablations"]
    assert "ccnn_full_2cell" in res["boundary_operator_ablations"]
    assert "TopoRingNet" in res["cycle_length_sensitivity"]
