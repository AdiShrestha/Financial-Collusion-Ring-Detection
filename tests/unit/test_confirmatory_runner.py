"""Unit tests for Confirmatory Test Set Prediction & Execution Pipeline."""

import json
import os
import pytest

from source.evidence.protocol_lock import ProtocolLock
from source.experiments.confirmatory_runner import ConfirmatoryPredictionRunner


def test_test_split_hash_verification():
    """Verify that runner verifies SHA-256 hash before running inference."""
    runner = ConfirmatoryPredictionRunner()
    is_valid, hash_val = runner.verify_test_checksum()
    assert is_valid is True
    assert hash_val == ProtocolLock.LOCKED_TEST_HASH


def test_prediction_archive_schema_and_persistence():
    """Verify runs/confirmatory/predictions.json schema and persistence."""
    pred_path = "runs/confirmatory/predictions.json"
    assert os.path.exists(pred_path), f"{pred_path} not found"

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "predictions" in data

    meta = data["metadata"]
    assert meta["evaluation_split"] == "test"
    assert meta["test_split_sha256"] in ProtocolLock.LOCKED_TEST_HASHES
    assert meta["seeds"] == [42, 43, 44, 45, 46]
    assert len(meta["models_evaluated"]) == 6

    preds = data["predictions"]
    for model_name in ["GCNBaseline", "GATBaseline", "GraphSAGEBaseline", "SimplicialComplexNet", "CellularComplexNet", "TopoRingNet"]:
        assert model_name in preds
        for seed_str in ["42", "43", "44", "45", "46"]:
            assert seed_str in preds[model_name]
            seed_list = preds[model_name][seed_str]
            assert len(seed_list) == meta["num_test_candidates"]
            for item in seed_list:
                assert "candidate_id" in item
                assert "group_id" in item
                assert "y_true" in item
                assert "y_prob" in item
                assert "typology" in item
                assert "dataset_track" in item


def test_prediction_probabilities_bounded():
    """Verify that all predicted probabilities in archive are bounded in [0, 1] without NaNs."""
    pred_path = "runs/confirmatory/predictions.json"
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for model_name, seed_dict in data["predictions"].items():
        for seed_str, cand_list in seed_dict.items():
            for item in cand_list:
                p = item["y_prob"]
                assert isinstance(p, (float, int))
                assert not (p != p)  # Not NaN
                assert 0.0 <= p <= 1.0, f"Probability {p} out of bounds in {model_name} seed {seed_str}"
