"""Unit tests for CandidateDatasetManager (Contract C11-04)."""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.candidate_dataset import CandidateDatasetManager, compute_split_hash


def test_candidate_serialization_and_deserialization():
    """Verify writing and reading candidate JSONL records."""
    cands = [
        {"candidate_id": "c1", "nodes": ["A", "B", "C"], "edges": [("A", "B"), ("B", "C"), ("C", "A")], "label": 1},
        {"candidate_id": "c2", "nodes": ["D", "E", "F"], "edges": [("D", "E"), ("E", "F"), ("F", "D")], "label": 0},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "test_candidates.jsonl")
        mgr = CandidateDatasetManager()
        n_written = mgr.serialize_candidates(cands, output_path=jsonl_path)
        assert n_written == 2

        loaded = mgr.load_candidates(input_path=jsonl_path)
        assert len(loaded) == 2
        assert loaded[0]["candidate_id"] == "c1"
        assert loaded[0]["label"] == 1
        assert loaded[1]["candidate_id"] == "c2"
        assert loaded[1]["label"] == 0


def test_group_safe_split_zero_account_overlap():
    """Verify Group-Safe splitting guarantees exact 0.0% account overlap across train, val, test."""
    cands = []
    for i in range(20):
        # Disjoint candidates
        nodes = [f"ACC_{i}_{j}" for j in range(3)]
        cands.append({
            "candidate_id": f"c_{i}",
            "group_id": f"grp_{i}",
            "nodes": nodes,
            "participants": nodes,
            "edges": [[nodes[0], nodes[1]], [nodes[1], nodes[2]], [nodes[2], nodes[0]]],
            "label": i % 2,
        })

    mgr = CandidateDatasetManager(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    train, val, test, leakage_info = mgr.group_safe_split(cands)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert leakage_info["is_disjoint"] is True
    assert leakage_info["train_val_overlap_count"] == 0
    assert leakage_info["train_test_overlap_count"] == 0
    assert leakage_info["val_test_overlap_count"] == 0


def test_split_manifest_export_and_hashes():
    """Verify split_manifest.json contains valid partition hashes and schema fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "candidates.jsonl")
        manifest_path = os.path.join(tmpdir, "split_manifest.json")

        mgr = CandidateDatasetManager()
        manifest = mgr.generate_and_export_cohort(output_jsonl=jsonl_path, output_manifest=manifest_path)

        assert os.path.exists(jsonl_path)
        assert os.path.exists(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["metadata"]["total_candidates"] >= 20
        assert len(data["splits"]["test"]["sha256_checksum"]) == 64
        assert data["disjointness_audit"]["is_disjoint"] is True
