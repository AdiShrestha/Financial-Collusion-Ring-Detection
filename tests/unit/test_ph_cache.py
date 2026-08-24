"""Unit tests for topological feature caching pipeline and split disjointness verification."""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pytest

from source.data.candidate_extractor import CandidateExample
from source.ph.cache_pipeline import TopologicalFeatureCache
from source.ph.persistence_extractor import PersistenceExtractor
from source.ph.ph_graph_view import PHGraphView
from source.ph.vectorizer import PersistenceVectorizer


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_disjoint_cohort() -> Tuple[list[CandidateExample], dict]:
    candidates = []
    splits = {"train": [], "val": [], "test": []}

    # Train candidates: accounts T0..T9
    for i in range(4):
        cid = f"train_c_{i}"
        nodes = [f"T_{i*2}", f"T_{i*2+1}"]
        cand = CandidateExample(
            candidate_id=cid,
            dataset_track="amlworld",
            temporal_bounds=(0.0, 10.0),
            participant_ids=nodes,
            edges=[(nodes[0], nodes[1], {"amount": 100.0, "timestamp": 5.0})],
            node_features={n: [0.0] * 56 for n in nodes},
            target_y=1,
            typology_label="CYCLE",
            group_id=f"g_train_{i}",
            metadata={"split": "train"},
        )
        candidates.append(cand)
        splits["train"].append(cid)

    # Val candidates: accounts V0..V3
    for i in range(2):
        cid = f"val_c_{i}"
        nodes = [f"V_{i*2}", f"V_{i*2+1}"]
        cand = CandidateExample(
            candidate_id=cid,
            dataset_track="amlworld",
            temporal_bounds=(0.0, 10.0),
            participant_ids=nodes,
            edges=[(nodes[0], nodes[1], {"amount": 50.0, "timestamp": 5.0})],
            node_features={n: [0.0] * 56 for n in nodes},
            target_y=0,
            typology_label="NEGATIVE_CANDIDATE",
            group_id=f"g_val_{i}",
            metadata={"split": "val"},
        )
        candidates.append(cand)
        splits["val"].append(cid)

    # Test candidates: accounts S0..S3
    for i in range(2):
        cid = f"test_c_{i}"
        nodes = [f"S_{i*2}", f"S_{i*2+1}"]
        cand = CandidateExample(
            candidate_id=cid,
            dataset_track="amlworld",
            temporal_bounds=(0.0, 10.0),
            participant_ids=nodes,
            edges=[(nodes[0], nodes[1], {"amount": 75.0, "timestamp": 5.0})],
            node_features={n: [0.0] * 56 for n in nodes},
            target_y=1,
            typology_label="CYCLE",
            group_id=f"g_test_{i}",
            metadata={"split": "test"},
        )
        candidates.append(cand)
        splits["test"].append(cid)

    return candidates, {"splits": splits}


def test_cache_generation_and_loading(temp_cache_dir):
    """Verify building and reloading feature cache with exact array fidelity."""
    candidates, split_manifest_data = create_disjoint_cohort()
    manifest_path = os.path.join(temp_cache_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest_data, f)

    cache = TopologicalFeatureCache()
    cache_meta = cache.build_and_save_cache(
        candidates=candidates,
        split_manifest_path=manifest_path,
        cache_dir=temp_cache_dir,
    )

    assert cache_meta["num_candidates"] == 8
    assert cache_meta["feature_dim"] == 372
    assert cache_meta["split_counts"]["train"] == 4
    assert cache_meta["split_counts"]["val"] == 2
    assert cache_meta["split_counts"]["test"] == 2

    # Load back
    loaded = TopologicalFeatureCache.load_cache(cache_dir=temp_cache_dir)
    assert loaded["x_topo"].shape == (8, 372)
    assert len(loaded["y"]) == 8
    assert len(loaded["candidate_ids"]) == 8
    assert len(loaded["split_assignments"]) == 8


def test_cache_manifest_split_disjointness(temp_cache_dir):
    """Verify that split leakage raises explicit error on overlapping accounts (INV-006)."""
    # Create candidates with overlapping account between train and val
    cand1 = CandidateExample(
        candidate_id="c1",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 10.0),
        participant_ids=["LEAK_ACCOUNT", "B"],
        edges=[("LEAK_ACCOUNT", "B", {"amount": 10.0, "timestamp": 5.0})],
        node_features={"LEAK_ACCOUNT": [0.0]*56, "B": [0.0]*56},
        target_y=1,
        typology_label="CYCLE",
        group_id="g1",
        metadata={"split": "train"},
    )
    cand2 = CandidateExample(
        candidate_id="c2",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 10.0),
        participant_ids=["LEAK_ACCOUNT", "C"],
        edges=[("LEAK_ACCOUNT", "C", {"amount": 10.0, "timestamp": 5.0})],
        node_features={"LEAK_ACCOUNT": [0.0]*56, "C": [0.0]*56},
        target_y=0,
        typology_label="NEGATIVE_CANDIDATE",
        group_id="g2",
        metadata={"split": "val"},
    )

    cache = TopologicalFeatureCache()
    with pytest.raises(ValueError, match="Split leakage detected during caching"):
        cache.build_and_save_cache([cand1, cand2], cache_dir=temp_cache_dir)


def test_cache_checksum_verification(temp_cache_dir):
    """Verify SHA-256 checksum detection on corrupted cache archive."""
    candidates, split_manifest_data = create_disjoint_cohort()
    manifest_path = os.path.join(temp_cache_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest_data, f)

    cache = TopologicalFeatureCache()
    cache.build_and_save_cache(candidates, split_manifest_path=manifest_path, cache_dir=temp_cache_dir)

    # Corrupt npz file
    npz_path = os.path.join(temp_cache_dir, "topological_features.npz")
    with open(npz_path, "ab") as f:
        f.write(b"corrupted_bytes")

    with pytest.raises(ValueError, match="Cache checksum mismatch"):
        TopologicalFeatureCache.load_cache(cache_dir=temp_cache_dir)


def test_semantic_ph_cache_consistency(temp_cache_dir):
    """Semantic test verifying cached feature vector matches direct live vectorization."""
    candidates, _ = create_disjoint_cohort()
    target_cand = candidates[0]

    cache = TopologicalFeatureCache()
    cache.build_and_save_cache([target_cand], cache_dir=temp_cache_dir)
    loaded = TopologicalFeatureCache.load_cache(cache_dir=temp_cache_dir)

    # Compute directly
    ph_view = PHGraphView.from_candidate_example(target_cand)
    diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)
    expected_vec = PersistenceVectorizer().vectorize(diag)

    np.testing.assert_allclose(loaded["x_topo"][0], expected_vec, atol=1e-6)
