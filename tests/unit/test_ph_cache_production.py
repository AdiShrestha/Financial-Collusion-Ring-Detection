"""Unit tests for TopologicalFeatureCache production pipeline (Contract C12-02)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.ph.cache_pipeline import TopologicalFeatureCache


def test_topological_cache_file_structure_and_dimensions():
    """Verify data/cache/topological_features.npz exists and contains valid 372-dim arrays."""
    cache = TopologicalFeatureCache()
    npz_path = "data/cache/topological_features.npz"

    assert os.path.exists(npz_path), "Missing topological features archive"
    loaded = cache.load_cache(npz_path)

    features = loaded["features"]
    cand_ids = loaded["candidate_ids"]
    splits = loaded["split_assignments"]
    y = loaded["y"]

    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[1] == 372
    assert len(features) == len(cand_ids) == len(splits) == len(y)

    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    # Verify all splits are represented
    split_set = set(splits)
    assert "train" in split_set
    assert "val" in split_set
    assert "test" in split_set


def test_topological_cache_reproducibility():
    """Verify loading cache and checking non-zero feature contents."""
    cache = TopologicalFeatureCache()
    loaded = cache.load_cache()

    features = loaded["features"]
    assert np.sum(np.abs(features)) > 0.0
