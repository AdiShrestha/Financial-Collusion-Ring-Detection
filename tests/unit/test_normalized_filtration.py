"""Unit tests for NormalizedPersistenceVectorizer (Contract C12-01)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.ph.normalized_filtration import NormalizedPersistenceVectorizer


def test_vectorizer_output_dimension_and_finite_values():
    """Verify output shape is strictly (372,) and contains no NaNs or Infs."""
    cand = {
        "candidate_id": "test_c1",
        "nodes": ["A", "B", "C", "D"],
        "transactions": [
            {"from_account": "A", "to_account": "B", "timestamp_epoch": 100.0},
            {"from_account": "B", "to_account": "C", "timestamp_epoch": 200.0},
            {"from_account": "C", "to_account": "D", "timestamp_epoch": 300.0},
            {"from_account": "D", "to_account": "A", "timestamp_epoch": 400.0},
        ],
    }

    vectorizer = NormalizedPersistenceVectorizer()
    vec = vectorizer.vectorize_candidate(cand)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (372,)
    assert vec.dtype == np.float32
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_h1_cycle_detection_in_unfilled_complex():
    """Verify H1 cycle produces positive Betti_1 entry on cycle formation."""
    cand_cycle = {
        "candidate_id": "cycle_3",
        "nodes": ["A", "B", "C"],
        "transactions": [
            {"from_account": "A", "to_account": "B", "timestamp_epoch": 100.0},
            {"from_account": "B", "to_account": "C", "timestamp_epoch": 200.0},
            {"from_account": "C", "to_account": "A", "timestamp_epoch": 300.0},
        ],
    }

    vectorizer = NormalizedPersistenceVectorizer()
    diag = vectorizer.extract_diagrams(cand_cycle)

    # Must contain exactly 1 H1 cycle
    assert len(diag["h1_raw"]) == 1
    birth, death = diag["h1_raw"][0]
    # Edge C->A added at normalized filtration 1.0 creates the cycle
    assert abs(birth - 1.0) < 1e-3
    assert np.isinf(death) or death >= 1e6


def test_zero_duration_candidate_handling():
    """Verify candidate with identical timestamps defaults smoothly without division by zero."""
    cand_instant = {
        "candidate_id": "instant_c",
        "nodes": ["X", "Y", "Z"],
        "transactions": [
            {"from_account": "X", "to_account": "Y", "timestamp_epoch": 500.0},
            {"from_account": "Y", "to_account": "Z", "timestamp_epoch": 500.0},
        ],
    }

    vectorizer = NormalizedPersistenceVectorizer()
    vec = vectorizer.vectorize_candidate(cand_instant)

    assert vec.shape == (372,)
    assert not np.isnan(vec).any()
