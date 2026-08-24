"""Unit tests for topological feature vectorizers."""

import numpy as np
import pytest

from source.ph.persistence_extractor import PersistenceDiagram
from source.ph.vectorizer import (
    PersistenceVectorizer,
    compute_betti_curve,
    compute_persistence_image,
    compute_persistence_landscape,
    compute_persistent_entropy,
    compute_summary_stats,
)


def test_betti_curve_shape_and_values():
    """Verify Betti curve non-negative values and discretization bin count."""
    intervals = np.array([[10.0, 30.0], [20.0, 40.0]], dtype=np.float32)
    curve = compute_betti_curve(intervals, num_bins=50, grid_min=0.0, grid_max=50.0)

    assert curve.shape == (50,)
    assert np.all(curve >= 0.0)
    # Peak at t in [20, 30] should be 2.0 (both intervals active)
    assert np.max(curve) == 2.0


def test_persistence_landscape_monotonicity():
    """Verify that landscape functions satisfy lambda_1(t) >= lambda_2(t) >= lambda_3(t)."""
    intervals = np.array([[5.0, 25.0], [10.0, 30.0], [15.0, 35.0]], dtype=np.float32)
    landscapes_flat = compute_persistence_landscape(
        intervals, num_landscapes=3, num_bins=30, grid_min=0.0, grid_max=40.0
    )
    assert landscapes_flat.shape == (90,)
    landscapes = landscapes_flat.reshape(3, 30)

    # Monotonicity check
    for t_idx in range(30):
        assert landscapes[0, t_idx] >= landscapes[1, t_idx] - 1e-6
        assert landscapes[1, t_idx] >= landscapes[2, t_idx] - 1e-6
        assert landscapes[2, t_idx] >= 0.0


def test_persistence_image_non_negative():
    """Verify that 2D Gaussian persistence image is non-negative and finite."""
    intervals = np.array([[10.0, 40.0]], dtype=np.float32)
    img_flat = compute_persistence_image(intervals, resolution=(10, 10), sigma=2.0)

    assert img_flat.shape == (100,)
    assert np.all(img_flat >= 0.0)
    assert not np.isnan(img_flat).any()
    assert np.sum(img_flat) > 0.0


def test_vectorizer_fixed_output_dimension():
    """Verify that diagrams with 0, 1, and 10 intervals produce identical output dimension without NaNs."""
    vectorizer = PersistenceVectorizer(
        betti_bins=20,
        landscape_levels=3,
        landscape_bins=20,
        image_res=(10, 10),
    )
    # Expected dim = (20 + 60 + 100 + 6) * 2 = 186 * 2 = 372
    assert vectorizer.output_dim == 372

    # Case 1: Empty diagram (0 intervals)
    diag_empty = PersistenceDiagram(
        candidate_id="empty",
        h0_intervals=np.zeros((0, 2)),
        h1_intervals=np.zeros((0, 2)),
        max_filtration=0.0,
    )
    vec_empty = vectorizer.vectorize(diag_empty)
    assert vec_empty.shape == (372,)
    assert not np.isnan(vec_empty).any()
    assert np.all(vec_empty == 0.0)

    # Case 2: Single cycle diagram
    diag_single = PersistenceDiagram(
        candidate_id="single",
        h0_intervals=np.array([[0.0, 10.0], [0.0, 20.0]]),
        h1_intervals=np.array([[15.0, 25.0]]),
        max_filtration=25.0,
    )
    vec_single = vectorizer.vectorize(diag_single)
    assert vec_single.shape == (372,)
    assert not np.isnan(vec_single).any()
    assert np.sum(np.abs(vec_single)) > 0.0

    # Case 3: Dense multi-interval diagram
    dense_h0 = np.array([[float(i), float(i + 5)] for i in range(10)])
    dense_h1 = np.array([[float(i + 2), float(i + 8)] for i in range(5)])
    diag_dense = PersistenceDiagram(
        candidate_id="dense",
        h0_intervals=dense_h0,
        h1_intervals=dense_h1,
        max_filtration=20.0,
    )
    vec_dense = vectorizer.vectorize(diag_dense)
    assert vec_dense.shape == (372,)
    assert not np.isnan(vec_dense).any()
