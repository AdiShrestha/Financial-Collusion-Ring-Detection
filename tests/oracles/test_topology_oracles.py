"""Semantic test suite for algebraic topology oracles and boundary operators."""

import numpy as np
import pytest

from source.topology.incidence import (
    compute_boundary_matrix_b1,
    compute_boundary_matrix_b2,
    compute_hodge_laplacian_0,
    compute_hodge_laplacian_1,
)
from source.topology.oracles import (
    compute_betti_numbers,
    verify_boundary_nilpotence,
    verify_hodge_laplacian_properties,
)


def test_boundary_nilpotence_simplicial_triangle():
    """Test boundary nilpotence B1 @ B2 == 0 on a 3-node 2-simplex (triangle)."""
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (0, 2)]
    # Oriented 2-simplex boundary: (0, 1) + (1, 2) - (0, 2)
    faces = [[0, 1, 2]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2 = compute_boundary_matrix_b2(edges, faces)

    assert b1.shape == (3, 3)
    assert b2.shape == (3, 1)
    # Check expected entries in B2
    assert b2[0, 0] == 1.0   # (0, 1) aligned
    assert b2[1, 0] == 1.0   # (1, 2) aligned
    assert b2[2, 0] == -1.0  # (0, 2) opposite

    assert verify_boundary_nilpotence(b1, b2), f"B1 @ B2 must be 0, got {b1 @ b2}"
    assert np.allclose(b1 @ b2, 0.0)


def test_boundary_nilpotence_cellular_square():
    """Test boundary nilpotence B1 @ B2 == 0 on a 4-node polygonal cell (square)."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    cells = [[0, 1, 2, 3]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2 = compute_boundary_matrix_b2(edges, cells)

    assert b1.shape == (4, 4)
    assert b2.shape == (4, 1)
    # Cell boundary has exactly 4 nonzero entries (INV-002)
    assert np.count_nonzero(b2) == 4

    assert verify_boundary_nilpotence(b1, b2), f"B1 @ B2 must be 0 for 4-gon, got {b1 @ b2}"
    assert np.allclose(b1 @ b2, 0.0)


def test_boundary_nilpotence_pentagon_cell():
    """Test boundary nilpotence B1 @ B2 == 0 on a 5-node polygonal cell (pentagon)."""
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    cells = [[0, 1, 2, 3, 4]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2 = compute_boundary_matrix_b2(edges, cells)

    assert b1.shape == (5, 5)
    assert b2.shape == (5, 1)
    assert np.count_nonzero(b2) == 5

    assert verify_boundary_nilpotence(b1, b2), f"B1 @ B2 must be 0 for 5-gon, got {b1 @ b2}"
    assert np.allclose(b1 @ b2, 0.0)


def test_boundary_nilpotence_hexagon_cell():
    """Test boundary nilpotence B1 @ B2 == 0 on a 6-node polygonal cell (hexagon)."""
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    cells = [[0, 1, 2, 3, 4, 5]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2 = compute_boundary_matrix_b2(edges, cells)

    assert b1.shape == (6, 6)
    assert b2.shape == (6, 1)
    assert np.count_nonzero(b2) == 6

    assert verify_boundary_nilpotence(b1, b2), f"B1 @ B2 must be 0 for 6-gon, got {b1 @ b2}"
    assert np.allclose(b1 @ b2, 0.0)


def test_betti_unfilled_vs_filled_triangle():
    """Semantic test: Unfilled 1-skeleton cycle has beta_1=1; filled 2-cell has beta_1=0."""
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (2, 0)]
    faces = [[0, 1, 2]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2_empty = np.zeros((3, 0))
    b2_filled = compute_boundary_matrix_b2(edges, faces)

    # 1. Unfilled 1-skeleton: 1 connected component (beta_0=1), 1 independent cycle (beta_1=1)
    beta0_unfilled, beta1_unfilled = compute_betti_numbers(b1, b2_empty)
    assert beta0_unfilled == 1
    assert beta1_unfilled == 1

    # 2. Filled 2-simplex: 1 connected component (beta_0=1), 0 uncontractible cycles (beta_1=0)
    beta0_filled, beta1_filled = compute_betti_numbers(b1, b2_filled)
    assert beta0_filled == 1
    assert beta1_filled == 0


def test_betti_unfilled_vs_filled_square():
    """Semantic test: Unfilled square has beta_1=1; filled square has beta_1=0."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    cells = [[0, 1, 2, 3]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2_empty = np.zeros((4, 0))
    b2_filled = compute_boundary_matrix_b2(edges, cells)

    beta0_unfilled, beta1_unfilled = compute_betti_numbers(b1, b2_empty)
    assert beta0_unfilled == 1
    assert beta1_unfilled == 1

    beta0_filled, beta1_filled = compute_betti_numbers(b1, b2_filled)
    assert beta0_filled == 1
    assert beta1_filled == 0


def test_hodge_laplacian_symmetry_and_psd():
    """Test symmetry and positive semi-definiteness of L0 and L1 Laplacians."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    faces = [[0, 1, 2], [0, 2, 3]]

    b1 = compute_boundary_matrix_b1(nodes, edges)
    b2 = compute_boundary_matrix_b2(edges, faces)

    # Verify nilpotence
    assert verify_boundary_nilpotence(b1, b2)

    l0 = compute_hodge_laplacian_0(b1)
    l1 = compute_hodge_laplacian_1(b1, b2)

    # Check L0
    is_sym_0, is_psd_0, eig_0 = verify_hodge_laplacian_properties(l0)
    assert is_sym_0, "L0 must be symmetric"
    assert is_psd_0, f"L0 must be PSD, min eigenvalue: {np.min(eig_0)}"

    # Check L1
    is_sym_1, is_psd_1, eig_1 = verify_hodge_laplacian_properties(l1)
    assert is_sym_1, "L1 must be symmetric"
    assert is_psd_1, f"L1 must be PSD, min eigenvalue: {np.min(eig_1)}"


def test_invalid_boundary_dimensions():
    """Test that dimension mismatches and malformed inputs raise explicit ValueErrors."""
    with pytest.raises(ValueError, match="unknown node"):
        compute_boundary_matrix_b1([0, 1], [(0, 2)])

    with pytest.raises(ValueError, match="at least 3 vertices"):
        compute_boundary_matrix_b2([(0, 1)], [[0, 1]])

    with pytest.raises(ValueError, match="not found in edge list"):
        compute_boundary_matrix_b2([(0, 1), (1, 2)], [[0, 1, 2]])
