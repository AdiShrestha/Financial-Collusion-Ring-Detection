"""Unit tests for CycleCellView polygonal cell complex lifter."""

import numpy as np
import pytest

from source.data.candidate_extractor import CandidateExample
from source.topology.cycle_cell_view import CycleCellView
from source.topology.oracles import (
    compute_betti_numbers,
    verify_boundary_nilpotence,
    verify_hodge_laplacian_properties,
)


def create_k_cycle_candidate(k: int) -> CandidateExample:
    nodes = [f"node_{i}" for i in range(k)]
    edges = [(nodes[i], nodes[(i + 1) % k], {"amount": 10.0, "timestamp": float(i + 1)}) for i in range(k)]
    return CandidateExample(
        candidate_id=f"cand_{k}cycle_001",
        dataset_track="amlworld",
        temporal_bounds=(1.0, float(k)),
        participant_ids=nodes,
        edges=edges,
        node_features={n: [0.0] * 56 for n in nodes},
        target_y=1,
        typology_label="CYCLE",
        group_id=f"group_{k}cycle",
        metadata={"cycle_nodes": nodes, "cycle_length": k},
    )


def test_boundary_nilpotence_cellular_arbitrary_k():
    """Verify algebraic nilpotence B1 @ B2 == 0 on polygonal cell complexes for k in {3, 4, 5, 6}."""
    for k in (3, 4, 5, 6):
        cand = create_k_cycle_candidate(k)
        cv = CycleCellView.from_candidate_example(cand, min_k=3, max_k=6)

        assert cv.num_nodes == k
        assert cv.num_edges == k
        assert cv.num_cells_2 == 1

        # Verify nilpotence
        assert verify_boundary_nilpotence(cv.B1, cv.B2) is True
        prod = np.dot(cv.B1, cv.B2)
        assert np.max(np.abs(prod)) == 0.0


def test_polygonal_cell_complex_4cycle_structure():
    """Verify that a 4-cycle is represented as a rank-2 cell with 4 boundary edges (INV-002)."""
    cand = create_k_cycle_candidate(4)
    cv = CycleCellView.from_candidate_example(cand)

    assert cv.num_cells_2 == 1
    assert len(cv.cells_2[0]) == 4

    # B2 matrix must have shape (|E|, |C_2|) = (4, 1)
    assert cv.B2.shape == (4, 1)
    col = cv.B2[:, 0]
    # Exactly 4 nonzero entries in the column
    nonzeros = np.nonzero(col)[0]
    assert len(nonzeros) == 4
    # All entries are either +1 or -1
    for val in col:
        assert val in (-1.0, 1.0)


def test_hodge_laplacian_cellular_properties():
    """Verify symmetry and PSD spectrum of cellular Hodge Laplacians."""
    cand = create_k_cycle_candidate(5)
    cv = CycleCellView.from_candidate_example(cand)

    # L0
    is_sym_0, is_psd_0, _ = verify_hodge_laplacian_properties(cv.L0)
    assert is_sym_0 is True
    assert is_psd_0 is True

    # L1
    is_sym_1, is_psd_1, _ = verify_hodge_laplacian_properties(cv.L1)
    assert is_sym_1 is True
    assert is_psd_1 is True


def test_filled_cell_betti_number():
    """Verify that a filled polygonal 4-cell has beta_1 = 0."""
    cand = create_k_cycle_candidate(4)
    cv = CycleCellView.from_candidate_example(cand)

    # Filled 4-cell: beta_0 = 1, beta_1 = 0
    beta_0, beta_1 = compute_betti_numbers(cv.B1, cv.B2)
    assert beta_0 == 1
    assert beta_1 == 0
