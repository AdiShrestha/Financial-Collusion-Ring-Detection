"""Unit tests for CliqueSimplicialView simplicial complex lifter."""

import numpy as np
import pytest

from source.data.candidate_extractor import CandidateExample
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.oracles import (
    compute_betti_numbers,
    verify_boundary_nilpotence,
    verify_hodge_laplacian_properties,
)


def create_triangle_candidate() -> CandidateExample:
    return CandidateExample(
        candidate_id="cand_tri_001",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 10.0),
        participant_ids=["A", "B", "C"],
        edges=[
            ("A", "B", {"amount": 100.0, "timestamp": 1.0}),
            ("B", "C", {"amount": 100.0, "timestamp": 2.0}),
            ("C", "A", {"amount": 100.0, "timestamp": 3.0}),
        ],
        node_features={"A": [1.0] * 56, "B": [1.0] * 56, "C": [1.0] * 56},
        target_y=1,
        typology_label="CYCLE",
        group_id="group_tri",
    )


def create_square_4cycle_candidate() -> CandidateExample:
    return CandidateExample(
        candidate_id="cand_sq_001",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 10.0),
        participant_ids=["1", "2", "3", "4"],
        edges=[
            ("1", "2", {"amount": 50.0, "timestamp": 1.0}),
            ("2", "3", {"amount": 50.0, "timestamp": 2.0}),
            ("3", "4", {"amount": 50.0, "timestamp": 3.0}),
            ("4", "1", {"amount": 50.0, "timestamp": 4.0}),
        ],
        node_features={str(i): [0.0] * 56 for i in range(1, 5)},
        target_y=1,
        typology_label="CYCLE",
        group_id="group_sq",
    )


def test_boundary_nilpotence_simplicial():
    """Verify B1 @ B2 == 0 on lifted simplicial triangle complex."""
    cand = create_triangle_candidate()
    sv = CliqueSimplicialView.from_candidate_example(cand)

    assert sv.num_nodes == 3
    assert sv.num_edges == 3
    assert sv.num_faces_2 == 1  # 1 filled 2-simplex
    assert sv.faces_2 == [("A", "B", "C")]

    # Assert exact nilpotence
    assert verify_boundary_nilpotence(sv.B1, sv.B2) is True
    prod = np.dot(sv.B1, sv.B2)
    assert np.max(np.abs(prod)) == 0.0


def test_simplicial_closure_enforcement():
    """Verify that 2-simplices are formed only when all boundary edges exist."""
    # Construct open 2-path A-B-C (missing A-C edge)
    open_cand = CandidateExample(
        candidate_id="cand_open_001",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 5.0),
        participant_ids=["A", "B", "C"],
        edges=[
            ("A", "B", {"amount": 10.0, "timestamp": 1.0}),
            ("B", "C", {"amount": 10.0, "timestamp": 2.0}),
        ],
        node_features={"A": [0.0] * 56, "B": [0.0] * 56, "C": [0.0] * 56},
        target_y=0,
        typology_label="NEGATIVE_CANDIDATE",
        group_id="group_open",
    )
    sv = CliqueSimplicialView.from_candidate_example(open_cand)
    assert sv.num_faces_2 == 0  # No 2-simplex formed because A-C is absent


def test_hodge_laplacian_simplicial_spectrum():
    """Verify that L0 and L1 are symmetric and positive semi-definite."""
    cand = create_triangle_candidate()
    sv = CliqueSimplicialView.from_candidate_example(cand)

    # L0 is |V| x |V| (3 x 3)
    is_sym_0, is_psd_0, eig_0 = verify_hodge_laplacian_properties(sv.L0)
    assert is_sym_0 is True
    assert is_psd_0 is True

    # L1 is |E| x |E| (3 x 3)
    is_sym_1, is_psd_1, eig_1 = verify_hodge_laplacian_properties(sv.L1)
    assert is_sym_1 is True
    assert is_psd_1 is True


def test_inv_002_4cycle_simplicial_exclusion():
    """Verify that an unfilled 4-cycle has 0 2-simplices in CliqueSimplicialView."""
    cand = create_square_4cycle_candidate()
    sv = CliqueSimplicialView.from_candidate_example(cand)

    # 4 nodes, 4 edges, but 0 2-simplices (because it has no 3-cliques)
    assert sv.num_nodes == 4
    assert sv.num_edges == 4
    assert sv.num_faces_2 == 0

    # In simplicial complex, unfilled 4-cycle has beta_1 = 1
    beta_0, beta_1 = compute_betti_numbers(sv.B1, sv.B2)
    assert beta_0 == 1
    assert beta_1 == 1
