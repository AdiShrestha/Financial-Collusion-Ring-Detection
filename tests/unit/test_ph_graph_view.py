"""Unit tests for PHGraphView persistent homology 1-skeleton filtration constructor."""

import pytest

from source.data.candidate_extractor import CandidateExample
from source.ph.ph_graph_view import PHGraphView


def create_sample_candidate() -> CandidateExample:
    return CandidateExample(
        candidate_id="cand_ph_001",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 40.0),
        participant_ids=["A", "B", "C", "D"],
        edges=[
            ("A", "B", {"amount": 100.0, "timestamp": 10.0}),
            ("B", "C", {"amount": 90.0, "timestamp": 20.0}),
            ("C", "D", {"amount": 80.0, "timestamp": 30.0}),
            ("D", "A", {"amount": 70.0, "timestamp": 40.0}),
        ],
        node_features={n: [0.0] * 56 for n in ["A", "B", "C", "D"]},
        target_y=1,
        typology_label="CYCLE",
        group_id="group_ph",
    )


def test_ph_graph_view_monotonicity():
    """Assert monotonic filtration values: filt(e) >= max(filt(u), filt(v))."""
    cand = create_sample_candidate()
    ph_view = PHGraphView.from_candidate_example(cand, filtration_type="temporal")

    assert ph_view.num_nodes == 4
    assert ph_view.num_edges == 4
    assert ph_view.num_simplices == 8  # 4 vertices + 4 edges

    simplex_map = {tuple(s[0]): s[1] for s in ph_view.simplices}

    # Check that each edge appears at or after its endpoint vertices
    for s_nodes, t_e in ph_view.simplices:
        if len(s_nodes) == 2:
            u, v = s_nodes
            t_u = simplex_map[(u,)]
            t_v = simplex_map[(v,)]
            assert t_e >= t_u - 1e-9
            assert t_e >= t_v - 1e-9


def test_ph_graph_view_no_2simplices_invariant():
    """Verify that attempting to construct a PHGraphView with 2-simplices raises ValueError (INV-003)."""
    cand = create_sample_candidate()
    ph_view = PHGraphView.from_candidate_example(cand)

    # Attempt to inject a 2-simplex into PHGraphView constructor
    invalid_simplices = ph_view.simplices + [(["A", "B", "C"], 50.0)]
    with pytest.raises(ValueError, match="2-simplex or higher-dimensional cell found in PHGraphView"):
        PHGraphView(
            candidate_id="cand_invalid",
            nodes=ph_view.nodes,
            edges=ph_view.edges,
            filtration_type="temporal",
            simplices=invalid_simplices,
            target_y=1,
        )


def test_alternative_filtration_types():
    """Verify amount-based and degree-based filtrations."""
    cand = create_sample_candidate()

    # Amount filtration
    ph_amt = PHGraphView.from_candidate_example(cand, filtration_type="amount")
    assert ph_amt.filtration_type == "amount"
    assert ph_amt.num_simplices == 8
    # Edge weights should be [100, 90, 80, 70]
    edge_amts = [s[1] for s in ph_amt.simplices if len(s[0]) == 2]
    assert sorted(edge_amts) == [70.0, 80.0, 90.0, 100.0]

    # Degree filtration
    ph_deg = PHGraphView.from_candidate_example(cand, filtration_type="degree")
    assert ph_deg.filtration_type == "degree"
    assert ph_deg.num_simplices == 8
    for s_nodes, deg_val in ph_deg.simplices:
        assert deg_val == 2.0  # In a 4-cycle, all nodes have degree 2
