"""Unit tests for higher-order topological feature aggregators."""

import numpy as np
import pytest

from source.data.candidate_extractor import CandidateExample
from source.features.cell_features import extract_cell_features, extract_batch_cell_features
from source.features.edge_features import extract_edge_features
from source.features.node_features import extract_node_features


def create_sample_candidate() -> CandidateExample:
    return CandidateExample(
        candidate_id="cand_feat_001",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 40.0),
        participant_ids=["N0", "N1", "N2", "N3"],
        edges=[
            ("N0", "N1", {"amount": 100.0, "timestamp": 10.0}),
            ("N1", "N2", {"amount": 95.0, "timestamp": 20.0}),
            ("N2", "N3", {"amount": 90.0, "timestamp": 30.0}),
            ("N3", "N0", {"amount": 85.0, "timestamp": 40.0}),
        ],
        node_features={
            "N0": [1.0] * 56,
            "N1": [2.0] * 56,
            "N2": [3.0] * 56,
            "N3": [4.0] * 56,
        },
        target_y=1,
        typology_label="CYCLE",
        group_id="group_feat",
    )


def test_node_and_edge_feature_shapes():
    """Verify node and edge feature matrix dimensions and non-empty content."""
    cand = create_sample_candidate()

    # Node features: 4 nodes x (56 + 4 structural) = (4, 60)
    node_x = extract_node_features(cand, base_dim=56, include_structural_stats=True)
    assert node_x.shape == (4, 60)
    assert not np.isnan(node_x).any()

    # Edge features: 4 edges x 4 attributes = (4, 4)
    edge_x = extract_edge_features(cand)
    assert edge_x.shape == (4, 4)
    assert not np.isnan(edge_x).any()
    assert edge_x[0, 0] == 100.0  # Amount
    assert edge_x[0, 1] == 10.0   # Timestamp


def test_cell_features_arbitrary_cycle_length():
    """Verify dynamic cell feature extraction across k in {3, 4, 5, 6} without tuple errors."""
    for k in (3, 4, 5, 6):
        nodes = [f"V_{i}" for i in range(k)]
        edges = [
            (nodes[i], nodes[(i + 1) % k], {"amount": float(10 * (i + 1)), "timestamp": float(i + 1)})
            for i in range(k)
        ]

        feats = extract_cell_features(cell_nodes=nodes, candidate_edges=edges)
        assert len(feats) == 8
        assert feats[0] == float(k)               # Cycle length
        assert feats[1] == sum(10 * (i + 1) for i in range(k))  # Total flow
        assert feats[7] == 1.0                    # Direction consistency (all aligned)

    # Batch test
    batch_cells = [["V_0", "V_1", "V_2"], ["V_0", "V_1", "V_2", "V_3"]]
    edges_4 = [
        ("V_0", "V_1", {"amount": 10.0, "timestamp": 1.0}),
        ("V_1", "V_2", {"amount": 20.0, "timestamp": 2.0}),
        ("V_2", "V_3", {"amount": 30.0, "timestamp": 3.0}),
        ("V_3", "V_0", {"amount": 40.0, "timestamp": 4.0}),
        ("V_2", "V_0", {"amount": 15.0, "timestamp": 2.5}),
    ]
    batch_x = extract_batch_cell_features(batch_cells, edges_4)
    assert batch_x.shape == (2, 8)


def test_direction_consistency_calculation():
    """Verify direction consistency calculation when cycle edges have mixed directions."""
    nodes = ["A", "B", "C", "D"]
    # Edges: A->B, B->C, D->C (reversed from C->D), D->A
    edges = [
        ("A", "B", {"amount": 10.0, "timestamp": 1.0}),
        ("B", "C", {"amount": 10.0, "timestamp": 2.0}),
        ("D", "C", {"amount": 10.0, "timestamp": 3.0}),  # Backward relative to A-B-C-D
        ("D", "A", {"amount": 10.0, "timestamp": 4.0}),
    ]

    feats = extract_cell_features(nodes, edges)
    # 3 forward edges out of 4 steps = 0.75 consistency
    assert feats[7] == 0.75
