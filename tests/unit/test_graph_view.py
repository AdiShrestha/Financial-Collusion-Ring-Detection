"""Unit tests for GraphView 1-skeleton baseline representation and PyG converters."""

import numpy as np
import pytest
import torch

from source.data.candidate_extractor import CandidateExample
from source.topology.graph_view import GraphView


def create_sample_candidate() -> CandidateExample:
    return CandidateExample(
        candidate_id="cand_test_001",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 30.0),
        participant_ids=["node_A", "node_B", "node_C"],
        edges=[
            ("node_A", "node_B", {"amount": 100.0, "timestamp": 10.0}),
            ("node_B", "node_C", {"amount": 90.0, "timestamp": 20.0}),
            ("node_C", "node_A", {"amount": 80.0, "timestamp": 30.0}),
        ],
        node_features={
            "node_A": [1.0, 0.0] + [0.0] * 54,
            "node_B": [0.0, 1.0] + [0.0] * 54,
            "node_C": [0.5, 0.5] + [0.0] * 54,
        },
        target_y=1,
        typology_label="CYCLE",
        group_id="group_001",
    )


def test_graph_view_construction():
    """Verify GraphView node indexing, edge list shapes, and feature matrices."""
    cand = create_sample_candidate()
    gv = GraphView.from_candidate_example(cand)

    assert gv.num_nodes == 3
    assert gv.num_edges == 3
    assert gv.candidate_id == "cand_test_001"
    assert gv.target_y == 1
    assert gv.node_features.shape == (3, 56)
    assert gv.edge_features.shape == (3, 2)
    assert gv.edge_index.shape == (2, 3)

    # Check node map
    assert gv.node_map == {"node_A": 0, "node_B": 1, "node_C": 2}

    # Check adjacency matrix
    adj = gv.to_adjacency_matrix()
    assert adj.shape == (3, 3)
    assert adj[0, 1] == 1.0  # A -> B
    assert adj[1, 2] == 1.0  # B -> C
    assert adj[2, 0] == 1.0  # C -> A
    assert adj[1, 0] == 0.0  # Directed (B -> A is 0)


def test_pyg_data_conversion():
    """Verify PyG Data conversion tensor shapes, dtypes, and NaN absence."""
    cand = create_sample_candidate()
    gv = GraphView.from_candidate_example(cand)
    pyg = gv.to_pyg_data()

    assert isinstance(pyg.x, torch.Tensor)
    assert pyg.x.shape == (3, 56)
    assert pyg.x.dtype == torch.float32
    assert not torch.isnan(pyg.x).any()

    assert isinstance(pyg.edge_index, torch.Tensor)
    assert pyg.edge_index.shape == (2, 3)
    assert pyg.edge_index.dtype == torch.int64

    assert isinstance(pyg.edge_attr, torch.Tensor)
    assert pyg.edge_attr.shape == (3, 2)
    assert pyg.edge_attr.dtype == torch.float32
    assert not torch.isnan(pyg.edge_attr).any()

    assert isinstance(pyg.y, torch.Tensor)
    assert pyg.y.item() == 1
    assert pyg.num_nodes == 3


def test_symmetrization_and_self_loops():
    """Verify undirected edge expansion and self-loop attachment."""
    cand = create_sample_candidate()
    gv = GraphView.from_candidate_example(cand, symmetrize=True, add_self_loops=True)

    # 3 original + 3 reversed + 3 self-loops = 9 edges
    assert gv.num_edges == 9
    assert gv.edge_index.shape == (2, 9)

    adj = gv.to_adjacency_matrix()
    # Check symmetry
    assert np.allclose(adj, adj.T)
    # Check diagonal has self loops
    for i in range(3):
        assert adj[i, i] == 1.0
