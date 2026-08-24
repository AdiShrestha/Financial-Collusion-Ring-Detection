"""Unit tests for standard GNN baselines (GCN, GAT, GraphSAGE)."""

import pytest
import torch
import torch.nn as nn

from source.data.candidate_extractor import CandidateExample
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline, global_pool
from source.topology.graph_view import GraphView


def create_sample_graph_view() -> GraphView:
    cand = CandidateExample(
        candidate_id="c_gnn_01",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 40.0),
        participant_ids=["N1", "N2", "N3", "N4"],
        edges=[
            ("N1", "N2", {"amount": 100.0, "timestamp": 10.0}),
            ("N2", "N3", {"amount": 90.0, "timestamp": 20.0}),
            ("N3", "N4", {"amount": 80.0, "timestamp": 30.0}),
            ("N4", "N1", {"amount": 70.0, "timestamp": 40.0}),
        ],
        node_features={n: [0.5] * 56 for n in ["N1", "N2", "N3", "N4"]},
        target_y=1,
        typology_label="CYCLE",
        group_id="g_gnn",
    )
    return GraphView.from_candidate_example(cand)


@pytest.mark.parametrize("model_cls", [GCNBaseline, GATBaseline, GraphSAGEBaseline])
def test_gnn_forward_pass_shapes(model_cls):
    """Verify output logits shape (B, 2) on single and batched graphs."""
    model = model_cls(in_dim=56, hidden_dim=32, num_layers=2, out_dim=2)
    model.eval()

    # Single graph (4 nodes, 4 edges)
    x = torch.randn(4, 56)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out_single = model(x, edge_index)
    assert out_single.shape == (1, 2)

    # Batched graphs: Graph 1 (4 nodes) + Graph 2 (3 nodes)
    x_batch = torch.randn(7, 56)
    edge_index_batch = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 0, 5, 6, 4]], dtype=torch.long
    )
    batch_vec = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    out_batch = model(x_batch, edge_index_batch, batch=batch_vec)
    assert out_batch.shape == (2, 2)


@pytest.mark.parametrize("model_cls", [GCNBaseline, GATBaseline, GraphSAGEBaseline])
def test_gnn_gradient_flow(model_cls):
    """Verify complete gradient backpropagation through all model parameters without NaNs."""
    model = model_cls(in_dim=56, hidden_dim=32, num_layers=2, out_dim=2)
    model.train()

    x = torch.randn(4, 56, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    target = torch.tensor([1], dtype=torch.long)

    logits = model(x, edge_index)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient contains NaN for {name}"
            assert torch.norm(param.grad) > 0.0, f"Gradient is all-zero for {name}"


def test_readout_pooling_variations():
    """Verify mean, sum, and max readout pooling modes."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    mean_pooled = global_pool(x, batch, mode="mean")
    assert torch.allclose(mean_pooled, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))

    sum_pooled = global_pool(x, batch, mode="sum")
    assert torch.allclose(sum_pooled, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))

    max_pooled = global_pool(x, batch, mode="max")
    assert torch.allclose(max_pooled, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_graph_view_input_direct():
    """Verify passing GraphView domain representation directly to GNN forward pass."""
    gv = create_sample_graph_view()
    model = GCNBaseline(in_dim=56, hidden_dim=32, num_layers=2, out_dim=2)
    model.eval()

    logits = model(gv)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()
