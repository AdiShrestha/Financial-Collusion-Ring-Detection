"""Unit tests for Cellular Complex Neural Network (CCNN / CIN) architecture."""

import pytest
import torch
import torch.nn as nn

from source.data.candidate_extractor import CandidateExample
from source.models.cell_net import CellularComplexNet
from source.topology.cycle_cell_view import CycleCellView


def create_polygon_candidate(k: int = 4) -> CycleCellView:
    nodes = [f"N_{i}" for i in range(k)]
    edges = []
    for i in range(k):
        next_i = (i + 1) % k
        edges.append((nodes[i], nodes[next_i], {"amount": 100.0 - i * 10, "timestamp": 10.0 * (i + 1)}))

    cand = CandidateExample(
        candidate_id=f"c_cell_{k}",
        dataset_track="amlworld",
        temporal_bounds=(10.0, float(k * 10)),
        participant_ids=nodes,
        edges=edges,
        node_features={
            n: [float(i + idx) for idx in range(56)] for i, n in enumerate(nodes)
        },
        target_y=1,
        typology_label="CYCLE",
        group_id=f"g_cell_{k}",
    )
    return CycleCellView.from_candidate_example(cand, max_k=6)


def create_open_chain_candidate() -> CycleCellView:
    nodes = ["C0", "C1", "C2", "C3"]
    edges = [
        ("C0", "C1", {"amount": 50.0, "timestamp": 10.0}),
        ("C1", "C2", {"amount": 50.0, "timestamp": 20.0}),
        ("C2", "C3", {"amount": 50.0, "timestamp": 30.0}),
    ]
    cand = CandidateExample(
        candidate_id="c_open_chain",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 30.0),
        participant_ids=nodes,
        edges=edges,
        node_features={n: [float(i) for i in range(56)] for n in nodes},
        target_y=0,
        typology_label="NEGATIVE_CANDIDATE",
        group_id="g_open",
    )
    return CycleCellView.from_candidate_example(cand, max_k=6)


def test_cell_net_forward_pass_4cycle():
    """Verify forward pass on authentic 4-cycle represented as rank-2 polygonal cell (INV-002)."""
    cell_view = create_polygon_candidate(k=4)
    assert cell_view.num_cells_2 == 1

    model = CellularComplexNet(
        in_dim_0=56,
        in_dim_1=2,
        in_dim_2=2,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.eval()

    logits = model(cell_view)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_cell_net_arbitrary_k_polygon(k):
    """Verify cellular message passing over k-gons without simplex distortion."""
    cell_view = create_polygon_candidate(k=k)
    assert cell_view.num_cells_2 == 1

    model = CellularComplexNet(
        in_dim_0=56,
        in_dim_1=2,
        in_dim_2=2,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.eval()

    logits = model(cell_view)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()


def test_cell_net_empty_cells_handling():
    """Verify forward and backward pass on subgraphs without 2-cells (zero-cell safety)."""
    cell_view = create_open_chain_candidate()
    assert cell_view.num_cells_2 == 0

    model = CellularComplexNet(
        in_dim_0=56,
        in_dim_1=2,
        in_dim_2=2,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.train()

    logits = model(cell_view)
    assert logits.shape == (1, 2)

    target = torch.tensor([0], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN for {name}"


def test_cell_net_gradient_flow():
    """Verify complete gradient flow through node, edge, and polygon cell parameters."""
    cell_view = create_polygon_candidate(k=4)
    model = CellularComplexNet(
        in_dim_0=56,
        in_dim_1=2,
        in_dim_2=2,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.train()

    logits = model(cell_view)
    target = torch.tensor([1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN for {name}"
            assert torch.norm(param.grad) > 0.0, f"Gradient is all-zero for {name}"
