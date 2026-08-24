"""Unit tests for GINEBaseline (Contract C12-03)."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.models.edge_aware_gnn import GINEBaseline


def test_gine_forward_pass_and_output_shapes():
    """Verify GINE forward pass with node and edge attributes returns correct logit shape [B, 1]."""
    model = GINEBaseline(node_dim=16, edge_dim=8, hidden_dim=64, num_layers=2)
    model.eval()

    num_nodes = 6
    x = torch.randn(num_nodes, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]], dtype=torch.long)
    edge_attr = torch.randn(6, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    logits = model(x, edge_index, edge_attr=edge_attr, batch=batch)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


def test_gine_backward_gradient_flow():
    """Verify gradient flow reaches both node encoder and edge projection parameters."""
    model = GINEBaseline(node_dim=16, edge_dim=8, hidden_dim=32, num_layers=2)
    model.train()

    x = torch.randn(4, 16, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 8, requires_grad=True)

    logits = model(x, edge_index, edge_attr=edge_attr)
    loss = logits.sum()
    loss.backward()

    # Check gradients on model parameters
    assert model.node_encoder.weight.grad is not None
    assert model.convs[0].edge_proj.weight.grad is not None
    assert torch.sum(torch.abs(model.convs[0].edge_proj.weight.grad)) > 0.0


def test_gine_parameter_count():
    """Verify parameter count helper method returns positive integer."""
    model = GINEBaseline(node_dim=16, edge_dim=8, hidden_dim=64)
    param_count = model.get_param_count()
    assert isinstance(param_count, int)
    assert param_count > 1000
