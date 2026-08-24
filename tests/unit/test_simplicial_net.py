"""Unit tests for Simplicial Neural Network (SCNN / Hodge Spectral Network)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from source.data.candidate_extractor import CandidateExample
from source.models.simplicial_net import SimplicialComplexNet
from source.topology.clique_simplicial_view import CliqueSimplicialView


def create_triangle_simplicial_view() -> CliqueSimplicialView:
    """Create a 3-node triangle with 1 2-simplex (3-clique)."""
    cand = CandidateExample(
        candidate_id="c_tri_01",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 30.0),
        participant_ids=["A", "B", "C"],
        edges=[
            ("A", "B", {"amount": 100.0, "timestamp": 10.0}),
            ("B", "C", {"amount": 90.0, "timestamp": 20.0}),
            ("C", "A", {"amount": 80.0, "timestamp": 30.0}),
        ],
        node_features={
            "A": [float(i) for i in range(56)],
            "B": [float(i * 2) for i in range(56)],
            "C": [float(i + 3) for i in range(56)],
        },
        target_y=1,
        typology_label="CYCLE",
        group_id="g_tri",
    )
    return CliqueSimplicialView.from_candidate_example(cand)


def create_square_simplicial_view() -> CliqueSimplicialView:
    """Create a 4-node square with 0 2-simplices (no 3-cliques)."""
    cand = CandidateExample(
        candidate_id="c_sq_01",
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
        target_y=0,
        typology_label="NEGATIVE_CANDIDATE",
        group_id="g_sq",
    )
    return CliqueSimplicialView.from_candidate_example(cand)


def test_simplicial_net_forward_and_backward():
    """Verify forward pass and gradient flow on simplicial complex with 2-simplices."""
    s_view = create_triangle_simplicial_view()
    assert s_view.num_faces_2 == 1

    model = SimplicialComplexNet(
        in_dim_0=56,
        in_dim_1=2,
        in_dim_2=2,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.train()

    logits = model(s_view)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()

    target = torch.tensor([1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN for {name}"
            assert torch.norm(param.grad) > 0.0, f"Gradient is all-zero for {name}"


def test_simplicial_net_empty_2simplices_handling():
    """Verify forward and backward pass executes gracefully on 4-cycle with zero 2-simplices."""
    s_view = create_square_simplicial_view()
    assert s_view.num_faces_2 == 0

    model = SimplicialComplexNet(
        in_dim_0=56,
        in_dim_1=2,
        in_dim_2=2,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.train()

    logits = model(s_view)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()

    target = torch.tensor([0], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # Verify gradients computed without NaN
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN for {name}"


def test_semantic_simplicial_boundary_gradient():
    """Semantic test verifying that boundary operators communicate gradients between simplicial dimensions."""
    x0 = torch.randn(3, 56, requires_grad=True)
    x1 = torch.randn(3, 2, requires_grad=True)
    x2 = torch.randn(1, 2, requires_grad=True)
    b1 = torch.tensor([[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=torch.float32)
    b2 = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)

    model = SimplicialComplexNet(in_dim_0=56, in_dim_1=2, in_dim_2=2, hidden_dim=32, num_layers=2, out_dim=2)
    logits = model(x0, x1, x2, b1, b2)

    loss = logits.sum()
    loss.backward()

    assert x0.grad is not None and torch.norm(x0.grad) > 0.0
    assert x1.grad is not None and torch.norm(x1.grad) > 0.0
    assert x2.grad is not None and torch.norm(x2.grad) > 0.0
