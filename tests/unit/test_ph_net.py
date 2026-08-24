"""Unit tests for Multi-Scale PH-Augmented Topological Network (TopoRingNet)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from source.data.candidate_extractor import CandidateExample
from source.models.ph_augmented_net import GatedTopologicalFusion, TopologicalVectorEncoder, TopoRingNet
from source.topology.cycle_cell_view import CycleCellView


def create_sample_cycle_candidate() -> CycleCellView:
    nodes = ["N0", "N1", "N2", "N3"]
    edges = [
        ("N0", "N1", {"amount": 100.0, "timestamp": 10.0}),
        ("N1", "N2", {"amount": 90.0, "timestamp": 20.0}),
        ("N2", "N3", {"amount": 80.0, "timestamp": 30.0}),
        ("N3", "N0", {"amount": 70.0, "timestamp": 40.0}),
    ]
    cand = CandidateExample(
        candidate_id="c_toporing_01",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 40.0),
        participant_ids=nodes,
        edges=edges,
        node_features={
            n: [float(idx + i * 3) for idx in range(56)]
            for i, n in enumerate(nodes)
        },
        target_y=1,
        typology_label="CYCLE",
        group_id="g_tr_01",
    )
    return CycleCellView.from_candidate_example(cand)


def test_topological_vector_encoder():
    """Verify TopologicalVectorEncoder shapes and gradient backprop."""
    encoder = TopologicalVectorEncoder(in_dim=372, hidden_dim=64, out_dim=64)
    z_topo = torch.randn(4, 372, requires_grad=True)
    h_topo = encoder(z_topo)
    assert h_topo.shape == (4, 64)

    loss = h_topo.sum()
    loss.backward()
    assert z_topo.grad is not None
    assert torch.norm(z_topo.grad) > 0.0


def test_gated_topological_fusion():
    """Verify GatedTopologicalFusion gating and output shapes."""
    fusion = GatedTopologicalFusion(hidden_dim=32)
    h_struct = torch.randn(2, 32, requires_grad=True)
    h_topo = torch.randn(2, 32, requires_grad=True)

    h_fused = fusion(h_struct, h_topo)
    assert h_fused.shape == (2, 32)

    loss = h_fused.sum()
    loss.backward()
    assert h_struct.grad is not None and torch.norm(h_struct.grad) > 0.0
    assert h_topo.grad is not None and torch.norm(h_topo.grad) > 0.0


def test_toporingnet_forward_pass():
    """Verify complete TopoRingNet forward pass with combined CycleCellView and 372-dim PH vector."""
    cell_view = create_sample_cycle_candidate()
    z_topo = torch.randn(1, 372)

    model = TopoRingNet(
        in_dim_node=56,
        in_dim_edge=2,
        in_dim_cell=2,
        in_dim_topo=372,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.eval()

    logits = model(cell_view, z_topo=z_topo)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()


def test_toporingnet_gradient_flow():
    """Verify simultaneous non-zero gradient flow into both structural and topological branches."""
    cell_view = create_sample_cycle_candidate()
    z_topo = torch.randn(1, 372, requires_grad=True)

    model = TopoRingNet(
        in_dim_node=56,
        in_dim_edge=2,
        in_dim_cell=2,
        in_dim_topo=372,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.train()

    logits = model(cell_view, z_topo=z_topo)
    target = torch.tensor([1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # Structural backbone gradients
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Backbone grad is None for {name}"
            assert torch.norm(param.grad) > 0.0, f"Backbone grad is 0 for {name}"

    # Topological encoder gradients
    for name, param in model.topo_encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Topo encoder grad is None for {name}"
            assert torch.norm(param.grad) > 0.0, f"Topo encoder grad is 0 for {name}"

    # Fusion gradients
    for name, param in model.fusion.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Fusion grad is None for {name}"
            assert torch.norm(param.grad) > 0.0, f"Fusion grad is 0 for {name}"


def test_toporingnet_ablation_modes():
    """Verify clean support for all ablation modes (full, struct_only, topo_only)."""
    cell_view = create_sample_cycle_candidate()
    z_topo = torch.randn(1, 372)

    model = TopoRingNet(
        in_dim_node=56,
        in_dim_edge=2,
        in_dim_cell=2,
        in_dim_topo=372,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.eval()

    # 1. Full mode
    logits_full = model(cell_view, z_topo=z_topo, ablation_mode="full")
    assert logits_full.shape == (1, 2)

    # 2. Structural only mode
    logits_struct = model(cell_view, z_topo=None, ablation_mode="struct_only")
    assert logits_struct.shape == (1, 2)

    # 3. Topo only mode
    logits_topo = model(cell_view, z_topo=z_topo, ablation_mode="topo_only")
    assert logits_topo.shape == (1, 2)


def test_semantic_ph_gated_fusion_gradient():
    """Semantic test verifying that topological feature changes actively alter fusion outputs."""
    cell_view = create_sample_cycle_candidate()
    z_topo_1 = torch.zeros(1, 372)
    z_topo_2 = torch.ones(1, 372) * 5.0

    model = TopoRingNet(
        in_dim_node=56,
        in_dim_edge=2,
        in_dim_cell=2,
        in_dim_topo=372,
        hidden_dim=32,
        num_layers=2,
        out_dim=2,
    )
    model.eval()

    logits_1 = model(cell_view, z_topo=z_topo_1)
    logits_2 = model(cell_view, z_topo=z_topo_2)

    # Different topological persistence must yield distinct logits
    assert not torch.allclose(logits_1, logits_2, atol=1e-3)
