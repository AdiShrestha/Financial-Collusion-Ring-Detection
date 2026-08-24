"""Unit tests for Boundary Operator & Higher-Order Adjacency Ablation."""

import numpy as np
import pytest
import torch

from source.data.candidate_extractor import CandidateExample
from source.experiments.boundary_ablation import (
    BoundaryAblationRunner,
    ablate_cell_view_b2,
    ablate_simplicial_view_b2,
)
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.oracles import verify_boundary_nilpotence


def _make_dummy_cell_candidate(k=4, label=1):
    nodes = [f"N_{i}" for i in range(k)]
    edges = []
    for i in range(k):
        next_i = (i + 1) % k
        edges.append((nodes[i], nodes[next_i], {"amount": 100.0, "timestamp": float(i + 1)}))

    cand = CandidateExample(
        candidate_id=f"c_cell_{k}_{label}",
        dataset_track="amlworld",
        temporal_bounds=(1.0, float(k)),
        participant_ids=nodes,
        edges=edges,
        node_features={n: [float(i + idx) for idx in range(56)] for i, n in enumerate(nodes)},
        target_y=label,
        typology_label="CYCLE" if label == 1 else "NEGATIVE_CANDIDATE",
        group_id=f"g_{k}_{label}",
    )
    cell_view = CycleCellView.from_candidate_example(cand, max_k=6)
    return {
        "cell_view": cell_view,
        "target_y": label,
    }


def _make_dummy_simplicial_candidate(label=1):
    nodes = ["N0", "N1", "N2"]
    edges = [
        ("N0", "N1", {"amount": 100.0, "timestamp": 1.0}),
        ("N1", "N2", {"amount": 100.0, "timestamp": 2.0}),
        ("N2", "N0", {"amount": 100.0, "timestamp": 3.0}),
    ]
    cand = CandidateExample(
        candidate_id=f"c_simp_3_{label}",
        dataset_track="amlworld",
        temporal_bounds=(1.0, 3.0),
        participant_ids=nodes,
        edges=edges,
        node_features={n: [float(i + idx) for idx in range(56)] for i, n in enumerate(nodes)},
        target_y=label,
        typology_label="CYCLE" if label == 1 else "NEGATIVE_CANDIDATE",
        group_id=f"g_s_{label}",
    )
    simp_view = CliqueSimplicialView.from_candidate_example(cand)
    return {
        "simplicial_view": simp_view,
        "target_y": label,
    }


def test_b2_zero_masking_disables_2cell_flow():
    """Verify that ablate_cell_view_b2 zeroes out B2 and retains nilpotence B1 B2 = 0."""
    item = _make_dummy_cell_candidate(4, 1)
    orig_view = item["cell_view"]
    assert orig_view.B2.size > 0
    assert not np.all(orig_view.B2 == 0.0)

    ablated_view = ablate_cell_view_b2(orig_view)
    assert np.all(ablated_view.B2 == 0.0)
    assert verify_boundary_nilpotence(ablated_view.B1, ablated_view.B2)


def test_boundary_ablation_execution():
    """Verify multi-seed boundary ablation execution across cell and simplicial candidate batches."""
    cell_train = [_make_dummy_cell_candidate(4, 0), _make_dummy_cell_candidate(4, 1), _make_dummy_cell_candidate(5, 0), _make_dummy_cell_candidate(5, 1)]
    cell_val = [_make_dummy_cell_candidate(4, 0), _make_dummy_cell_candidate(4, 1), _make_dummy_cell_candidate(5, 0), _make_dummy_cell_candidate(5, 1)]

    runner = BoundaryAblationRunner(in_dim_node=56, in_dim_edge=2, in_dim_cell=2, hidden_dim=16)
    results = runner.run_boundary_ablations(
        train_data=cell_train,
        val_data=cell_val,
        seeds=[42, 43],
        modes=["ccnn_full_2cell", "ccnn_ablated_1skeleton"],
        epochs=3,
    )

    assert results["evaluated_split"] == "validation"
    assert "ccnn_full_2cell" in results["summary"]
    assert "ccnn_ablated_1skeleton" in results["summary"]
    assert "ccnn_b2_contribution_delta_pr" in results["summary"]
    assert results["summary"]["ccnn_full_2cell"]["num_seeds"] == 2
