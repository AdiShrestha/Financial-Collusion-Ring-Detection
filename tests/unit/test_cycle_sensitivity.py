"""Unit tests for Cycle Length Sensitivity Analysis."""

import numpy as np
import pytest
import torch

from source.data.candidate_extractor import CandidateExample
from source.experiments.cycle_sensitivity import (
    CycleSensitivityAnalyzer,
    stratify_candidates_by_cycle_length,
)
from source.topology.cycle_cell_view import CycleCellView


def _make_dummy_cycle_item(k=4, label=1):
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
        "cycle_length": k,
        "z_topo": np.ones(372, dtype=np.float32),
        "target_y": label,
    }


def test_candidate_stratification_by_k():
    """Verify that candidates are cleanly partitioned by cycle length k in {3,4,5,6}."""
    items = [
        _make_dummy_cycle_item(3, 1),
        _make_dummy_cycle_item(4, 1),
        _make_dummy_cycle_item(5, 1),
        _make_dummy_cycle_item(6, 1),
        _make_dummy_cycle_item(4, 0),
    ]

    strata = stratify_candidates_by_cycle_length(items, k_values=[3, 4, 5, 6])

    assert len(strata[3]) == 1
    assert len(strata[4]) == 2
    assert len(strata[5]) == 1
    assert len(strata[6]) == 1


def test_sensitivity_analyzer_execution():
    """Verify stratified sensitivity analyzer execution across cycle length strata."""
    train_data = [
        _make_dummy_cycle_item(3, 0),
        _make_dummy_cycle_item(3, 1),
        _make_dummy_cycle_item(4, 0),
        _make_dummy_cycle_item(4, 1),
        _make_dummy_cycle_item(5, 0),
        _make_dummy_cycle_item(5, 1),
        _make_dummy_cycle_item(6, 0),
        _make_dummy_cycle_item(6, 1),
    ]
    val_data = [
        _make_dummy_cycle_item(3, 0),
        _make_dummy_cycle_item(3, 1),
        _make_dummy_cycle_item(4, 0),
        _make_dummy_cycle_item(4, 1),
        _make_dummy_cycle_item(5, 0),
        _make_dummy_cycle_item(5, 1),
        _make_dummy_cycle_item(6, 0),
        _make_dummy_cycle_item(6, 1),
    ]

    analyzer = CycleSensitivityAnalyzer(in_dim_node=56, in_dim_edge=2, in_dim_cell=2, hidden_dim=16)
    results = analyzer.run_sensitivity_analysis(
        train_data=train_data,
        val_data=val_data,
        model_names=["GCNBaseline", "CellularComplexNet", "TopoRingNet"],
        seeds=[42, 43],
        k_values=[3, 4, 5, 6],
        epochs=2,
    )

    assert results["evaluated_split"] == "validation"
    assert "summary" in results
    assert "GCNBaseline" in results["summary"]
    assert "CellularComplexNet" in results["summary"]
    assert "TopoRingNet" in results["summary"]
    assert 4 in results["summary"]["TopoRingNet"]
    assert "mean_pr_auc" in results["summary"]["TopoRingNet"][4]


def test_sensitivity_analyzer_rejects_test_split():
    """Verify that sensitivity analyzer rejects test_data (INV-006)."""
    train_data = [_make_dummy_cycle_item(4, 0)]
    val_data = [_make_dummy_cycle_item(4, 1)]
    test_data = [_make_dummy_cycle_item(4, 0)]

    analyzer = CycleSensitivityAnalyzer()
    with pytest.raises(ValueError, match="Test data cannot be passed"):
        analyzer.run_sensitivity_analysis(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
