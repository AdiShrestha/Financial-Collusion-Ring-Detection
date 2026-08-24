"""Unit tests for Topological Feature Ablation Experiment Runner."""

import numpy as np
import pytest
import torch

from source.data.candidate_extractor import CandidateExample
from source.experiments.ablation_runner import (
    TopologicalAblationRunner,
    apply_ablation_mask,
)
from source.topology.cycle_cell_view import CycleCellView


def _make_dummy_cell_item(k=4, label=1):
    nodes = [f"N_{i}" for i in range(k)]
    edges = []
    for i in range(k):
        next_i = (i + 1) % k
        edges.append((nodes[i], nodes[next_i], {"amount": 100.0 - i * 10, "timestamp": 10.0 * (i + 1)}))

    cand = CandidateExample(
        candidate_id=f"c_cell_{k}_{label}",
        dataset_track="amlworld",
        temporal_bounds=(10.0, float(k * 10)),
        participant_ids=nodes,
        edges=edges,
        node_features={n: [float(i + idx) for idx in range(56)] for i, n in enumerate(nodes)},
        target_y=label,
        typology_label="CYCLE" if label == 1 else "NEGATIVE_CANDIDATE",
        group_id=f"g_cell_{k}_{label}",
    )
    cell_view = CycleCellView.from_candidate_example(cand, max_k=6)
    return {
        "cell_view": cell_view,
        "z_topo": np.ones(372, dtype=np.float32),
        "target_y": label,
    }


def test_ablation_masks_zero_target_slices():
    """Verify that feature ablation masks zero-out exact slice boundaries."""
    z_orig = np.ones(372, dtype=np.float32)

    # Full: unaltered
    z_full = apply_ablation_mask(z_orig, "full")
    assert np.all(z_full == 1.0)

    # No Betti: 0..20 and 186..206 zeroed
    z_no_betti = apply_ablation_mask(z_orig, "no_betti")
    assert np.all(z_no_betti[0:20] == 0.0)
    assert np.all(z_no_betti[186:206] == 0.0)
    assert np.all(z_no_betti[20:186] == 1.0)
    assert np.all(z_no_betti[206:372] == 1.0)

    # No Landscapes: 20..80 and 206..266 zeroed
    z_no_land = apply_ablation_mask(z_orig, "no_landscapes")
    assert np.all(z_no_land[20:80] == 0.0)
    assert np.all(z_no_land[206:266] == 0.0)
    assert np.all(z_no_land[0:20] == 1.0)

    # No Images: 80..180 and 266..366 zeroed
    z_no_img = apply_ablation_mask(z_orig, "no_images")
    assert np.all(z_no_img[80:180] == 0.0)
    assert np.all(z_no_img[266:366] == 0.0)

    # No Entropy / Stats: 180..186 and 366..372 zeroed
    z_no_stat = apply_ablation_mask(z_orig, "no_entropy_stats")
    assert np.all(z_no_stat[180:186] == 0.0)
    assert np.all(z_no_stat[366:372] == 0.0)

    # Struct only: all zero
    z_struct = apply_ablation_mask(z_orig, "struct_only")
    assert np.all(z_struct == 0.0)


def test_ablation_runner_multi_seed_execution():
    """Verify multi-seed ablation runner execution across synthetic training batches."""
    train_data = [_make_dummy_cell_item(4, 0), _make_dummy_cell_item(4, 1), _make_dummy_cell_item(5, 0), _make_dummy_cell_item(5, 1)]
    val_data = [_make_dummy_cell_item(4, 0), _make_dummy_cell_item(4, 1), _make_dummy_cell_item(5, 0), _make_dummy_cell_item(5, 1)]

    runner = TopologicalAblationRunner(in_dim_node=56, in_dim_edge=2, in_dim_cell=2, hidden_dim=16)
    results = runner.run_ablation_suite(
        train_data=train_data,
        val_data=val_data,
        seeds=[42, 43],
        modes=["full", "no_betti", "struct_only"],
        epochs=3,
    )

    assert results["evaluated_split"] == "validation"
    assert "summary" in results
    assert "full" in results["summary"]
    assert "no_betti" in results["summary"]
    assert "struct_only" in results["summary"]
    assert "mean_pr_auc" in results["summary"]["full"]
    assert results["summary"]["full"]["num_seeds"] == 2


def test_ablation_runner_rejects_test_split():
    """Verify that passing test_data triggers a hard ValueError (INV-006)."""
    train_data = [_make_dummy_cell_item(4, 0)]
    val_data = [_make_dummy_cell_item(4, 1)]
    test_data = [_make_dummy_cell_item(4, 0)]

    runner = TopologicalAblationRunner()
    with pytest.raises(ValueError, match="Test data cannot be passed"):
        runner.run_ablation_suite(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
