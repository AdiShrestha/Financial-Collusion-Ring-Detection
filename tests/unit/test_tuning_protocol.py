"""Unit tests for Exploratory Hyperparameter Tuning Engine."""

import json
import os
import pytest
import torch

from source.models.cell_net import CellularComplexNet
from source.models.gate_c_verifier import generate_gate_c_synthetic_batch
from source.models.gnn_baselines import GCNBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.training.tuning_protocol import HyperparameterTuner, compute_binary_metrics


def test_compute_binary_metrics():
    """Verify calculation of PR-AUC, ROC-AUC, and F1-Macro."""
    y_true = [1, 0, 1, 0, 1]
    y_probs = [0.9, 0.1, 0.8, 0.2, 0.7]
    metrics = compute_binary_metrics(y_true, y_probs)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1_macro" in metrics
    assert metrics["pr_auc"] >= 0.9
    assert metrics["roc_auc"] >= 0.9
    assert metrics["f1_macro"] == 1.0


def test_tuner_executes_on_validation_split(tmp_path):
    """Verify tuning executes over validation partition and selects best configuration."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]
    val_data = batch[6:]

    param_grid = {
        "lr": [1e-2, 1e-3],
        "hidden_dim": [32],
        "dropout": [0.0],
    }

    tuner = HyperparameterTuner(
        model_class=GCNBaseline,
        param_grid=param_grid,
        num_trials=2,
        max_epochs_per_trial=5,
        random_seed=42,
    )

    results = tuner.tune(train_data=train_data, val_data=val_data)
    assert results["model_name"] == "GCNBaseline"
    assert results["best_config"] is not None
    assert results["best_score"] >= 0.0
    assert len(results["trial_history"]) == 2

    # Verify saving summary
    cache_file = str(tmp_path / "tuned_hyperparameters.json")
    tuner.save_tuning_summary(cache_file)
    assert os.path.exists(cache_file)

    with open(cache_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert "GCNBaseline" in saved_data
    assert saved_data["GCNBaseline"]["best_config"] == results["best_config"]


def test_tuner_rejects_test_set_access():
    """Verify strict prohibition of test set access during hyperparameter tuning (INV-006)."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]
    val_data = batch[6:8]
    test_data = batch[8:]

    tuner = HyperparameterTuner(
        model_class=CellularComplexNet,
        num_trials=1,
    )

    with pytest.raises(ValueError, match="Test data strictly forbidden"):
        tuner.tune(train_data=train_data, val_data=val_data, test_data=test_data)


def test_deterministic_tuning_results():
    """Verify reproducibility of tuning trials under identical random seed."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]
    val_data = batch[6:]

    param_grid = {
        "lr": [1e-2, 1e-3],
        "hidden_dim": [32],
    }

    tuner1 = HyperparameterTuner(
        model_class=TopoRingNet,
        param_grid=param_grid,
        num_trials=2,
        max_epochs_per_trial=3,
        random_seed=123,
    )
    res1 = tuner1.tune(train_data, val_data)

    tuner2 = HyperparameterTuner(
        model_class=TopoRingNet,
        param_grid=param_grid,
        num_trials=2,
        max_epochs_per_trial=3,
        random_seed=123,
    )
    res2 = tuner2.tune(train_data, val_data)

    assert res1["best_config"] == res2["best_config"]
    assert res1["best_score"] == pytest.approx(res2["best_score"], abs=1e-5)
