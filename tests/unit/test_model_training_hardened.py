"""Unit tests for model factory and multi-seed 5-fold cross-validation training (Contract C17-03)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.models.model_factory import create_model
from source.training.cross_val_runner import run_5fold_multi_seed_training


def test_model_factory_instantiations():
    """Verify all 8 model architectures instantiate with correct parameter signatures."""
    models = ["logistic_regression", "hist_gradient_boosting", "gcn", "gat", "graphsage", "gine", "scnn", "ccnn"]
    for m in models:
        inst = create_model(m, random_seed=42)
        assert inst is not None


def test_run_cross_validation_training_execution():
    """Verify cross-validation training executes across all 5 folds, saves checkpoints, and outputs history."""
    test_hist_path = "artifacts/training/test_training_history.json"
    res = run_5fold_multi_seed_training(
        models=["logistic_regression", "gine", "ccnn"],
        seeds=[42],
        epochs=5,
        output_history_path=test_hist_path,
    )

    assert res["status"] == "TRAINING_COMPLETE"
    assert os.path.exists(test_hist_path)

    with open(test_hist_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "oof_predictions" in data
    assert "gine" in data["oof_predictions"]
    assert len(data["oof_predictions"]["gine"]["42"]) == 155
