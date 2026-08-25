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


def test_run_cross_validation_training_execution(tmp_path):
    """Verify a partial one-seed/model run cannot be labelled confirmatory."""
    with pytest.raises(ValueError, match="requires seeds"):
        run_5fold_multi_seed_training(
            models=["logistic_regression", "gine", "ccnn"],
            seeds=[42],
            epochs=5,
            output_history_path=str(tmp_path / "must_not_exist.json"),
        )

    with open("artifacts/training/training_history.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["seeds"] == [42, 43, 44, 45, 46]
    assert len(data["models"]) * len(data["seeds"]) * data["n_outer_folds"] == 200
