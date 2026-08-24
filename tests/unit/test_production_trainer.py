"""Unit tests for ProductionTrainer (Contract C12-04)."""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.training.production_trainer import LOCKED_SEEDS, MODEL_NAMES, ProductionTrainer


def test_production_trainer_checkpoints_exist_and_valid():
    """Verify all 35 model checkpoints exist and contain valid state dicts and training history."""
    ckpt_dir = "checkpoints"
    assert os.path.exists(ckpt_dir), "Checkpoints directory missing"

    for model_name in MODEL_NAMES:
        for seed in LOCKED_SEEDS:
            ckpt_path = os.path.join(ckpt_dir, f"{model_name}_seed{seed}.pt")
            assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
            assert os.path.getsize(ckpt_path) > 1000, f"Checkpoint too small: {ckpt_path}"

            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            assert "model_state_dict" in ckpt
            assert "training_history" in ckpt
            assert "best_val_auprc" in ckpt
            assert ckpt["model_name"] == model_name
            assert ckpt["seed"] == seed


def test_production_trainer_single_run():
    """Verify single model training runs and produces a valid checkpoint."""
    trainer = ProductionTrainer(checkpoint_dir="checkpoints")
    train_items, val_items, _ = trainer._prepare_data_items()

    res = trainer.train_single_model("gine", 42, train_items, val_items, max_epochs=2, patience=2)
    assert res["model_name"] == "gine"
    assert res["seed"] == 42
    assert os.path.exists(res["checkpoint_path"])
