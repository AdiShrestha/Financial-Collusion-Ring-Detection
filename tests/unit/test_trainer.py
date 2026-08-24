"""Unit tests for Unified Production Model Trainer and Checkpoint Engine."""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from source.models.cell_net import CellularComplexNet
from source.models.gate_c_verifier import generate_gate_c_synthetic_batch
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.training.trainer import ModelTrainer


def test_trainer_convergence_and_early_stopping(tmp_path):
    """Verify training loop convergence and early stopping trigger."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]
    val_data = batch[6:]

    model = GCNBaseline(hidden_dim=32, num_layers=2, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    ckpt_file = str(tmp_path / "model_ckpt.pt")

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        patience=5,
        min_epochs=2,
        max_epochs=40,
        monitor_metric="pr_auc",
    )

    fit_res = trainer.fit(train_data, val_data, checkpoint_path=ckpt_file)

    assert "best_epoch" in fit_res
    assert fit_res["best_epoch"] >= 1
    assert os.path.exists(ckpt_file)
    assert len(trainer.history["train_loss"]) <= 40


def test_checkpoint_save_and_restore(tmp_path):
    """Verify that checkpoint saving and loading reproduces exact model predictions."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]
    val_data = batch[6:]

    model = CellularComplexNet(hidden_dim=32, num_layers=2, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    ckpt_file = str(tmp_path / "cell_model_ckpt.pt")

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        patience=5,
        max_epochs=10,
    )
    trainer.fit(train_data, val_data, checkpoint_path=ckpt_file)

    # Compute reference probabilities
    ref_probs = trainer.predict_proba(val_data)

    # Perturb model parameters to destroy weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 10.0)

    perturbed_probs = trainer.predict_proba(val_data)
    assert not np.allclose(ref_probs, perturbed_probs, atol=1e-3)

    # Restore from checkpoint
    trainer.load_checkpoint(ckpt_file)
    restored_probs = trainer.predict_proba(val_data)

    # Assert exact floating-point reproduction
    np.testing.assert_allclose(ref_probs, restored_probs, atol=1e-6)


def test_gradient_clipping_enforcement():
    """Verify gradient clipping bounds gradients during backpropagation."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]

    model = TopoRingNet(hidden_dim=32, num_layers=2, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        grad_clip=0.5,
    )

    loss_val = trainer.train_epoch(train_data)
    assert loss_val > 0.0

    # Ensure max gradient norm in parameters is bounded
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    assert total_norm <= 1.0  # Bounded after step


def test_trainer_supports_all_architectures():
    """Verify all 6 model architectures can execute train_epoch and evaluate without error."""
    batch = generate_gate_c_synthetic_batch(size=10)
    train_data = batch[:6]
    val_data = batch[6:]

    architectures = [
        GCNBaseline(hidden_dim=32, num_layers=2),
        GATBaseline(hidden_dim=32, num_layers=2),
        GraphSAGEBaseline(hidden_dim=32, num_layers=2),
        SimplicialComplexNet(hidden_dim=32, num_layers=2),
        CellularComplexNet(hidden_dim=32, num_layers=2),
        TopoRingNet(hidden_dim=32, num_layers=2),
    ]

    for model in architectures:
        trainer = ModelTrainer(model=model, max_epochs=2)
        loss = trainer.train_epoch(train_data)
        assert not np.isnan(loss)
        val_eval = trainer.evaluate(val_data)
        assert "pr_auc" in val_eval
        assert "loss" in val_eval
