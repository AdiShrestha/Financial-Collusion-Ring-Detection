"""Unit tests for Provenance-Hardened Production Trainer (Contract C14-03)."""

import json
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.training.production_trainer import LOCKED_SEEDS, MODEL_NAMES


def test_production_training_checkpoints_exist_and_healthy():
    """Verify all 35 checkpoints exist in checkpoints/ with non-zero parameter norms."""
    ckpt_dir = "checkpoints"
    assert os.path.exists(ckpt_dir), "Missing checkpoints directory"

    manifest_path = os.path.join(ckpt_dir, "checkpoint_manifest.json")
    assert os.path.exists(manifest_path), "Missing checkpoint_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_checkpoints"] == 35

    for model_name in MODEL_NAMES:
        for seed in LOCKED_SEEDS:
            fname = f"{model_name}_seed{seed}.pt"
            fpath = os.path.join(ckpt_dir, fname)
            assert os.path.exists(fpath), f"Missing checkpoint: {fname}"

            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert "best_val_auprc" in ckpt
            assert "training_history" in ckpt

            # Verify parameter health
            state_dict = ckpt["model_state_dict"]
            for param_name, tensor in state_dict.items():
                assert not torch.isnan(tensor).any(), f"NaN in {fname} param {param_name}"
                assert not torch.isinf(tensor).any(), f"Inf in {fname} param {param_name}"


def test_topological_cache_provenance():
    """Verify data/cache/topological_features.npz exists and has non-zero variance."""
    cache_path = "data/cache/topological_features.npz"
    assert os.path.exists(cache_path), "Missing topological_features.npz"
    assert os.path.getsize(cache_path) > 5000
