"""Unit tests for ProductionConfirmatoryRunner (Contract C13-01)."""

import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.experiments.production_confirmatory_runner import ProductionConfirmatoryRunner
from source.training.production_trainer import LOCKED_SEEDS, MODEL_NAMES


def test_production_confirmatory_predictions_file_valid():
    """Verify runs/production_confirmatory/predictions.json exists and is well-formed."""
    pred_path = "runs/production_confirmatory/predictions.json"
    assert os.path.exists(pred_path), "Missing predictions.json"

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["total_test_candidates"] == 28
    assert meta["models"] == MODEL_NAMES
    assert meta["seeds"] == LOCKED_SEEDS

    preds = data["predictions"]
    assert len(preds) == 28

    for p in preds:
        assert "candidate_id" in p
        assert "group_id" in p
        assert "label" in p
        assert p["label"] in (0, 1)
        assert "cycle_length" in p
        assert p["cycle_length"] in (3, 4, 5, 6)

        m_preds = p["model_predictions"]
        for m in MODEL_NAMES:
            assert m in m_preds
            assert len(m_preds[m]) == 5
            for prob in m_preds[m]:
                assert 0.0 <= prob <= 1.0


def test_production_confirmatory_runner_execution():
    """Verify ProductionConfirmatoryRunner executes without error."""
    runner = ProductionConfirmatoryRunner()
    test_items, manifest_data, _ = runner._prepare_test_items()
    assert len(test_items) == 28
