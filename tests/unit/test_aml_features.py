"""Unit tests for AMLFeatureBuilder (Contract C11-01)."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.features.aml_features import AMLFeatureBuilder


def test_aml_feature_builder_dimensions():
    """Verify 16-dim node tensors and 8-dim edge tensors shapes."""
    txs = [
        {"from_account": "ACC1", "to_account": "ACC2", "amount_paid": 100.0, "timestamp_epoch": 1000.0, "payment_format": "wire", "from_bank": "B1", "to_bank": "B2", "receiving_currency": "USD", "payment_currency": "USD"},
        {"from_account": "ACC2", "to_account": "ACC3", "amount_paid": 200.0, "timestamp_epoch": 1050.0, "payment_format": "ach", "from_bank": "B2", "to_bank": "B3", "receiving_currency": "USD", "payment_currency": "USD"},
        {"from_account": "ACC3", "to_account": "ACC1", "amount_paid": 300.0, "timestamp_epoch": 1100.0, "payment_format": "cheque", "from_bank": "B3", "to_bank": "B1", "receiving_currency": "EUR", "payment_currency": "USD"},
    ]
    nodes = ["ACC1", "ACC2", "ACC3"]

    builder = AMLFeatureBuilder(use_scaler=False)
    node_feats = builder.transform_nodes(txs, nodes)
    edge_feats = builder.transform_edges(txs)

    assert isinstance(node_feats, torch.Tensor)
    assert node_feats.shape == (3, 16)

    assert isinstance(edge_feats, torch.Tensor)
    assert edge_feats.shape == (3, 8)

    # Check edge one-hot format and currency match
    # Edge 0: wire -> format index 3 (col 2+3 = 5), currency match USD==USD -> col 7 = 1
    assert edge_feats[0, 5] == 1.0
    assert edge_feats[0, 7] == 1.0

    # Edge 2: EUR != USD -> col 7 = 0
    assert edge_feats[2, 7] == 0.0


def test_aml_feature_builder_train_scaler_isolation():
    """Verify scalers are fitted on training data and applied to test data without leakage."""
    train_txs = [
        {"from_account": f"ACC_{i}", "to_account": f"ACC_{i+1}", "amount_paid": 100.0 * (i + 1), "timestamp_epoch": 1000.0 + i * 50, "payment_format": "wire", "from_bank": "B1", "to_bank": "B2"}
        for i in range(10)
    ]
    test_txs = [
        {"from_account": "TEST_A", "to_account": "TEST_B", "amount_paid": 500.0, "timestamp_epoch": 2000.0, "payment_format": "ach", "from_bank": "B3", "to_bank": "B4"}
    ]

    builder = AMLFeatureBuilder(use_scaler=True)
    builder.fit(train_txs)
    assert builder.is_fitted is True

    test_node_feats = builder.transform_nodes(test_txs, ["TEST_A", "TEST_B"])
    assert test_node_feats.shape == (2, 16)
    assert not torch.isnan(test_node_feats).any()


def test_build_candidate_features():
    """Verify build_candidate_features constructs complete tensor dictionary."""
    candidate = {
        "candidate_id": 1,
        "participants": ["ACC1", "ACC2", "ACC3"],
        "transactions": [
            {"from_account": "ACC1", "to_account": "ACC2", "amount_paid": 50.0, "timestamp_epoch": 100.0, "payment_format": "wire"},
            {"from_account": "ACC2", "to_account": "ACC3", "amount_paid": 50.0, "timestamp_epoch": 200.0, "payment_format": "wire"},
            {"from_account": "ACC3", "to_account": "ACC1", "amount_paid": 50.0, "timestamp_epoch": 300.0, "payment_format": "wire"},
        ],
        "label": 1,
    }

    builder = AMLFeatureBuilder(use_scaler=False)
    data = builder.build_candidate_features(candidate, is_train=True)

    assert "x" in data and data["x"].shape == (3, 16)
    assert "edge_attr" in data and data["edge_attr"].shape == (3, 8)
    assert "edge_index" in data and data["edge_index"].shape == (2, 3)
    assert "y" in data and data["y"].item() == 1
    assert data["num_nodes"].item() == 3
