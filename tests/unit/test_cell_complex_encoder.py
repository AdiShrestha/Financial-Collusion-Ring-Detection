"""Unit tests for real topological cell complex encoder (Contract C17-02)."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.topology.cell_complex_encoder import (
    CellComplexCandidateEncoder,
    encode_and_serialize_fold_tensors,
)


def test_cell_complex_encoder_boundary_nilpotency():
    """Verify B1 @ B2 == 0 invariant (INV-004) and non-empty tensor dimensions."""
    cand = {
        "candidate_id": "test_c_01",
        "label": 1,
        "cycle_length": 3,
        "ordered_cycle_accounts": ["A", "B", "C"],
    }
    txs = [
        {"from_account": "A", "to_account": "B", "amount_paid": 100.0, "timestamp_epoch": 1000.0, "payment_format": "ACH", "from_bank": 1, "to_bank": 2},
        {"from_account": "B", "to_account": "C", "amount_paid": 95.0, "timestamp_epoch": 2000.0, "payment_format": "Wire", "from_bank": 2, "to_bank": 3},
        {"from_account": "C", "to_account": "A", "amount_paid": 90.0, "timestamp_epoch": 3000.0, "payment_format": "ACH", "from_bank": 3, "to_bank": 1},
    ]

    encoder = CellComplexCandidateEncoder()
    encoded = encoder.encode_candidate(cand, txs)

    B1 = encoded["B1"].numpy()
    B2 = encoded["B2"].numpy()

    # Invariant INV-004: B1 @ B2 == 0
    assert np.all(np.dot(B1, B2) == 0)
    assert encoded["X0"].shape == (3, 5)
    assert encoded["X1"].shape == (3, 9)
    assert encoded["X2"].shape == (1, 4)


def test_encode_and_serialize_fold_tensors():
    """Verify encoding and serialization of all 5 fold tensor bundles."""
    res = encode_and_serialize_fold_tensors(
        candidates_parquet_path="artifacts/candidates/candidates.parquet",
        candidate_txs_parquet_path="artifacts/candidates/candidate_transactions.parquet",
        fold_manifest_path="artifacts/splits/fold_manifest.json",
        output_dir="artifacts/features",
    )

    assert res["status"] == "TENSORS_ENCODED"
    assert res["total_encoded_candidates"] == 155

    for f_id in range(5):
        pt_path = f"artifacts/features/fold_{f_id}_tensors.pt"
        assert os.path.exists(pt_path)

        bundle = torch.load(pt_path, weights_only=False)
        assert "train_candidates" in bundle
        assert "test_candidates" in bundle
        assert len(bundle["train_candidates"]) + len(bundle["test_candidates"]) == 155
