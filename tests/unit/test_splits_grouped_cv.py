"""Unit tests for Union-Find grouping and stratified 5-fold nested CV manifest (Contract C16-03)."""

import json
import os
import sys
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.splits import generate_grouped_nested_splits


def test_grouped_nested_splits_generation_and_zero_leakage():
    """Verify 5-fold outer / 3-fold inner split generation enforces strict 0.0% account leakage."""
    res = generate_grouped_nested_splits(
        candidates_parquet_path="artifacts/candidates/candidates.parquet",
        candidate_txs_parquet_path="artifacts/candidates/candidate_transactions.parquet",
        covariate_balance_path="artifacts/candidates/covariate_balance.json",
        output_manifest_path="artifacts/splits/fold_manifest.json",
        n_outer_folds=5,
        n_inner_folds=3,
        random_seed=42,
    )

    assert res["status"] == "SPLITS_GENERATED"
    assert os.path.exists("artifacts/splits/fold_manifest.json")

    with open("artifacts/splits/fold_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["n_outer_folds"] == 5
    assert len(manifest["outer_folds"]) == 5

    # Load candidate account maps
    c_table = pq.read_table("artifacts/candidates/candidates.parquet")
    cand_accounts = {
        c["candidate_id"]: set(c["ordered_cycle_accounts"])
        for c in c_table.to_pylist()
    }

    # Verify strict 0.0% account leakage across outer folds
    for of in manifest["outer_folds"]:
        train_accs = set()
        for cid in of["train_candidate_ids"]:
            train_accs.update(cand_accounts[cid])

        test_accs = set()
        for cid in of["test_candidate_ids"]:
            test_accs.update(cand_accounts[cid])

        leak = train_accs.intersection(test_accs)
        assert len(leak) == 0, f"Outer fold {of['outer_fold_id']} has account leakage: {leak}"

        # Verify inner folds structure
        assert len(of["inner_folds"]) == 3
        for inf in of["inner_folds"]:
            in_train_accs = set()
            for cid in inf["train_candidate_ids"]:
                in_train_accs.update(cand_accounts[cid])
            in_val_accs = set()
            for cid in inf["val_candidate_ids"]:
                in_val_accs.update(cand_accounts[cid])

            inner_leak = in_train_accs.intersection(in_val_accs)
            assert len(inner_leak) == 0, f"Inner fold {inf['inner_fold_id']} has account leakage: {inner_leak}"
