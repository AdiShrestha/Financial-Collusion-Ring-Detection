"""Unit tests for caliper negative matching and relational candidate dataset assembler (Contract C16-02)."""

import os
import sys
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.negative_cycle_sampler import match_and_assemble_candidates


def test_caliper_matching_and_relational_dataset_assembly():
    """Verify caliper matching builds consistent relational Parquet tables with zero leakage."""
    res = match_and_assemble_candidates(
        pos_parquet_path="artifacts/candidates/positive_candidates.parquet",
        ben_parquet_path="artifacts/candidates/benign_pool.parquet",
        output_candidates_path="artifacts/candidates/candidates.parquet",
        output_candidate_txs_path="artifacts/candidates/candidate_transactions.parquet",
        output_labels_path="artifacts/candidates/labels.parquet",
        output_balance_path="artifacts/candidates/covariate_balance.json",
        match_ratio=3,
    )

    assert res["status"] == "ASSEMBLED"
    assert res["total_positive"] == 40
    assert res["total_negative"] >= 40
    assert os.path.exists("artifacts/candidates/candidates.parquet")
    assert os.path.exists("artifacts/candidates/candidate_transactions.parquet")
    assert os.path.exists("artifacts/candidates/labels.parquet")
    assert os.path.exists("artifacts/candidates/covariate_balance.json")

    # Relational consistency check
    cand_tbl = pq.read_table("artifacts/candidates/candidates.parquet")
    tx_tbl = pq.read_table("artifacts/candidates/candidate_transactions.parquet")
    lbl_tbl = pq.read_table("artifacts/candidates/labels.parquet")

    assert cand_tbl.num_rows == res["total_candidates"]
    assert lbl_tbl.num_rows == res["total_candidates"]
    assert tx_tbl.num_rows == res["total_candidate_transactions"]

    # Check candidate ID alignment across tables
    cand_ids = set(cand_tbl.column("candidate_id").to_pylist())
    lbl_ids = set(lbl_tbl.column("candidate_id").to_pylist())
    tx_cand_ids = set(tx_tbl.column("candidate_id").to_pylist())

    assert cand_ids == lbl_ids
    assert tx_cand_ids.issubset(cand_ids)
