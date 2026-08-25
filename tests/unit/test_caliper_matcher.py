"""Unit tests for caliper negative matching and relational candidate dataset assembler (Contract C16-02)."""

import os
import sys
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.negative_cycle_sampler import match_and_assemble_candidates


def test_caliper_matching_and_relational_dataset_assembly(tmp_path):
    """Verify caliper matching builds consistent relational Parquet tables with zero leakage."""
    out_cands = str(tmp_path / "candidates.parquet")
    out_txs = str(tmp_path / "candidate_transactions.parquet")
    out_lbls = str(tmp_path / "labels.parquet")
    out_bal = str(tmp_path / "covariate_balance.json")

    res = match_and_assemble_candidates(
        pos_parquet_path="artifacts/candidates/positive_candidates.parquet",
        ben_parquet_path="artifacts/candidates/benign_pool.parquet",
        output_candidates_path=out_cands,
        output_candidate_txs_path=out_txs,
        output_labels_path=out_lbls,
        output_balance_path=out_bal,
        match_ratio=3,
    )

    assert res["status"] == "ASSEMBLED"
    assert res["total_positive"] == 40
    assert res["total_negative"] >= 40
    assert os.path.exists(out_cands)
    assert os.path.exists(out_txs)
    assert os.path.exists(out_lbls)
    assert os.path.exists(out_bal)

    # Relational consistency check
    cand_tbl = pq.read_table(out_cands)
    tx_tbl = pq.read_table(out_txs)
    lbl_tbl = pq.read_table(out_lbls)

    assert cand_tbl.num_rows == res["total_candidates"]
    assert lbl_tbl.num_rows == res["total_candidates"]
    assert tx_tbl.num_rows == res["total_candidate_transactions"]

    # Check candidate ID alignment across tables
    cand_ids = set(cand_tbl.column("candidate_id").to_pylist())
    lbl_ids = set(lbl_tbl.column("candidate_id").to_pylist())
    tx_cand_ids = set(tx_tbl.column("candidate_id").to_pylist())

    assert cand_ids == lbl_ids
    assert tx_cand_ids.issubset(cand_ids)
