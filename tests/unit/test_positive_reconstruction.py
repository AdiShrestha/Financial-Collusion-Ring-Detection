"""Unit tests for exact positive cycle candidate reconstruction (Contract C15-03)."""

import os
import sys
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.positive_reconstruction import reconstruct_positive_candidates


def test_positive_candidate_reconstruction_integrity():
    """Verify 40 positive cycle candidates of length 3..12 are reconstructed with 100% master joins."""
    res = reconstruct_positive_candidates(
        patterns_path="data/raw/HI-Small_Patterns.txt",
        transactions_parquet_path="artifacts/raw/transactions.parquet",
        output_parquet_path="artifacts/candidates/positive_candidates.parquet",
    )

    assert res["status"] == "RECONSTRUCTED"
    assert res["total_positive_candidates"] == 40
    assert res["total_matched_transactions"] == res["total_expected_transactions"]

    # Verify Parquet table structure
    table = pq.read_table("artifacts/candidates/positive_candidates.parquet")
    assert table.num_rows == 40

    cycle_lengths = table.column("cycle_length").to_pylist()
    assert all(3 <= k <= 12 for k in cycle_lengths)

    ordered_accounts = table.column("ordered_cycle_accounts").to_pylist()
    for accs in ordered_accounts:
        assert len(accs) == len(set(accs)), "Accounts within simple cycle must be unique"
