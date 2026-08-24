"""Unit tests for hardened label-blind bounded cycle extractor (Contract C16-01)."""

import os
import sys
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.bounded_extractor import BoundedCycleExtractor, canonical_cycle


def test_canonical_cycle_rotation():
    """Verify canonical_cycle returns the lexicographically smallest rotation."""
    c1 = ("B", "C", "A")
    c2 = ("C", "A", "B")
    c3 = ("A", "B", "C")
    assert canonical_cycle(c1) == ("A", "B", "C")
    assert canonical_cycle(c2) == ("A", "B", "C")
    assert canonical_cycle(c3) == ("A", "B", "C")


def test_benign_pool_extraction_and_invariants():
    """Verify benign pool extraction extracts >100 genuine cycles with zero laundering/pattern overlap."""
    extractor = BoundedCycleExtractor(min_cycle_len=3, max_cycle_len=12, max_candidates=1000)
    res = extractor.extract_benign_pool(
        transactions_parquet_path="artifacts/raw/transactions.parquet",
        patterns_txt_path="data/raw/HI-Small_Patterns.txt",
        output_parquet_path="artifacts/candidates/benign_pool.parquet",
    )

    assert res["status"] == "EXTRACTED"
    assert res["total_benign_candidates"] >= 100
    assert os.path.exists("artifacts/candidates/benign_pool.parquet")

    table = pq.read_table("artifacts/candidates/benign_pool.parquet")
    assert table.num_rows == res["total_benign_candidates"]

    # Verify all labels are 0
    labels = table.column("label").to_pylist()
    assert all(lbl == 0 for lbl in labels)

    # Verify all cycle lengths in [3..12]
    lengths = table.column("cycle_length").to_pylist()
    assert all(3 <= k <= 12 for k in lengths)
