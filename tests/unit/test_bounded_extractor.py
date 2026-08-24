"""Unit tests for BoundedCycleExtractor (Contract C11-02)."""

import os
import sys
import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.bounded_extractor import BoundedCycleExtractor, canonical_cycle


def test_canonical_cycle_permutation_invariance():
    """Verify that different cyclic rotations map to the identical canonical tuple."""
    c1 = ["A", "B", "C", "D"]
    c2 = ["B", "C", "D", "A"]
    c3 = ["C", "D", "A", "B"]
    c4 = ["D", "A", "B", "C"]

    canon = ("A", "B", "C", "D")
    assert canonical_cycle(c1) == canon
    assert canonical_cycle(c2) == canon
    assert canonical_cycle(c3) == canon
    assert canonical_cycle(c4) == canon


def test_bounded_cycle_extraction_lengths():
    """Verify extraction strictly captures cycles with k in [3, 6] and skips 2-cycles."""
    # Build graph with:
    # 2-cycle: N1 -> N2 -> N1 (should NOT be extracted with min_len=3)
    # 3-cycle: A -> B -> C -> A (k=3)
    # 4-cycle: D -> E -> F -> G -> D (k=4)
    # 7-cycle: H1 -> H2 -> H3 -> H4 -> H5 -> H6 -> H7 -> H1 (k=7, should NOT be extracted with max_len=6)
    txs = [
        # 2-cycle
        {"from_account": "N1", "to_account": "N2", "amount": 10.0, "timestamp_epoch": 100.0},
        {"from_account": "N2", "to_account": "N1", "amount": 10.0, "timestamp_epoch": 110.0},
        # 3-cycle
        {"from_account": "A", "to_account": "B", "amount": 100.0, "timestamp_epoch": 200.0, "is_laundering": 1},
        {"from_account": "B", "to_account": "C", "amount": 100.0, "timestamp_epoch": 210.0, "is_laundering": 1},
        {"from_account": "C", "to_account": "A", "amount": 100.0, "timestamp_epoch": 220.0, "is_laundering": 1},
        # 4-cycle
        {"from_account": "D", "to_account": "E", "amount": 50.0, "timestamp_epoch": 300.0, "is_laundering": 0},
        {"from_account": "E", "to_account": "F", "amount": 50.0, "timestamp_epoch": 310.0, "is_laundering": 0},
        {"from_account": "F", "to_account": "G", "amount": 50.0, "timestamp_epoch": 320.0, "is_laundering": 0},
        {"from_account": "G", "to_account": "D", "amount": 50.0, "timestamp_epoch": 330.0, "is_laundering": 0},
        # 7-cycle
        {"from_account": "H1", "to_account": "H2", "amount": 20.0},
        {"from_account": "H2", "to_account": "H3", "amount": 20.0},
        {"from_account": "H3", "to_account": "H4", "amount": 20.0},
        {"from_account": "H4", "to_account": "H5", "amount": 20.0},
        {"from_account": "H5", "to_account": "H6", "amount": 20.0},
        {"from_account": "H6", "to_account": "H7", "amount": 20.0},
        {"from_account": "H7", "to_account": "H1", "amount": 20.0},
    ]

    extractor = BoundedCycleExtractor(min_cycle_len=3, max_cycle_len=6)
    candidates = extractor.extract_candidates(txs)

    # Exactly 2 candidates should be extracted: k=3 and k=4
    assert len(candidates) == 2

    lengths = sorted([c["cycle_length"] for c in candidates])
    assert lengths == [3, 4]

    c3 = next(c for c in candidates if c["cycle_length"] == 3)
    assert set(c3["nodes"]) == {"A", "B", "C"}
    assert c3["is_laundering"] == 1
    assert c3["duration_seconds"] == 20.0

    c4 = next(c for c in candidates if c["cycle_length"] == 4)
    assert set(c4["nodes"]) == {"D", "E", "F", "G"}
    assert c4["is_laundering"] == 0


def test_pattern_recall_evaluation():
    """Verify recall evaluation against ground-truth pattern blocks."""
    txs = [
        {"from_account": "ACC1", "to_account": "ACC2", "amount": 100.0},
        {"from_account": "ACC2", "to_account": "ACC3", "amount": 100.0},
        {"from_account": "ACC3", "to_account": "ACC1", "amount": 100.0},
    ]
    patterns = [
        {
            "pattern_id": 1,
            "typology": "CYCLE",
            "participants": ["ACC1", "ACC2", "ACC3"],
        }
    ]

    extractor = BoundedCycleExtractor()
    candidates = extractor.extract_candidates(txs)
    recall_stats = extractor.evaluate_pattern_recall(candidates, patterns)

    assert recall_stats["recall"] == 1.0
    assert recall_stats["total_cycle_patterns"] == 1
    assert recall_stats["matched_cycle_patterns"] == 1
