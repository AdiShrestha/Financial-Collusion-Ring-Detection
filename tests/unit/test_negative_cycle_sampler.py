"""Unit tests for NegativeCycleSampler (Contract C11-03)."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.negative_cycle_sampler import NegativeCycleSampler


def test_filter_benign_candidates_excludes_laundering_and_forbidden():
    """Verify filter_benign_candidates excludes any candidate with laundering label or forbidden accounts."""
    candidates = [
        # Pure benign
        {
            "candidate_id": "c1",
            "cycle_length": 3,
            "nodes": ["B1", "B2", "B3"],
            "participants": ["B1", "B2", "B3"],
            "is_laundering": 0,
            "transactions": [{"is_laundering": 0}],
        },
        # Laundering candidate
        {
            "candidate_id": "c2",
            "cycle_length": 3,
            "nodes": ["L1", "L2", "L3"],
            "participants": ["L1", "L2", "L3"],
            "is_laundering": 1,
            "transactions": [{"is_laundering": 1}],
        },
        # Candidate sharing account with laundering
        {
            "candidate_id": "c3",
            "cycle_length": 4,
            "nodes": ["B4", "B5", "L1", "B6"],
            "participants": ["B4", "B5", "L1", "B6"],
            "is_laundering": 0,
            "transactions": [{"is_laundering": 0}],
        },
    ]

    sampler = NegativeCycleSampler()
    benign = sampler.filter_benign_candidates(candidates, forbidden_accounts={"L1", "L2", "L3"})

    assert len(benign) == 1
    assert benign[0]["candidate_id"] == "c1"


def test_stratified_length_matching():
    """Verify negative candidates are sampled with exact matching cycle lengths k in {3,4,5,6}."""
    positives = [
        {"candidate_id": "p3_1", "cycle_length": 3, "nodes": ["P1", "P2", "P3"], "is_laundering": 1},
        {"candidate_id": "p3_2", "cycle_length": 3, "nodes": ["P4", "P5", "P6"], "is_laundering": 1},
        {"candidate_id": "p4_1", "cycle_length": 4, "nodes": ["P7", "P8", "P9", "P10"], "is_laundering": 1},
    ]

    all_candidates = [
        # Benign k=3 (3 available)
        {"candidate_id": "b3_1", "cycle_length": 3, "nodes": ["B1", "B2", "B3"], "is_laundering": 0, "transactions": [{"is_laundering": 0}]},
        {"candidate_id": "b3_2", "cycle_length": 3, "nodes": ["B4", "B5", "B6"], "is_laundering": 0, "transactions": [{"is_laundering": 0}]},
        {"candidate_id": "b3_3", "cycle_length": 3, "nodes": ["B7", "B8", "B9"], "is_laundering": 0, "transactions": [{"is_laundering": 0}]},
        # Benign k=4 (2 available)
        {"candidate_id": "b4_1", "cycle_length": 4, "nodes": ["B10", "B11", "B12", "B13"], "is_laundering": 0, "transactions": [{"is_laundering": 0}]},
        {"candidate_id": "b4_2", "cycle_length": 4, "nodes": ["B14", "B15", "B16", "B17"], "is_laundering": 0, "transactions": [{"is_laundering": 0}]},
    ]

    sampler = NegativeCycleSampler(negative_ratio=1.0)
    negatives = sampler.sample_matched_negatives(positives, all_candidates)

    # We requested ratio 1.0 -> 2 of k=3 and 1 of k=4
    assert len(negatives) == 3
    neg_k3 = [n for n in negatives if n["cycle_length"] == 3]
    neg_k4 = [n for n in negatives if n["cycle_length"] == 4]
    assert len(neg_k3) == 2
    assert len(neg_k4) == 1

    # Check all sampled negatives have label 0 and is_laundering 0
    for n in negatives:
        assert n["label"] == 0
        assert n["is_laundering"] == 0

    report = sampler.get_sampling_report()
    assert report["total_positives"] == 3
    assert report["total_sampled_negatives"] == 3
    assert report["positive_counts_by_length"][3] == 2
    assert report["negative_counts_by_length"][3] == 2
