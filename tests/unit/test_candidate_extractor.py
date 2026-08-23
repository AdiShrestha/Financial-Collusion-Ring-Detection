"""Unit tests for production LabelBlindCandidateExtractor."""

import networkx as nx
import pytest

from source.data.amlworld_patterns import LaunderingPatternGroup
from source.data.candidate_extractor import (
    CandidateExample,
    LabelBlindCandidateExtractor,
)


def test_label_blind_extraction_on_toy_network():
    """Verify extractor detects directed cycles without accessing label annotations."""
    g = nx.DiGraph()

    # Cycle 1: 3-cycle [A -> B -> C -> A]
    g.add_edge("A", "B", timestamp=10.0, amount=100.0)
    g.add_edge("B", "C", timestamp=12.0, amount=100.0)
    g.add_edge("C", "A", timestamp=15.0, amount=100.0)

    # Cycle 2: 4-cycle [X -> Y -> Z -> W -> X]
    g.add_edge("X", "Y", timestamp=20.0, amount=50.0)
    g.add_edge("Y", "Z", timestamp=21.0, amount=50.0)
    g.add_edge("Z", "W", timestamp=22.0, amount=50.0)
    g.add_edge("W", "X", timestamp=25.0, amount=50.0)

    # Benign path (no cycle): [M -> N -> P]
    g.add_edge("M", "N", timestamp=30.0, amount=10.0)
    g.add_edge("N", "P", timestamp=31.0, amount=10.0)

    extractor = LabelBlindCandidateExtractor(
        dataset_track="amlworld",
        min_cycle_length=3,
        max_cycle_length=6,
        include_1hop_context=False,
    )

    candidates = extractor.extract_candidates(g)
    assert len(candidates) == 2

    # Check candidates
    cand_cycle_lens = {len(c.metadata["cycle_nodes"]) for c in candidates}
    assert cand_cycle_lens == {3, 4}

    for c in candidates:
        assert c.target_y == 0  # Unlabeled at extraction time
        assert c.typology_label == "UNLABELED"


def test_temporal_window_bounding():
    """Verify that cycles exceeding max_temporal_span are filtered."""
    g = nx.DiGraph()

    # Cycle spanning 100 seconds
    g.add_edge("1", "2", timestamp=0.0)
    g.add_edge("2", "3", timestamp=50.0)
    g.add_edge("3", "1", timestamp=100.0)

    extractor_strict = LabelBlindCandidateExtractor(max_temporal_span=50.0)
    candidates_strict = extractor_strict.extract_candidates(g)
    assert len(candidates_strict) == 0

    extractor_loose = LabelBlindCandidateExtractor(max_temporal_span=120.0)
    candidates_loose = extractor_loose.extract_candidates(g)
    assert len(candidates_loose) == 1


def test_coverage_and_ground_truth_matching():
    """Verify post-hoc ground truth matching and recall statistics."""
    g = nx.DiGraph()
    # Cycle 1
    g.add_edge("A", "B", timestamp=1.0)
    g.add_edge("B", "C", timestamp=2.0)
    g.add_edge("C", "A", timestamp=3.0)

    # Cycle 2
    g.add_edge("D", "E", timestamp=1.0)
    g.add_edge("E", "F", timestamp=2.0)
    g.add_edge("F", "D", timestamp=3.0)

    extractor = LabelBlindCandidateExtractor(include_1hop_context=False)
    candidates = extractor.extract_candidates(g)
    assert len(candidates) == 2

    # Define true pattern group matching only Cycle 1
    true_patterns = [
        LaunderingPatternGroup(
            pattern_id=101,
            typology="CYCLE",
            participant_ids=["A", "B", "C"],
            transactions=[],
            start_timestamp=1.0,
            end_timestamp=3.0,
        )
    ]

    labeled_cands, stats = LabelBlindCandidateExtractor.match_ground_truth(
        candidates=candidates,
        true_patterns=true_patterns,
        iou_threshold=0.5,
    )

    assert stats["total_candidates"] == 2
    assert stats["positive_candidates"] == 1
    assert stats["negative_candidates"] == 1
    assert stats["matched_true_patterns"] == 1
    assert stats["pattern_recall"] == 1.0

    pos_cand = [c for c in labeled_cands if c.target_y == 1][0]
    assert pos_cand.typology_label == "CYCLE"
    assert pos_cand.group_id == "pattern_101"


def test_1hop_context_subgraph():
    """Verify that 1-hop neighbors are attached to candidate subgraph."""
    g = nx.DiGraph()
    # Core 3-cycle
    g.add_edge("A", "B", timestamp=1.0)
    g.add_edge("B", "C", timestamp=2.0)
    g.add_edge("C", "A", timestamp=3.0)

    # 1-hop feeder and receiver
    g.add_edge("Feeder", "A", timestamp=0.5)
    g.add_edge("C", "Receiver", timestamp=3.5)

    extractor = LabelBlindCandidateExtractor(include_1hop_context=True)
    cands = extractor.extract_candidates(g)
    assert len(cands) == 1

    cand = cands[0]
    assert "Feeder" in cand.participant_ids
    assert "Receiver" in cand.participant_ids
    assert cand.metadata["subgraph_node_count"] == 5
