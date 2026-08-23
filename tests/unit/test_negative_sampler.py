"""Unit tests for HardNegativeSampler and structural cohort builder."""

import pytest

from source.data.candidate_extractor import CandidateExample
from source.data.negative_sampler import HardNegativeSampler


def create_mock_candidate(cand_id: str, k: int, is_pos: bool, accounts: list[str]) -> CandidateExample:
    return CandidateExample(
        candidate_id=cand_id,
        dataset_track="amlworld",
        temporal_bounds=(0.0, 10.0),
        participant_ids=accounts,
        edges=[(accounts[i], accounts[(i + 1) % k], {"is_laundering": 1 if is_pos else 0}) for i in range(k)],
        node_features={a: [0.0] * 56 for a in accounts},
        target_y=1 if is_pos else 0,
        typology_label="CYCLE" if is_pos else "NEGATIVE_CANDIDATE",
        group_id=f"group_{cand_id}",
        metadata={"cycle_length": k},
    )


def test_negative_sampling_label_purity():
    """Verify sampled hard negatives contain zero positive laundering transactions or accounts."""
    pos_candidates = [
        create_mock_candidate("p1", 3, True, ["pos_1", "pos_2", "pos_3"]),
        create_mock_candidate("p2", 4, True, ["pos_4", "pos_5", "pos_6", "pos_7"]),
    ]

    neg_pool = [
        create_mock_candidate("n1", 3, False, ["neg_1", "neg_2", "neg_3"]),
        create_mock_candidate("n2", 4, False, ["neg_4", "neg_5", "neg_6", "neg_7"]),
        create_mock_candidate("n3_dirty", 3, False, ["pos_1", "neg_8", "neg_9"]),  # Contaminated with pos_1
    ]

    sampler = HardNegativeSampler(match_ratio=1.0, seed=42)
    sampled_negs, manifest = sampler.sample_hard_negatives(
        candidate_pool=neg_pool,
        positive_candidates=pos_candidates,
    )

    assert len(sampled_negs) == 2
    assert manifest["negative_contamination_count"] == 0

    sampled_ids = {c.candidate_id for c in sampled_negs}
    assert "n1" in sampled_ids
    assert "n2" in sampled_ids
    assert "n3_dirty" not in sampled_ids

    for neg in sampled_negs:
        assert neg.target_y == 0
        assert neg.typology_label.startswith("HARD_NEGATIVE_")
        for p in neg.participant_ids:
            assert not p.startswith("pos_")


def test_structural_property_matching():
    """Verify that sampled negatives match cycle lengths of positive candidates."""
    pos_candidates = [
        create_mock_candidate("p1", 3, True, ["p_a1", "p_a2", "p_a3"]),
        create_mock_candidate("p2", 5, True, ["p_b1", "p_b2", "p_b3", "p_b4", "p_b5"]),
    ]

    neg_pool = [
        create_mock_candidate("n_k3", 3, False, ["n_a1", "n_a2", "n_a3"]),
        create_mock_candidate("n_k4", 4, False, ["n_b1", "n_b2", "n_b3", "n_b4"]),
        create_mock_candidate("n_k5", 5, False, ["n_c1", "n_c2", "n_c3", "n_c4", "n_c5"]),
    ]

    sampler = HardNegativeSampler(match_ratio=1.0, seed=42)
    sampled_negs, manifest = sampler.sample_hard_negatives(neg_pool, pos_candidates)

    assert len(sampled_negs) == 2
    lens = {c.metadata["cycle_length"] for c in sampled_negs}
    assert lens == {3, 5}


def test_sampling_determinism():
    """Verify that two runs with identical seed return identical results."""
    pos = [create_mock_candidate(f"p_{i}", 3, True, [f"pos_{i}_{j}" for j in range(3)]) for i in range(5)]
    neg_pool = [create_mock_candidate(f"n_{i}", 3, False, [f"neg_{i}_{j}" for j in range(3)]) for i in range(20)]

    s1 = HardNegativeSampler(match_ratio=1.0, seed=1234)
    res1, _ = s1.sample_hard_negatives(neg_pool, pos)

    s2 = HardNegativeSampler(match_ratio=1.0, seed=1234)
    res2, _ = s2.sample_hard_negatives(neg_pool, pos)

    assert [c.candidate_id for c in res1] == [c.candidate_id for c in res2]
