"""Unit tests for CandidateExample data schema, label-blind extraction interface, and group-safe splitters."""

import pytest

from source.data.candidate_extractor import CandidateExample, CandidateExtractor
from source.data.splits import GroupSafeSplitter, verify_split_disjointness


def test_candidate_example_serialization():
    """Verify CandidateExample dataclass creation, serialization, and round-trip fidelity."""
    c = CandidateExample(
        candidate_id="cand_001",
        dataset_track="amlworld",
        temporal_bounds=(100.0, 200.0),
        participant_ids=["acc_1", "acc_2", "acc_3"],
        edges=[
            ("acc_1", "acc_2", {"amount": 100.0, "timestamp": 120.0}),
            ("acc_2", "acc_3", {"amount": 95.0, "timestamp": 150.0}),
        ],
        node_features={"acc_1": [1.0, 0.0], "acc_2": [0.5, 0.5], "acc_3": [0.0, 1.0]},
        target_y=1,
        typology_label="FAN-OUT",
        group_id="group_alpha",
        metadata={"source": "HI-Small"},
    )

    d = c.to_dict()
    assert d["candidate_id"] == "cand_001"
    assert d["target_y"] == 1
    assert d["typology_label"] == "FAN-OUT"
    assert len(d["edges"]) == 2

    # Round trip
    c_reconstructed = CandidateExample.from_dict(d)
    assert c_reconstructed.candidate_id == c.candidate_id
    assert c_reconstructed.participant_ids == c.participant_ids
    assert c_reconstructed.temporal_bounds == c.temporal_bounds
    assert c_reconstructed.target_y == c.target_y


def test_group_safe_split_disjointness():
    """Verify GroupSafeSplitter partitions candidates with zero participant account leakage."""
    candidates = []
    # Create 12 distinct groups with some internal sharing
    for grp_idx in range(12):
        grp_id = f"grp_{grp_idx}"
        base_acc = f"acc_hub_{grp_idx}"
        for i in range(3):
            cand_id = f"c_{grp_idx}_{i}"
            accs = [base_acc, f"acc_leaf_{grp_idx}_{i}", f"acc_leaf_{grp_idx}_{(i+1)%3}"]
            cand = CandidateExample(
                candidate_id=cand_id,
                dataset_track="amlworld",
                temporal_bounds=(0.0, 10.0),
                participant_ids=accs,
                edges=[(accs[0], accs[1], {}), (accs[1], accs[2], {})],
                node_features={a: [1.0] for a in accs},
                target_y=1 if grp_idx % 2 == 0 else 0,
                typology_label="CYCLE" if grp_idx % 2 == 0 else "HARD-NEGATIVE",
                group_id=grp_id,
            )
            candidates.append(cand)

    splitter = GroupSafeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=123)
    train, val, test = splitter.split(candidates)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == len(candidates)

    audit = verify_split_disjointness(train, val, test)
    assert audit["is_disjoint"] is True, f"Overlap detected: {audit}"
    assert len(audit["train_val_overlap"]) == 0
    assert len(audit["train_test_overlap"]) == 0
    assert len(audit["val_test_overlap"]) == 0


def test_leakage_detection():
    """Verify verify_split_disjointness detects and flags deliberate account leakage."""
    c1 = CandidateExample(
        candidate_id="c1",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 1.0),
        participant_ids=["shared_account", "node_a"],
        edges=[],
        node_features={},
        target_y=1,
        typology_label="CYCLE",
        group_id="g1",
    )
    c2 = CandidateExample(
        candidate_id="c2",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 1.0),
        participant_ids=["shared_account", "node_b"],
        edges=[],
        node_features={},
        target_y=0,
        typology_label="HARD-NEGATIVE",
        group_id="g2",
    )
    c3 = CandidateExample(
        candidate_id="c3",
        dataset_track="amlworld",
        temporal_bounds=(0.0, 1.0),
        participant_ids=["node_c", "node_d"],
        edges=[],
        node_features={},
        target_y=0,
        typology_label="HARD-NEGATIVE",
        group_id="g3",
    )

    # Deliberately place c1 in train and c2 in val (shares 'shared_account')
    audit = verify_split_disjointness([c1], [c2], [c3])
    assert audit["is_disjoint"] is False
    assert "shared_account" in audit["train_val_overlap"]


def test_candidate_extractor_base_class():
    """Verify CandidateExtractor raises NotImplementedError when not subclassed."""
    extractor = CandidateExtractor(dataset_track="amlworld")
    with pytest.raises(NotImplementedError):
        extractor.extract_candidates(graph=None)
