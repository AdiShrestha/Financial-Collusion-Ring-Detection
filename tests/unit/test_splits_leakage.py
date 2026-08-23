"""Unit tests for group-safe dataset partitioning and leakage invariant verification."""

import json
import os
import pytest

from source.data.candidate_extractor import CandidateExample
from source.data.splits import GroupSafeSplitter, verify_split_disjointness


def generate_synthetic_cohort(num_groups: int = 50, cands_per_group: int = 4) -> list[CandidateExample]:
    """Generate multi-group cohort with overlapping participant accounts within groups."""
    candidates = []
    typologies = ["CYCLE", "FAN-OUT", "BIPARTITE", "HARD_NEGATIVE_CYCLE"]

    for g_idx in range(num_groups):
        grp_id = f"group_{g_idx:04d}"
        hub_account = f"acc_hub_{g_idx}"
        typology = typologies[g_idx % len(typologies)]
        is_pos = not typology.startswith("HARD_NEGATIVE")

        for c_idx in range(cands_per_group):
            cand_id = f"cand_{g_idx:04d}_{c_idx:02d}"
            leaf_1 = f"acc_leaf_{g_idx}_{c_idx * 2}"
            leaf_2 = f"acc_leaf_{g_idx}_{c_idx * 2 + 1}"
            accounts = [hub_account, leaf_1, leaf_2]

            cand = CandidateExample(
                candidate_id=cand_id,
                dataset_track="amlworld" if g_idx % 2 == 0 else "elliptic_actors",
                temporal_bounds=(float(g_idx), float(g_idx + 1)),
                participant_ids=accounts,
                edges=[(accounts[0], accounts[1], {}), (accounts[1], accounts[2], {})],
                node_features={a: [0.0] * 56 for a in accounts},
                target_y=1 if is_pos else 0,
                typology_label=typology,
                group_id=grp_id,
            )
            candidates.append(cand)

    return candidates


def test_full_cohort_split_disjointness():
    """Verify GroupSafeSplitter partitions a full 200-candidate cohort with zero participant leakage."""
    cohort = generate_synthetic_cohort(num_groups=50, cands_per_group=4)  # 200 candidates
    assert len(cohort) == 200

    splitter = GroupSafeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    train, val, test = splitter.split(cohort)

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train) + len(val) + len(test) == 200

    audit = verify_split_disjointness(train, val, test)
    assert audit["is_disjoint"] is True
    assert len(audit["train_val_overlap"]) == 0
    assert len(audit["train_test_overlap"]) == 0
    assert len(audit["val_test_overlap"]) == 0


def test_split_manifest_generation_and_integrity(tmp_path):
    """Verify split manifest generation, JSON serialization, checksums, and audit fields."""
    cohort = generate_synthetic_cohort(num_groups=40, cands_per_group=3)
    splitter = GroupSafeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    train, val, test = splitter.split(cohort)

    manifest_path = os.path.join(tmp_path, "split_manifest.json")
    manifest = splitter.generate_and_save_split_manifest(train, val, test, manifest_path)

    assert os.path.isfile(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["disjointness_audit"]["is_disjoint"] is True
    assert "splits" in loaded
    assert "train" in loaded["splits"]
    assert "validation" in loaded["splits"]
    assert "test" in loaded["splits"]
    assert len(loaded["splits"]["train"]["sha256_checksum"]) == 64


def test_typology_representation_across_splits():
    """Verify that primary typologies are represented across train, val, and test partitions."""
    cohort = generate_synthetic_cohort(num_groups=60, cands_per_group=3)
    splitter = GroupSafeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    train, val, test = splitter.split(cohort)

    train_typos = {c.typology_label for c in train}
    val_typos = {c.typology_label for c in val}
    test_typos = {c.typology_label for c in test}

    assert "CYCLE" in train_typos
    assert len(train_typos) >= 3
    assert len(val_typos) >= 2
    assert len(test_typos) >= 2
