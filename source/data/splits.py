"""Group-safe dataset splitting utilities guaranteeing zero participant leakage across partitions."""

import random
from typing import Any, Dict, List, Sequence, Set, Tuple
import networkx as nx

from source.data.candidate_extractor import CandidateExample


def verify_split_disjointness(
    train_candidates: Sequence[CandidateExample],
    val_candidates: Sequence[CandidateExample],
    test_candidates: Sequence[CandidateExample],
) -> Dict[str, Any]:
    """Verify that participant account sets are completely disjoint across train, val, and test splits.

    In accordance with INV-006, zero accounts may overlap between splits.

    Returns:
        Dictionary with is_disjoint boolean, pairwise overlap sets, and candidate/account counts.
    """
    train_accounts: Set[str] = set()
    for c in train_candidates:
        train_accounts.update(c.participant_ids)

    val_accounts: Set[str] = set()
    for c in val_candidates:
        val_accounts.update(c.participant_ids)

    test_accounts: Set[str] = set()
    for c in test_candidates:
        test_accounts.update(c.participant_ids)

    overlap_train_val = train_accounts.intersection(val_accounts)
    overlap_train_test = train_accounts.intersection(test_accounts)
    overlap_val_test = val_accounts.intersection(test_accounts)

    is_disjoint = (
        len(overlap_train_val) == 0
        and len(overlap_train_test) == 0
        and len(overlap_val_test) == 0
    )

    return {
        "is_disjoint": is_disjoint,
        "train_val_overlap": sorted(list(overlap_train_val)),
        "train_test_overlap": sorted(list(overlap_train_test)),
        "val_test_overlap": sorted(list(overlap_val_test)),
        "counts": {
            "train_candidates": len(train_candidates),
            "val_candidates": len(val_candidates),
            "test_candidates": len(test_candidates),
            "train_participants": len(train_accounts),
            "val_participants": len(val_accounts),
            "test_participants": len(test_accounts),
        },
    }


class GroupSafeSplitter:
    """Partitions CandidateExamples into leakage-free train, validation, and test sets.

    Guarantees:
    1. All candidates sharing accounts or group_ids belong to the same split.
    2. Zero account leakage between splits (INV-006).
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self,
        candidates: Sequence[CandidateExample],
    ) -> Tuple[List[CandidateExample], List[CandidateExample], List[CandidateExample]]:
        """Group candidates into connected account clusters and assign to splits."""
        if not candidates:
            return [], [], []

        # Build candidate overlap graph
        g = nx.Graph()
        account_to_candidates: Dict[str, List[int]] = {}
        group_to_candidates: Dict[str, List[int]] = {}

        for idx, c in enumerate(candidates):
            g.add_node(idx)
            for acc in c.participant_ids:
                account_to_candidates.setdefault(acc, []).append(idx)
            if c.group_id:
                group_to_candidates.setdefault(c.group_id, []).append(idx)

        # Add edges between candidates sharing an account
        for acc, c_indices in account_to_candidates.items():
            for i in range(len(c_indices) - 1):
                g.add_edge(c_indices[i], c_indices[i + 1])

        # Add edges between candidates sharing a group_id
        for grp, c_indices in group_to_candidates.items():
            for i in range(len(c_indices) - 1):
                g.add_edge(c_indices[i], c_indices[i + 1])

        # Find connected components (clusters)
        clusters = list(nx.connected_components(g))

        # Deterministic shuffle with seed
        rng = random.Random(self.seed)
        cluster_list = [list(comp) for comp in clusters]
        rng.shuffle(cluster_list)

        # Sort clusters by size descending for smooth bin-packing
        cluster_list.sort(key=lambda c: len(c), reverse=True)

        total_candidates = len(candidates)
        target_train = int(total_candidates * self.train_ratio)
        target_val = int(total_candidates * self.val_ratio)
        target_test = total_candidates - target_train - target_val

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        # Greedy allocation by remaining deficit
        for cluster in cluster_list:
            c_len = len(cluster)
            train_deficit = target_train - len(train_indices)
            val_deficit = target_val - len(val_indices)
            test_deficit = target_test - len(test_indices)

            # Assign to bucket with largest positive deficit
            deficits = [
                (train_deficit, "train"),
                (val_deficit, "val"),
                (test_deficit, "test"),
            ]
            deficits.sort(key=lambda x: x[0], reverse=True)
            chosen_bucket = deficits[0][1]

            if chosen_bucket == "train":
                train_indices.extend(cluster)
            elif chosen_bucket == "val":
                val_indices.extend(cluster)
            else:
                test_indices.extend(cluster)

        train_set = [candidates[i] for i in train_indices]
        val_set = [candidates[i] for i in val_indices]
        test_set = [candidates[i] for i in test_indices]

        return train_set, val_set, test_set
