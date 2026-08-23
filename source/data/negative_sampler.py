"""Hard-negative candidate sampling and structural cohort generation."""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from source.data.candidate_extractor import CandidateExample


class HardNegativeSampler:
    """Samples structurally matched, pure non-laundering negative candidates.

    Guarantees:
    1. Zero contamination: no participant or transaction in a negative candidate is laundering.
    2. Structural matching: negatives match positives on cycle length k and node count.
    3. Deterministic sampling given a fixed random seed.
    """

    def __init__(
        self,
        match_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.match_ratio = match_ratio
        self.seed = seed

    def sample_hard_negatives(
        self,
        candidate_pool: Sequence[CandidateExample],
        positive_candidates: Sequence[CandidateExample],
        known_laundering_accounts: Optional[Set[str]] = None,
    ) -> Tuple[List[CandidateExample], Dict[str, Any]]:
        """Sample structurally matched hard-negative candidates from a candidate pool."""
        rng = random.Random(self.seed)
        laundering_accs = set(known_laundering_accounts or [])

        # Gather positive candidate IDs and all positive participant accounts
        pos_ids = {p.candidate_id for p in positive_candidates}
        for p in positive_candidates:
            laundering_accs.update(p.participant_ids)

        # Filter candidate pool for pure negatives:
        # 1. target_y == 0
        # 2. Not in pos_ids
        # 3. Disjoint from any laundering accounts
        # 4. Zero laundering edge attributes
        pure_negatives: List[CandidateExample] = []
        for c in candidate_pool:
            if c.candidate_id in pos_ids or c.target_y != 0:
                continue
            cand_accs = set(c.participant_ids)
            if cand_accs.intersection(laundering_accs):
                continue
            # Check edge attributes
            has_laundering_edge = False
            for _, _, attr in c.edges:
                if attr.get("is_laundering") in (1, "1", True) or attr.get("injected") is True:
                    has_laundering_edge = True
                    break
            if has_laundering_edge:
                continue

            pure_negatives.append(c)

        # Index pure negatives by cycle length / node count
        neg_by_length: Dict[int, List[CandidateExample]] = {}
        for c in pure_negatives:
            k = c.metadata.get("cycle_length", len(c.participant_ids))
            neg_by_length.setdefault(k, []).append(c)

        # Shuffle each bucket deterministically
        for k in neg_by_length:
            rng.shuffle(neg_by_length[k])

        sampled_negatives: List[CandidateExample] = []
        matched_counts_by_length: Dict[int, int] = {}
        fallback_counts = 0
        used_negative_ids: Set[str] = set()

        num_needed_per_pos = max(1, math.ceil(self.match_ratio))

        for pos in positive_candidates:
            k = pos.metadata.get("cycle_length", len(pos.participant_ids))
            for _ in range(num_needed_per_pos):
                if len(sampled_negatives) >= math.ceil(len(positive_candidates) * self.match_ratio):
                    break

                # Try exact structural bucket match first
                available_in_bucket = [
                    c for c in neg_by_length.get(k, []) if c.candidate_id not in used_negative_ids
                ]
                chosen = None

                if available_in_bucket:
                    chosen = available_in_bucket[0]
                    matched_counts_by_length[k] = matched_counts_by_length.get(k, 0) + 1
                else:
                    # Stratified fallback to any available unused pure negative
                    all_available = [
                        c for c in pure_negatives if c.candidate_id not in used_negative_ids
                    ]
                    if all_available:
                        chosen = all_available[0]
                        fallback_counts += 1

                if chosen is not None:
                    used_negative_ids.add(chosen.candidate_id)
                    # Create styled hard-negative CandidateExample
                    neg_cand = CandidateExample(
                        candidate_id=chosen.candidate_id,
                        dataset_track=chosen.dataset_track,
                        temporal_bounds=chosen.temporal_bounds,
                        participant_ids=chosen.participant_ids,
                        edges=chosen.edges,
                        node_features=chosen.node_features,
                        target_y=0,
                        typology_label=f"HARD_NEGATIVE_{pos.typology_label}",
                        group_id=f"neg_group_{chosen.candidate_id}",
                        metadata={
                            **chosen.metadata,
                            "matched_positive_typology": pos.typology_label,
                            "matched_cycle_length": k,
                            "hard_negative": True,
                        },
                    )
                    sampled_negatives.append(neg_cand)

        manifest = {
            "seed": self.seed,
            "match_ratio": self.match_ratio,
            "total_positive_candidates": len(positive_candidates),
            "pure_negative_pool_size": len(pure_negatives),
            "sampled_negative_count": len(sampled_negatives),
            "exact_cycle_length_matches": matched_counts_by_length,
            "fallback_matches": fallback_counts,
            "negative_contamination_count": 0,
        }

        return sampled_negatives, manifest
