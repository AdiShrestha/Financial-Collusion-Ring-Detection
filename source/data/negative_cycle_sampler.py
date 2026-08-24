"""Length-matched benign negative cycle sampler for AML candidate subgraphs.

Contract C11-03 (T-DESC): Samples pure benign directed cycle candidate subgraphs
matched to positive laundering rings on exact cycle length k in {3,4,5,6}, edge count,
and duration, ensuring unconfounded topological evaluation.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple


class NegativeCycleSampler:
    """Samples length-matched benign directed cycle candidate subgraphs."""

    def __init__(self, negative_ratio: float = 1.0, seed: int = 42):
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.rng = random.Random(seed)
        self.last_sampling_report: Dict[str, Any] = {}

    def filter_benign_candidates(
        self,
        candidates: List[Dict[str, Any]],
        forbidden_accounts: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter candidates to retain only 100% benign cycles with zero laundering overlap."""
        forbidden = set(str(a) for a in (forbidden_accounts or set()))
        benign = []

        for c in candidates:
            # Must not be labeled laundering
            if int(c.get("is_laundering", 0)) != 0 or int(c.get("label", 0)) != 0:
                continue

            # Must have zero laundering transactions
            txs = c.get("transactions", [])
            if any(int(tx.get("is_laundering", 0)) != 0 for tx in txs):
                continue

            # Must not contain forbidden/laundering accounts
            participants = set(str(p) for p in c.get("participants", c.get("nodes", [])))
            if participants & forbidden:
                continue

            benign.append(c)

        return benign

    def sample_matched_negatives(
        self,
        positive_candidates: List[Dict[str, Any]],
        all_candidates: List[Dict[str, Any]],
        forbidden_accounts: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample negative candidates stratified by exact cycle length k matching positive candidates."""
        # Separate positives and pool of benign candidates
        positives_by_k: Dict[int, List[Dict[str, Any]]] = {}
        for p in positive_candidates:
            k = int(p.get("cycle_length", len(p.get("nodes", []))))
            if k not in positives_by_k:
                positives_by_k[k] = []
            positives_by_k[k].append(p)

        # Collect forbidden accounts from positives if not explicitly supplied
        if forbidden_accounts is None:
            forbidden_accounts = set()
            for p in positive_candidates:
                for a in p.get("participants", p.get("nodes", [])):
                    forbidden_accounts.add(str(a))

        benign_pool = self.filter_benign_candidates(all_candidates, forbidden_accounts=forbidden_accounts)
        benign_by_k: Dict[int, List[Dict[str, Any]]] = {}
        for b in benign_pool:
            k = int(b.get("cycle_length", len(b.get("nodes", []))))
            if k not in benign_by_k:
                benign_by_k[k] = []
            benign_by_k[k].append(b)

        sampled_negatives: List[Dict[str, Any]] = []
        pos_counts: Dict[int, int] = {}
        neg_counts: Dict[int, int] = {}
        neg_counter = 1

        for k, pos_list in sorted(positives_by_k.items()):
            n_pos = len(pos_list)
            pos_counts[k] = n_pos
            target_n_neg = max(1, int(round(n_pos * self.negative_ratio)))

            available_benign = benign_by_k.get(k, [])
            if len(available_benign) <= target_n_neg:
                chosen = list(available_benign)
            else:
                # Deterministic selection with closest duration match
                pos_durations = [float(p.get("duration_seconds", 0.0)) for p in pos_list]
                mean_pos_dur = sum(pos_durations) / len(pos_durations) if pos_durations else 0.0

                # Sort by distance to mean positive duration, then sample deterministically
                sorted_benign = sorted(
                    available_benign,
                    key=lambda b: (abs(float(b.get("duration_seconds", 0.0)) - mean_pos_dur), b.get("candidate_id", "")),
                )
                chosen = sorted_benign[:target_n_neg]

            for cand in chosen:
                cand_copy = dict(cand)
                cand_copy["candidate_id"] = f"cand_neg_{k}_{neg_counter:04d}"
                cand_copy["is_laundering"] = 0
                cand_copy["label"] = 0
                sampled_negatives.append(cand_copy)
                neg_counter += 1

            neg_counts[k] = len(chosen)

        # Generate report
        self.last_sampling_report = {
            "total_positives": len(positive_candidates),
            "total_sampled_negatives": len(sampled_negatives),
            "positive_counts_by_length": pos_counts,
            "negative_counts_by_length": neg_counts,
            "negative_ratio_target": self.negative_ratio,
            "laundering_leakage_detected": False,
        }

        return sampled_negatives

    def get_sampling_report(self) -> Dict[str, Any]:
        """Return metrics and matching balance report from the last sampling pass."""
        return dict(self.last_sampling_report)
