"""
Group-Safe Stratified Splitting Engine for Financial Collusion-Ring Detection.

Upholds Invariant INV-007 (Leakage-Free Group-Safe Splitting):
Splitting is performed at the pattern/cluster level such that no transaction
or account overlaps between training, validation, and test splits.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class DataLeakageError(Exception):
    """Raised when accounts or transactions are shared across cross-validation folds."""
    pass


@dataclass
class FoldAuditSummary:
    """Audit summary proving zero leakage and reporting fold balance."""
    n_splits: int
    total_candidates: int
    zero_node_leakage: bool
    zero_edge_leakage: bool
    fold_candidate_counts: Dict[int, int]
    fold_positive_counts: Dict[int, int]
    fold_positive_ratios: Dict[int, float]
    fold_node_counts: Dict[int, int]
    fold_edge_counts: Dict[int, int]


class DisjointSetUnion:
    """Disjoint Set Union (DSU) with path compression and union by rank."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


class GroupSafeSplitter:
    """Partitions candidate subgraphs into leakage-free stratified folds."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.seed = seed

    @staticmethod
    def _extract_candidate_attrs(
        cand: Any,
    ) -> Tuple[str, Set[str], Set[str], int, Optional[str], Optional[str]]:
        """Extracts standard attributes from either CandidateSubgraph or dict."""
        if hasattr(cand, "candidate_id"):
            cand_id = cand.candidate_id
            nodes = set(cand.nodes)
            edges = set()
            for e in cand.edges:
                if isinstance(e, dict):
                    edges.add(str(e.get("tx_id", f"{e.get('source')}_{e.get('target')}_{e.get('timestamp', '')}")))
                else:
                    edges.add(str(e))
            label = getattr(cand, "label", None)
            pattern_id = getattr(cand, "pattern_id", None)
            matched_group_id = getattr(cand, "matched_group_id", None)
        elif isinstance(cand, dict):
            cand_id = cand["candidate_id"]
            nodes = set(cand.get("nodes", []))
            edges = set()
            for e in cand.get("edges", []):
                if isinstance(e, dict):
                    edges.add(str(e.get("tx_id", f"{e.get('source')}_{e.get('target')}_{e.get('timestamp', '')}")))
                else:
                    edges.add(str(e))
            label = cand.get("label")
            pattern_id = cand.get("pattern_id")
            matched_group_id = cand.get("matched_group_id")
        else:
            raise TypeError(f"Unsupported candidate object type: {type(cand)}")
        if label not in (0, 1):
            raise ValueError(f"Candidate {cand_id} requires an explicit binary label, got {label!r}")
        return (
            str(cand_id), {str(n) for n in nodes}, edges, int(label),
            (str(pattern_id) if pattern_id else None),
            (str(matched_group_id) if matched_group_id else None),
        )

    def compute_connected_groups(self, candidates: List[Any]) -> List[List[str]]:
        """
        Computes disjoint connected groups of candidates based on shared nodes,
        shared transactions, or shared pattern IDs using Disjoint Set Union.
        """
        if not candidates:
            return []

        dsu = DisjointSetUnion()
        node_to_cand: Dict[str, str] = {}
        edge_to_cand: Dict[str, str] = {}
        pattern_to_cand: Dict[str, str] = {}
        matched_group_to_cand: Dict[str, str] = {}

        for cand in candidates:
            cand_id, nodes, edges, _, pattern_id, matched_group_id = self._extract_candidate_attrs(cand)
            dsu.find(cand_id)

            for node in nodes:
                if node in node_to_cand:
                    dsu.union(cand_id, node_to_cand[node])
                else:
                    node_to_cand[node] = cand_id

            for edge in edges:
                if edge in edge_to_cand:
                    dsu.union(cand_id, edge_to_cand[edge])
                else:
                    edge_to_cand[edge] = cand_id

            if pattern_id:
                if pattern_id in pattern_to_cand:
                    dsu.union(cand_id, pattern_to_cand[pattern_id])
                else:
                    pattern_to_cand[pattern_id] = cand_id

            if matched_group_id:
                if matched_group_id in matched_group_to_cand:
                    dsu.union(cand_id, matched_group_to_cand[matched_group_id])
                else:
                    matched_group_to_cand[matched_group_id] = cand_id

        # Group candidate IDs by their root component
        groups_by_root: Dict[str, List[str]] = defaultdict(list)
        for cand in candidates:
            cand_id, _, _, _, _, _ = self._extract_candidate_attrs(cand)
            root = dsu.find(cand_id)
            groups_by_root[root].append(cand_id)

        # Return sorted list of groups (deterministic order)
        groups = list(groups_by_root.values())
        groups.sort(key=lambda g: (len(g), sorted(g)[0]), reverse=True)
        return groups

    def split(self, candidates: List[Any]) -> Dict[str, int]:
        """
        Assigns each candidate to a fold in [0, n_splits - 1] such that:
        1. All candidates in a connected group belong to the exact same fold.
        2. Positive/negative candidate distributions are balanced across folds.
        3. Zero node or transaction overlap exists across distinct folds.
        4. Different seeds produce genuinely different fold assignments
           (via seeded shuffling of connected components before bin-packing).
        """
        if not candidates:
            return {}

        cand_map = {}
        for cand in candidates:
            cid, nodes, edges, label, pat_id, matched_group_id = self._extract_candidate_attrs(cand)
            if cid in cand_map:
                raise ValueError(f"Duplicate candidate ID: {cid}")
            cand_map[cid] = {
                "nodes": nodes, "edges": edges, "label": label,
                "pattern_id": pat_id, "matched_group_id": matched_group_id,
            }

        groups = self.compute_connected_groups(candidates)

        # Characterize each group by positive and negative counts
        group_profiles = []
        for gid, grp in enumerate(groups):
            pos_count = sum(cand_map[cid]["label"] for cid in grp)
            neg_count = len(grp) - pos_count
            group_profiles.append({
                "group_id": gid,
                "candidates": grp,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "total_count": len(grp)
            })

        # ── Seed-conditioned shuffling ──
        # Shuffle components using the seed BEFORE sorting by size.
        # This ensures different seeds produce different fold assignments
        # while the subsequent sort-by-size stabilizes bin-packing quality.
        import random as _random
        rng = _random.Random(self.seed)
        rng.shuffle(group_profiles)

        # Stable sort by descending size after shuffle (ties broken by shuffle order)
        group_profiles.sort(
            key=lambda p: (p["pos_count"], p["total_count"]),
            reverse=True
        )

        # Multi-objective greedy bin-packing into n_splits folds
        fold_pos = [0] * self.n_splits
        fold_tot = [0] * self.n_splits
        fold_neg = [0] * self.n_splits
        candidate_to_fold: Dict[str, int] = {}

        for profile in group_profiles:
            # Pick fold with minimum positive count; break ties with minimum total count
            if profile["pos_count"] and not profile["neg_count"]:
                key = lambda f: (fold_pos[f], fold_tot[f], f)
            elif profile["neg_count"] and not profile["pos_count"]:
                key = lambda f: (fold_neg[f], fold_tot[f], f)
            else:
                key = lambda f: (fold_tot[f], fold_pos[f], fold_neg[f], f)
            best_fold = min(range(self.n_splits), key=key)

            for cid in profile["candidates"]:
                candidate_to_fold[cid] = best_fold

            fold_pos[best_fold] += profile["pos_count"]
            fold_neg[best_fold] += profile["neg_count"]
            fold_tot[best_fold] += profile["total_count"]

        # Run mandatory leakage audit before returning
        self.verify_zero_leakage(candidates, candidate_to_fold)
        return candidate_to_fold

    def three_way_split(self, candidates: List[Any], test_fold: int) -> Tuple[List[Any], List[Any], List[Any], Dict[str, int]]:
        """Hold out two whole, disjoint folds for validation and test."""
        if self.n_splits < 3:
            raise ValueError("Three-way evaluation requires at least three folds")
        assignments = self.split(candidates)
        if not 0 <= test_fold < self.n_splits:
            raise ValueError(f"Invalid test fold: {test_fold}")
        val_fold = (test_fold + 1) % self.n_splits
        test = [c for c in candidates if assignments[self._extract_candidate_attrs(c)[0]] == test_fold]
        val = [c for c in candidates if assignments[self._extract_candidate_attrs(c)[0]] == val_fold]
        train = [c for c in candidates if assignments[self._extract_candidate_attrs(c)[0]] not in (test_fold, val_fold)]
        for name, partition in (("train", train), ("validation", val), ("test", test)):
            labels = {self._extract_candidate_attrs(c)[3] for c in partition}
            if labels != {0, 1}:
                raise ValueError(f"{name} fold lacks both classes: labels={sorted(labels)}; revise the cohort or split protocol")
        return train, val, test, assignments

    def verify_zero_leakage(
        self,
        candidates: List[Any],
        split_assignments: Dict[str, int]
    ) -> FoldAuditSummary:
        """
        Mechanically audits split assignments to verify Invariant INV-007.
        Raises DataLeakageError if ANY node or edge appears in more than one fold.
        """
        fold_nodes: Dict[int, Set[str]] = defaultdict(set)
        fold_edges: Dict[int, Set[str]] = defaultdict(set)
        fold_cand_counts: Dict[int, int] = defaultdict(int)
        fold_pos_counts: Dict[int, int] = defaultdict(int)
        matched_group_folds: Dict[str, int] = {}

        for cand in candidates:
            cid, nodes, edges, label, _, matched_group_id = self._extract_candidate_attrs(cand)
            if cid not in split_assignments:
                raise ValueError(f"Candidate {cid} missing from split assignments.")
            f = split_assignments[cid]
            if f < 0 or f >= self.n_splits:
                raise ValueError(f"Fold index {f} out of bounds for n_splits={self.n_splits}")
            if matched_group_id:
                previous_fold = matched_group_folds.setdefault(matched_group_id, f)
                if previous_fold != f:
                    raise DataLeakageError(
                        f"Matched group {matched_group_id} split across folds {previous_fold} and {f}"
                    )

            fold_cand_counts[f] += 1
            if label == 1:
                fold_pos_counts[f] += 1

            # Check for node leakage against all other folds
            for other_f, other_nodes in fold_nodes.items():
                if other_f != f:
                    leaked_nodes = nodes & other_nodes
                    if leaked_nodes:
                        raise DataLeakageError(
                            f"NODE LEAKAGE DETECTED between fold {f} and fold {other_f}: "
                            f"{sorted(leaked_nodes)[:5]} ({len(leaked_nodes)} nodes leaked)."
                        )

            # Check for edge leakage against all other folds
            for other_f, other_edges in fold_edges.items():
                if other_f != f:
                    leaked_edges = edges & other_edges
                    if leaked_edges:
                        raise DataLeakageError(
                            f"TRANSACTION LEAKAGE DETECTED between fold {f} and fold {other_f}: "
                            f"{sorted(leaked_edges)[:5]} ({len(leaked_edges)} transactions leaked)."
                        )

            fold_nodes[f].update(nodes)
            fold_edges[f].update(edges)

        # Build summary
        fold_pos_ratios = {}
        for f in range(self.n_splits):
            tot = fold_cand_counts[f]
            pos = fold_pos_counts[f]
            fold_pos_ratios[f] = float(pos / tot) if tot > 0 else 0.0

        return FoldAuditSummary(
            n_splits=self.n_splits,
            total_candidates=len(candidates),
            zero_node_leakage=True,
            zero_edge_leakage=True,
            fold_candidate_counts=dict(fold_cand_counts),
            fold_positive_counts=dict(fold_pos_counts),
            fold_positive_ratios=fold_pos_ratios,
            fold_node_counts={f: len(nodes) for f, nodes in fold_nodes.items()},
            fold_edge_counts={f: len(edges) for f, edges in fold_edges.items()}
        )
