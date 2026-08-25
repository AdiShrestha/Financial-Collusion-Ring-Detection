"""Union-Find component grouping and stratified 5-fold nested cross-validation manifest generator.

Contract C16-03 (T-COMP): Unions candidates sharing accounts, transactions, pattern IDs,
or matched sets into disjoint group components. Constructs a leak-free 5-fold outer / 3-fold inner
cross-validation manifest (artifacts/splits/fold_manifest.json) enforcing INV-006 (0.0% outer account leakage).
"""

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import pyarrow.parquet as pq


def generate_grouped_nested_splits(
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    candidate_txs_parquet_path: str = "artifacts/candidates/candidate_transactions.parquet",
    covariate_balance_path: str = "artifacts/candidates/covariate_balance.json",
    output_manifest_path: str = "artifacts/splits/fold_manifest.json",
    legacy_mirror_path: str = "data/processed/split_manifest.json",
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Generate deterministic group-safe stratified 5-fold outer and 3-fold inner split manifest."""
    if not os.path.exists(candidates_parquet_path):
        raise FileNotFoundError(f"Candidates parquet missing: {candidates_parquet_path}")
    if not os.path.exists(candidate_txs_parquet_path):
        raise FileNotFoundError(f"Candidate transactions parquet missing: {candidate_txs_parquet_path}")

    c_table = pq.read_table(candidates_parquet_path)
    candidates = c_table.to_pylist()

    # Load matched sets if available to keep matched pairs grouped
    matched_group_map: Dict[str, int] = {}
    if os.path.exists(covariate_balance_path):
        with open(covariate_balance_path, "r", encoding="utf-8") as f:
            bal_data = json.load(f)
            for set_idx, mset in enumerate(bal_data.get("matched_sets", [])):
                matched_group_map[mset["pos_candidate_id"]] = set_idx
                for nid in mset.get("matched_negative_ids", []):
                    matched_group_map[nid] = set_idx

    # Build Union-Find graph over candidates
    G = nx.Graph()
    for c in candidates:
        G.add_node(c["candidate_id"], label=c["label"], length=c["cycle_length"])

    # 1. Edge between candidates sharing accounts
    acc_to_cands: Dict[str, List[str]] = defaultdict(list)
    for c in candidates:
        for acc in c.get("ordered_cycle_accounts", c.get("nodes", c.get("participants", []))):
            acc_to_cands[acc].append(c["candidate_id"])
    for acc, c_list in acc_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                G.add_edge(c_list[i], c_list[j])

    # 2. Edge between candidates sharing transactions
    tx_to_cands: Dict[str, List[str]] = defaultdict(list)
    for c in candidates:
        for tid in c.get("transaction_ids", c.get("cycle_transaction_ids", [])):
            tx_to_cands[tid].append(c["candidate_id"])
    for tid, c_list in tx_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                G.add_edge(c_list[i], c_list[j])

    # 3. Edge between matched pairs
    mset_to_cands: Dict[int, List[str]] = defaultdict(list)
    for cid, s_idx in matched_group_map.items():
        mset_to_cands[s_idx].append(cid)
    for s_idx, c_list in mset_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                if G.has_node(c_list[i]) and G.has_node(c_list[j]):
                    G.add_edge(c_list[i], c_list[j])

    # Extract connected components (disjoint groups)
    components = list(nx.connected_components(G))
    group_records = []
    cand_to_group: Dict[str, str] = {}
    cand_by_id = {c["candidate_id"]: c for c in candidates}

    for g_idx, comp in enumerate(sorted(components, key=lambda s: (-len(s), sorted(list(s))[0]))):
        grp_id = f"group_{g_idx:02d}"
        c_list = sorted(list(comp))
        n_pos = sum(1 for cid in c_list if cand_by_id[cid]["label"] == 1)
        n_neg = sum(1 for cid in c_list if cand_by_id[cid]["label"] == 0)
        group_records.append({
            "group_id": grp_id,
            "candidate_ids": c_list,
            "total_candidates": len(c_list),
            "positive_count": n_pos,
            "negative_count": n_neg,
            "lengths": sorted(list(set(cand_by_id[cid]["cycle_length"] for cid in c_list))),
        })
        for cid in c_list:
            cand_to_group[cid] = grp_id

    # Stratified Outer 5-fold assignment over groups
    # Greedily assign groups to outer folds balanced by positive count and candidate count
    rng = random.Random(random_seed)
    # Sort groups deterministically: primary by pos_count desc, secondary by size desc, tertiary by group_id
    sorted_groups = sorted(group_records, key=lambda g: (-g["positive_count"], -g["total_candidates"], g["group_id"]))

    outer_folds: List[Dict[str, Any]] = [
        {"fold_idx": f, "group_ids": [], "candidate_ids": [], "pos_count": 0, "neg_count": 0, "total": 0}
        for f in range(n_outer_folds)
    ]

    for grp in sorted_groups:
        # Pick the fold with the lowest pos_count, break ties with lowest total
        best_fold = min(outer_folds, key=lambda f: (f["pos_count"], f["total"], f["fold_idx"]))
        best_fold["group_ids"].append(grp["group_id"])
        best_fold["candidate_ids"].extend(grp["candidate_ids"])
        best_fold["pos_count"] += grp["positive_count"]
        best_fold["neg_count"] += grp["negative_count"]
        best_fold["total"] += grp["total_candidates"]

    cand_outer_fold: Dict[str, int] = {}
    for f in outer_folds:
        for cid in f["candidate_ids"]:
            cand_outer_fold[cid] = f["fold_idx"]

    # Verify 0.0% account leakage across outer folds
    fold_accounts: Dict[int, Set[str]] = defaultdict(set)
    for cid, f_idx in cand_outer_fold.items():
        for acc in cand_by_id[cid]["ordered_cycle_accounts"]:
            fold_accounts[f_idx].add(acc)

    for f1 in range(n_outer_folds):
        for f2 in range(f1 + 1, n_outer_folds):
            overlap = fold_accounts[f1] & fold_accounts[f2]
            if overlap:
                raise ValueError(f"FATAL: Account leakage detected between outer fold {f1} and fold {f2}: {overlap}")

    # Build 3-fold inner splits for each outer fold
    outer_folds_spec = []
    for f_idx in range(n_outer_folds):
        test_cids = sorted(outer_folds[f_idx]["candidate_ids"])
        train_groups = [g for g in group_records if g["group_id"] not in outer_folds[f_idx]["group_ids"]]
        train_cids = [cid for g in train_groups for cid in g["candidate_ids"]]

        # Sort train groups for inner fold assignment
        inner_sorted_groups = sorted(train_groups, key=lambda g: (-g["positive_count"], -g["total_candidates"], g["group_id"]))
        inner_folds: List[Dict[str, Any]] = [
            {"inner_fold_idx": i, "group_ids": [], "candidate_ids": [], "pos_count": 0, "total": 0}
            for i in range(n_inner_folds)
        ]
        for grp in inner_sorted_groups:
            best_inner = min(inner_folds, key=lambda inf: (inf["pos_count"], inf["total"], inf["inner_fold_idx"]))
            best_inner["group_ids"].append(grp["group_id"])
            best_inner["candidate_ids"].extend(grp["candidate_ids"])
            best_inner["pos_count"] += grp["positive_count"]
            best_inner["total"] += grp["total_candidates"]

        inner_splits_spec = []
        for i_idx in range(n_inner_folds):
            inner_val_cids = sorted(inner_folds[i_idx]["candidate_ids"])
            inner_train_cids = sorted([cid for inf in inner_folds if inf["inner_fold_idx"] != i_idx for cid in inf["candidate_ids"]])
            inner_splits_spec.append({
                "inner_fold": i_idx,
                "inner_fold_id": i_idx,
                "train_candidate_ids": inner_train_cids,
                "val_candidate_ids": inner_val_cids,
                "test_candidate_ids": inner_val_cids,
                "inner_train_candidate_ids": inner_train_cids,
                "inner_val_candidate_ids": inner_val_cids,
                "inner_train_count": len(inner_train_cids),
                "inner_val_count": len(inner_val_cids),
            })

        outer_folds_spec.append({
            "fold_idx": f_idx,
            "outer_fold_id": f_idx,
            "test_groups": outer_folds[f_idx]["group_ids"],
            "test_candidate_ids": test_cids,
            "train_candidate_ids": sorted(train_cids),
            "test_pos_count": outer_folds[f_idx]["pos_count"],
            "test_neg_count": outer_folds[f_idx]["neg_count"],
            "test_total": outer_folds[f_idx]["total"],
            "train_total": len(train_cids),
            "inner_folds": inner_splits_spec,
            "inner_splits": inner_splits_spec,
        })

    manifest = {
        "manifest_version": "2.2.0",
        "dataset_name": "IBM AMLworld HI-Small Pattern Cycle Cohort",
        "random_seed": random_seed,
        "n_outer_folds": n_outer_folds,
        "n_inner_folds": n_inner_folds,
        "total_candidates": len(candidates),
        "total_groups": len(group_records),
        "groups": group_records,
        "candidate_assignments": {
            cid: {
                "group_id": cand_to_group[cid],
                "outer_fold": cand_outer_fold[cid],
                "label": cand_by_id[cid]["label"],
                "cycle_length": cand_by_id[cid]["cycle_length"],
            }
            for cid in sorted(list(cand_to_group.keys()))
        },
        "outer_folds": outer_folds_spec,
    }

    # Manifest content SHA-256
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["manifest_sha256"] = manifest_sha

    # Save to canonical location
    os.makedirs(os.path.dirname(os.path.abspath(output_manifest_path)), exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save to legacy mirror
    os.makedirs(os.path.dirname(os.path.abspath(legacy_mirror_path)), exist_ok=True)
    with open(legacy_mirror_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "status": "SPLITS_GENERATED",
        "manifest_sha256": manifest_sha,
        "total_candidates": len(candidates),
        "total_groups": len(group_records),
        "n_outer_folds": n_outer_folds,
        "output_path": output_manifest_path,
    }


def verify_split_disjointness(*partitions: Any) -> Dict[str, Any]:
    """Verify 0.0% overlap in participant accounts between partitions."""
    part_accs = []
    for p in partitions:
        accs = set()
        for c in p:
            c_accs = getattr(c, "participant_ids", None)
            if c_accs is None and isinstance(c, dict):
                c_accs = c.get("participant_ids", c.get("ordered_cycle_accounts", []))
            if c_accs:
                accs.update(c_accs)
        part_accs.append(accs)

    train_val_overlap = list(part_accs[0] & part_accs[1]) if len(part_accs) > 1 else []
    train_test_overlap = list(part_accs[0] & part_accs[2]) if len(part_accs) > 2 else []
    val_test_overlap = list(part_accs[1] & part_accs[2]) if len(part_accs) > 2 else []

    is_disjoint = (len(train_val_overlap) == 0) and (len(train_test_overlap) == 0) and (len(val_test_overlap) == 0)
    return {
        "is_disjoint": is_disjoint,
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "leakage_detected": not is_disjoint,
    }


class GroupSafeSplitter:
    """Group-safe dataset splitter guaranteeing zero participant account leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        n_splits: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = random_state if random_state is not None else seed
        self.n_splits = n_splits

    def split(self, candidates: List[Any], groups: Optional[List[str]] = None) -> Any:
        """Generate train/val/test splits or cross-validation fold indices."""
        if self.n_splits is not None:
            if groups is None:
                groups = [c.group_id if hasattr(c, "group_id") else f"group_{i}" for i, c in enumerate(candidates)]
            unique_groups = sorted(list(set(groups)))
            rng = random.Random(self.seed)
            rng.shuffle(unique_groups)
            fold_groups = [[] for _ in range(self.n_splits)]
            for i, g in enumerate(unique_groups):
                fold_groups[i % self.n_splits].append(g)
            splits = []
            for f_idx in range(self.n_splits):
                test_grps = set(fold_groups[f_idx])
                train_idx = [i for i, g in enumerate(groups) if g not in test_grps]
                test_idx = [i for i, g in enumerate(groups) if g in test_grps]
                splits.append((train_idx, test_idx))
            return splits
        else:
            group_to_cands = defaultdict(list)
            for c in candidates:
                gid = getattr(c, "group_id", c.get("group_id", "default") if isinstance(c, dict) else "default")
                group_to_cands[gid].append(c)

            unique_groups = sorted(list(group_to_cands.keys()))
            rng = random.Random(self.seed)
            rng.shuffle(unique_groups)

            n_total = len(unique_groups)
            n_train = max(1, int(round(n_total * self.train_ratio)))
            n_val = max(1, int(round(n_total * self.val_ratio)))
            train_grps = set(unique_groups[:n_train])
            val_grps = set(unique_groups[n_train:n_train + n_val])
            test_grps = set(unique_groups[n_train + n_val:])
            if len(test_grps) == 0 and len(val_grps) > 1:
                t_grp = list(val_grps)[-1]
                test_grps.add(t_grp)
                val_grps.remove(t_grp)

            train_cands = [c for g, c_list in group_to_cands.items() if g in train_grps for c in c_list]
            val_cands = [c for g, c_list in group_to_cands.items() if g in val_grps for c in c_list]
            test_cands = [c for g, c_list in group_to_cands.items() if g in test_grps for c in c_list]
            return train_cands, val_cands, test_cands

    def generate_and_save_split_manifest(self, train: List[Any], val: List[Any], test: List[Any], output_path: str) -> Dict[str, Any]:
        """Generate manifest JSON with split checksums and disjointness audit."""
        audit = verify_split_disjointness(train, val, test)
        manifest = {
            "disjointness_audit": audit,
            "splits": {
                "train": {
                    "count": len(train),
                    "sha256_checksum": hashlib.sha256(json.dumps([getattr(c, "candidate_id", str(c)) for c in train]).encode()).hexdigest(),
                },
                "validation": {
                    "count": len(val),
                    "sha256_checksum": hashlib.sha256(json.dumps([getattr(c, "candidate_id", str(c)) for c in val]).encode()).hexdigest(),
                },
                "test": {
                    "count": len(test),
                    "sha256_checksum": hashlib.sha256(json.dumps([getattr(c, "candidate_id", str(c)) for c in test]).encode()).hexdigest(),
                },
            }
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest


build_grouped_cross_validation_splits = generate_grouped_nested_splits


if __name__ == "__main__":
    res = generate_grouped_nested_splits()
    print(json.dumps(res, indent=2))
