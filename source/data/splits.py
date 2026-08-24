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
        for acc in c["ordered_cycle_accounts"]:
            acc_to_cands[acc].append(c["candidate_id"])
    for acc, c_list in acc_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                G.add_edge(c_list[i], c_list[j])

    # 2. Edge between candidates sharing transactions
    tx_to_cands: Dict[str, List[str]] = defaultdict(list)
    for c in candidates:
        for txid in c["cycle_transaction_ids"]:
            tx_to_cands[txid].append(c["candidate_id"])
    for txid, c_list in tx_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                G.add_edge(c_list[i], c_list[j])

    # 3. Edge between candidates sharing pattern ID (if > 0)
    pat_to_cands: Dict[int, List[str]] = defaultdict(list)
    for c in candidates:
        pid = c.get("pattern_id", -1)
        if pid > 0:
            pat_to_cands[pid].append(c["candidate_id"])
    for pid, c_list in pat_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                G.add_edge(c_list[i], c_list[j])

    # 4. Edge between matched positive-negative sets
    mset_to_cands: Dict[int, List[str]] = defaultdict(list)
    for cid, s_idx in matched_group_map.items():
        mset_to_cands[s_idx].append(cid)
    for s_idx, c_list in mset_to_cands.items():
        for i in range(len(c_list)):
            for j in range(i + 1, len(c_list)):
                G.add_edge(c_list[i], c_list[j])

    # Extract connected components (Groups)
    components = list(nx.connected_components(G))
    # Sort deterministically by sorted candidate IDs
    components.sort(key=lambda comp: sorted(list(comp))[0])

    group_records = []
    cand_to_group = {}
    cand_by_id = {c["candidate_id"]: c for c in candidates}

    for g_idx, comp in enumerate(components):
        g_id = f"group_{g_idx:04d}"
        comp_cands = [cand_by_id[cid] for cid in comp]
        pos_cnt = sum(c["label"] for c in comp_cands)
        neg_cnt = len(comp_cands) - pos_cnt
        all_accs = set()
        for c in comp_cands:
            all_accs.update(c["ordered_cycle_accounts"])
        
        for cid in comp:
            cand_to_group[cid] = g_id

        group_records.append({
            "group_id": g_id,
            "candidate_ids": sorted(list(comp)),
            "positive_count": pos_cnt,
            "negative_count": neg_cnt,
            "total_count": len(comp),
            "accounts": sorted(list(all_accs)),
        })

    # Stratified partition of group components across n_outer_folds
    rng = random.Random(random_seed)
    # Sort groups by (pos_cnt > 0, total_count) descending, then shuffle with seed
    sorted_groups = sorted(group_records, key=lambda g: (g["positive_count"], g["total_count"]), reverse=True)
    
    outer_fold_groups: List[List[Dict[str, Any]]] = [[] for _ in range(n_outer_folds)]
    outer_fold_pos = [0] * n_outer_folds
    outer_fold_tot = [0] * n_outer_folds

    for g in sorted_groups:
        # Assign to fold with lowest positive count (or lowest total count)
        best_fold = min(range(n_outer_folds), key=lambda f: (outer_fold_pos[f], outer_fold_tot[f]))
        outer_fold_groups[best_fold].append(g)
        outer_fold_pos[best_fold] += g["positive_count"]
        outer_fold_tot[best_fold] += g["total_count"]

    # Assign outer_fold to each candidate
    cand_outer_fold: Dict[str, int] = {}
    for f_idx, g_list in enumerate(outer_fold_groups):
        for g in g_list:
            for cid in g["candidate_ids"]:
                cand_outer_fold[cid] = f_idx

    # Build 3 inner folds for each outer fold
    outer_folds_spec = []
    for test_fold in range(n_outer_folds):
        train_groups = []
        for f_idx in range(n_outer_folds):
            if f_idx != test_fold:
                train_groups.extend(outer_fold_groups[f_idx])

        # Partition train_groups into n_inner_folds
        inner_fold_groups: List[List[Dict[str, Any]]] = [[] for _ in range(n_inner_folds)]
        inner_pos = [0] * n_inner_folds
        inner_tot = [0] * n_inner_folds

        for g in train_groups:
            best_inner = min(range(n_inner_folds), key=lambda i: (inner_pos[i], inner_tot[i]))
            inner_fold_groups[best_inner].append(g)
            inner_pos[best_inner] += g["positive_count"]
            inner_tot[best_inner] += g["total_count"]

        inner_spec = []
        for val_inner in range(n_inner_folds):
            val_cands = []
            for g in inner_fold_groups[val_inner]:
                val_cands.extend(g["candidate_ids"])

            train_inner_cands = []
            for i_idx in range(n_inner_folds):
                if i_idx != val_inner:
                    for g in inner_fold_groups[i_idx]:
                        train_inner_cands.extend(g["candidate_ids"])

            inner_spec.append({
                "inner_fold_id": val_inner,
                "train_candidate_ids": sorted(train_inner_cands),
                "val_candidate_ids": sorted(val_cands),
            })

        test_cands = []
        for g in outer_fold_groups[test_fold]:
            test_cands.extend(g["candidate_ids"])

        train_outer_cands = []
        for f_idx in range(n_outer_folds):
            if f_idx != test_fold:
                for g in outer_fold_groups[f_idx]:
                    train_outer_cands.extend(g["candidate_ids"])

        outer_folds_spec.append({
            "outer_fold_id": test_fold,
            "train_candidate_ids": sorted(train_outer_cands),
            "test_candidate_ids": sorted(test_cands),
            "inner_folds": inner_spec,
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


if __name__ == "__main__":
    res = generate_grouped_nested_splits()
    print(json.dumps(res, indent=2))
