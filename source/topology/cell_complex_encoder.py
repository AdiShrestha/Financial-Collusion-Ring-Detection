"""Real topological cell complex and graph candidate encoder.

Contract C17-02 (T-COMP): Converts candidate transaction subgraphs into node (X0),
edge (X1), and 2-cell (X2) feature tensors, with oriented boundary matrices B1 in {-1,0,1}^(|V|x|E|)
and B2 in {-1,0,1}^(|E|x|C|) satisfying exact nilpotency B1 B2 = 0 (INV-004). Serializes fold tensor bundles.
"""

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.topology.incidence import compute_boundary_matrix_b1, compute_boundary_matrix_b2


class CellComplexCandidateEncoder:
    """Encodes transaction cycle subgraphs into cell complexes and native PyTorch tensor dictionaries."""

    FORMAT_VOCAB = ["ACH", "Wire", "Cheque", "Credit Card", "Reinvestment", "Cash"]

    def encode_candidate(
        self,
        candidate_record: Dict[str, Any],
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Encode a single candidate into verified cell complex and graph tensor representation."""
        ordered_accs = candidate_record["ordered_cycle_accounts"]
        k = len(ordered_accs)
        acc_to_idx = {acc: i for i, acc in enumerate(ordered_accs)}

        # 1. Build Node Features (X0)
        # Features: in_degree, out_degree, total_amount_in, total_amount_out, bank_id_mod
        in_amt = defaultdict(float)
        out_amt = defaultdict(float)
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        node_banks = {}

        for tx in transactions:
            u = tx["from_account"]
            v = tx["to_account"]
            amt = float(tx.get("amount_paid", 0.0))
            out_amt[u] += amt
            in_amt[v] += amt
            out_deg[u] += 1
            in_deg[v] += 1
            node_banks[u] = float(tx.get("from_bank", 0)) % 100.0
            node_banks[v] = float(tx.get("to_bank", 0)) % 100.0

        node_feats = []
        for acc in ordered_accs:
            node_feats.append([
                float(in_deg[acc]),
                float(out_deg[acc]),
                math.log1p(max(0.0, in_amt[acc])),
                math.log1p(max(0.0, out_amt[acc])),
                node_banks.get(acc, 0.0) / 100.0,
            ])
        X0 = torch.tensor(node_feats, dtype=torch.float32)

        # 2. Build Edge Index, Edge Features (X1) & Boundary Matrix B1
        # Edges sorted by cycle step: e_i = (u_i, u_{i+1})
        edge_src = []
        edge_dst = []
        edge_feats = []
        num_edges = len(transactions)

        # B1 matrix: |V| x |E|
        B1 = np.zeros((k, num_edges), dtype=np.int64)

        epochs = [tx["timestamp_epoch"] for tx in transactions if "timestamp_epoch" in tx]
        min_epoch = min(epochs) if epochs else 0.0

        for e_idx, tx in enumerate(transactions):
            u = tx["from_account"]
            v = tx["to_account"]
            u_idx = acc_to_idx.get(u, 0)
            v_idx = acc_to_idx.get(v, (u_idx + 1) % k)

            edge_src.append(u_idx)
            edge_dst.append(v_idx)

            # Oriented boundary: head (+1), tail (-1)
            B1[v_idx, e_idx] = 1
            B1[u_idx, e_idx] = -1

            amt = float(tx.get("amount_paid", 0.0))
            log_amt = math.log1p(max(0.0, amt))
            rel_epoch = (tx.get("timestamp_epoch", min_epoch) - min_epoch) / 86400.0  # in days
            fb = tx.get("from_bank", 0)
            tb = tx.get("to_bank", 0)
            cross_bank = 1.0 if fb != tb else 0.0

            # One-hot format
            fmt = tx.get("payment_format", "ACH")
            fmt_onehot = [1.0 if fmt == f else 0.0 for f in self.FORMAT_VOCAB]

            edge_feats.append([log_amt, rel_epoch, cross_bank] + fmt_onehot)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        X1 = torch.tensor(edge_feats, dtype=torch.float32)

        # 3. Build 2-Cell Features (X2) & Boundary Matrix B2
        # Single 2-cell representing the closed cycle polygon (boundary = all backbone edges with +1)
        B2 = np.ones((num_edges, 1), dtype=np.int64)
        
        # Verify boundary nilpotency: B1 @ B2 must equal 0 (|V| x 1 vector of zeros)
        B1_B2 = np.dot(B1, B2)
        if not np.all(B1_B2 == 0):
            raise ValueError(f"Boundary nilpotency violation (INV-004) on candidate {candidate_record['candidate_id']}: B1 @ B2 = {B1_B2.flatten()}")

        # 2-cell features: cycle length, log duration, median amount, min/max amount ratio
        duration = max(epochs) - min(epochs) if len(epochs) > 1 else 0.0
        amounts = [float(tx.get("amount_paid", 0.0)) for tx in transactions]
        med_amt = float(np.median(amounts)) if amounts else 0.0
        amt_ratio = (min(amounts) / (max(amounts) + 1e-9)) if amounts else 1.0

        X2 = torch.tensor([[
            float(k),
            math.log1p(max(0.0, duration)),
            math.log1p(max(0.0, med_amt)),
            amt_ratio,
        ]], dtype=torch.float32)

        label = int(candidate_record["label"])
        y = torch.tensor([label], dtype=torch.long)

        return {
            "candidate_id": candidate_record["candidate_id"],
            "label": label,
            "cycle_length": k,
            "num_nodes": k,
            "x": X0,
            "edge_index": edge_index,
            "edge_attr": X1,
            "y": y,
            "X0": X0,
            "X1": X1,
            "X2": X2,
            "B1": torch.tensor(B1, dtype=torch.float32),
            "B2": torch.tensor(B2, dtype=torch.float32),
        }


def encode_and_serialize_fold_tensors(
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    candidate_txs_parquet_path: str = "artifacts/candidates/candidate_transactions.parquet",
    fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    output_dir: str = "artifacts/features",
) -> Dict[str, Any]:
    """Encode all candidates and serialize tensor bundles per fold."""
    if not os.path.exists(candidates_parquet_path):
        raise FileNotFoundError(f"Missing candidates: {candidates_parquet_path}")
    if not os.path.exists(candidate_txs_parquet_path):
        raise FileNotFoundError(f"Missing candidate txs: {candidate_txs_parquet_path}")
    if not os.path.exists(fold_manifest_path):
        raise FileNotFoundError(f"Missing fold manifest: {fold_manifest_path}")

    c_table = pq.read_table(candidates_parquet_path)
    tx_table = pq.read_table(candidate_txs_parquet_path)

    candidates = c_table.to_pylist()
    tx_records = tx_table.to_pylist()

    cand_by_id = {c["candidate_id"]: c for c in candidates}
    txs_by_cand: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tx in tx_records:
        txs_by_cand[tx["candidate_id"]].append(tx)

    encoder = CellComplexCandidateEncoder()
    encoded_all: Dict[str, Dict[str, Any]] = {}

    for c in candidates:
        cid = c["candidate_id"]
        txs = txs_by_cand[cid]
        encoded = encoder.encode_candidate(c, txs)
        encoded_all[cid] = encoded

    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for of in manifest["outer_folds"]:
        f_id = of["outer_fold_id"]
        train_ids = of["train_candidate_ids"]
        test_ids = of["test_candidate_ids"]

        bundle = {
            "fold_id": f_id,
            "train_candidates": [encoded_all[cid] for cid in train_ids],
            "test_candidates": [encoded_all[cid] for cid in test_ids],
        }

        out_path = os.path.join(output_dir, f"fold_{f_id}_tensors.pt")
        torch.save(bundle, out_path)

        results[f"fold_{f_id}"] = {
            "tensor_path": out_path,
            "train_count": len(train_ids),
            "test_count": len(test_ids),
        }

    return {
        "status": "TENSORS_ENCODED",
        "total_encoded_candidates": len(encoded_all),
        "folds": results,
    }


if __name__ == "__main__":
    res = encode_and_serialize_fold_tensors()
    print(json.dumps(res, indent=2))
