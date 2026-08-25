"""Topological Cell and Simplicial Complex Feature Encoder.

Contract C17-02 & Scientific Remediation:
- Cellular Complexes (CWN): Encodes polygonal k-gons (k in [3..12]) as native 2-cells with exact signed incidence matrices (B1 in {-1,0,1}^{|V|x|E|}, B2 in {-1,0,1}^{|E|x|C|}, B1 @ B2 == 0).
- Simplicial Complexes (MPSN): Encodes cycles via simplicial complexes (triangulated 2-simplices for k >= 4, or pure 3-clique 2-simplices for k=3).
- Information-Fairness: 2-cell features (X2) are derived strictly from boundary aggregation of constituent edge features (X2 = |B2|^T X1 / k), avoiding privileged tabular statistics.
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


class CandidateCellComplexEncoder:
    """Encodes candidate transaction subgraphs into mathematically exact cell and simplicial complexes."""

    FORMAT_VOCAB = ["ACH", "WIRE", "CHECK", "CREDIT_CARD", "CASH", "INTERNAL"]
    BANK_VOCAB = [0, 1, 2]

    def __init__(self, node_feat_dim: int = 6, edge_feat_dim: int = 9, cell_feat_dim: int = 9):
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.cell_feat_dim = cell_feat_dim

    def encode_candidate(
        self,
        candidate_record: Dict[str, Any],
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Encode candidate subgraph into both Cellular and Simplicial complex representations."""
        k = len(candidate_record["ordered_cycle_accounts"])
        accounts = candidate_record["ordered_cycle_accounts"]
        acc_to_idx = {acc: i for i, acc in enumerate(accounts)}
        num_nodes = k

        # 1. Build Node Features (X0)
        epochs = [tx.get("timestamp_epoch", 0.0) for tx in transactions if "timestamp_epoch" in tx]
        min_epoch = min(epochs) if epochs else 0.0

        node_flow = defaultdict(float)
        node_in_deg = defaultdict(int)
        node_out_deg = defaultdict(int)
        node_banks = {}

        for tx in transactions:
            u, v = tx["from_account"], tx["to_account"]
            amt = float(tx.get("amount_paid", 0.0))
            node_flow[u] += amt
            node_flow[v] += amt
            node_out_deg[u] += 1
            node_in_deg[v] += 1
            if u in acc_to_idx:
                node_banks[u] = tx.get("from_bank", 0)
            if v in acc_to_idx:
                node_banks[v] = tx.get("to_bank", 0)

        node_feats = []
        for acc in accounts:
            flow_val = math.log1p(max(0.0, node_flow[acc]))
            in_d = float(node_in_deg[acc])
            out_d = float(node_out_deg[acc])
            b_id = node_banks.get(acc, 0)
            b_onehot = [1.0 if b_id == b else 0.0 for b in self.BANK_VOCAB]
            node_feats.append([flow_val, in_d, out_d] + b_onehot)

        X0 = torch.tensor(node_feats, dtype=torch.float32)

        # 2. Build 1-Cell / Edge Features (X1) & 1-Boundary Matrix (B1)
        num_edges = len(transactions)
        B1 = np.zeros((num_nodes, num_edges), dtype=np.int64)
        edge_src = []
        edge_dst = []
        edge_feats = []

        for e_idx, tx in enumerate(transactions):
            u, v = tx["from_account"], tx["to_account"]
            u_idx = acc_to_idx[u]
            v_idx = acc_to_idx[v]
            edge_src.append(u_idx)
            edge_dst.append(v_idx)

            # B1: target is +1, source is -1
            B1[v_idx, e_idx] = 1
            B1[u_idx, e_idx] = -1

            amt = float(tx.get("amount_paid", 0.0))
            log_amt = math.log1p(max(0.0, amt))
            rel_epoch = (tx.get("timestamp_epoch", min_epoch) - min_epoch) / 86400.0  # in days
            fb = tx.get("from_bank", 0)
            tb = tx.get("to_bank", 0)
            cross_bank = 1.0 if fb != tb else 0.0

            fmt = tx.get("payment_format", "ACH")
            fmt_onehot = [1.0 if fmt == f else 0.0 for f in self.FORMAT_VOCAB]

            edge_feats.append([log_amt, rel_epoch, cross_bank] + fmt_onehot)

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        X1 = torch.tensor(edge_feats, dtype=torch.float32)

        # 3. Cellular Complex Encoding (CCNN / CWN)
        # Single native polygonal 2-cell over all k boundary edges
        B2_cell = np.ones((num_edges, 1), dtype=np.int64)

        # Verify boundary nilpotency: B1 @ B2 == 0
        B1_B2 = np.dot(B1, B2_cell)
        if not np.all(B1_B2 == 0):
            raise ValueError(f"Cellular boundary nilpotency violation (INV-004) on candidate {candidate_record['candidate_id']}")

        # Information-fair 2-cell feature: mean-pooled boundary edge features
        X2_cell = (torch.tensor(np.abs(B2_cell).T, dtype=torch.float32) @ X1) / float(k)

        # 4. Simplicial Complex Encoding (SCNN / MPSN)
        # For k=3: identical triangle 2-simplex
        # For k>=4: triangulated simplicial complex via fan triangulation from vertex 0
        if k == 3:
            B1_simp = B1.copy()
            B2_simp = B2_cell.copy()
            X1_simp = X1.clone()
            X2_simp = X2_cell.clone()
        else:
            # Triangulate polygon (v0..vk-1) into (k-2) 2-simplices
            # Add (k-3) chord edges: (v0 -> v2), (v0 -> v3), ..., (v0 -> vk-2)
            n_chords = k - 3
            n_simp_edges = num_edges + n_chords
            n_2simplices = k - 2

            B1_simp = np.zeros((num_nodes, n_simp_edges), dtype=np.int64)
            B1_simp[:, :num_edges] = B1

            chord_feats = []
            chord_src = []
            chord_dst = []

            for c_idx in range(n_chords):
                e_id = num_edges + c_idx
                u_i = 0
                v_i = c_idx + 2
                B1_simp[v_i, e_id] = 1
                B1_simp[u_i, e_id] = -1
                chord_src.append(u_i)
                chord_dst.append(v_i)
                # Chord feature interpolated from endpoint node features
                mean_edge_feat = (X0[u_i] + X0[v_i]) / 2.0
                # Pad to edge_feat_dim
                chord_feat = list(mean_edge_feat[:3].numpy()) + [0.0] * (self.edge_feat_dim - 3)
                chord_feats.append(chord_feat)

            if chord_feats:
                X1_simp = torch.cat([X1, torch.tensor(chord_feats, dtype=torch.float32)], dim=0)
            else:
                X1_simp = X1.clone()

            # Build simplicial B2: each 2-simplex has 3 oriented boundary edges
            B2_simp = np.zeros((n_simp_edges, n_2simplices), dtype=np.int64)
            # Triangles: (v0, v1, v2), (v0, v2, v3), ..., (v0, vk-2, vk-1)
            for t_idx in range(n_2simplices):
                if t_idx == 0:
                    # (v0->v1): edge 0, (v1->v2): edge 1, (v0->v2): chord 0
                    B2_simp[0, 0] = 1
                    B2_simp[1, 0] = 1
                    B2_simp[num_edges, 0] = -1
                elif t_idx == n_2simplices - 1:
                    # (v0->vk-2): last chord, (vk-2->vk-1): edge k-2, (vk-1->v0): edge k-1
                    last_chord = num_edges + n_chords - 1
                    B2_simp[last_chord, t_idx] = 1
                    B2_simp[k - 2, t_idx] = 1
                    B2_simp[k - 1, t_idx] = 1
                else:
                    # (v0->vi): chord i-2, (vi->vi+1): edge i, (v0->vi+1): chord i-1
                    c_in = num_edges + t_idx - 1
                    c_out = num_edges + t_idx
                    edge_i = t_idx + 1
                    B2_simp[c_in, t_idx] = 1
                    B2_simp[edge_i, t_idx] = 1
                    B2_simp[c_out, t_idx] = -1

            # Verify simplicial nilpotency
            B1_simp_B2_simp = np.dot(B1_simp, B2_simp)
            if not np.all(B1_simp_B2_simp == 0):
                raise ValueError(f"Simplicial boundary nilpotency violation on candidate {candidate_record['candidate_id']}")

            X2_simp = (torch.tensor(np.abs(B2_simp).T, dtype=torch.float32) @ X1_simp) / 3.0

        label = int(candidate_record["label"])
        y = torch.tensor([label], dtype=torch.long)

        return {
            "candidate_id": candidate_record["candidate_id"],
            "label": label,
            "cycle_length": k,
            "num_nodes": k,
            # GNN representation
            "x": X0,
            "edge_index": edge_index,
            "edge_attr": X1,
            # Cellular representation (CCNN)
            "X0": X0,
            "X1": X1,
            "X2": X2_cell,
            "B1": torch.tensor(B1, dtype=torch.float32),
            "B2": torch.tensor(B2_cell, dtype=torch.float32),
            # Simplicial representation (SCNN)
            "X0_simp": X0,
            "X1_simp": X1_simp,
            "X2_simp": X2_simp,
            "B1_simp": torch.tensor(B1_simp, dtype=torch.float32),
            "B2_simp": torch.tensor(B2_simp, dtype=torch.float32),
            "y": y,
        }


def encode_and_serialize_fold_tensors(
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    candidate_txs_parquet_path: str = "artifacts/candidates/candidate_transactions.parquet",
    fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    output_dir: str = "artifacts/features",
) -> Dict[str, Any]:
    """Encode all candidates and serialize fold tensor bundles."""
    os.makedirs(output_dir, exist_ok=True)
    c_tbl = pq.read_table(candidates_parquet_path)
    candidates = c_tbl.to_pylist()

    tx_tbl = pq.read_table(candidate_txs_parquet_path)
    txs = tx_tbl.to_pylist()
    txs_by_cid = defaultdict(list)
    for t in txs:
        txs_by_cid[t["candidate_id"]].append(t)

    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        mf = json.load(f)

    cand_by_id = {c["candidate_id"]: c for c in candidates}
    encoder = CandidateCellComplexEncoder()

    # Pre-encode all candidates
    encoded_all = {}
    nilpotent_count = 0
    for c in candidates:
        cid = c["candidate_id"]
        c_txs = txs_by_cid[cid]
        enc = encoder.encode_candidate(c, c_txs)
        encoded_all[cid] = enc
        nilpotent_count += 1

    fold_summaries = {}
    for fold_idx in range(mf["n_outer_folds"]):
        of_spec = mf["outer_folds"][fold_idx]
        train_cids = of_spec["train_candidate_ids"]
        test_cids = of_spec["test_candidate_ids"]

        train_bundle = [encoded_all[cid] for cid in train_cids]
        test_bundle = [encoded_all[cid] for cid in test_cids]

        fold_data = {
            "outer_fold_id": fold_idx,
            "train_candidates": train_bundle,
            "test_candidates": test_bundle,
            "inner_folds": of_spec.get("inner_folds", []),
        }

        save_path = os.path.join(output_dir, f"fold_{fold_idx}_tensors.pt")
        torch.save(fold_data, save_path)
        fold_summaries[f"fold_{fold_idx}"] = {
            "train_count": len(train_bundle),
            "test_count": len(test_bundle),
            "file_size": os.path.getsize(save_path),
        }

    return {
        "status": "TENSORS_ENCODED",
        "total_candidates": len(candidates),
        "total_encoded_candidates": len(candidates),
        "nilpotency_verified_count": nilpotent_count,
        "folds": fold_summaries,
    }


CellComplexCandidateEncoder = CandidateCellComplexEncoder


if __name__ == "__main__":
    res = encode_and_serialize_fold_tensors()
    print(json.dumps(res, indent=2))
