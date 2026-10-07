"""
Node, Edge, and Structural Feature Encoders for Graph Baselines.

Implements feature extraction for:
- GNN-basic: Raw node transaction volume, in/out degree, and temporal activity span (8 dims).
- GNN-structure-aware: Base features augmented with non-neural structural motif counts
  (clustering coefficient, triangles, 4/5/6-cycles) (13 dims total).
- Edge features: Normalized log amount and timestamp (2 dims).

Upholds Invariants:
- INV-001 (No Mock Data in Production): Validated on candidate graph schemas.
- INV-005 (Blind Candidate Extraction & Feature Integrity): Features built purely from
  graph topology and transaction attributes; zero label contamination.
- INV-008 (Self-Contained Verification Scripts): Numerical stability with zero NaNs.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))


class NodeFeatureEncoder:
    """Extracts raw transaction statistics per node."""

    def encode(
        self,
        nodes: List[str],
        edges: List[Dict[str, Any]],
        node_map: Dict[str, int],
    ) -> np.ndarray:
        """
        Computes 8-dimensional base node feature matrix:
        [in_degree, out_degree, log_vol_in, log_vol_out, first_ts, last_ts, span_ts, txn_count]
        """
        num_nodes = len(nodes)
        feats = np.zeros((num_nodes, 8), dtype=np.float32)

        if not edges:
            return feats

        in_deg = np.zeros(num_nodes, dtype=np.float32)
        out_deg = np.zeros(num_nodes, dtype=np.float32)
        vol_in = np.zeros(num_nodes, dtype=np.float32)
        vol_out = np.zeros(num_nodes, dtype=np.float32)
        min_ts = np.full(num_nodes, np.inf, dtype=np.float32)
        max_ts = np.full(num_nodes, -np.inf, dtype=np.float32)

        for e in edges:
            u = str(e.get("source", ""))
            v = str(e.get("target", ""))
            amt = float(e.get("amount", 0.0) or 0.0)
            ts = float(e.get("timestamp", 0.0) or 0.0)

            if u in node_map and v in node_map:
                u_idx = node_map[u]
                v_idx = node_map[v]

                out_deg[u_idx] += 1.0
                in_deg[v_idx] += 1.0
                vol_out[u_idx] += amt
                vol_in[v_idx] += amt

                min_ts[u_idx] = min(min_ts[u_idx], ts)
                max_ts[u_idx] = max(max_ts[u_idx], ts)
                min_ts[v_idx] = min(min_ts[v_idx], ts)
                max_ts[v_idx] = max(max_ts[v_idx], ts)

        # Sanitize timestamps
        min_ts[np.isinf(min_ts)] = 0.0
        max_ts[np.isinf(max_ts)] = 0.0
        span = np.maximum(0.0, max_ts - min_ts)
        txn_count = in_deg + out_deg

        feats[:, 0] = in_deg
        feats[:, 1] = out_deg
        feats[:, 2] = np.log10(vol_in + 1.0)
        feats[:, 3] = np.log10(vol_out + 1.0)
        feats[:, 4] = min_ts
        feats[:, 5] = max_ts
        feats[:, 6] = span
        feats[:, 7] = txn_count

        if not np.isfinite(feats).all():
            raise ValueError("Non-finite node features")
        return feats


class StructuralMotifEncoder:
    """Extracts non-neural topological motif counts for GNN-structure-aware baseline."""

    def encode(
        self,
        nodes: List[str],
        edges: List[Dict[str, Any]],
        node_map: Dict[str, int],
    ) -> np.ndarray:
        """
        Computes 5-dimensional structural motif matrix:
        [clustering_coeff, triangle_count, cycle4_count, cycle5_count, cycle6_count]
        """
        num_nodes = len(nodes)
        feats = np.zeros((num_nodes, 5), dtype=np.float32)

        # Construct undirected graph for motif counting
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for e in edges:
            u = str(e.get("source", ""))
            v = str(e.get("target", ""))
            if u != v and u in node_map and v in node_map:
                G.add_edge(u, v)

        # 1. Clustering coefficient
        clustering = nx.clustering(G)

        # 2. Triangles (3-cliques)
        triangles = nx.triangles(G)

        # 3. Simple undirected cycles up to length 6. NetworkX receives a
        # bidirected encoding, so canonicalize both orientations to count each
        # undirected cycle exactly once.
        cycle4_counts = {n: 0 for n in nodes}
        cycle5_counts = {n: 0 for n in nodes}
        cycle6_counts = {n: 0 for n in nodes}
        seen_cycles = set()

        # The same exact bounded computation is used for every candidate.
        for c in nx.simple_cycles(G.to_directed(), length_bound=6):
            k = len(c)
            target = {4: cycle4_counts, 5: cycle5_counts, 6: cycle6_counts}.get(k)
            if target is not None:
                rotations = [tuple(c[i:] + c[:i]) for i in range(k)]
                reversed_cycle = list(reversed(c))
                rotations.extend(tuple(reversed_cycle[i:] + reversed_cycle[:i]) for i in range(k))
                canonical = min(rotations)
                if canonical in seen_cycles:
                    continue
                seen_cycles.add(canonical)
                for n in canonical:
                    target[n] += 1

        for n in nodes:
            idx = node_map[n]
            feats[idx, 0] = float(clustering.get(n, 0.0))
            feats[idx, 1] = float(triangles.get(n, 0.0))
            feats[idx, 2] = float(cycle4_counts.get(n, 0.0))
            feats[idx, 3] = float(cycle5_counts.get(n, 0.0))
            feats[idx, 4] = float(cycle6_counts.get(n, 0.0))

        if not np.isfinite(feats).all():
            raise ValueError("Non-finite structural features")
        return feats


class EdgeFeatureEncoder:
    """Extracts normalized log amount and timestamp per directed edge."""

    def encode(self, edges: List[Dict[str, Any]]) -> np.ndarray:
        num_edges = len(edges)
        feats = np.zeros((num_edges, 2), dtype=np.float32)

        for i, e in enumerate(edges):
            amt = float(e.get("amount", 0.0) or 0.0)
            ts = float(e.get("timestamp", 0.0) or 0.0)
            feats[i, 0] = np.log10(max(0.0, amt) + 1.0)
            feats[i, 1] = ts

        if not np.isfinite(feats).all():
            raise ValueError("Non-finite edge features")
        return feats


class CandidateFeatureExtractor:
    """End-to-end feature extractor producing PyTorch tensors for GNN baselines."""

    def __init__(self):
        self.node_encoder = NodeFeatureEncoder()
        self.struct_encoder = StructuralMotifEncoder()
        self.edge_encoder = EdgeFeatureEncoder()

    def extract_features(
        self, candidate: Any, mode: str = "structure_aware"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts features for a CandidateSubgraph record.
        
        Args:
            candidate: Dict or object with nodes, edges, label.
            mode: "basic" (8 node features) or "structure_aware" (13 node features).
            
        Returns:
            Tuple of:
            - x_nodes: (N, d_v) float32 tensor
            - edge_index: (2, M) int64 tensor
            - edge_attr: (M, 2) float32 tensor
            - y: scalar float32 tensor
        """
        if hasattr(candidate, "nodes"):
            raw_nodes = list(candidate.nodes)
            raw_edges = list(candidate.edges)
            label = getattr(candidate, "label", None)
        elif isinstance(candidate, dict):
            raw_nodes = list(candidate.get("nodes", []))
            raw_edges = list(candidate.get("edges", []))
            label = candidate.get("label")
        else:
            raise TypeError(f"Unsupported candidate type: {type(candidate)}")
        if label not in (0, 1):
            raise ValueError(f"Feature extraction requires an explicit binary label, got {label!r}")

        nodes = sorted(list(set(str(n) for n in raw_nodes)))
        node_map = {n: i for i, n in enumerate(nodes)}

        # Standardize edges to dict
        clean_edges: List[Dict[str, Any]] = []
        src_indices: List[int] = []
        dst_indices: List[int] = []

        for e in raw_edges:
            if isinstance(e, dict):
                if "amount" not in e or "timestamp" not in e:
                    raise ValueError("Edge amount/time must be explicit; missing values need a declared static-graph protocol")
                u = str(e.get("source", ""))
                v = str(e.get("target", ""))
                amt = float(e["amount"])
                ts = float(e["timestamp"])
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                raise ValueError("Tuple edges omit observed amount/time metadata")
            else:
                continue

            if u in node_map and v in node_map:
                clean_edges.append({"source": u, "target": v, "amount": amt, "timestamp": ts})
                src_indices.append(node_map[u])
                dst_indices.append(node_map[v])

        if clean_edges:
            origin = min(e["timestamp"] for e in clean_edges)
            for e in clean_edges:
                e["timestamp"] -= origin

        # Node features
        base_feats = self.node_encoder.encode(nodes, clean_edges, node_map)
        if mode == "structure_aware":
            struct_feats = self.struct_encoder.encode(nodes, clean_edges, node_map)
            node_feats = np.concatenate([base_feats, struct_feats], axis=1)
        elif mode == "basic":
            node_feats = base_feats
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'basic' or 'structure_aware'.")

        # Edge features
        edge_feats = self.edge_encoder.encode(clean_edges)

        # Convert to PyTorch tensors
        x_nodes = torch.from_numpy(node_feats).to(torch.float32)
        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.from_numpy(edge_feats).to(torch.float32)
        y = torch.tensor(float(label), dtype=torch.float32)

        return x_nodes, edge_index, edge_attr, y
