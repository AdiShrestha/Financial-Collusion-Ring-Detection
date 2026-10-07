"""Standard Graph Domain Lifting Engine.

Converts CandidateSubgraph instances into standard PyTorch / PyG graph representations
with 0-cell and 1-cell feature tensors and target labels.
Traces to Contract C03-01, FR-003, INV-008, INV-010.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np

try:
    from source.src.data.candidate_extractor import CandidateSubgraph
except ModuleNotFoundError:
    from src.data.candidate_extractor import CandidateSubgraph


@dataclass
class GraphData:
    x: torch.Tensor          # (num_nodes, node_feat_dim) float32
    edge_index: torch.Tensor # (2, num_edges) int64
    edge_attr: torch.Tensor  # (num_edges, edge_feat_dim) float32
    y: torch.Tensor          # (1,) int64
    node_map: Dict[str, int]
    num_nodes: int
    num_edges: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x.cpu().numpy().tolist(),
            "edge_index": self.edge_index.cpu().numpy().tolist(),
            "edge_attr": self.edge_attr.cpu().numpy().tolist(),
            "y": int(self.y.item()),
            "node_map": self.node_map,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }


def lift_to_graph(candidate: CandidateSubgraph) -> GraphData:
    """Converts a CandidateSubgraph into a canonical GraphData tensor representation.

    Args:
        candidate: Input candidate subgraph.

    Returns:
        GraphData containing x, edge_index, edge_attr, y, and node_map.

    Raises:
        IndexError: If an edge references a node outside candidate.nodes.
    """
    # 1. Deterministic node mapping
    unique_nodes = sorted(list(set(candidate.nodes)))
    num_nodes = len(unique_nodes)
    node_map: Dict[str, int] = {node: i for i, node in enumerate(unique_nodes)}

    # Node-level statistics accumulators
    in_degrees = [0] * num_nodes
    out_degrees = [0] * num_nodes
    in_flow = [0.0] * num_nodes
    out_flow = [0.0] * num_nodes
    cycle_nodes_set = set(candidate.cycle_nodes)

    # 2. Build edge index and edge features
    src_indices: List[int] = []
    dst_indices: List[int] = []
    edge_features: List[List[float]] = []

    for edge in candidate.edges:
        src = edge["source"]
        dst = edge["target"]

        if src not in node_map:
            raise IndexError(f"Edge source '{src}' outside candidate node set [0, {num_nodes-1}]")
        if dst not in node_map:
            raise IndexError(f"Edge target '{dst}' outside candidate node set [0, {num_nodes-1}]")

        u = node_map[src]
        v = node_map[dst]

        src_indices.append(u)
        dst_indices.append(v)

        out_degrees[u] += 1
        in_degrees[v] += 1

        raw_amt = float(edge.get("amount", 0.0))
        log_amt = math.log1p(max(0.0, raw_amt))
        in_flow[v] += log_amt
        out_flow[u] += log_amt

        # Is cycle edge (both endpoints in cycle_nodes)
        is_cycle_edge = 1.0 if (src in cycle_nodes_set and dst in cycle_nodes_set) else 0.0

        edge_features.append([
            log_amt,
            raw_amt,
            is_cycle_edge,
        ])

    num_edges = len(src_indices)

    # 3. Construct node feature tensor
    node_features: List[List[float]] = []
    for i, node in enumerate(unique_nodes):
        is_cycle_node = 1.0 if node in cycle_nodes_set else 0.0
        tot_deg = in_degrees[i] + out_degrees[i]
        cycle_ratio = in_degrees[i] / (tot_deg + 1e-6)

        node_features.append([
            float(in_degrees[i]),
            float(out_degrees[i]),
            float(in_flow[i]),
            float(out_flow[i]),
            is_cycle_node,
            float(cycle_ratio),
        ])

    # 4. Assemble PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float32) if node_features else torch.empty((0, 6), dtype=torch.float32)
    if num_edges > 0:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)

    if candidate.label not in (0, 1):
        raise ValueError("Graph lifting requires an explicit binary label")
    label_val = candidate.label
    y = torch.tensor([label_val], dtype=torch.long)

    # Check for NaN / Inf
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError("NaN or Inf detected in node feature matrix x")
    if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
        raise ValueError("NaN or Inf detected in edge attribute matrix edge_attr")

    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_map=node_map,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )
