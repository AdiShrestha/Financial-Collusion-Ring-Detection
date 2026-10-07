"""Clique Simplicial Complex Domain Lifter.

Converts CandidateSubgraph instances into 2-simplicial complexes (Flag 2-complexes),
where 3-cliques form 2-simplices, leaving chordless k-cycles (k >= 4) as unfilled 1-skeletons.
Traces to Contract C03-03, FR-003, INV-002, INV-004, INV-008.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

try:
    from source.src.data.candidate_extractor import CandidateSubgraph
    from source.src.topology.incidence import (
        build_b1,
        build_b2_simplicial,
        compute_hodge_laplacians,
        verify_nilpotency,
    )
except ModuleNotFoundError:
    from src.data.candidate_extractor import CandidateSubgraph
    from src.topology.incidence import (
        build_b1,
        build_b2_simplicial,
        compute_hodge_laplacians,
        verify_nilpotency,
    )


@dataclass
class SimplicialComplex:
    nodes: List[str]                           # List of 0-simplices
    edges: List[Tuple[str, str]]               # List of 1-simplices (canonical sorted pairs u < v)
    triangles: List[Tuple[str, str, str]]      # List of 2-simplices (canonical sorted triplets u < v < w)
    x_0: torch.Tensor                          # Node features (num_nodes, d_0) float32
    x_1: torch.Tensor                          # Edge features (num_edges, d_1) float32
    x_2: torch.Tensor                          # Triangle features (num_triangles, d_2) float32
    B1: sp.csr_matrix                          # Simplicial boundary-1 matrix (|V| x |E|)
    B2: sp.csr_matrix                          # Simplicial boundary-2 matrix (|E| x |T|)
    L0: sp.csr_matrix                          # Simplicial 0-Laplacian (|V| x |V|)
    L1: sp.csr_matrix                          # Simplicial 1-Laplacian (|E| x |E|)
    label: int                                 # Target binary label
    node_map: Dict[str, int]
    edge_map: Dict[Tuple[str, str], int]
    triangle_map: Dict[Tuple[str, str, str], int]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    def compute_betti_numbers(self) -> Tuple[int, int]:
        """Computes Betti numbers (beta_0, beta_1) via matrix ranks."""
        n_v = len(self.nodes)
        n_e = len(self.edges)
        n_t = len(self.triangles)

        rank_b1 = int(np.linalg.matrix_rank(self.B1.toarray())) if n_e > 0 and n_v > 0 else 0
        rank_b2 = int(np.linalg.matrix_rank(self.B2.toarray())) if n_t > 0 and n_e > 0 else 0

        beta_0 = n_v - rank_b1
        beta_1 = (n_e - rank_b1) - rank_b2
        return max(0, beta_0), max(0, beta_1)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complex topology metadata and statistics."""
        b0, b1 = self.compute_betti_numbers()
        _, max_res = verify_nilpotency(self.B1, self.B2, tol=1e-14)
        return {
            "num_0_simplices": self.num_nodes,
            "num_1_simplices": self.num_edges,
            "num_2_simplices": self.num_triangles,
            "betti_0": b0,
            "betti_1": b1,
            "nilpotency_residual": max_res,
            "label": self.label,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "triangles": [list(t) for t in self.triangles],
        }


def lift_to_simplicial_complex(candidate: CandidateSubgraph) -> SimplicialComplex:
    """Converts a CandidateSubgraph into a 2-simplicial complex.

    Only genuine 3-cliques form 2-simplices. Higher-order chordless cycles
    (k >= 4) remain unfilled 1-skeletons without artificial triangulation.

    Args:
        candidate: Input CandidateSubgraph instance.

    Returns:
        SimplicialComplex with boundary matrices B1, B2, Hodge Laplacians L0, L1,
        and multi-cell feature tensors x_0, x_1, x_2.

    Raises:
        IndexError: If an edge references a node outside candidate.nodes.
        ValueError: If algebraic nilpotency ||B1 B2||_inf > 10^-14.
    """
    # 1. Deterministic node mapping (0-simplices)
    nodes = sorted(list(set(candidate.nodes)))
    num_nodes = len(nodes)
    node_map: Dict[str, int] = {node: i for i, node in enumerate(nodes)}

    # Check for invalid node references in edges
    for edge in candidate.edges:
        src = edge["source"]
        dst = edge["target"]
        if src not in node_map:
            raise IndexError(f"Edge source '{src}' outside candidate node set [0, {num_nodes-1}]")
        if dst not in node_map:
            raise IndexError(f"Edge target '{dst}' outside candidate node set [0, {num_nodes-1}]")

    # 2. Extract canonical 1-simplices (u < v) and aggregate undirected edge data
    cycle_nodes_set = set(candidate.cycle_nodes)
    edge_amounts: Dict[Tuple[str, str], float] = {}
    in_degrees = [0] * num_nodes
    out_degrees = [0] * num_nodes
    in_flow = [0.0] * num_nodes
    out_flow = [0.0] * num_nodes

    for edge in candidate.edges:
        src = edge["source"]
        dst = edge["target"]
        u = node_map[src]
        v = node_map[dst]

        out_degrees[u] += 1
        in_degrees[v] += 1

        raw_amt = float(edge.get("amount", 0.0))
        log_amt = math.log1p(max(0.0, raw_amt))
        in_flow[v] += log_amt
        out_flow[u] += log_amt

        if src != dst:
            canon_edge = (min(src, dst), max(src, dst))
            edge_amounts[canon_edge] = edge_amounts.get(canon_edge, 0.0) + raw_amt

    canonical_edges: List[Tuple[str, str]] = sorted(list(edge_amounts.keys()))
    num_edges = len(canonical_edges)
    edge_map: Dict[Tuple[str, str], int] = {e: i for i, e in enumerate(canonical_edges)}

    # 3. Enumerate 3-cliques (2-simplices) using NetworkX on undirected 1-skeleton
    g_undirected = nx.Graph()
    g_undirected.add_nodes_from(nodes)
    g_undirected.add_edges_from(canonical_edges)

    triangles: List[Tuple[str, str, str]] = []
    for clq in nx.enumerate_all_cliques(g_undirected):
        if len(clq) == 3:
            sorted_tri = tuple(sorted(clq))
            triangles.append(sorted_tri)  # type: ignore[arg-type]
        elif len(clq) > 3:
            # NetworkX yields cliques in nondecreasing size, so no triangles
            # remain after the first higher-dimensional clique.
            break

    triangles.sort()
    num_triangles = len(triangles)
    triangle_map: Dict[Tuple[str, str, str], int] = {t: i for i, t in enumerate(triangles)}

    # 4. Construct boundary matrices B1 and B2
    B1 = build_b1(nodes, canonical_edges)
    B2 = build_b2_simplicial(canonical_edges, triangles)

    # 5. Nilpotency check (Stop Condition / INV-004)
    valid_nilpotency, max_residual = verify_nilpotency(B1, B2, tol=1e-14)
    if not valid_nilpotency or max_residual > 1e-14:
        raise ValueError(
            f"Stop Condition violated: simplicial nilpotency ||B1 B2||_inf = {max_residual:.2e} > 1e-14"
        )

    # 6. Compute simplicial Hodge Laplacians L0 and L1
    L0, L1, _ = compute_hodge_laplacians(B1, B2)

    # 7. Construct feature tensors x_0, x_1, x_2
    # Node features x_0: (num_nodes, 6)
    node_features: List[List[float]] = []
    for i, node in enumerate(nodes):
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
    x_0 = torch.tensor(node_features, dtype=torch.float32) if node_features else torch.empty((0, 6), dtype=torch.float32)

    # Edge features x_1: (num_edges, 3)
    edge_features: List[List[float]] = []
    for u_str, v_str in canonical_edges:
        tot_amt = edge_amounts[(u_str, v_str)]
        log_amt = math.log1p(max(0.0, tot_amt))
        is_cycle_edge = 1.0 if (u_str in cycle_nodes_set and v_str in cycle_nodes_set) else 0.0
        edge_features.append([
            log_amt,
            tot_amt,
            is_cycle_edge,
        ])
    x_1 = torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.empty((0, 3), dtype=torch.float32)

    # Triangle features x_2: (num_triangles, 3)
    triangle_features: List[List[float]] = []
    for v0, v1, v2 in triangles:
        # 3 boundary edges
        e1 = (v0, v1)
        e2 = (v0, v2)
        e3 = (v1, v2)
        amt1 = edge_amounts.get(e1, 0.0)
        amt2 = edge_amounts.get(e2, 0.0)
        amt3 = edge_amounts.get(e3, 0.0)
        tot_tri_amt = amt1 + amt2 + amt3
        mean_log_amt = (math.log1p(max(0.0, amt1)) + math.log1p(max(0.0, amt2)) + math.log1p(max(0.0, amt3))) / 3.0
        is_cycle_tri = 1.0 if (v0 in cycle_nodes_set and v1 in cycle_nodes_set and v2 in cycle_nodes_set) else 0.0
        triangle_features.append([
            mean_log_amt,
            tot_tri_amt,
            is_cycle_tri,
        ])
    x_2 = torch.tensor(triangle_features, dtype=torch.float32) if triangle_features else torch.empty((0, 3), dtype=torch.float32)

    # Check for NaN / Inf
    if torch.isnan(x_0).any() or torch.isinf(x_0).any():
        raise ValueError("NaN or Inf detected in 0-simplex feature matrix x_0")
    if torch.isnan(x_1).any() or torch.isinf(x_1).any():
        raise ValueError("NaN or Inf detected in 1-simplex feature matrix x_1")
    if torch.isnan(x_2).any() or torch.isinf(x_2).any():
        raise ValueError("NaN or Inf detected in 2-simplex feature matrix x_2")

    if candidate.label not in (0, 1):
        raise ValueError("Simplicial lifting requires an explicit binary label")
    label_val = candidate.label

    return SimplicialComplex(
        nodes=nodes,
        edges=canonical_edges,
        triangles=triangles,
        x_0=x_0,
        x_1=x_1,
        x_2=x_2,
        B1=B1,
        B2=B2,
        L0=L0,
        L1=L1,
        label=int(label_val),
        node_map=node_map,
        edge_map=edge_map,
        triangle_map=triangle_map,
    )
