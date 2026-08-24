"""CycleCellView polygonal cell complex lifting module."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import networkx as nx
import numpy as np

from source.data.candidate_extractor import CandidateExample
from source.topology.incidence import (
    compute_boundary_matrix_b1,
    compute_boundary_matrix_b2,
    compute_hodge_laplacian_0,
    compute_hodge_laplacian_1,
)
from source.topology.oracles import verify_boundary_nilpotence


class CycleCellView:
    """Lifts a candidate subgraph into a polygonal cell complex (CW complex of rank <= 2).

    In accordance with INV-002:
    - 0-cells: Vertices.
    - 1-cells: 1-skeleton edges.
    - 2-cells: Polygonal simple cycles of length k in {3, 4, 5, 6}, where each k-cycle
      forms a single rank-2 cell bounded by k edges.
    - 4-cycles, 5-cycles, and 6-cycles are NEVER decomposed or distorted into simplices.
    """

    def __init__(
        self,
        candidate_id: str,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        cells_2: List[List[str]],
        B1: np.ndarray,
        B2: np.ndarray,
        L0: np.ndarray,
        L1: np.ndarray,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        cell_features: np.ndarray,
        target_y: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.candidate_id = candidate_id
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.cells_2 = [list(c) for c in cells_2]
        self.B1 = np.asarray(B1, dtype=np.float32)
        self.B2 = np.asarray(B2, dtype=np.float32)
        self.L0 = np.asarray(L0, dtype=np.float32)
        self.L1 = np.asarray(L1, dtype=np.float32)
        self.node_features = np.asarray(node_features, dtype=np.float32)
        self.edge_features = np.asarray(edge_features, dtype=np.float32)
        self.cell_features = np.asarray(cell_features, dtype=np.float32)
        self.target_y = int(target_y)
        self.metadata = dict(metadata or {})

        # Self-verify algebraic boundary nilpotence on construction
        if not verify_boundary_nilpotence(self.B1, self.B2):
            raise ValueError(
                f"Cellular boundary nilpotence failed for candidate {self.candidate_id}: B1 @ B2 != 0"
            )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_cells_2(self) -> int:
        return len(self.cells_2)

    @classmethod
    def from_candidate_example(
        cls,
        candidate: CandidateExample,
        min_k: int = 3,
        max_k: int = 6,
        default_node_dim: int = 56,
    ) -> "CycleCellView":
        """Lift CandidateExample into CycleCellView."""
        # 1. 0-cells (canonical node list)
        nodes = sorted(list(candidate.participant_ids))
        node_set = set(nodes)

        # 2. 1-cells (canonical undirected edges)
        edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for u, v, attr in candidate.edges:
            u_str, v_str = str(u), str(v)
            if u_str not in node_set or v_str not in node_set or u_str == v_str:
                continue
            e_canon = (min(u_str, v_str), max(u_str, v_str))
            edge_map.setdefault(e_canon, dict(attr))

        edges = sorted(list(edge_map.keys()))
        edge_set = set(edges)

        # 3. 2-cells (polygonal cycles of length k in [min_k, max_k])
        cells_2: List[List[str]] = []

        # If metadata has explicit cycle_nodes, include it
        if "cycle_nodes" in candidate.metadata and candidate.metadata["cycle_nodes"]:
            meta_cycle = [str(n) for n in candidate.metadata["cycle_nodes"]]
            if min_k <= len(meta_cycle) <= max_k:
                cells_2.append(meta_cycle)

        # Also search for chordless cycle basis in graph
        dg = nx.DiGraph()
        dg.add_nodes_from(nodes)
        for u, v, _ in candidate.edges:
            if str(u) in node_set and str(v) in node_set:
                dg.add_edge(str(u), str(v))

        # Enumerate simple cycles
        raw_cycles = list(nx.simple_cycles(dg))
        seen_cycles = set()

        for c in cells_2:
            # Canonical cycle representation
            min_node = min(c)
            idx = c.index(min_node)
            can_c = tuple(c[idx:] + c[:idx])
            seen_cycles.add(can_c)

        for cycle in raw_cycles:
            k = len(cycle)
            if min_k <= k <= max_k:
                c_str = [str(n) for n in cycle]
                min_node = min(c_str)
                idx = c_str.index(min_node)
                can_c = tuple(c_str[idx:] + c_str[:idx])
                if can_c not in seen_cycles:
                    seen_cycles.add(can_c)
                    cells_2.append(c_str)

        # 4. Compute signed incidence matrices
        B1 = compute_boundary_matrix_b1(nodes, edges)
        B2 = compute_boundary_matrix_b2(edges, cells_2)

        # 5. Compute Hodge Laplacians
        L0 = compute_hodge_laplacian_0(B1)
        L1 = compute_hodge_laplacian_1(B1, B2)

        # 6. Feature matrices
        node_feat_list = []
        for n in nodes:
            if n in candidate.node_features and candidate.node_features[n]:
                node_feat_list.append(candidate.node_features[n])
            else:
                node_feat_list.append([0.0] * default_node_dim)
        node_features = np.array(node_feat_list, dtype=np.float32)

        edge_feat_list = []
        for e in edges:
            attr = edge_map.get(e, {})
            amt = float(attr.get("amount", attr.get("amount_paid", 0.0)))
            ts = float(attr.get("timestamp", 0.0))
            edge_feat_list.append([amt, ts])
        edge_features = np.array(edge_feat_list, dtype=np.float32) if edge_feat_list else np.zeros((0, 2), dtype=np.float32)

        cell_feat_list = []
        for cell in cells_2:
            k_len = float(len(cell))
            cell_feat_list.append([2.0, k_len])
        cell_features = np.array(cell_feat_list, dtype=np.float32) if cell_feat_list else np.zeros((0, 2), dtype=np.float32)

        return cls(
            candidate_id=candidate.candidate_id,
            nodes=nodes,
            edges=edges,
            cells_2=cells_2,
            B1=B1,
            B2=B2,
            L0=L0,
            L1=L1,
            node_features=node_features,
            edge_features=edge_features,
            cell_features=cell_features,
            target_y=candidate.target_y,
            metadata={
                **candidate.metadata,
                "typology_label": candidate.typology_label,
                "dataset_track": candidate.dataset_track,
                "cell_lengths": [len(c) for c in cells_2],
            },
        )
