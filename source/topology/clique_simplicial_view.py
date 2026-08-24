"""CliqueSimplicialView simplicial complex lifting module."""

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


class CliqueSimplicialView:
    """Lifts a candidate subgraph into a flag simplicial complex (dim <= 2).

    In accordance with INV-002:
    - 0-simplices: Vertices (nodes).
    - 1-simplices: Undirected edges (1-skeleton).
    - 2-simplices: Filled 3-cliques (triangles) with all 3 boundary edges present.
    - Polygonal cycles of length k >= 4 are NOT treated as 2-simplices.
    """

    def __init__(
        self,
        candidate_id: str,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        faces_2: List[Tuple[str, str, str]],
        B1: np.ndarray,
        B2: np.ndarray,
        L0: np.ndarray,
        L1: np.ndarray,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        face_features: np.ndarray,
        target_y: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.candidate_id = candidate_id
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.faces_2 = list(faces_2)
        self.B1 = np.asarray(B1, dtype=np.float32)
        self.B2 = np.asarray(B2, dtype=np.float32)
        self.L0 = np.asarray(L0, dtype=np.float32)
        self.L1 = np.asarray(L1, dtype=np.float32)
        self.node_features = np.asarray(node_features, dtype=np.float32)
        self.edge_features = np.asarray(edge_features, dtype=np.float32)
        self.face_features = np.asarray(face_features, dtype=np.float32)
        self.target_y = int(target_y)
        self.metadata = dict(metadata or {})

        # Self-verify nilpotence on construction
        if not verify_boundary_nilpotence(self.B1, self.B2):
            raise ValueError(
                f"Simplicial boundary nilpotence failed for candidate {self.candidate_id}: B1 @ B2 != 0"
            )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_faces_2(self) -> int:
        return len(self.faces_2)

    @classmethod
    def from_candidate_example(
        cls,
        candidate: CandidateExample,
        max_dim: int = 2,
        default_node_dim: int = 56,
    ) -> "CliqueSimplicialView":
        """Lift CandidateExample into CliqueSimplicialView."""
        # 1. 0-simplices (canonical node list)
        nodes = sorted(list(candidate.participant_ids))
        node_set = set(nodes)

        # 2. 1-simplices (canonical undirected edges: (min(u, v), max(u, v)))
        edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for u, v, attr in candidate.edges:
            u_str, v_str = str(u), str(v)
            if u_str not in node_set or v_str not in node_set or u_str == v_str:
                continue
            e_canon = (min(u_str, v_str), max(u_str, v_str))
            edge_map.setdefault(e_canon, dict(attr))

        edges = sorted(list(edge_map.keys()))
        edge_set = set(edges)

        # 3. 2-simplices (3-cliques: (u, v, w) with all 3 pairs in edge_set)
        faces_2: List[Tuple[str, str, str]] = []
        if max_dim >= 2:
            # Build undirected graph for clique search
            ug = nx.Graph()
            ug.add_nodes_from(nodes)
            ug.add_edges_from(edges)

            # Find all 3-cliques
            all_cliques = list(nx.enumerate_all_cliques(ug))
            triangles = [c for c in all_cliques if len(c) == 3]

            for tri in triangles:
                u, v, w = sorted(tri)
                # Verify simplicial closure
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, w), max(v, w))
                e3 = (min(u, w), max(u, w))
                if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                    faces_2.append((u, v, w))

        faces_2 = sorted(faces_2)

        # 4. Compute signed incidence matrices
        B1 = compute_boundary_matrix_b1(nodes, edges)
        B2 = compute_boundary_matrix_b2(edges, faces_2)

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

        face_feat_list = []
        for f in faces_2:
            # 2-simplex features: [simplex_dimension=2, num_boundary_edges=3]
            face_feat_list.append([2.0, 3.0])
        face_features = np.array(face_feat_list, dtype=np.float32) if face_feat_list else np.zeros((0, 2), dtype=np.float32)

        return cls(
            candidate_id=candidate.candidate_id,
            nodes=nodes,
            edges=edges,
            faces_2=faces_2,
            B1=B1,
            B2=B2,
            L0=L0,
            L1=L1,
            node_features=node_features,
            edge_features=edge_features,
            face_features=face_features,
            target_y=candidate.target_y,
            metadata={
                **candidate.metadata,
                "typology_label": candidate.typology_label,
                "dataset_track": candidate.dataset_track,
                "num_triangles": len(faces_2),
            },
        )
