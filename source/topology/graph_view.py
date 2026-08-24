"""GraphView 1-skeleton baseline representation and PyG data converters."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from source.data.candidate_extractor import CandidateExample

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    # Deterministic native PyG Data surrogate when torch_geometric is not installed
    class PyGData:
        def __init__(
            self,
            x: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.y = y
            self._num_nodes = kwargs.pop("num_nodes", None)
            for k, v in kwargs.items():
                setattr(self, k, v)

        @property
        def num_nodes(self) -> int:
            if self._num_nodes is not None:
                return self._num_nodes
            return self.x.size(0) if self.x is not None else 0

        @property
        def num_edges(self) -> int:
            return self.edge_index.size(1) if self.edge_index is not None else 0

        def __repr__(self) -> str:
            cls = "Data"
            fields = []
            if self.x is not None:
                fields.append(f"x={list(self.x.shape)}")
            if self.edge_index is not None:
                fields.append(f"edge_index={list(self.edge_index.shape)}")
            if self.edge_attr is not None:
                fields.append(f"edge_attr={list(self.edge_attr.shape)}")
            if self.y is not None:
                fields.append(f"y={list(self.y.shape)}")
            return f"{cls}({', '.join(fields)})"


class GraphView:
    """Standard 1-skeleton graph representation for GNN baselines (GCN, GAT, GraphSAGE)."""

    def __init__(
        self,
        candidate_id: str,
        nodes: List[str],
        edges: List[Tuple[str, str, Dict[str, Any]]],
        node_features: np.ndarray,
        edge_features: np.ndarray,
        target_y: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.candidate_id = candidate_id
        self.nodes = list(nodes)
        self.node_map = {n: i for i, n in enumerate(self.nodes)}
        self.edges = list(edges)
        self.node_features = np.asarray(node_features, dtype=np.float32)
        self.edge_features = np.asarray(edge_features, dtype=np.float32)
        self.target_y = int(target_y)
        self.metadata = dict(metadata or {})

        # Build edge index (2, |E|)
        if self.edges:
            srcs = [self.node_map[e[0]] for e in self.edges if e[0] in self.node_map and e[1] in self.node_map]
            dsts = [self.node_map[e[1]] for e in self.edges if e[0] in self.node_map and e[1] in self.node_map]
            self.edge_index = np.array([srcs, dsts], dtype=np.int64)
        else:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @classmethod
    def from_candidate_example(
        cls,
        candidate: CandidateExample,
        symmetrize: bool = False,
        add_self_loops: bool = False,
        default_feature_dim: int = 56,
    ) -> "GraphView":
        """Construct GraphView from CandidateExample."""
        nodes = list(candidate.participant_ids)
        node_map = {n: i for i, n in enumerate(nodes)}

        # Build node feature matrix
        node_feat_list = []
        for n in nodes:
            if n in candidate.node_features and candidate.node_features[n]:
                node_feat_list.append(candidate.node_features[n])
            else:
                node_feat_list.append([0.0] * default_feature_dim)
        node_features = np.array(node_feat_list, dtype=np.float32)

        # Process edges
        raw_edges = list(candidate.edges)
        final_edges: List[Tuple[str, str, Dict[str, Any]]] = []
        edge_feat_list: List[List[float]] = []

        for u, v, attr in raw_edges:
            u_str, v_str = str(u), str(v)
            if u_str not in node_map or v_str not in node_map:
                continue

            # Standard edge features: [amount, timestamp]
            amt = float(attr.get("amount", attr.get("amount_paid", attr.get("amount_received", 0.0))))
            ts = float(attr.get("timestamp", 0.0))
            e_feat = [amt, ts]

            final_edges.append((u_str, v_str, dict(attr)))
            edge_feat_list.append(e_feat)

            if symmetrize and u_str != v_str:
                # Add reverse edge
                rev_attr = dict(attr)
                rev_attr["reversed"] = True
                final_edges.append((v_str, u_str, rev_attr))
                edge_feat_list.append(e_feat)

        if add_self_loops:
            for n in nodes:
                loop_attr = {"self_loop": True, "timestamp": 0.0, "amount": 0.0}
                final_edges.append((n, n, loop_attr))
                edge_feat_list.append([0.0, 0.0])

        if edge_feat_list:
            edge_features = np.array(edge_feat_list, dtype=np.float32)
        else:
            edge_features = np.zeros((0, 2), dtype=np.float32)

        return cls(
            candidate_id=candidate.candidate_id,
            nodes=nodes,
            edges=final_edges,
            node_features=node_features,
            edge_features=edge_features,
            target_y=candidate.target_y,
            metadata={
                **candidate.metadata,
                "typology_label": candidate.typology_label,
                "dataset_track": candidate.dataset_track,
                "temporal_bounds": candidate.temporal_bounds,
            },
        )

    def to_adjacency_matrix(self) -> np.ndarray:
        """Return dense |V| x |V| binary adjacency matrix."""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for u, v, _ in self.edges:
            if u in self.node_map and v in self.node_map:
                i, j = self.node_map[u], self.node_map[v]
                adj[i, j] = 1.0
        return adj

    def to_pyg_data(self) -> PyGData:
        """Convert GraphView to PyG Data instance."""
        x_tensor = torch.tensor(self.node_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.int64)
        edge_attr_tensor = torch.tensor(self.edge_features, dtype=torch.float32)
        y_tensor = torch.tensor([self.target_y], dtype=torch.int64)

        # Check for NaNs
        if torch.isnan(x_tensor).any():
            raise ValueError(f"NaN detected in node features for candidate {self.candidate_id}")
        if torch.isnan(edge_attr_tensor).any():
            raise ValueError(f"NaN detected in edge features for candidate {self.candidate_id}")

        return PyGData(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            y=y_tensor,
            candidate_id=self.candidate_id,
            num_nodes=self.num_nodes,
        )
