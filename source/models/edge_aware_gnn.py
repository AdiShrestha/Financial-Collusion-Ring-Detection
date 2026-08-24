"""Edge-Aware Graph Neural Baseline (GINEBaseline).

Contract C12-03 (T-DESC): Implements a competitive edge-aware graph neural baseline
incorporating 8-dimensional transfer attributes (amount, timestamp, payment format, currency)
directly into neighborhood message-passing equations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.gnn_baselines import global_pool


class GINELayer(nn.Module):
    """Graph Isomorphism Network layer with Edge feature conditioning (GINE)."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0

        # Message aggregation
        agg = torch.zeros_like(x)

        if num_edges > 0:
            src, dst = edge_index[0], edge_index[1]
            x_src = x[src]

            if edge_attr is not None and edge_attr.numel() > 0:
                e_proj = self.edge_proj(edge_attr)
                msg = F.relu(x_src + e_proj)
            else:
                msg = F.relu(x_src)

            # Scatter add into destination nodes
            agg.index_add_(0, dst, msg)

        # GIN update
        out = (1.0 + self.eps) * x + agg
        return self.mlp(out)


class GINEBaseline(nn.Module):
    """Competitive edge-aware GNN baseline for transaction graph classification."""

    def __init__(
        self,
        node_dim: int = 16,
        edge_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINELayer(hidden_dim, edge_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass accepting node features, edge indices, and edge attributes."""
        # Handle input edge attributes fallback
        if edge_attr is None:
            num_edges = edge_index.size(1) if edge_index.numel() > 0 else 0
            edge_attr = torch.zeros((num_edges, self.edge_dim), device=x.device, dtype=x.dtype)

        h = self.node_encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            if h.size(0) > 1:
                h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in

        # Dual pooling: mean + max
        mean_p = global_pool(h, batch, mode="mean")
        max_p = global_pool(h, batch, mode="max")
        pooled = torch.cat([mean_p, max_p], dim=-1)

        logits = self.classifier(pooled)
        return logits

    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
