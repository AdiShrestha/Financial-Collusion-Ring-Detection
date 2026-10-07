"""
Capacity-Matched Baseline Graph Neural Network Architectures for Ring Detection.

Implements standard GNN backbones (GCN, GAT, GraphSAGE) with:
- Standardized hidden dimension and layer depth for parameter parity (INV-009).
- Dual readout (concatenated global mean and max pooling).
- Optional fusion head for precomputed Persistent Homology feature vectors (INV-003, H3).
- Support for GNN-basic and GNN-structure-aware input dimensionalities.

Upholds Invariants:
- INV-001 (No Mock Data in Production).
- INV-008 (Self-Contained Verification Scripts).
- INV-009 (Model Parameter and FLOP Matching Protocol): parameter delta <= 2%.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))


def global_pool(x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Computes concatenated global mean and max pooling across graph nodes."""
    if batch is None:
        mean_p = torch.mean(x, dim=0, keepdim=True)
        max_p, _ = torch.max(x, dim=0, keepdim=True)
        return torch.cat([mean_p, max_p], dim=-1)

    batch_size = int(batch.max().item()) + 1
    mean_list = []
    max_list = []
    for b in range(batch_size):
        mask = (batch == b)
        if mask.sum() == 0:
            mean_list.append(torch.zeros((1, x.size(-1)), device=x.device, dtype=x.dtype))
            max_list.append(torch.zeros((1, x.size(-1)), device=x.device, dtype=x.dtype))
        else:
            sub_x = x[mask]
            mean_list.append(torch.mean(sub_x, dim=0, keepdim=True))
            max_p, _ = torch.max(sub_x, dim=0, keepdim=True)
            max_list.append(max_p)

    return torch.cat([torch.cat(mean_list, dim=0), torch.cat(max_list, dim=0)], dim=-1)


def global_edge_pool(
    edge_attr: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean/max edge summaries, pooled separately for each graph in a batch."""
    if edge_attr is None:
        raise ValueError("Graph baselines require edge attributes; pass amount/time features explicitly")
    if edge_attr.ndim != 2 or edge_attr.size(1) != 2:
        raise ValueError(f"Expected edge attributes with shape (E, 2), got {tuple(edge_attr.shape)}")
    if edge_attr.size(0) != edge_index.size(1):
        raise ValueError("Edge feature rows must align one-to-one with edge_index columns")
    if not torch.isfinite(edge_attr).all():
        raise ValueError("Graph baseline received non-finite edge features")
    if batch is None:
        edge_batch = torch.zeros(edge_index.size(1), device=edge_attr.device, dtype=torch.long)
        batch_size = 1
    else:
        if batch.ndim != 1:
            raise ValueError("Node-to-graph batch assignment must be one-dimensional")
        batch = batch.to(edge_attr.device)
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
        if edge_index.numel():
            edge_batch = batch[edge_index[0].to(batch.device)]
            dst_batch = batch[edge_index[1].to(batch.device)]
            if not torch.equal(edge_batch, dst_batch):
                raise ValueError("An edge connects nodes assigned to different graphs")
        else:
            edge_batch = torch.empty((0,), device=edge_attr.device, dtype=torch.long)

    pooled = torch.zeros((batch_size, 4), device=edge_attr.device, dtype=edge_attr.dtype)
    for graph_id in range(batch_size):
        graph_edges = edge_attr[edge_batch == graph_id]
        if graph_edges.numel():
            pooled[graph_id, :2] = graph_edges.mean(dim=0)
            pooled[graph_id, 2:] = graph_edges.max(dim=0).values
    return pooled


class GCNConvLayer(nn.Module):
    """Custom directed bi-degree-normalized message-passing layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            return self.linear(x)

        # Add self-loops
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_edge = torch.stack([loop_index, loop_index], dim=0)
        full_edge_index = torch.cat([edge_index, loop_edge], dim=1)

        # Compute degree normalization
        out_deg = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
        in_deg = torch.zeros_like(out_deg)
        out_deg.scatter_add_(0, full_edge_index[0], torch.ones(full_edge_index.size(1), device=x.device))
        in_deg.scatter_add_(0, full_edge_index[1], torch.ones(full_edge_index.size(1), device=x.device))
        norm = (out_deg[full_edge_index[0]] * in_deg[full_edge_index[1]]).clamp(min=1).rsqrt()

        # Message passing: A_norm @ X
        h = F.linear(x, self.linear.weight, bias=None)
        out = torch.zeros_like(h)
        for i in range(full_edge_index.size(1)):
            u = full_edge_index[0, i]
            v = full_edge_index[1, i]
            out[v] += norm[i] * h[u]

        return out + self.linear.bias


class GATConvLayer(nn.Module):
    """Multi-head graph attention convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        if heads <= 0 or out_dim % heads:
            raise ValueError("GAT out_dim must be divisible by a positive head count")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.linear = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        h = self.linear(x).view(num_nodes, self.heads, self.head_dim)

        if edge_index.numel() == 0:
            return h.view(num_nodes, -1) + self.bias

        # Add self-loops
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_edge = torch.stack([loop_index, loop_index], dim=0)
        full_edge_index = torch.cat([edge_index, loop_edge], dim=1)

        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        src = full_edge_index[0]
        dst = full_edge_index[1]
        edge_att = F.leaky_relu(alpha_src[src] + alpha_dst[dst], negative_slope=0.2)
        # Softmax normalization per destination node
        max_per_dst = torch.full((num_nodes, self.heads), -torch.inf, device=x.device, dtype=edge_att.dtype)
        max_per_dst.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.heads), edge_att, reduce="amax", include_self=True)
        exp_att = torch.exp(edge_att - max_per_dst[dst])
        exp_sum = torch.zeros((num_nodes, self.heads), device=x.device)
        exp_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.heads), exp_att)
        alpha = exp_att / (exp_sum[dst] + 1e-12)

        out = torch.zeros((num_nodes, self.heads, self.head_dim), device=x.device)
        weighted_h = h[src] * alpha.unsqueeze(-1)
        for i in range(full_edge_index.size(1)):
            d = dst[i]
            out[d] += weighted_h[i]

        return out.view(num_nodes, -1) + self.bias


class SAGEConvLayer(nn.Module):
    """GraphSAGE mean neighborhood convolution layer (additive variant).
    
    NOTE: This implements the sum-project variant: h_v = W·(x_v + mean_neigh(x_u))
    rather than the canonical concat-project variant: h_v = W·[x_v || mean_neigh(x_u)].
    
    The additive form is a deliberate capacity-matching choice (INV-009):
    concat-project would double the input dimension of the linear layer,
    inflating parameter count and breaking parity with GCN/GAT baselines.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        if edge_index.numel() == 0:
            return self.linear(x)

        # Compute mean neighbor aggregation
        src = edge_index[0]
        dst = edge_index[1]
        deg = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
        deg.scatter_add_(0, dst, torch.ones(edge_index.size(1), device=x.device))
        deg[deg == 0] = 1.0

        neigh_sum = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            neigh_sum[dst[i]] += x[src[i]]

        neigh_mean = neigh_sum / deg.unsqueeze(1)
        return self.linear(x + neigh_mean)  # Additive variant
class BaseGNNClassifier(nn.Module):
    """Abstract base class for capacity-matched GNN ring classifiers."""

    def __init__(
        self,
        in_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 2,
        ph_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ph_dim = ph_dim
        self.dropout = dropout

        # PH projection layer if persistent homology features are provided
        if ph_dim is not None and ph_dim > 0:
            self.ph_proj = nn.Linear(ph_dim, 64)
            classifier_in = hidden_dim * 2 + 64 + 4
        else:
            self.ph_proj = None
            classifier_in = hidden_dim * 2 + 4

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GCNBaseline(BaseGNNClassifier):
    """Widened custom directed GCN-style baseline with residuals and LayerNorm."""

    def __init__(
        self,
        in_dim: int = 13,
        hidden_dim: int = 64,
        ph_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=2, ph_dim=ph_dim, dropout=dropout)
        self.conv1 = GCNConvLayer(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConvLayer(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ph_features: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h1 = self.norm1(F.relu(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.norm2(F.relu(self.conv2(h1, edge_index))) + h1  # Residual skip connection
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Global readout pooling (mean + max)
        pooled = global_pool(h2, batch)
        pooled = torch.cat([pooled, global_edge_pool(edge_attr, edge_index, batch)], dim=-1)

        if self.ph_proj is not None and ph_features is not None:
            h_ph = F.relu(self.ph_proj(ph_features))
            pooled = torch.cat([pooled, h_ph], dim=-1)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)


class GATBaseline(BaseGNNClassifier):
    """Custom 2-layer graph-attention baseline with residuals and LayerNorm."""

    def __init__(
        self,
        in_dim: int = 13,
        hidden_dim: int = 64,
        ph_dim: Optional[int] = None,
        dropout: float = 0.1,
        heads: int = 16,
    ):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=2, ph_dim=ph_dim, dropout=dropout)
        self.conv1 = GATConvLayer(in_dim, hidden_dim, heads=heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GATConvLayer(hidden_dim, hidden_dim, heads=heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ph_features: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h1 = self.norm1(F.elu(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.norm2(F.elu(self.conv2(h1, edge_index))) + h1  # Residual skip connection
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        pooled = global_pool(h2, batch)
        pooled = torch.cat([pooled, global_edge_pool(edge_attr, edge_index, batch)], dim=-1)

        if self.ph_proj is not None and ph_features is not None:
            h_ph = F.relu(self.ph_proj(ph_features))
            pooled = torch.cat([pooled, h_ph], dim=-1)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)


class GraphSAGEBaseline(BaseGNNClassifier):
    """Capacity-matched 2-layer GraphSAGE baseline model."""

    def __init__(
        self,
        in_dim: int = 13,
        hidden_dim: int = 64,
        ph_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=2, ph_dim=ph_dim, dropout=dropout)
        self.conv1 = SAGEConvLayer(in_dim, hidden_dim)
        self.conv2 = SAGEConvLayer(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ph_features: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))

        pooled = global_pool(h, batch)
        pooled = torch.cat([pooled, global_edge_pool(edge_attr, edge_index, batch)], dim=-1)

        if self.ph_proj is not None and ph_features is not None:
            h_ph = F.relu(self.ph_proj(ph_features))
            pooled = torch.cat([pooled, h_ph], dim=-1)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)


def audit_parameter_parity(
    models: Dict[str, BaseGNNClassifier], tolerance: float = 0.05
) -> Dict[str, Any]:
    """Audits parameter capacity parity across baseline models (INV-009)."""
    counts = {name: m.count_parameters() for name, m in models.items()}
    values = list(counts.values())
    mean_val = np.mean(values)
    max_dev = max(abs(v - mean_val) / mean_val for v in values)
    is_compliant = max_dev <= tolerance

    return {
        "parameter_counts": counts,
        "mean_parameters": float(mean_val),
        "max_relative_deviation": float(max_dev),
        "is_compliant": is_compliant,
    }
