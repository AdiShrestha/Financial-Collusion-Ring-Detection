"""Standard 1-skeleton Graph Neural Network baselines (GCN, GAT, GraphSAGE)."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.topology.graph_view import GraphView


def global_pool(x: torch.Tensor, batch: Optional[torch.Tensor], mode: str = "mean") -> torch.Tensor:
    """Pool node representations to graph-level representation."""
    if batch is None:
        if mode == "mean":
            return torch.mean(x, dim=0, keepdim=True)
        elif mode == "sum" or mode == "add":
            return torch.sum(x, dim=0, keepdim=True)
        elif mode == "max":
            return torch.max(x, dim=0, keepdim=True)[0]
        else:
            raise ValueError(f"Unsupported pooling mode: '{mode}'")

    num_graphs = int(batch.max().item()) + 1
    out_dim = x.size(1)
    pooled = torch.zeros(num_graphs, out_dim, device=x.device, dtype=x.dtype)

    if mode == "mean":
        counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        for i in range(x.size(0)):
            b = int(batch[i].item())
            pooled[b] += x[i]
            counts[b] += 1.0
        pooled = pooled / torch.clamp(counts, min=1.0)
    elif mode == "sum" or mode == "add":
        for i in range(x.size(0)):
            b = int(batch[i].item())
            pooled[b] += x[i]
    elif mode == "max":
        pooled.fill_(-1e9)
        for i in range(x.size(0)):
            b = int(batch[i].item())
            pooled[b] = torch.maximum(pooled[b], x[i])
        # Replace unassigned graphs if any
        pooled[pooled == -1e9] = 0.0
    else:
        raise ValueError(f"Unsupported pooling mode: '{mode}'")

    return pooled


class GCNLayer(nn.Module):
    """Symmetric normalized graph convolution layer with self-loops."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        # Build adjacency with self-loops
        adj = torch.eye(num_nodes, device=x.device, dtype=x.dtype)
        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # undirected / symmetrized

        # Degree normalization: D^(-1/2) * A * D^(-1/2)
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-8), -0.5)
        d_mat = torch.diag(deg_inv_sqrt)
        norm_adj = d_mat @ adj @ d_mat

        # Message passing + linear transform
        ax = norm_adj @ x
        return self.linear(ax)


class GATLayer(nn.Module):
    """Multi-head Graph Attention Network layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.w = nn.Parameter(torch.empty(num_heads, in_features, out_features))
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.bias = nn.Parameter(torch.zeros(out_features * num_heads if concat else out_features))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        # Self-loops included
        adj = torch.eye(num_nodes, device=x.device, dtype=torch.bool)
        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = True
            adj[edge_index[1], edge_index[0]] = True

        head_outs = []
        for h in range(self.num_heads):
            # h_x: (N, out_features)
            h_x = x @ self.w[h]
            attn_src = h_x @ self.a_src[h]  # (N, 1)
            attn_dst = h_x @ self.a_dst[h]  # (N, 1)
            e_mat = attn_src + attn_dst.T  # (N, N)
            e_mat = self.leaky_relu(e_mat)

            # Mask non-edges with -1e9
            mask_val = torch.tensor(-1e9, device=x.device, dtype=x.dtype)
            e_mat = torch.where(adj, e_mat, mask_val)
            alpha = F.softmax(e_mat, dim=1)
            alpha = self.dropout(alpha)

            h_out = alpha @ h_x  # (N, out_features)
            head_outs.append(h_out)

        if self.concat:
            out = torch.cat(head_outs, dim=-1) + self.bias
        else:
            out = torch.mean(torch.stack(head_outs, dim=0), dim=0) + self.bias
        return out


class SAGELayer(nn.Module):
    """GraphSAGE layer with mean neighborhood aggregation."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        adj = torch.zeros((num_nodes, num_nodes), device=x.device, dtype=x.dtype)
        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            adj[edge_index[1], edge_index[0]] = 1.0

        deg = torch.sum(adj, dim=1, keepdim=True)
        norm_adj = adj / torch.clamp(deg, min=1.0)

        neigh_agg = norm_adj @ x
        out = self.lin_self(x) + self.lin_neigh(neigh_agg)
        return out


class GCNBaseline(nn.Module):
    """Multi-layer Graph Convolutional Network baseline."""

    def __init__(
        self,
        in_dim: int = 56,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 2,
        dropout: float = 0.1,
        readout: str = "mean",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.readout = readout
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.convs.append(GCNLayer(in_dim, hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: Union[torch.Tensor, GraphView, Any],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, GraphView):
            data = x.to_pyg_data()
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
        elif hasattr(x, "edge_index") and hasattr(x, "x"):
            # PyG Data object
            batch = getattr(x, "batch", None)
            edge_index = x.edge_index
            x = x.x

        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.lns[i](h)
            h = F.relu(h)
            h = self.dropout(h)

        g = global_pool(h, batch=batch, mode=self.readout)
        logits = self.classifier(g)
        return logits


class GATBaseline(nn.Module):
    """Multi-head Graph Attention Network baseline."""

    def __init__(
        self,
        in_dim: int = 56,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        out_dim: int = 2,
        dropout: float = 0.1,
        readout: str = "mean",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.readout = readout
        self.dropout = nn.Dropout(dropout)

        head_dim = max(1, hidden_dim // num_heads)
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.convs.append(GATLayer(in_dim, head_dim, num_heads=num_heads, dropout=dropout, concat=True))
        self.lns.append(nn.LayerNorm(head_dim * num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATLayer(head_dim * num_heads, head_dim, num_heads=num_heads, dropout=dropout, concat=True))
            self.lns.append(nn.LayerNorm(head_dim * num_heads))
        if num_layers > 1:
            self.convs.append(GATLayer(head_dim * num_heads, hidden_dim, num_heads=1, dropout=dropout, concat=False))
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: Union[torch.Tensor, GraphView, Any],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, GraphView):
            data = x.to_pyg_data()
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
        elif hasattr(x, "edge_index") and hasattr(x, "x"):
            batch = getattr(x, "batch", None)
            edge_index = x.edge_index
            x = x.x

        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.lns[i](h)
            h = F.elu(h)
            h = self.dropout(h)

        g = global_pool(h, batch=batch, mode=self.readout)
        logits = self.classifier(g)
        return logits


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE baseline architecture with mean neighborhood aggregation."""

    def __init__(
        self,
        in_dim: int = 56,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 2,
        dropout: float = 0.1,
        readout: str = "mean",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.readout = readout
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.convs.append(SAGELayer(in_dim, hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGELayer(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: Union[torch.Tensor, GraphView, Any],
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, GraphView):
            data = x.to_pyg_data()
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
        elif hasattr(x, "edge_index") and hasattr(x, "x"):
            batch = getattr(x, "batch", None)
            edge_index = x.edge_index
            x = x.x

        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.lns[i](h)
            h = F.relu(h)
            h = self.dropout(h)

        g = global_pool(h, batch=batch, mode=self.readout)
        logits = self.classifier(g)
        return logits
