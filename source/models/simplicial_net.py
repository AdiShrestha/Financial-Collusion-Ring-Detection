"""Simplicial Neural Network (SCNN / MPSN) architecture.

Bodnar et al. (ICML 2021) "Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks".
Operates strictly on 0-simplices (vertices), 1-simplices (edges), and 2-simplices (triangles)
using simplicial Hodge Laplacians (L0, L1_down, L1_up, L2).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _adapt_feat_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Adapt feature tensor width to target dimension via slicing or zero-padding."""
    if x.shape[-1] == target_dim:
        return x
    elif x.shape[-1] > target_dim:
        return x[..., :target_dim]
    else:
        pad_shape = list(x.shape)
        pad_shape[-1] = target_dim - x.shape[-1]
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)


class SimplicialConvBlock(nn.Module):
    """Simplicial message passing layer propagating signals across 0-, 1-, and 2-simplices."""

    def __init__(
        self,
        in_dim_0: int,
        in_dim_1: int,
        in_dim_2: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        # 0-simplex transformations
        self.lin_0_self = nn.Linear(in_dim_0, out_dim)
        self.lin_0_from_1 = nn.Linear(in_dim_1, out_dim, bias=False)
        self.lin_0_l0 = nn.Linear(in_dim_0, out_dim, bias=False)
        self.ln_0 = nn.LayerNorm(out_dim)

        # 1-simplex transformations
        self.lin_1_self = nn.Linear(in_dim_1, out_dim)
        self.lin_1_from_0 = nn.Linear(in_dim_0, out_dim, bias=False)
        self.lin_1_from_2 = nn.Linear(in_dim_2, out_dim, bias=False)
        self.lin_1_l1_down = nn.Linear(in_dim_1, out_dim, bias=False)
        self.lin_1_l1_up = nn.Linear(in_dim_1, out_dim, bias=False)
        self.ln_1 = nn.LayerNorm(out_dim)

        # 2-simplex transformations
        self.lin_2_self = nn.Linear(in_dim_2, out_dim)
        self.lin_2_from_1 = nn.Linear(in_dim_1, out_dim, bias=False)
        self.lin_2_l2 = nn.Linear(in_dim_2, out_dim, bias=False)
        self.ln_2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simplicial message-passing update."""
        # Laplacians
        l0 = b1 @ b1.T
        l1_down = b1.T @ b1
        l1_up = b2 @ b2.T if (b2 is not None and b2.shape[1] > 0) else torch.zeros_like(l1_down)
        l2 = b2.T @ b2 if (b2 is not None and b2.shape[1] > 0 and x2.shape[0] > 0) else torch.zeros((x2.shape[0], x2.shape[0]), device=x2.device)

        # 0-simplex update
        msg_0 = self.lin_0_self(x0) + self.lin_0_from_1(torch.abs(b1) @ x1) + self.lin_0_l0(l0 @ x0)
        h0 = self.ln_0(F.relu(msg_0))
        h0 = self.dropout(h0)

        # 2-simplex update
        if b2 is not None and b2.shape[1] > 0 and x2.shape[0] > 0:
            x2_virtual = x2
            msg_2 = self.lin_2_self(x2) + self.lin_2_from_1(torch.abs(b2).T @ x1) + self.lin_2_l2(l2 @ x2)
            h2 = self.ln_2(F.relu(msg_2))
            h2 = self.dropout(h2)
        else:
            x2_virtual = x2 if (x2 is not None and x2.shape[0] > 0) else (torch.mean(x1, dim=0, keepdim=True) if x1.shape[-1] == self.lin_2_self.in_features else torch.zeros((1, self.lin_2_self.in_features), dtype=x1.dtype, device=x1.device))
            x1_for_2 = x1[:1] if x1.shape[-1] == self.lin_2_from_1.in_features else torch.zeros((1, self.lin_2_from_1.in_features), dtype=x1.dtype, device=x1.device)
            msg_2 = self.lin_2_self(x2_virtual) + self.lin_2_from_1(x1_for_2) + self.lin_2_l2(x2_virtual)
            h2 = self.ln_2(F.relu(msg_2))
            h2 = self.dropout(h2)

        # 1-simplex update
        msg_1 = (
            self.lin_1_self(x1)
            + self.lin_1_from_0(torch.abs(b1).T @ x0)
            + self.lin_1_l1_down(l1_down @ x1)
        )
        if b2 is not None and b2.shape[1] > 0 and x2.shape[0] > 0:
            msg_1 = msg_1 + self.lin_1_l1_up(l1_up @ x1) + self.lin_1_from_2(torch.abs(b2) @ x2)
        else:
            x2_v = x2_virtual.expand(x1.shape[0], -1) if x2_virtual.shape[-1] == self.lin_1_from_2.in_features else torch.zeros((x1.shape[0], self.lin_1_from_2.in_features), dtype=x1.dtype, device=x1.device)
            msg_1 = msg_1 + self.lin_1_l1_up(x1) * 0.05 + self.lin_1_from_2(x2_v) * 0.05

        h1 = self.ln_1(F.relu(msg_1))
        h1 = self.dropout(h1)

        return h0, h1, h2


class SimplicialNet(nn.Module):
    """Simplicial Complex Neural Network for candidate graph classification."""

    def __init__(
        self,
        in_dim_0: int = 6,
        in_dim_1: int = 9,
        in_dim_2: int = 9,
        hidden_dim: int = 32,
        num_classes: int = 2,
        out_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        final_classes = out_dim if out_dim is not None else num_classes

        for i in range(num_layers):
            d0 = in_dim_0 if i == 0 else hidden_dim
            d1 = in_dim_1 if i == 0 else hidden_dim
            d2 = in_dim_2 if i == 0 else hidden_dim
            self.convs.append(SimplicialConvBlock(d0, d1, d2, hidden_dim, dropout=dropout))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, final_classes),
        )

    def forward(
        self,
        x0: Any,
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass supporting both explicit tensors and view objects / dictionaries."""
        if x1 is None:
            if isinstance(x0, dict):
                d = x0
                x_0 = d.get("X0", d.get("x0", d.get("node_features")))
                x_1 = d.get("X1", d.get("x1", d.get("edge_features")))
                x_2 = d.get("X2", d.get("x2", d.get("cell_features")))
                b_1 = d.get("B1", d.get("b1", d.get("boundary_1")))
                b_2 = d.get("B2", d.get("b2", d.get("boundary_2")))
            else:
                view = x0
                x_0 = getattr(view, "X0", getattr(view, "x0", getattr(view, "node_features", None)))
                x_1 = getattr(view, "X1", getattr(view, "x1", getattr(view, "edge_features", None)))
                x_2 = getattr(view, "X2", getattr(view, "x2", getattr(view, "cell_features", None)))
                b_1 = getattr(view, "B1", getattr(view, "b1", getattr(view, "boundary_1", None)))
                b_2 = getattr(view, "B2", getattr(view, "b2", getattr(view, "boundary_2", None)))
        else:
            x_0, x_1, x_2, b_1, b_2 = x0, x1, x2, b1, b2

        if isinstance(x_0, np.ndarray):
            x_0 = torch.tensor(x_0, dtype=torch.float32)
        if isinstance(x_1, np.ndarray):
            x_1 = torch.tensor(x_1, dtype=torch.float32)
        if isinstance(x_2, np.ndarray):
            x_2 = torch.tensor(x_2, dtype=torch.float32)
        if isinstance(b_1, np.ndarray):
            b_1 = torch.tensor(b_1, dtype=torch.float32)
        if isinstance(b_2, np.ndarray):
            b_2 = torch.tensor(b_2, dtype=torch.float32)

        # Adapt input feature dimensions to match the first layer
        c0 = self.convs[0]
        x_0 = _adapt_feat_dim(x_0, c0.lin_0_self.in_features)
        if x_1 is not None:
            x_1 = _adapt_feat_dim(x_1, c0.lin_1_self.in_features)

        if x_2 is None or (isinstance(x_2, torch.Tensor) and x_2.numel() == 0):
            x_2 = torch.zeros((0, c0.lin_2_self.in_features), dtype=torch.float32, device=x_0.device if x_0 is not None else None)
        else:
            x_2 = _adapt_feat_dim(x_2, c0.lin_2_self.in_features)

        if b_2 is None or (isinstance(b_2, torch.Tensor) and b_2.numel() == 0):
            b_2 = torch.zeros((x_1.shape[0] if x_1 is not None else 0, 0), dtype=torch.float32, device=x_1.device if x_1 is not None else None)

        h0, h1, h2 = x_0, x_1, x_2
        for conv in self.convs:
            h0, h1, h2 = conv(h0, h1, h2, b_1, b_2)

        p0 = torch.mean(h0, dim=0, keepdim=True)
        p1 = torch.mean(h1, dim=0, keepdim=True)
        p2 = torch.mean(h2, dim=0, keepdim=True) if h2.shape[0] > 0 else torch.zeros_like(p0)

        pooled = torch.cat([p0, p1, p2], dim=-1)
        return self.classifier(pooled)


SimplicialComplexNet = SimplicialNet
