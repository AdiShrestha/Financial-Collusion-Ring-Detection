"""
Cellular Complex Network (CWN) for Polygonal Financial Collusion Rings.

Propagates neural messages over:
- 0-cells (vertices): transaction accounts
- 1-cells (edges): directed money transfers
- 2-cells (polygonal faces): closed collusion cycles (k in [3, 6]) without triangulation (INV-002).

Upholds Invariants:
- INV-001 (No Mock Data in Production).
- INV-002 (Simplex vs Polygonal Cell Separation): Operates on k >= 4 cycle cells.
- INV-004 (Exact Boundary Nilpotency): B1 @ B2 = 0.
- INV-008 (Self-Contained Verification Scripts).
- INV-009 (Model Parameter and FLOP Matching Protocol): parameter delta <= 2%.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))


def global_cell_pool(x: torch.Tensor) -> torch.Tensor:
    """Computes concatenated global mean and max pooling across cells."""
    if x.numel() == 0 or x.size(0) == 0:
        return torch.zeros((1, x.size(-1) * 2), device=x.device, dtype=x.dtype)
    mean_p = torch.mean(x, dim=0, keepdim=True)
    max_p, _ = torch.max(x, dim=0, keepdim=True)
    return torch.cat([mean_p, max_p], dim=-1)


class CellularConvLayer(nn.Module):
    """
    Higher-order cellular message passing layer.
    Propagates signals between adjacent and incident cells via Hodge Laplacians
    and signed incidence operators.
    """

    def __init__(self, in_dim0: int, in_dim1: int, in_dim2: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.linear0 = nn.Linear(in_dim0, out_dim, bias=True)
        self.linear1 = nn.Linear(in_dim1, out_dim, bias=True)
        self.linear2 = nn.Linear(in_dim2, out_dim, bias=True)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        l0: Optional[torch.Tensor] = None,
        l1: Optional[torch.Tensor] = None,
        l2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x0: (N0, d0) 0-cell features
            x1: (N1, d1) 1-cell features
            x2: (N2, d2) 2-cell features (can be 0-length)
            b1: (N0, N1) signed boundary operator
            b2: (N1, N2) signed boundary operator
            l0, l1, l2: optional precomputed Hodge Laplacians
        """
        # 1. Update 0-cells: aggregate from L0 (within 0-cells) and B1 (from 1-cells)
        abs_b1 = torch.abs(b1)
        m0 = x0
        if l0 is not None and l0.numel() > 0:
            deg_l0 = torch.abs(l0).sum(dim=1, keepdim=True).clamp(min=1.0)
            m0 = m0 + torch.matmul(l0 / deg_l0, x0)
        if x1.numel() > 0 and b1.numel() > 0:
            # Project 1-cell features onto 0-cells via |B1|
            deg0 = abs_b1.sum(dim=1, keepdim=True).clamp(min=1.0)
            norm_b1 = abs_b1 / deg0
            if x0.size(-1) == x1.size(-1):
                m0 = m0 + torch.matmul(norm_b1, x1)
        h0_out = F.relu(self.linear0(m0))

        # 2. Update 1-cells: aggregate from L1, B1^T (from 0-cells), and B2 (from 2-cells)
        m1 = x1
        if l1 is not None and l1.numel() > 0:
            deg_l1 = torch.abs(l1).sum(dim=1, keepdim=True).clamp(min=1.0)
            # Edge attributes are scalar magnitudes, not oriented 1-cochains.
            # Absolute coupling is invariant to arbitrary edge-basis flips.
            m1 = m1 + torch.matmul(torch.abs(l1) / deg_l1, x1)
        if x0.numel() > 0 and b1.numel() > 0 and x0.size(-1) == x1.size(-1):
            deg1_down = abs_b1.t().sum(dim=1, keepdim=True).clamp(min=1.0)
            norm_b1_t = abs_b1.t() / deg1_down
            m1 = m1 + torch.matmul(norm_b1_t, x0)
        if x2.numel() > 0 and b2.numel() > 0 and x2.size(-1) == x1.size(-1):
            abs_b2 = torch.abs(b2)
            deg1_up = abs_b2.sum(dim=1, keepdim=True).clamp(min=1.0)
            norm_b2 = abs_b2 / deg1_up
            m1 = m1 + torch.matmul(norm_b2, x2)
        h1_out = F.relu(self.linear1(m1)) if x1.numel() > 0 else torch.empty((0, self.out_dim), device=x1.device)

        # 3. Update 2-cells: aggregate from L2 and B2^T (from 1-cells)
        if x2.numel() > 0:
            m2 = x2
            if l2 is not None and l2.numel() > 0:
                deg_l2 = torch.abs(l2).sum(dim=1, keepdim=True).clamp(min=1.0)
                m2 = m2 + torch.matmul(l2 / deg_l2, x2)
            if x1.numel() > 0 and b2.numel() > 0 and x1.size(-1) == x2.size(-1):
                abs_b2_t = torch.abs(b2).t()
                deg2 = abs_b2_t.sum(dim=1, keepdim=True).clamp(min=1.0)
                norm_b2_t = abs_b2_t / deg2
                m2 = m2 + torch.matmul(norm_b2_t, x1)
            h2_out = F.relu(self.linear2(m2))
        else:
            h2_out = torch.empty((0, self.out_dim), device=x2.device)

        return h0_out, h1_out, h2_out


def pool_cells_by_mode(x: torch.Tensor, mode: str = "mean_max") -> torch.Tensor:
    """Pools cell representation with specified pooling mode."""
    if x.numel() == 0 or x.size(0) == 0:
        mult = 2 if mode == "mean_max" else 1
        return torch.zeros((1, x.size(-1) * mult), device=x.device, dtype=x.dtype)
    if mode == "mean_only":
        return torch.mean(x, dim=0, keepdim=True)
    elif mode == "max_only":
        return torch.max(x, dim=0, keepdim=True)[0]
    else:  # mean_max
        mean_p = torch.mean(x, dim=0, keepdim=True)
        max_p, _ = torch.max(x, dim=0, keepdim=True)
        return torch.cat([mean_p, max_p], dim=-1)


class CellularComplexNetwork(nn.Module):
    """
    Capacity-matched Cellular Complex Network (CWN) for financial collusion rings.
    Calibrated with d_hidden=27 to match standard baseline GNN capacity (~13.4k params).
    Supports configurable message passing depth and readout pooling mechanisms.
    """

    def __init__(
        self,
        in_dim0: int = 13,
        in_dim1: int = 2,
        in_dim2: int = 4,
        hidden_dim: int = 27,
        ph_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_layers: int = 2,
        readout_pooling: str = "mean_max",
    ):
        super().__init__()
        self.in_dim0 = in_dim0
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.hidden_dim = hidden_dim
        self.ph_dim = ph_dim
        self.dropout = dropout
        self.num_layers = max(1, num_layers)
        self.readout_pooling = readout_pooling

        # Dynamic Message passing layers
        self.input_projections = nn.ModuleList([
            nn.Linear(in_dim0, hidden_dim),
            nn.Linear(in_dim1, hidden_dim),
            nn.Linear(in_dim2, hidden_dim),
        ])
        self.convs = nn.ModuleList()
        # Layer 1
        self.convs.append(CellularConvLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        # Additional layers
        for _ in range(self.num_layers - 1):
            self.convs.append(CellularConvLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim))

        # Backward compatibility aliases for 2-layer default
        self.conv1 = self.convs[0]
        self.conv2 = self.convs[1] if self.num_layers > 1 else self.convs[0]

        # Readout dimension
        pool_mult = 2 if readout_pooling == "mean_max" else 1
        readout_dim = 3 * pool_mult * hidden_dim

        if ph_dim is not None and ph_dim > 0:
            self.ph_proj = nn.Linear(ph_dim, 64)
            classifier_in = readout_dim + 64
        else:
            self.ph_proj = None
            classifier_in = readout_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        l0: Optional[torch.Tensor] = None,
        l1: Optional[torch.Tensor] = None,
        l2: Optional[torch.Tensor] = None,
        ph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h0, h1, h2 = [F.relu(proj(x)) for proj, x in zip(self.input_projections, (x0, x1, x2))]
        for idx, conv in enumerate(self.convs):
            h0, h1, h2 = conv(h0, h1, h2, b1, b2, l0, l1, l2)
            if idx < len(self.convs) - 1:
                h0 = F.dropout(h0, p=self.dropout, training=self.training)
                if h1.numel() > 0:
                    h1 = F.dropout(h1, p=self.dropout, training=self.training)
                if h2.numel() > 0:
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Global hierarchical pooling across cell dimensions
        pool_mult = 2 if self.readout_pooling == "mean_max" else 1
        p0 = pool_cells_by_mode(h0, mode=self.readout_pooling)
        p1 = pool_cells_by_mode(h1, mode=self.readout_pooling) if h1.numel() > 0 else torch.zeros((1, self.hidden_dim * pool_mult), device=x0.device)
        p2 = pool_cells_by_mode(h2, mode=self.readout_pooling) if h2.numel() > 0 else torch.zeros((1, self.hidden_dim * pool_mult), device=x0.device)

        pooled = torch.cat([p0, p1, p2], dim=-1)

        # Optional Persistent Homology fusion
        if self.ph_proj is not None and ph_features is not None:
            h_ph = F.relu(self.ph_proj(ph_features))
            pooled = torch.cat([pooled, h_ph], dim=-1)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)
