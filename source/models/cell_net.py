"""Cellular Complex Neural Network (CCNN / CIN) architecture for polygonal complexes."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from source.topology.cycle_cell_view import CycleCellView


class CellularMessagePassingBlock(nn.Module):
    """Cellular message passing layer propagating signals across 0-cells, 1-cells, and 2-cells (polygons).

    Operates on arbitrary k-gons (k in {3, 4, 5, 6}) without simplicial distortion (INV-002).
    """

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

        # 0-cell (vertex) updates
        self.lin_0_self = nn.Linear(in_dim_0, out_dim)
        self.lin_0_from_1 = nn.Linear(in_dim_1, out_dim, bias=False)
        self.lin_0_l0 = nn.Linear(in_dim_0, out_dim, bias=False)
        self.ln_0 = nn.LayerNorm(out_dim)

        # 1-cell (edge) updates
        self.lin_1_self = nn.Linear(in_dim_1, out_dim)
        self.lin_1_from_0 = nn.Linear(in_dim_0, out_dim, bias=False)
        self.lin_1_from_2 = nn.Linear(in_dim_2, out_dim, bias=False)
        self.lin_1_l1_down = nn.Linear(in_dim_1, out_dim, bias=False)
        self.lin_1_l1_up = nn.Linear(in_dim_1, out_dim, bias=False)
        self.ln_1 = nn.LayerNorm(out_dim)

        # 2-cell (polygon) updates
        self.lin_2_self = nn.Linear(in_dim_2, out_dim)
        self.lin_2_from_1 = nn.Linear(in_dim_1, out_dim, bias=False)
        self.lin_2_l2 = nn.Linear(in_dim_2, out_dim, bias=False)
        self.ln_2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor],
        b1: torch.Tensor,
        b2: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Compute Laplacians
        l0 = b1 @ b1.T
        l1_down = b1.T @ b1

        has_cells = x2 is not None and x2.numel() > 0 and b2 is not None and b2.numel() > 0 and b2.size(1) > 0

        if has_cells:
            l1_up = b2 @ b2.T
            l2 = b2.T @ b2
        else:
            l1_up = torch.zeros((x1.size(0), x1.size(0)), device=x1.device, dtype=x1.dtype)
            l2 = None

        # 1. Update 0-cells
        msg_0_from_1 = b1 @ x1 if x1.numel() > 0 else torch.zeros_like(x0)
        h0 = self.lin_0_self(x0) + self.lin_0_from_1(msg_0_from_1) + self.lin_0_l0(l0 @ x0)
        h0 = self.ln_0(h0)
        h0 = F.relu(h0)
        h0 = self.dropout(h0)

        # 2. Update 1-cells
        msg_1_from_0 = b1.T @ x0
        h1 = self.lin_1_self(x1) + self.lin_1_from_0(msg_1_from_0) + self.lin_1_l1_down(l1_down @ x1) + self.lin_1_l1_up(l1_up @ x1)
        if has_cells:
            msg_1_from_2 = b2 @ x2
            h1 = h1 + self.lin_1_from_2(msg_1_from_2)
        h1 = self.ln_1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # 3. Update 2-cells
        if has_cells:
            msg_2_from_1 = b2.T @ x1
            h2 = self.lin_2_self(x2) + self.lin_2_from_1(msg_2_from_1) + self.lin_2_l2(l2 @ x2)
            h2 = self.ln_2(h2)
            h2 = F.relu(h2)
            h2 = self.dropout(h2)
        else:
            # Ensure cell parameters participate in autograd graph
            dummy_2 = 0.0 * (
                self.lin_2_self.weight.sum()
                + self.lin_2_self.bias.sum()
                + self.lin_2_from_1.weight.sum()
                + self.lin_2_l2.weight.sum()
                + self.lin_1_from_2.weight.sum()
                + self.ln_2.weight.sum()
                + self.ln_2.bias.sum()
            )
            h0 = h0 + dummy_2
            h2 = None

        return h0, h1, h2


class CellularComplexNet(nn.Module):
    """Cellular Complex Neural Network (CCNN / CIN) for candidate polygonal subgraphs."""

    def __init__(
        self,
        in_dim_0: int = 56,
        in_dim_1: int = 2,
        in_dim_2: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim_0 = in_dim_0
        self.in_dim_1 = in_dim_1
        self.in_dim_2 = in_dim_2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # Initial projection layers
        self.proj_0 = nn.Linear(in_dim_0, hidden_dim)
        self.proj_1 = nn.Linear(in_dim_1, hidden_dim)
        self.proj_2 = nn.Linear(in_dim_2, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                CellularMessagePassingBlock(
                    in_dim_0=hidden_dim,
                    in_dim_1=hidden_dim,
                    in_dim_2=hidden_dim,
                    out_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        # Candidate-level readout classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        cell_input: Union[CycleCellView, Tuple[torch.Tensor, ...]],
        x1: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass accepting either a CycleCellView or explicit tensors."""
        if isinstance(cell_input, CycleCellView):
            x0 = torch.tensor(cell_input.node_features, dtype=torch.float32)
            x1 = torch.tensor(cell_input.edge_features, dtype=torch.float32)
            if cell_input.num_cells_2 > 0:
                x2 = torch.tensor(cell_input.cell_features, dtype=torch.float32)
                b2 = torch.tensor(cell_input.B2, dtype=torch.float32)
            else:
                x2 = torch.zeros((0, self.in_dim_2), dtype=torch.float32)
                b2 = torch.zeros((x1.size(0), 0), dtype=torch.float32)
            b1 = torch.tensor(cell_input.B1, dtype=torch.float32)
        else:
            x0 = cell_input
            if x2 is None:
                x2 = torch.zeros((0, self.in_dim_2), dtype=torch.float32, device=x0.device)
            if b2 is None:
                x1_len = x1.size(0) if x1 is not None else 0
                b2 = torch.zeros((x1_len, 0), dtype=torch.float32, device=x0.device)

        # Projections
        h0 = self.proj_0(x0)
        h1 = self.proj_1(x1)
        if x2 is not None and x2.numel() > 0:
            h2 = self.proj_2(x2)
        else:
            dummy_proj = (self.proj_2.weight.sum() + self.proj_2.bias.sum()) * 0.0
            h0 = h0 + dummy_proj
            h2 = None

        # Cellular message passing
        for block in self.blocks:
            h0, h1, h2 = block(h0, h1, h2, b1, b2)

        # Multi-scale global pooling
        g0 = torch.mean(h0, dim=0, keepdim=True)
        g1 = torch.mean(h1, dim=0, keepdim=True) if h1.numel() > 0 else torch.zeros_like(g0)
        g2 = torch.mean(h2, dim=0, keepdim=True) if h2 is not None and h2.numel() > 0 else torch.zeros_like(g0)

        g_cat = torch.cat([g0, g1, g2], dim=-1)
        logits = self.classifier(g_cat)
        return logits
