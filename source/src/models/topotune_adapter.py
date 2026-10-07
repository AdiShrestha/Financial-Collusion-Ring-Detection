"""
TopoTune-Inspired Combinatorial Adapter & Hybrid Topological Readouts.

Provides:
- TopoTuneInspiredAdapter: A modular message passing adapter for general combinatorial complexes,
  labeled 'TopoTune-inspired' per Section 9.3 of the project specification.
- HybridTopologicalReadout: Invariant multi-cell global readout pooling module with modular
  fusion of precomputed Persistent Homology feature vectors (for Hypothesis H3 ablation testing).

Upholds Invariants:
- INV-001 (No Mock Data in Production): Validated against combinatorial complex schemas.
- INV-008 (Self-Contained Verification Scripts).
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


def pool_rank(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Computes concatenated global mean and max pooling for a specific rank tensor."""
    if x.numel() == 0 or x.size(0) == 0:
        return torch.zeros((1, dim * 2), device=x.device, dtype=x.dtype)
    mean_p = torch.mean(x, dim=0, keepdim=True)
    max_p, _ = torch.max(x, dim=0, keepdim=True)
    return torch.cat([mean_p, max_p], dim=-1)


class TopoTuneInspiredAdapter(nn.Module):
    """
    TopoTune-inspired generalized combinatorial complex adapter layer.
    Allows flexible message aggregation across arbitrary cell ranks and incidence relations.
    (Excludes claims of baseline equivalence per project_description.md Section 9.3).
    """

    def __init__(self, in_dims: Dict[int, int], out_dim: int):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.proj_layers = nn.ModuleDict({
            str(r): nn.Linear(dim, out_dim, bias=True) for r, dim in in_dims.items()
        })

    def forward(
        self,
        features: Dict[int, torch.Tensor],
        incidences: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Propagates messages within and across combinatorial ranks.
        features: Dict mapping rank (e.g. 0, 1, 2) to feature tensor (N_r, d_r).
        incidences: Dict mapping (r_src, r_dst) to incidence matrix (N_dst, N_src).
        """
        out_features: Dict[int, torch.Tensor] = {}

        for r, x in features.items():
            if x.numel() == 0:
                out_features[r] = torch.empty((0, self.out_dim), device=x.device)
                continue

            # Base transformation
            h = self.proj_layers[str(r)](x)

            # Aggregate from incident ranks if incidence matrix provided
            if incidences is not None:
                for (r_src, r_dst), inc in incidences.items():
                    if r_dst == r and r_src in features and inc.numel() > 0:
                        x_src = features[r_src]
                        if x_src.numel() > 0:
                            abs_inc = torch.abs(inc)
                            deg = abs_inc.sum(dim=1, keepdim=True).clamp(min=1.0)
                            norm_inc = abs_inc / deg
                            proj_src = self.proj_layers[str(r_src)](x_src)
                            h = h + torch.matmul(norm_inc, proj_src)

            out_features[r] = F.relu(h)

        return out_features


class HybridTopologicalReadout(nn.Module):
    """
    Multi-dimensional pooling module that combines learned cell representations
    (ranks 0, 1, 2) with fixed precomputed persistent homology vectors.
    """

    def __init__(
        self,
        cell_hidden_dim: int = 27,
        num_ranks: int = 3,
        ph_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_hidden_dim = cell_hidden_dim
        self.num_ranks = num_ranks
        self.ph_dim = ph_dim

        # Dual readout (mean + max) per active rank
        total_cell_dim = num_ranks * (cell_hidden_dim * 2)

        if ph_dim is not None and ph_dim > 0:
            self.ph_proj = nn.Linear(ph_dim, 64)
            classifier_in = total_cell_dim + 64
        else:
            self.ph_proj = None
            classifier_in = total_cell_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        cell_embeddings: List[torch.Tensor],
        ph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pools cell embeddings across all ranks, concatenates optional PH vector,
        and evaluates classification head.
        """
        pooled_ranks = []
        for x in cell_embeddings:
            p = pool_rank(x, self.cell_hidden_dim)
            pooled_ranks.append(p)

        pooled = torch.cat(pooled_ranks, dim=-1)

        if self.ph_proj is not None and ph_features is not None:
            h_ph = F.relu(self.ph_proj(ph_features))
            pooled = torch.cat([pooled, h_ph], dim=-1)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)


class TopoTuneInspiredModel(nn.Module):
    """TopoTune-inspired model combining adapter layer and hybrid readout."""
    def __init__(
        self,
        in_dim0: int = 13,
        in_dim1: int = 2,
        in_dim2: int = 4,
        hidden_dim: int = 27,
        ph_dim: Optional[int] = None,
    ):
        super().__init__()
        self.adapter = TopoTuneInspiredAdapter(
            in_dims={0: in_dim0, 1: in_dim1, 2: in_dim2},
            out_dim=hidden_dim,
        )
        self.readout = HybridTopologicalReadout(
            cell_hidden_dim=hidden_dim,
            num_ranks=3,
            ph_dim=ph_dim,
        )

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
        features = {0: x0, 1: x1, 2: x2}
        incidences = {(1, 0): b1, (2, 1): b2}
        h = self.adapter(features, incidences)
        return self.readout([h[0], h[1], h[2]], ph_features=ph_features)
