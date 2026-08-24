"""Boundary operator and higher-order adjacency ablation experiment module."""

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from source.evidence.statistics import compute_classification_metrics
from source.models.cell_net import CellularComplexNet
from source.models.simplicial_net import SimplicialComplexNet
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView


def ablate_cell_view_b2(view: CycleCellView) -> CycleCellView:
    """Return a clone of CycleCellView with 2-cell boundary operator zeroed out."""
    b2_zero = np.zeros_like(view.B2)
    l1_down = view.B1.T @ view.B1
    return CycleCellView(
        candidate_id=view.candidate_id,
        nodes=view.nodes,
        edges=view.edges,
        cells_2=view.cells_2,
        B1=view.B1,
        B2=b2_zero,
        L0=view.L0,
        L1=l1_down,
        node_features=view.node_features,
        edge_features=view.edge_features,
        cell_features=np.zeros_like(view.cell_features),
        target_y=view.target_y,
        metadata=dict(view.metadata),
    )


def ablate_simplicial_view_b2(view: CliqueSimplicialView) -> CliqueSimplicialView:
    """Return a clone of CliqueSimplicialView with 2-simplex boundary operator zeroed out."""
    b2_zero = np.zeros_like(view.B2)
    l1_down = view.B1.T @ view.B1
    return CliqueSimplicialView(
        candidate_id=view.candidate_id,
        nodes=view.nodes,
        edges=view.edges,
        triangles=view.triangles,
        B1=view.B1,
        B2=b2_zero,
        L0=view.L0,
        L1=l1_down,
        L2=np.zeros_like(view.L2),
        node_features=view.node_features,
        edge_features=view.edge_features,
        face_features=np.zeros_like(view.face_features),
        target_y=view.target_y,
        metadata=dict(view.metadata),
    )


class BoundaryAblationRunner:
    """Runner for boundary operator and Hodge Laplacian ablation experiments."""

    MODES = [
        "ccnn_full_2cell",
        "ccnn_ablated_1skeleton",
        "scnn_full_hodge",
        "scnn_lower_only",
    ]

    def __init__(
        self,
        in_dim_node: int = 56,
        in_dim_edge: int = 2,
        in_dim_cell: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
        out_dim: int = 2,
    ):
        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.in_dim_cell = in_dim_cell
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

    def run_boundary_ablations(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: Optional[List[Dict[str, Any]]] = None,
        seeds: Sequence[int] = (42, 43, 44, 45, 46),
        modes: Optional[Sequence[str]] = None,
        epochs: int = 10,
        lr: float = 0.01,
    ) -> Dict[str, Any]:
        """Execute boundary operator ablations across seeds on validation split."""
        if test_data is not None:
            raise ValueError("Test data cannot be passed to boundary ablation runner (INV-006)")

        target_modes = list(modes) if modes is not None else self.MODES
        results_by_mode: Dict[str, List[Dict[str, float]]] = {m: [] for m in target_modes}

        for mode in target_modes:
            is_scnn = mode.startswith("scnn")
            is_ablated = "ablated" in mode or "lower_only" in mode

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                if is_scnn:
                    model = SimplicialComplexNet(
                        in_dim_0=self.in_dim_node,
                        in_dim_1=self.in_dim_edge,
                        in_dim_2=self.in_dim_cell,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        out_dim=self.out_dim,
                    )
                else:
                    model = CellularComplexNet(
                        in_dim_0=self.in_dim_node,
                        in_dim_1=self.in_dim_edge,
                        in_dim_2=self.in_dim_cell,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers,
                        out_dim=self.out_dim,
                    )

                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                # Train loop
                model.train()
                for _ in range(epochs):
                    for item in train_data:
                        optimizer.zero_grad()
                        raw_view = item.get("cell_view" if not is_scnn else "simplicial_view")
                        if raw_view is None:
                            raw_view = item.get("domain_view")

                        if is_scnn:
                            view = ablate_simplicial_view_b2(raw_view) if is_ablated else raw_view
                        else:
                            view = ablate_cell_view_b2(raw_view) if is_ablated else raw_view

                        logits = model(view)
                        y_val = item.get("target_y", item.get("y"))
                        if isinstance(y_val, torch.Tensor):
                            y_target = y_val if y_val.dim() > 0 else y_val.unsqueeze(0)
                        else:
                            y_target = torch.tensor([int(y_val)], dtype=torch.long)

                        loss = criterion(logits, y_target)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # Validation evaluation
                model.eval()
                val_probs = []
                val_trues = []
                with torch.no_grad():
                    for item in val_data:
                        raw_view = item.get("cell_view" if not is_scnn else "simplicial_view")
                        if raw_view is None:
                            raw_view = item.get("domain_view")

                        if is_scnn:
                            view = ablate_simplicial_view_b2(raw_view) if is_ablated else raw_view
                        else:
                            view = ablate_cell_view_b2(raw_view) if is_ablated else raw_view

                        logits = model(view)
                        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                        val_probs.extend(prob.tolist())

                        y_val = item.get("target_y", item.get("y"))
                        val_trues.append(int(y_val.item() if isinstance(y_val, torch.Tensor) else y_val))

                metrics = compute_classification_metrics(val_trues, val_probs)
                results_by_mode[mode].append(metrics)

        # Summary aggregation
        summary = {}
        for mode, run_metrics in results_by_mode.items():
            pr_scores = [m["pr_auc"] for m in run_metrics]
            roc_scores = [m["roc_auc"] for m in run_metrics]
            f1_scores = [m["f1_macro"] for m in run_metrics]

            summary[mode] = {
                "mean_pr_auc": float(np.mean(pr_scores)),
                "std_pr_auc": float(np.std(pr_scores)),
                "mean_roc_auc": float(np.mean(roc_scores)),
                "std_roc_auc": float(np.std(roc_scores)),
                "mean_f1_macro": float(np.mean(f1_scores)),
                "std_f1_macro": float(np.std(f1_scores)),
                "num_seeds": len(run_metrics),
                "runs": run_metrics,
            }

        # Calculate deltas between full and ablated pairs
        if "ccnn_full_2cell" in summary and "ccnn_ablated_1skeleton" in summary:
            summary["ccnn_b2_contribution_delta_pr"] = float(
                summary["ccnn_full_2cell"]["mean_pr_auc"] - summary["ccnn_ablated_1skeleton"]["mean_pr_auc"]
            )
        if "scnn_full_hodge" in summary and "scnn_lower_only" in summary:
            summary["scnn_upper_hodge_contribution_delta_pr"] = float(
                summary["scnn_full_hodge"]["mean_pr_auc"] - summary["scnn_lower_only"]["mean_pr_auc"]
            )

        return {
            "summary": summary,
            "seeds": list(seeds),
            "modes_evaluated": target_modes,
            "evaluated_split": "validation",
        }
