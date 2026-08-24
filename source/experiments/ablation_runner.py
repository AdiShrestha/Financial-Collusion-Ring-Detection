"""Topological feature ablation experiment runner on validation partition."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from source.evidence.statistics import compute_classification_metrics
from source.models.ph_augmented_net import TopoRingNet
from source.topology.cycle_cell_view import CycleCellView


def apply_ablation_mask(z_topo: np.ndarray, ablation_mode: str) -> np.ndarray:
    """Apply feature ablation mask to 372-dim persistent homology vector."""
    z_masked = np.array(z_topo, dtype=np.float32).copy()
    dim_per_homology = 186

    if ablation_mode == "full":
        return z_masked
    elif ablation_mode == "no_betti":
        # Betti: 0..20 in H0, 186..206 in H1
        z_masked[..., 0:20] = 0.0
        z_masked[..., dim_per_homology : dim_per_homology + 20] = 0.0
    elif ablation_mode == "no_landscapes":
        # Landscapes: 20..80 in H0, 206..266 in H1
        z_masked[..., 20:80] = 0.0
        z_masked[..., dim_per_homology + 20 : dim_per_homology + 80] = 0.0
    elif ablation_mode == "no_images":
        # Images: 80..180 in H0, 266..366 in H1
        z_masked[..., 80:180] = 0.0
        z_masked[..., dim_per_homology + 80 : dim_per_homology + 180] = 0.0
    elif ablation_mode == "no_entropy_stats":
        # Stats & Entropy: 180..186 in H0, 366..372 in H1
        z_masked[..., 180:186] = 0.0
        z_masked[..., dim_per_homology + 180 : dim_per_homology + 186] = 0.0
    elif ablation_mode in ("struct_only", "no_ph"):
        # Zero out entire 372-dim topological feature vector
        z_masked[...] = 0.0
    else:
        raise ValueError(f"Unknown ablation mode: {ablation_mode}")

    return z_masked


class TopologicalAblationRunner:
    """Multi-seed topological ablation runner evaluating on validation candidates."""

    ABLATION_MODES = [
        "full",
        "no_betti",
        "no_landscapes",
        "no_images",
        "no_entropy_stats",
        "struct_only",
    ]

    def __init__(
        self,
        in_dim_node: int = 56,
        in_dim_edge: int = 2,
        in_dim_cell: int = 2,
        in_dim_topo: int = 372,
        hidden_dim: int = 32,
        num_layers: int = 2,
        out_dim: int = 2,
    ):
        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.in_dim_cell = in_dim_cell
        self.in_dim_topo = in_dim_topo
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

    def _forward_item(
        self,
        model: TopoRingNet,
        item: Dict[str, Any],
        ablation_mode: str,
    ) -> torch.Tensor:
        """Helper to forward a single candidate or batch through TopoRingNet."""
        z_raw = item.get("z_topo")
        if z_raw is not None:
            z_masked = apply_ablation_mask(z_raw, ablation_mode)
            z_topo_t = torch.tensor(z_masked, dtype=torch.float32)
        else:
            z_topo_t = torch.zeros(self.in_dim_topo, dtype=torch.float32)

        view = item.get("cell_view", item.get("domain_view"))
        if view is not None:
            return model(
                view,
                z_topo=z_topo_t,
                ablation_mode="struct_only" if ablation_mode == "struct_only" else "full",
            )
        else:
            raise KeyError("Item must contain 'cell_view' or 'domain_view'")

    def run_ablation_suite(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: Optional[List[Dict[str, Any]]] = None,
        seeds: Sequence[int] = (42, 43, 44, 45, 46),
        modes: Optional[Sequence[str]] = None,
        epochs: int = 10,
        lr: float = 0.01,
    ) -> Dict[str, Any]:
        """Execute ablation suite across locked seeds exclusively on validation split."""
        if test_data is not None:
            raise ValueError("Test data cannot be passed to ablation runner (INV-006)")

        target_modes = list(modes) if modes is not None else self.ABLATION_MODES
        results_by_mode: Dict[str, List[Dict[str, float]]] = {m: [] for m in target_modes}

        for mode in target_modes:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = TopoRingNet(
                    in_dim_node=self.in_dim_node,
                    in_dim_edge=self.in_dim_edge,
                    in_dim_cell=self.in_dim_cell,
                    in_dim_topo=self.in_dim_topo,
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
                        logits = self._forward_item(model, item, mode)
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
                        logits = self._forward_item(model, item, mode)
                        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                        val_probs.extend(prob.tolist())

                        y_val = item.get("target_y", item.get("y"))
                        val_trues.append(int(y_val.item() if isinstance(y_val, torch.Tensor) else y_val))

                metrics = compute_classification_metrics(val_trues, val_probs)
                results_by_mode[mode].append(metrics)

        # Aggregate summary statistics
        summary = {}
        full_mean_pr = float(np.mean([m["pr_auc"] for m in results_by_mode.get("full", [{}])])) if "full" in results_by_mode else 1.0

        for mode, run_metrics in results_by_mode.items():
            pr_scores = [m["pr_auc"] for m in run_metrics]
            roc_scores = [m["roc_auc"] for m in run_metrics]
            f1_scores = [m["f1_macro"] for m in run_metrics]

            mean_pr = float(np.mean(pr_scores))
            std_pr = float(np.std(pr_scores))
            mean_roc = float(np.mean(roc_scores))
            std_roc = float(np.std(roc_scores))
            mean_f1 = float(np.mean(f1_scores))
            std_f1 = float(np.std(f1_scores))
            delta_pr = float(mean_pr - full_mean_pr)

            summary[mode] = {
                "mean_pr_auc": mean_pr,
                "std_pr_auc": std_pr,
                "mean_roc_auc": mean_roc,
                "std_roc_auc": std_roc,
                "mean_f1_macro": mean_f1,
                "std_f1_macro": std_f1,
                "delta_pr_auc": delta_pr,
                "num_seeds": len(run_metrics),
                "runs": run_metrics,
            }

        return {
            "summary": summary,
            "seeds": list(seeds),
            "modes_evaluated": target_modes,
            "evaluated_split": "validation",
        }
