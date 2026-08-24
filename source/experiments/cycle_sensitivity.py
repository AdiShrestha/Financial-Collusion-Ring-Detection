"""Cycle length sensitivity benchmark analyzing detection across cycle lengths k in {3,4,5,6}."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from source.evidence.statistics import compute_classification_metrics
from source.models.cell_net import CellularComplexNet
from source.models.gnn_baselines import GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.graph_view import GraphView


def get_candidate_cycle_length(item: Dict[str, Any]) -> int:
    """Extract or infer cycle length k for candidate example."""
    if "cycle_length" in item:
        return int(item["cycle_length"])

    cell_v = item.get("cell_view", item.get("domain_view"))
    if cell_v is not None and isinstance(cell_v, CycleCellView):
        if len(cell_v.cells_2) > 0:
            return len(cell_v.cells_2[0])
        return len(cell_v.nodes)

    cand = item.get("candidate")
    if cand is not None and hasattr(cand, "participant_ids"):
        return len(cand.participant_ids)

    nodes = item.get("nodes")
    if nodes is not None:
        return len(nodes)

    return 4  # Default fallback


def stratify_candidates_by_cycle_length(
    candidates: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (3, 4, 5, 6),
) -> Dict[int, List[Dict[str, Any]]]:
    """Partition candidate examples into strata by cycle length k."""
    strata: Dict[int, List[Dict[str, Any]]] = {k: [] for k in k_values}

    for item in candidates:
        k = get_candidate_cycle_length(item)
        if k in strata:
            strata[k].append(item)
        elif k < min(k_values):
            strata[min(k_values)].append(item)
        elif k > max(k_values):
            strata[max(k_values)].append(item)

    return strata


class CycleSensitivityAnalyzer:
    """Evaluates detection performance across cycle length strata k in {3,4,5,6}."""

    MODEL_CLASSES = {
        "GCNBaseline": GCNBaseline,
        "GraphSAGEBaseline": GraphSAGEBaseline,
        "SimplicialComplexNet": SimplicialComplexNet,
        "CellularComplexNet": CellularComplexNet,
        "TopoRingNet": TopoRingNet,
    }

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

    def _init_model(self, model_name: str) -> nn.Module:
        if model_name == "GCNBaseline":
            return GCNBaseline(
                in_dim=self.in_dim_node,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                num_layers=self.num_layers,
            )
        elif model_name == "GraphSAGEBaseline":
            return GraphSAGEBaseline(
                in_dim=self.in_dim_node,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                num_layers=self.num_layers,
            )
        elif model_name == "SimplicialComplexNet":
            return SimplicialComplexNet(
                in_dim_0=self.in_dim_node,
                in_dim_1=self.in_dim_edge,
                in_dim_2=self.in_dim_cell,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                out_dim=self.out_dim,
            )
        elif model_name == "CellularComplexNet":
            return CellularComplexNet(
                in_dim_0=self.in_dim_node,
                in_dim_1=self.in_dim_edge,
                in_dim_2=self.in_dim_cell,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                out_dim=self.out_dim,
            )
        elif model_name == "TopoRingNet":
            return TopoRingNet(
                in_dim_node=self.in_dim_node,
                in_dim_edge=self.in_dim_edge,
                in_dim_cell=self.in_dim_cell,
                in_dim_topo=372,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                out_dim=self.out_dim,
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def _forward_model(self, model: nn.Module, item: Dict[str, Any]) -> torch.Tensor:
        if isinstance(model, TopoRingNet):
            view = item.get("cell_view", item.get("domain_view"))
            z_t = item.get("z_topo")
            if z_t is not None:
                z_topo_t = torch.tensor(z_t, dtype=torch.float32)
            else:
                z_topo_t = torch.zeros(372, dtype=torch.float32)
            return model(view, z_topo=z_topo_t)
        elif isinstance(model, CellularComplexNet):
            view = item.get("cell_view", item.get("domain_view"))
            return model(view)
        elif isinstance(model, SimplicialComplexNet):
            simp_v = item.get("simplicial_view")
            if simp_v is not None:
                return model(simp_v)
            # Adapt from cell_view if simplicial_view not explicitly supplied
            cell_v = item.get("cell_view", item.get("domain_view"))
            cand = item.get("candidate")
            if cand is not None:
                return model(CliqueSimplicialView.from_candidate_example(cand))
            return model(cell_v)
        else:
            # GNN Baselines (GCN, GraphSAGE)
            g_v = item.get("graph_view")
            if g_v is not None:
                return model(g_v)
            cell_v = item.get("cell_view", item.get("domain_view"))
            if cell_v is not None and isinstance(cell_v, CycleCellView):
                x = torch.tensor(cell_v.node_features, dtype=torch.float32)
                node_to_idx = {n: i for i, n in enumerate(cell_v.nodes)}
                if cell_v.edges:
                    src = [node_to_idx[u] for u, v in cell_v.edges]
                    dst = [node_to_idx[v] for u, v in cell_v.edges]
                    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                return model(x, edge_index=edge_index)
            cand = item.get("candidate")
            if cand is not None:
                return model(GraphView.from_candidate_example(cand))
            raise ValueError("Could not extract GNN inputs from item")

    def run_sensitivity_analysis(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: Optional[List[Dict[str, Any]]] = None,
        model_names: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42, 43, 44, 45, 46),
        k_values: Sequence[int] = (3, 4, 5, 6),
        epochs: int = 5,
        lr: float = 0.01,
    ) -> Dict[str, Any]:
        """Run stratified sensitivity analysis across cycle lengths on validation split."""
        if test_data is not None:
            raise ValueError("Test data cannot be passed to sensitivity analyzer (INV-006)")

        target_models = list(model_names) if model_names is not None else list(self.MODEL_CLASSES.keys())
        val_strata = stratify_candidates_by_cycle_length(val_data, k_values=k_values)

        results: Dict[str, Dict[int, List[Dict[str, float]]]] = {
            m: {k: [] for k in k_values} for m in target_models
        }

        for model_name in target_models:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = self._init_model(model_name)
                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                # Train model on train_data
                model.train()
                for _ in range(epochs):
                    for item in train_data:
                        optimizer.zero_grad()
                        logits = self._forward_model(model, item)
                        y_val = item.get("target_y", item.get("y"))
                        if isinstance(y_val, torch.Tensor):
                            y_target = y_val if y_val.dim() > 0 else y_val.unsqueeze(0)
                        else:
                            y_target = torch.tensor([int(y_val)], dtype=torch.long)

                        loss = criterion(logits, y_target)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # Evaluate on each validation stratum k
                model.eval()
                with torch.no_grad():
                    for k in k_values:
                        stratum_items = val_strata[k]
                        if not stratum_items:
                            continue

                        val_probs = []
                        val_trues = []
                        for item in stratum_items:
                            logits = self._forward_model(model, item)
                            prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                            val_probs.extend(prob.tolist())

                            y_val = item.get("target_y", item.get("y"))
                            val_trues.append(int(y_val.item() if isinstance(y_val, torch.Tensor) else y_val))

                        metrics = compute_classification_metrics(val_trues, val_probs)
                        results[model_name][k].append(metrics)

        # Aggregate summary statistics
        summary = {}
        for model_name, k_dict in results.items():
            summary[model_name] = {}
            for k, run_metrics in k_dict.items():
                if not run_metrics:
                    summary[model_name][k] = {"count": 0, "status": "EMPTY_STRATUM"}
                    continue

                pr_scores = [m["pr_auc"] for m in run_metrics]
                f1_scores = [m["f1_macro"] for m in run_metrics]
                roc_scores = [m["roc_auc"] for m in run_metrics]

                summary[model_name][k] = {
                    "mean_pr_auc": float(np.mean(pr_scores)),
                    "std_pr_auc": float(np.std(pr_scores)),
                    "mean_f1_macro": float(np.mean(f1_scores)),
                    "std_f1_macro": float(np.std(f1_scores)),
                    "mean_roc_auc": float(np.mean(roc_scores)),
                    "num_runs": len(run_metrics),
                }

        return {
            "summary": summary,
            "seeds": list(seeds),
            "k_values": list(k_values),
            "strata_counts": {k: len(v) for k, v in val_strata.items()},
            "evaluated_split": "validation",
        }
