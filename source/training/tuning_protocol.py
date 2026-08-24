"""Exploratory Hyperparameter Tuning Engine operating exclusively on validation partitions."""

import itertools
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from source.models.cell_net import CellularComplexNet
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.graph_view import GraphView


def compute_binary_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute PR-AUC, ROC-AUC, and F1-Macro for binary classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    # Fallbacks if only 1 class present in batch
    if len(np.unique(y_true)) < 2:
        return {
            "pr_auc": 1.0 if np.all(y_true == (y_probs >= threshold)) else 0.5,
            "roc_auc": 1.0 if np.all(y_true == (y_probs >= threshold)) else 0.5,
            "f1_macro": 1.0 if np.all(y_true == (y_probs >= threshold)) else 0.5,
        }

    pr_auc = float(average_precision_score(y_true, y_probs))
    roc_auc = float(roc_auc_score(y_true, y_probs))
    y_pred = (y_probs >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_macro": f1,
    }


class HyperparameterTuner:
    """Systematic and randomized hyperparameter search engine strictly isolated from test partitions (INV-006)."""

    DEFAULT_PARAM_GRID = {
        "lr": [1e-4, 1e-3, 1e-2],
        "hidden_dim": [32, 64],
        "num_layers": [2, 3],
        "dropout": [0.0, 0.2],
        "weight_decay": [0.0, 1e-4],
    }

    def __init__(
        self,
        model_class: Type[nn.Module],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        metric: str = "pr_auc",
        num_trials: int = 5,
        max_epochs_per_trial: int = 20,
        random_seed: int = 42,
    ):
        self.model_class = model_class
        self.param_grid = dict(param_grid or self.DEFAULT_PARAM_GRID)
        self.metric = metric
        self.num_trials = num_trials
        self.max_epochs_per_trial = max_epochs_per_trial
        self.random_seed = random_seed
        self.trial_history: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: float = -1.0

    def _forward_model(self, model: nn.Module, item: Dict[str, Any]) -> torch.Tensor:
        """Helper to invoke forward pass based on model architecture and available domain view."""
        if isinstance(model, TopoRingNet):
            cell_v = item.get("cell_view", item.get("domain_view"))
            z_t = item.get("z_topo")
            return model(cell_v, z_topo=z_t)
        elif isinstance(model, CellularComplexNet):
            cell_v = item.get("cell_view", item.get("domain_view"))
            return model(cell_v)
        elif isinstance(model, SimplicialComplexNet):
            simp_v = item.get("simplicial_view", item.get("domain_view"))
            return model(simp_v)
        elif isinstance(model, (GCNBaseline, GATBaseline, GraphSAGEBaseline)):
            g_v = item.get("graph_view", item.get("domain_view"))
            return model(g_v)
        else:
            d_v = item.get("domain_view", item.get("graph_view"))
            return model(d_v)

    def _evaluate(
        self,
        model: nn.Module,
        val_data: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate model on validation candidates."""
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for item in val_data:
                logits = self._forward_model(model, item)
                probs = F.softmax(logits, dim=-1)[:, 1]
                all_probs.append(float(probs[0].item()))
                all_targets.append(int(item["target_y"]))

        return compute_binary_metrics(np.array(all_targets), np.array(all_probs))

    def _train_single_trial(
        self,
        config: Dict[str, Any],
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Train a single hyperparameter configuration trial."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 1e-4)
        hidden_dim = config.get("hidden_dim", 32)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)

        # Instantiate model
        if self.model_class == TopoRingNet:
            model = TopoRingNet(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        elif self.model_class in [CellularComplexNet, SimplicialComplexNet]:
            model = self.model_class(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        elif self.model_class in [GCNBaseline, GATBaseline, GraphSAGEBaseline]:
            model = self.model_class(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            model = self.model_class()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_targets = torch.tensor([item["target_y"] for item in train_data], dtype=torch.long)

        best_val_metric = -1.0
        best_metrics_dict: Dict[str, float] = {}

        for epoch in range(self.max_epochs_per_trial):
            model.train()
            optimizer.zero_grad()

            logits_list = []
            for item in train_data:
                logits_list.append(self._forward_model(model, item))

            batch_logits = torch.cat(logits_list, dim=0)
            loss = criterion(batch_logits, train_targets)

            if torch.isnan(loss) or torch.isinf(loss):
                # Prune diverging trial
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Evaluate on validation split
            val_metrics = self._evaluate(model, val_data)
            score = val_metrics.get(self.metric, val_metrics.get("pr_auc", 0.0))

            if score > best_val_metric:
                best_val_metric = score
                best_metrics_dict = val_metrics

        return {
            "config": config,
            "best_val_score": best_val_metric,
            "metrics": best_metrics_dict,
        }

    def tune(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute hyperparameter tuning. Enforces INV-006: test_data is strictly forbidden."""
        if test_data is not None:
            raise ValueError(
                "INV-006 Violation: Test data strictly forbidden during hyperparameter tuning. "
                "Exploratory tuning is restricted exclusively to validation splits."
            )

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Sample combinations deterministically up to num_trials
        rng = np.random.RandomState(self.random_seed)
        if len(all_combinations) > self.num_trials:
            indices = rng.choice(len(all_combinations), size=self.num_trials, replace=False)
            sampled_configs = [all_combinations[i] for i in sorted(indices)]
        else:
            sampled_configs = all_combinations

        self.trial_history = []
        self.best_score = -1.0
        self.best_config = None

        for trial_idx, config in enumerate(sampled_configs):
            trial_res = self._train_single_trial(config, train_data, val_data)
            self.trial_history.append({
                "trial_index": trial_idx,
                **trial_res,
            })

            score = trial_res["best_val_score"]
            if score > self.best_score or self.best_config is None:
                self.best_score = score
                self.best_config = config

        return {
            "model_name": self.model_class.__name__,
            "best_config": self.best_config,
            "best_score": self.best_score,
            "metric": self.metric,
            "num_trials_evaluated": len(self.trial_history),
            "trial_history": self.trial_history,
        }

    def save_tuning_summary(
        self,
        output_path: str = "data/cache/tuned_hyperparameters.json",
    ) -> None:
        """Save optimal hyperparameters and trial history to persistent JSON cache."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        summary: Dict[str, Any] = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

        model_key = self.model_class.__name__
        summary[model_key] = {
            "best_config": self.best_config,
            "best_val_score": self.best_score,
            "metric": self.metric,
            "trial_history": self.trial_history,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
