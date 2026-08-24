"""Unified Production Model Trainer supporting all 6 architectures with early stopping and checkpointing."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from source.models.cell_net import CellularComplexNet
from source.models.gnn_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from source.models.ph_augmented_net import TopoRingNet
from source.models.simplicial_net import SimplicialComplexNet
from source.training.tuning_protocol import compute_binary_metrics


class ModelTrainer:
    """Production training engine with early stopping, scheduling, and exact checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        lr_scheduler: Optional[Any] = None,
        patience: int = 15,
        min_epochs: int = 5,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        monitor_metric: str = "pr_auc",
        mode: str = "max",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.monitor_metric = monitor_metric
        self.mode = mode  # "max" or "min"

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_roc_auc": [],
            "val_f1_macro": [],
            "lr": [],
        }
        self.best_metric_val: float = -float("inf") if mode == "max" else float("inf")
        self.best_epoch: int = -1
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None

    def _forward_item(self, item: Dict[str, Any]) -> torch.Tensor:
        """Forward pass adaptively dispatching domain complex representation."""
        if isinstance(self.model, TopoRingNet):
            cell_v = item.get("cell_view", item.get("domain_view"))
            z_t = item.get("z_topo")
            return self.model(cell_v, z_topo=z_t)
        elif isinstance(self.model, CellularComplexNet):
            cell_v = item.get("cell_view", item.get("domain_view"))
            return self.model(cell_v)
        elif isinstance(self.model, SimplicialComplexNet):
            simp_v = item.get("simplicial_view", item.get("domain_view"))
            return self.model(simp_v)
        elif isinstance(self.model, (GCNBaseline, GATBaseline, GraphSAGEBaseline)):
            g_v = item.get("graph_view", item.get("domain_view"))
            return self.model(g_v)
        else:
            d_v = item.get("domain_view", item.get("graph_view"))
            return self.model(d_v)

    def train_epoch(self, train_loader: Sequence[Dict[str, Any]]) -> float:
        """Execute one training epoch across all candidates."""
        self.model.train()
        self.optimizer.zero_grad()

        logits_list = []
        targets_list = []

        for item in train_loader:
            logits = self._forward_item(item)
            logits_list.append(logits)
            targets_list.append(item["target_y"])

        batch_logits = torch.cat(logits_list, dim=0)
        batch_targets = torch.tensor(targets_list, dtype=torch.long, device=self.device)

        loss = self.criterion(batch_logits, batch_targets)

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError("Training loss diverged to NaN or Inf")

        loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()
        return float(loss.item())

    def evaluate(self, eval_loader: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model on candidate dataset."""
        self.model.eval()
        logits_list = []
        targets_list = []

        with torch.no_grad():
            for item in eval_loader:
                logits = self._forward_item(item)
                logits_list.append(logits)
                targets_list.append(item["target_y"])

            batch_logits = torch.cat(logits_list, dim=0)
            batch_targets = torch.tensor(targets_list, dtype=torch.long, device=self.device)
            loss = self.criterion(batch_logits, batch_targets)

            probs = F.softmax(batch_logits, dim=-1)[:, 1].cpu().numpy()
            y_true = np.array(targets_list, dtype=int)

        metrics = compute_binary_metrics(y_true, probs)
        metrics["loss"] = float(loss.item())
        return metrics

    def fit(
        self,
        train_loader: Sequence[Dict[str, Any]],
        val_loader: Sequence[Dict[str, Any]],
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train model with early stopping and save optimal checkpoint."""
        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics["loss"])
                else:
                    self.lr_scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])
            self.history["val_f1_macro"].append(val_metrics["f1_macro"])
            self.history["lr"].append(current_lr)

            # Monitor metric for early stopping
            if self.monitor_metric == "loss":
                current_score = val_metrics["loss"]
                is_better = current_score < self.best_metric_val
            else:
                current_score = val_metrics.get(self.monitor_metric, val_metrics["pr_auc"])
                is_better = current_score > self.best_metric_val

            if is_better:
                self.best_metric_val = current_score
                self.best_epoch = epoch
                epochs_no_improve = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, epoch=epoch, val_metrics=val_metrics)
            else:
                if epoch >= self.min_epochs:
                    epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        # Restore best model state if saved
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        final_val_metrics = self.evaluate(val_loader)
        return {
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_metric_val,
            "final_val_metrics": final_val_metrics,
            "total_epochs": len(self.history["train_loss"]),
            "history": self.history,
        }

    def predict_proba(self, loader: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Compute positive class probabilities."""
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for item in loader:
                logits = self._forward_item(item)
                prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(prob.tolist())

        return np.array(all_probs, dtype=float)

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        val_metrics: Dict[str, float],
    ) -> None:
        """Save atomic training checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "val_metrics": val_metrics,
            "best_metric_val": self.best_metric_val,
            "monitor_metric": self.monitor_metric,
            "history": self.history,
        }
        # Atomic write
        temp_path = f"{checkpoint_path}.tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Restore training checkpoint with exact model weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler and checkpoint.get("lr_scheduler_state_dict"):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.best_metric_val = checkpoint.get("best_metric_val", self.best_metric_val)
        self.best_epoch = checkpoint.get("epoch", self.best_epoch)
        return checkpoint
