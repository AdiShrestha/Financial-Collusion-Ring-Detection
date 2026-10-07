"""
Baseline Training & Validation Harness for Graph Ring Detection.

Provides:
- Imbalanced binary classification loss (BCEWithLogitsLoss with positive weighting).
- Training loop with gradient norm clipping and weight decay.
- Evaluation metrics: AUPRC, AUROC, optimal-threshold F1-score, Precision, Recall.
- Validation AUPRC early stopping with state checkpointing.

Upholds Invariants:
- INV-001 (No Mock Data in Production): Validated on candidate data structures.
- INV-008 (Self-Contained Verification Scripts).
"""

import copy
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))


def compute_optimal_threshold(
    y_true: Union[np.ndarray, torch.Tensor],
    y_probs: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Sweeps thresholds on VALIDATION data to find the optimal F1 threshold.
    This threshold should then be frozen and passed to compute_classification_metrics()
    at test time via the val_threshold parameter.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    raw_labels = np.asarray(y_true).flatten()
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("Validation labels must be binary")
    y_true = raw_labels.astype(int)
    y_probs = np.asarray(y_probs).flatten().astype(float)
    if not np.isfinite(y_probs).all() or np.any((y_probs < 0) | (y_probs > 1)):
        raise ValueError("Validation probabilities must be finite and in [0, 1]")
    if len(y_true) != len(y_probs) or len(y_true) == 0:
        raise ValueError("Validation labels and probabilities must be nonempty and aligned")
    if not np.isin(y_true, [0, 1]).all() or len(np.unique(y_true)) < 2:
        raise ValueError("Threshold calibration requires both binary classes in validation")

    best_f1 = 0.0
    best_thresh = 0.5

    for t in np.unique(np.r_[0.0, y_probs, 1.0]):
        preds = (y_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    return best_thresh


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_probs: Union[np.ndarray, torch.Tensor],
    val_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes AUPRC, AUROC, and F1-score for binary classification.
    Safe against single-class or degenerate batches.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities.
        val_threshold: If provided, F1/precision/recall are computed at this
            frozen threshold (calibrated on validation data) instead of sweeping.
            This eliminates optimistic test-set threshold-tuning bias.
            Average precision and AUROC are threshold-free ranking metrics; they
            remain sensitive to cohort construction, class prevalence, and model selection.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    raw_labels = np.asarray(y_true).flatten()
    if not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary")
    y_true = raw_labels.astype(int)
    y_probs = np.asarray(y_probs).flatten().astype(float)

    if len(y_true) != len(y_probs) or len(y_true) == 0:
        raise ValueError("Evaluation labels and probabilities must be nonempty and aligned")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Evaluation labels must be binary")
    if not np.isfinite(y_probs).all() or np.any((y_probs < 0) | (y_probs > 1)):
        raise ValueError("Evaluation probabilities must be finite and in [0, 1]")

    threshold = 0.5 if val_threshold is None else float(val_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must lie in [0, 1]")

    # Accuracy (used by downstream scripts)
    y_pred_binary = (y_probs >= threshold).astype(int)
    accuracy = float(np.mean(y_true == y_pred_binary))

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    auprc_val = float(average_precision_score(y_true, y_probs)) if n_pos and n_neg else None
    auroc_val = float(roc_auc_score(y_true, y_probs)) if n_pos and n_neg else None
    preds = y_pred_binary
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    best_prec = tp / (tp + fp) if tp + fp else 0.0
    best_rec = tp / (tp + fn) if tp + fn else 0.0
    best_f1 = 2 * best_prec * best_rec / (best_prec + best_rec) if best_prec + best_rec else 0.0

    return {
        "auprc": auprc_val,
        "auroc": auroc_val,
        "f1": float(best_f1),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "best_threshold": threshold,
        "threshold_source": "val_frozen" if val_threshold is not None else "fixed_0.5",
        "accuracy": accuracy,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "ranking_metrics_valid": bool(n_pos and n_neg),
    }


class Trainer:
    """GNN trainer with validation AUPRC early stopping."""

    @staticmethod
    def compute_pos_weight(labels: Union[np.ndarray, torch.Tensor, list]) -> float:
        """Computes pos_weight = n_neg / n_pos from training fold labels.
        
        Requires both classes in the training fold.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels).flatten()
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Cannot train a binary classifier without both classes")
        return float(n_neg / n_pos)

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.pos_weight = torch.tensor([pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.grad_clip = grad_clip

    def train_epoch(self, dataloader: Any) -> float:
        """Runs one epoch of training over dataloader."""
        self.model.train()
        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            y = batch["y"].to(self.device)
            b_idx = batch.get("batch")
            if b_idx is not None:
                b_idx = b_idx.to(self.device)
            ph_vec = batch.get("ph_features")
            if ph_vec is not None:
                ph_vec = ph_vec.to(self.device)
            edge_attr = batch.get("edge_attr")
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, edge_index, b_idx, ph_features=ph_vec, edge_attr=edge_attr)
            loss = self.criterion(logits, y)

            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return total_loss / max(1, batches)

    @torch.no_grad()
    def evaluate(
        self, dataloader: Any
    ) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluates model on dataloader, returning loss and performance metrics."""
        self.model.eval()
        total_loss = 0.0
        batches = 0
        all_y: List[float] = []
        all_probs: List[float] = []

        for batch in dataloader:
            x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            y = batch["y"].to(self.device)
            b_idx = batch.get("batch")
            if b_idx is not None:
                b_idx = b_idx.to(self.device)
            ph_vec = batch.get("ph_features")
            if ph_vec is not None:
                ph_vec = ph_vec.to(self.device)
            edge_attr = batch.get("edge_attr")
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)

            logits = self.model(x, edge_index, b_idx, ph_features=ph_vec, edge_attr=edge_attr)
            loss = self.criterion(logits, y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            batches += 1

            all_y.extend(y.cpu().numpy().flatten().tolist())
            all_probs.extend(probs.cpu().numpy().flatten().tolist())

        avg_loss = total_loss / max(1, batches)
        y_arr = np.array(all_y, dtype=np.float32)
        p_arr = np.array(all_probs, dtype=np.float32)
        metrics = compute_classification_metrics(y_arr, p_arr)

        return avg_loss, metrics, y_arr, p_arr

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        max_epochs: int = 50,
        patience: int = 20,
    ) -> Dict[str, Any]:
        """Executes full training with early stopping on validation AUPRC."""
        history: List[Dict[str, Any]] = []
        best_auprc = -1.0
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        stagnant_epochs = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)

            val_auprc = val_metrics["auprc"]
            if val_auprc is None:
                raise ValueError("Validation fold requires both classes for AUPRC early stopping")
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auprc": val_auprc,
                "val_auroc": val_metrics["auroc"],
                "val_f1": val_metrics["f1"],
            })

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1

            if stagnant_epochs >= patience:
                break

        # Restore best model weights
        self.model.load_state_dict(best_state)

        return {
            "best_epoch": best_epoch,
            "best_val_auprc": best_auprc,
            "history": history,
            "stopped_early": stagnant_epochs >= patience,
        }
