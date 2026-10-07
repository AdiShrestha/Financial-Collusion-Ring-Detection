"""
Baseline Benchmarking Pipeline & Metrics Logger for Financial Collusion Detection.

Executes cross-validation training and evaluation of GNN baselines (GCN, GAT, GraphSAGE)
with support for:
- GNN-basic vs. GNN-structure-aware feature tiers.
- Modular Persistent Homology feature fusion (H3 ablation testing).
- Cryptographic run manifests and metric logging.

Upholds Invariants:
- INV-001 (No Mock Data in Production): Structured candidate graphs.
- INV-006 (Cryptographic Lineage Tracking): Output predictions and metrics hashed with SHA-256.
- INV-007 (Leakage-Free Group-Safe Splitting): Folds evaluated independently.
- INV-008 (Self-Contained Verification Scripts).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.models.feature_encoders import CandidateFeatureExtractor
    from source.src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from source.src.models.train_harness import Trainer, compute_optimal_threshold, compute_classification_metrics
except ModuleNotFoundError:
    from src.models.feature_encoders import CandidateFeatureExtractor
    from src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from src.models.train_harness import Trainer, compute_optimal_threshold, compute_classification_metrics


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class CandidateBatchCollator:
    """Collates CandidateSubgraph objects into batched PyTorch tensors."""

    def __init__(self, mode: str = "structure_aware"):
        self.extractor = CandidateFeatureExtractor()
        self.mode = mode

    def collate(
        self, candidates: List[Any], ph_vectors: Optional[Dict[str, np.ndarray]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Converts candidates into minibatches for GNN forward pass."""
        batches = []
        for cand in candidates:
            x, edge_index, edge_attr, y = self.extractor.extract_features(cand, mode=self.mode)
            batch_dict = {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "y": y.unsqueeze(0),
                "batch": torch.zeros(x.size(0), dtype=torch.long),
            }

            if ph_vectors is not None:
                cid = str(cand["candidate_id"] if isinstance(cand, dict) else cand.candidate_id)
                if cid not in ph_vectors:
                    raise KeyError(f"PH vector missing for candidate {cid}")
                ph_tensor = torch.from_numpy(ph_vectors[cid]).unsqueeze(0).to(torch.float32)
                batch_dict["ph_features"] = ph_tensor

            batches.append(batch_dict)
        return batches


class BaselineBenchmarkRunner:
    """Runs cross-validation fold training and records metrics with cryptographic manifests."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        model_name: str = "gcn",
        tier: str = "structure_aware",
        use_ph: bool = False,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        pos_weight: Optional[float] = None,
        max_epochs: int = 30,
        patience: int = 15,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name.lower()
        self.tier = tier
        self.use_ph = use_ph
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.patience = patience

        self.in_dim = 13 if tier == "structure_aware" else 8
        self.ph_dim = 310 if use_ph else None
        self.collator = CandidateBatchCollator(mode=tier)

    def _create_model(self) -> torch.nn.Module:
        if self.model_name == "gcn":
            return GCNBaseline(in_dim=self.in_dim, hidden_dim=self.hidden_dim, ph_dim=self.ph_dim)
        elif self.model_name == "gat":
            return GATBaseline(in_dim=self.in_dim, hidden_dim=self.hidden_dim, ph_dim=self.ph_dim)
        elif self.model_name == "graphsage":
            return GraphSAGEBaseline(in_dim=self.in_dim, hidden_dim=self.hidden_dim, ph_dim=self.ph_dim)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def run_fold(
        self,
        train_candidates: List[Any],
        val_candidates: List[Any],
        test_candidates: List[Any],
        fold_id: Union[int, str] = 0,
        ph_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Trains model on train fold, validates with early stopping, evaluates on test."""
        model = self._create_model()
        train_labels = [int(c.label if hasattr(c, "label") else c["label"]) for c in train_candidates]
        fold_pos_weight = self.pos_weight if self.pos_weight is not None else Trainer.compute_pos_weight(train_labels)
        trainer = Trainer(
            model=model,
            lr=self.lr,
            pos_weight=fold_pos_weight,
        )

        train_batches = self.collator.collate(train_candidates, ph_vectors)
        val_batches = self.collator.collate(val_candidates, ph_vectors)
        test_batches = self.collator.collate(test_candidates, ph_vectors)

        t_start = time.perf_counter()
        fit_res = trainer.fit(
            train_loader=train_batches,
            val_loader=val_batches,
            max_epochs=self.max_epochs,
            patience=self.patience,
        )
        train_time = time.perf_counter() - t_start

        _, _, val_y, val_probs = trainer.evaluate(val_batches)
        threshold = compute_optimal_threshold(val_y, val_probs)
        # Evaluate on test split
        t_eval_start = time.perf_counter()
        test_loss, test_metrics, y_true, y_probs = trainer.evaluate(test_batches)
        test_metrics = compute_classification_metrics(y_true, y_probs, val_threshold=threshold)
        eval_time = time.perf_counter() - t_eval_start

        # Checkpoint model and predictions
        model_filename = f"{self.model_name}_{self.tier}_fold_{fold_id}.pt"
        model_path = self.output_dir / model_filename
        torch.save(model.state_dict(), model_path)

        preds_filename = f"{self.model_name}_{self.tier}_fold_{fold_id}_preds.json"
        preds_path = self.output_dir / preds_filename
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump({
                "y_true": y_true.tolist(),
                "y_probs": y_probs.tolist(),
            }, f)

        fold_report = {
            "fold_id": str(fold_id),
            "model": self.model_name,
            "tier": self.tier,
            "use_ph": self.use_ph,
            "parameters": model.count_parameters(),
            "train_candidates": len(train_candidates),
            "val_candidates": len(val_candidates),
            "test_candidates": len(test_candidates),
            "best_epoch": fit_res["best_epoch"],
            "stopped_early": fit_res["stopped_early"],
            "train_wall_sec": float(train_time),
            "eval_wall_sec": float(eval_time),
            "test_loss": float(test_loss),
            "metrics": test_metrics,
            "model_sha256": compute_file_sha256(model_path),
            "preds_sha256": compute_file_sha256(preds_path),
        }

        report_filename = f"{self.model_name}_{self.tier}_fold_{fold_id}_report.json"
        report_path = self.output_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(fold_report, f, indent=2)

        return fold_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline GNN Evaluation CLI")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "graphsage"])
    parser.add_argument("--tier", type=str, default="structure_aware", choices=["basic", "structure_aware"])
    parser.add_argument("--use-ph", action="store_true")
    parser.add_argument("--output-dir", type=str, default="source/results/baselines")
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    runner = BaselineBenchmarkRunner(
        output_dir=args.output_dir,
        model_name=args.model,
        tier=args.tier,
        use_ph=args.use_ph,
        max_epochs=args.epochs,
    )
    print(f"Initialized BaselineBenchmarkRunner: {args.model} ({args.tier}, use_ph={args.use_ph})")


if __name__ == "__main__":
    main()
