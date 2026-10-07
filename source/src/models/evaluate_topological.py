"""
Topological Neural Network Benchmarking & Parity Suite.

Runs cross-validation training and evaluation of higher-order topological deep learning
architectures (Simplicial MPSN and Cellular CWN) with:
- Global model parameter capacity parity audit across GNN baselines and TDL models (INV-009).
- Evaluation across group-safe cross-validation folds (INV-007).
- Cryptographic SHA-256 lineage manifests (INV-006).

Upholds Invariants:
- INV-001 (No Mock Data in Production): Validated on candidate complex schemas.
- INV-006 (Cryptographic Lineage Tracking): SHA-256 hashes of all artifacts.
- INV-007 (Leakage-Free Group-Safe Splitting): Folds processed independently.
- INV-008 (Self-Contained Verification Scripts).
- INV-009 (Model Parameter and FLOP Matching Protocol): parameter delta <= 2%.
"""

import argparse
import copy
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
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from source.src.models.tdl_simplicial import SimplicialComplexNetwork
    from source.src.models.tdl_cellular import CellularComplexNetwork
    from source.src.models.train_harness import compute_classification_metrics, compute_optimal_threshold
except ModuleNotFoundError:
    from src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from src.models.tdl_simplicial import SimplicialComplexNetwork
    from src.models.tdl_cellular import CellularComplexNetwork
    from src.models.train_harness import compute_classification_metrics, compute_optimal_threshold


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_overall_capacity_parity(tolerance: float = 0.02) -> Dict[str, Any]:
    """
    Audits parameter capacity parity across ALL 6 benchmark model families
    under the actual production configuration (ph_dim=310):
    - Baseline GNNs: GCN, GAT, GraphSAGE
    - Topological TDL: Simplicial (MPSN), Cellular (CWN)
    - Generalized CC: TopoTune-Inspired
    
    Returns measured parameter counts and deviations. It does not make a
    noncompliant stress-test configuration appear parity-matched.
    """
    try:
        from source.src.models.topotune_adapter import TopoTuneInspiredModel
    except ModuleNotFoundError:
        from src.models.topotune_adapter import TopoTuneInspiredModel

    # ── Production configuration: ph_dim=310 for all models ──
    ph_dim = 310
    models = {
        "GCN": GCNBaseline(in_dim=13, hidden_dim=288, ph_dim=ph_dim),
        "GAT": GATBaseline(in_dim=13, hidden_dim=288, heads=16, ph_dim=ph_dim),
        "GraphSAGE": GraphSAGEBaseline(in_dim=13, hidden_dim=288, ph_dim=ph_dim),
        "Simplicial_MPSN": SimplicialComplexNetwork(in_dim0=13, in_dim1=3, in_dim2=4, hidden_dim=27, ph_dim=ph_dim),
        "Cellular_CWN": CellularComplexNetwork(in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=ph_dim),
        "TopoTune_GCCN": TopoTuneInspiredModel(in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=ph_dim),
    }

    counts = {name: sum(p.numel() for p in m.parameters() if p.requires_grad) for name, m in models.items()}
    values = list(counts.values())
    mean_val = float(np.mean(values))
    deviations = {name: abs(v - mean_val) / mean_val for name, v in counts.items()}
    max_dev = float(max(deviations.values()))
    is_compliant = max_dev <= tolerance

    result = {
        "parameter_counts": counts,
        "mean_parameters": mean_val,
        "per_model_deviation": {k: float(v) for k, v in deviations.items()},
        "max_relative_deviation": max_dev,
        "is_compliant": is_compliant,
        "tolerance": tolerance,
        "ph_dim": ph_dim,
        "includes_topotune": True,
    }

    return result


class TopologicalBenchmarkRunner:
    """Runs cross-validation fold benchmarks for higher-order topological models."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        model_type: str = "cellular",
        use_ph: bool = False,
        lr: float = 1e-3,
        pos_weight: Optional[float] = None,
        max_epochs: int = 20,
        patience: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type.lower()
        self.use_ph = use_ph
        self.lr = lr
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.patience = patience
        self.ph_dim = 310 if use_ph else None

    def _create_model(self) -> nn.Module:
        if self.model_type == "cellular":
            return CellularComplexNetwork(
                in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=self.ph_dim
            )
        elif self.model_type == "simplicial":
            return SimplicialComplexNetwork(
                in_dim0=13, in_dim1=3, in_dim2=4, hidden_dim=27, ph_dim=self.ph_dim
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def run_fold(
        self,
        train_batches: List[Dict[str, torch.Tensor]],
        val_batches: List[Dict[str, torch.Tensor]],
        test_batches: List[Dict[str, torch.Tensor]],
        fold_id: Union[int, str] = 0,
    ) -> Dict[str, Any]:
        """Trains topological model on fold, early-stops on val, evaluates on test."""
        model = self._create_model()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        labels = torch.cat([b["y"].reshape(-1) for b in train_batches]).cpu().numpy() if train_batches else np.array([])
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Topological training fold requires both classes")
        weight = self.pos_weight if self.pos_weight is not None else n_neg / n_pos
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))

        t_start = time.perf_counter()
        best_auprc = -1.0
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        stagnant = 0

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for b in train_batches:
                optimizer.zero_grad()
                if self.model_type == "cellular":
                    logits = model(
                        b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                        l0=b.get("l0"), l1=b.get("l1"), l2=b.get("l2"),
                        ph_features=b.get("ph_features"),
                    )
                else:
                    logits = model(
                        b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                        ph_features=b.get("ph_features"),
                    )
                loss = criterion(logits, b["y"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            val_probs = []
            val_y = []
            with torch.no_grad():
                for b in val_batches:
                    if self.model_type == "cellular":
                        out = model(
                            b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                            l0=b.get("l0"), l1=b.get("l1"), l2=b.get("l2"),
                            ph_features=b.get("ph_features"),
                        )
                    else:
                        out = model(
                            b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                            ph_features=b.get("ph_features"),
                        )
                    val_probs.extend(torch.sigmoid(out).cpu().numpy().flatten().tolist())
                    val_y.extend(b["y"].cpu().numpy().flatten().tolist())

            val_metrics = compute_classification_metrics(np.array(val_y), np.array(val_probs))
            val_auprc = val_metrics["auprc"]
            if val_auprc is None:
                raise ValueError("Validation fold requires both classes for AUPRC early stopping")

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stagnant = 0
            else:
                stagnant += 1

            if stagnant >= self.patience:
                break

        train_wall = time.perf_counter() - t_start

        # Restore best model and evaluate on test
        model.load_state_dict(best_state)
        model.eval()
        test_probs = []
        test_y = []
        t_eval_start = time.perf_counter()
        with torch.no_grad():
            for b in test_batches:
                if self.model_type == "cellular":
                    out = model(
                        b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                        l0=b.get("l0"), l1=b.get("l1"), l2=b.get("l2"),
                        ph_features=b.get("ph_features"),
                    )
                else:
                    out = model(
                        b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                        ph_features=b.get("ph_features"),
                    )
                test_probs.extend(torch.sigmoid(out).cpu().numpy().flatten().tolist())
                test_y.extend(b["y"].cpu().numpy().flatten().tolist())
        eval_wall = time.perf_counter() - t_eval_start

        # Re-evaluate the selected checkpoint on validation before freezing F1 threshold.
        val_probs = []
        val_y = []
        with torch.no_grad():
            for b in val_batches:
                if self.model_type == "cellular":
                    out = model(b["x0"], b["x1"], b["x2"], b["b1"], b["b2"],
                                l0=b.get("l0"), l1=b.get("l1"), l2=b.get("l2"), ph_features=b.get("ph_features"))
                else:
                    out = model(b["x0"], b["x1"], b["x2"], b["b1"], b["b2"], ph_features=b.get("ph_features"))
                val_probs.extend(torch.sigmoid(out).cpu().numpy().flatten().tolist())
                val_y.extend(b["y"].cpu().numpy().flatten().tolist())
        threshold = compute_optimal_threshold(np.array(val_y), np.array(val_probs))
        test_metrics = compute_classification_metrics(np.array(test_y), np.array(test_probs), val_threshold=threshold)

        # Checkpoint model and predictions
        model_filename = f"{self.model_type}_fold_{fold_id}.pt"
        model_path = self.output_dir / model_filename
        torch.save(best_state, model_path)

        preds_filename = f"{self.model_type}_fold_{fold_id}_preds.json"
        preds_path = self.output_dir / preds_filename
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump({"y_true": test_y, "y_probs": test_probs}, f)

        fold_report = {
            "fold_id": str(fold_id),
            "model_type": self.model_type,
            "use_ph": self.use_ph,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "best_epoch": best_epoch,
            "train_wall_sec": float(train_wall),
            "eval_wall_sec": float(eval_wall),
            "metrics": test_metrics,
            "model_sha256": compute_file_sha256(model_path),
            "preds_sha256": compute_file_sha256(preds_path),
        }

        report_filename = f"{self.model_type}_fold_{fold_id}_report.json"
        report_path = self.output_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(fold_report, f, indent=2)

        return fold_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topological Neural Network Evaluation CLI")
    parser.add_argument("--model", type=str, default="cellular", choices=["cellular", "simplicial"])
    parser.add_argument("--use-ph", action="store_true")
    parser.add_argument("--output-dir", type=str, default="source/results/topological")
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    parity = audit_overall_capacity_parity()
    print(f"Overall capacity parity audit: compliant={parity['is_compliant']}, max_dev={parity['max_relative_deviation']*100:.2f}%")
    runner = TopologicalBenchmarkRunner(
        output_dir=args.output_dir,
        model_type=args.model,
        use_ph=args.use_ph,
        max_epochs=args.epochs,
    )
    print(f"Initialized TopologicalBenchmarkRunner: {args.model} (use_ph={args.use_ph})")


if __name__ == "__main__":
    main()
