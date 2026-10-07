#!/usr/bin/env python3
"""
Exploratory AMLworld Cohort Benchmark Evaluation Engine.

Executes group-safe cross-validation runs on IBM AMLworld candidate subgraphs
using explicitly supplied cohort, model, seed, and training configuration.

Produces:
1. Lossless columnar Parquet predictions in runs/<run_id>/predictions.parquet.
2. Cryptographic SHA-256 run manifest in runs/<run_id>/run_manifest.json.

Upholds Invariants:
- INV-001 (No Mock Data in Production): Real AMLworld schema candidate graphs.
- INV-002 (Simplex vs Polygonal Cell Separation): k >= 4 cycle complexes.
- INV-003 (Unfilled 1-Skeleton for Persistent Homology).
- INV-004 (Exact Boundary Nilpotency).
- INV-006 (Cryptographic Lineage Tracking): SHA-256 manifests.
- INV-007 (Leakage-Free Group-Safe Splitting).
- INV-008 (Self-Contained Verification Scripts).
- INV-009 (Model Parameter Capacity Parity & Pre-Registration Protocol).
- INV-012 (Real File Schema Grounding).
- Declared-seed provenance; software output does not itself certify preregistration.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.data.build_candidates import build_candidate_pool
    from source.src.data.candidate_extractor import CandidateSubgraph
    from source.src.data.splits import GroupSafeSplitter, DataLeakageError
    from source.src.topology.lift_dataset import lift_candidate
    from source.src.ph.graph_filtration import build_temporal_filtration
    from source.src.ph.gudhi_backend import GudhiPersistenceEngine
    from source.src.ph.vectorize import PersistenceLandscapeVectorizer
    from source.src.models.feature_encoders import CandidateFeatureExtractor
    from source.src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from source.src.models.tdl_simplicial import SimplicialComplexNetwork
    from source.src.models.tdl_cellular import CellularComplexNetwork
    from source.src.models.topotune_adapter import TopoTuneInspiredModel
    from source.src.models.train_harness import compute_classification_metrics
    from source.src.analysis.statistical_engine import (
        cluster_bootstrap_paired_difference,
        holm_bonferroni_correction,
    )
except ModuleNotFoundError:
    from src.data.build_candidates import build_candidate_pool
    from src.data.candidate_extractor import CandidateSubgraph
    from src.data.splits import GroupSafeSplitter, DataLeakageError
    from src.topology.lift_dataset import lift_candidate
    from src.ph.graph_filtration import build_temporal_filtration
    from src.ph.gudhi_backend import GudhiPersistenceEngine
    from src.ph.vectorize import PersistenceLandscapeVectorizer
    from src.models.feature_encoders import CandidateFeatureExtractor
    from src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from src.models.tdl_simplicial import SimplicialComplexNetwork
    from src.models.tdl_cellular import CellularComplexNetwork
    from src.models.topotune_adapter import TopoTuneInspiredModel
    from src.models.train_harness import compute_classification_metrics
    from src.analysis.statistical_engine import (
        cluster_bootstrap_paired_difference,
        holm_bonferroni_correction,
    )


TRAINING_PROTOCOL = {
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "no_decay_parameters": "bias, LayerNorm, and attention vectors",
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "learning_rate_schedule": {
        "type": "linear_warmup_then_cosine",
        "warmup_fraction": 0.15,
        "warmup_start_factor": 0.1,
        "cosine_minimum_learning_rate": 1e-6,
    },
    "label_smoothing_epsilon": 0.005,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 8,
    "early_stopping_minimum_epoch": 6,
    "checkpoint_selection": "minimum_validation_binary_cross_entropy",
}


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_engine_provenance() -> Dict[str, Any]:
    """Hashes executable source/config files and records active dependency versions."""
    source_root = Path(__file__).resolve().parents[1]
    tracked = sorted(
        p for p in source_root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and (p.suffix == ".py" or p.name in {"pyproject.toml", "requirements_lock.txt"})
    )
    digest = hashlib.sha256()
    for path in tracked:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "torch", "networkx", "scikit-learn", "scipy", "gudhi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"engine_source_sha256": digest.hexdigest(), "runtime_versions": versions}




class BenchmarkRunner:
    """Executes the requested exploratory AMLworld cohort evaluation."""

    def __init__(
        self,
        candidates: List[CandidateSubgraph],
        runs_dir: str = "runs",
        device: Optional[torch.device] = None,
        baseline_hidden_dim: int = 288,
        baseline_heads: int = 16,
    ):
        self.candidates = candidates
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cpu")
        self.baseline_hidden_dim = baseline_hidden_dim
        self.baseline_heads = baseline_heads
        self.encoder = CandidateFeatureExtractor()
        self.ph_engine = GudhiPersistenceEngine()
        self.ph_vectorizer = PersistenceLandscapeVectorizer()
        self.prepared_data = self._prepare_data()

    def _prepare_data(self) -> List[Dict[str, Any]]:
        """Precomputes topological liftings and normalized features."""
        prepared = []
        for c in self.candidates:
            if c.label not in (0, 1):
                raise ValueError(f"Candidate {c.candidate_id} has no explicit binary label")
            lifted = lift_candidate(c)
            x_node, edge_index, edge_attr, _ = self.encoder.extract_features(c, mode="structure_aware")
                
            ph_graph = build_temporal_filtration(c)
            diag = self.ph_engine.compute_persistence(ph_graph)
            ph_vec = self.ph_vectorizer.vectorize(diag)
            ph_tensor = torch.tensor(ph_vec, dtype=torch.float32).unsqueeze(0)
            
            b1 = torch.tensor(lifted.cell.B1.toarray(), dtype=torch.float32)
            b2 = torch.tensor(lifted.cell.B2.toarray(), dtype=torch.float32)
            l0 = torch.tensor(lifted.cell.L0.toarray(), dtype=torch.float32)
            l1 = torch.tensor(lifted.cell.L1.toarray(), dtype=torch.float32)
            l2 = torch.tensor(lifted.cell.L2.toarray(), dtype=torch.float32)
            
            raw_x2 = lifted.cell.x_2
            if raw_x2.shape[0] > 0:
                if raw_x2.shape[1] < 4:
                    x2 = torch.cat([
                        torch.tensor(raw_x2, dtype=torch.float32),
                        torch.zeros((raw_x2.shape[0], 4 - raw_x2.shape[1]), dtype=torch.float32)
                    ], dim=-1)
                else:
                    x2 = torch.tensor(raw_x2[:, :4], dtype=torch.float32)
            else:
                x2 = torch.empty((0, 4), dtype=torch.float32)
                
            simp_b1 = torch.tensor(lifted.simplicial.B1.toarray(), dtype=torch.float32)
            simp_b2 = torch.tensor(lifted.simplicial.B2.toarray(), dtype=torch.float32)
            simp_x1 = lifted.simplicial.x_1
            simp_face = lifted.simplicial.x_2
            simp_x2 = F.pad(simp_face, (0, 4 - simp_face.shape[1]))
            prepared.append({
                "cand": c,
                "x_node": x_node,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "cell_x1": torch.tensor(lifted.cell.x_1, dtype=torch.float32),
                "ph_tensor": ph_tensor,
                "b1": b1,
                "b2": b2,
                "l0": l0,
                "l1": l1,
                "l2": l2,
                "x2": x2,
                "simp_b1": simp_b1, "simp_b2": simp_b2,
                "simp_x1": simp_x1, "simp_x2": simp_x2,
                "label": int(c.label),
                "pattern_id": c.pattern_id or "NONE"
            })
        return prepared

    def instantiate_model(self, model_family: str, ph_dim: int = 310) -> nn.Module:
        """Instantiates capacity-matched or stress-test baseline model."""
        fam = model_family.lower()
        if fam == "gcn":
            return GCNBaseline(in_dim=13, hidden_dim=self.baseline_hidden_dim, ph_dim=ph_dim).to(self.device)
        elif fam == "gat":
            return GATBaseline(in_dim=13, hidden_dim=self.baseline_hidden_dim, ph_dim=ph_dim, heads=self.baseline_heads).to(self.device)
        elif fam == "sage":
            return GraphSAGEBaseline(in_dim=13, hidden_dim=self.baseline_hidden_dim, ph_dim=ph_dim).to(self.device)
        elif fam == "mpsn":
            return SimplicialComplexNetwork(in_dim0=13, in_dim1=3, in_dim2=4, hidden_dim=27, ph_dim=ph_dim).to(self.device)
        elif fam == "cwn":
            return CellularComplexNetwork(in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=ph_dim).to(self.device)
        elif fam == "topotune":
            return TopoTuneInspiredModel(in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=ph_dim).to(self.device)
        else:
            raise ValueError(f"Unknown family: {model_family}")

    def evaluate_model_seed(
        self,
        model_family: str,
        seed: int,
        fold: int = 0,
        epochs: int = 3,
        lr: float = 0.001,
        data_provenance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs training and test evaluation for one model and seed, logging predictions to Parquet."""
        if epochs <= 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("epochs and learning rate must be positive and finite")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        ph_dim = self.prepared_data[0]["ph_tensor"].size(-1) if self.prepared_data else 310
        model = self.instantiate_model(model_family, ph_dim=ph_dim)
        
        # Fixed AdamW parameter grouping; the exact settings are recorded in each run manifest.
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "att_src" in name or "att_dst" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.BCEWithLogitsLoss()
        
        # ── INV-007: Group-safe splitting via DSU-based splitter ──
        splitter = GroupSafeSplitter(n_splits=5, seed=seed)
        train_cands, val_cands, test_cands, assignments = splitter.three_way_split(
            [item["cand"] for item in self.prepared_data], fold % 5
        )
        train_ids = {c.candidate_id for c in train_cands}
        val_ids = {c.candidate_id for c in val_cands}
        test_ids = {c.candidate_id for c in test_cands}
        actual_train_items = [item for item in self.prepared_data if item["cand"].candidate_id in train_ids]
        val_items = [item for item in self.prepared_data if item["cand"].candidate_id in val_ids]
        test_items = [item for item in self.prepared_data if item["cand"].candidate_id in test_ids]
            
        # Training loop with Early Stopping, Warmup-Cosine Annealing, Label Smoothing, and Gradient Clipping
        model.train()
        random.Random(seed).shuffle(actual_train_items)

        warmup_epochs = max(1, int(epochs * 0.15))
        sched1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup_epochs])

        best_val_loss = float("inf")
        best_model_weights = None
        best_epoch = None
        patience = 8
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for item in actual_train_items:
                optimizer.zero_grad()
                x_node = item["x_node"].to(self.device)
                ph = item["ph_tensor"].to(self.device)
                target = torch.tensor([item["label"]], dtype=torch.float32, device=self.device)
                
                fam = model_family.lower()
                if fam in ["gcn", "gat", "sage"]:
                    ei = item["edge_index"].to(self.device)
                    b = torch.zeros(x_node.size(0), dtype=torch.long, device=self.device)
                    logits = model(x_node, ei, batch=b, ph_features=ph, edge_attr=item["edge_attr"].to(self.device))
                elif fam in ["cwn", "topotune"]:
                    logits = model(x_node, item["cell_x1"].to(self.device), item["x2"].to(self.device),
                                   item["b1"].to(self.device), item["b2"].to(self.device),
                                   item["l0"].to(self.device), item["l1"].to(self.device),
                                   item["l2"].to(self.device), ph_features=ph)
                elif fam == "mpsn":
                    logits = model(x_node, item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                   item["simp_b1"].to(self.device), item["simp_b2"].to(self.device),
                                   ph_features=ph)
                                   
                # Symmetric binary label smoothing with epsilon=0.005.
                smooth_target = target * 0.99 + 0.005
                loss = criterion(logits.view(-1), smooth_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Early stopping validation
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for item in val_items:
                    x_node = item["x_node"].to(self.device)
                    ph = item["ph_tensor"].to(self.device)
                    target = torch.tensor([item["label"]], dtype=torch.float32, device=self.device)
                    fam = model_family.lower()
                    if fam in ["gcn", "gat", "sage"]:
                        ei = item["edge_index"].to(self.device)
                        b = torch.zeros(x_node.size(0), dtype=torch.long, device=self.device)
                        lgt = model(x_node, ei, batch=b, ph_features=ph, edge_attr=item["edge_attr"].to(self.device))
                    elif fam in ["cwn", "topotune"]:
                        lgt = model(x_node, item["cell_x1"].to(self.device), item["x2"].to(self.device),
                                    item["b1"].to(self.device), item["b2"].to(self.device),
                                    item["l0"].to(self.device), item["l1"].to(self.device),
                                    item["l2"].to(self.device), ph_features=ph)
                    elif fam == "mpsn":
                        lgt = model(x_node, item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                    item["simp_b1"].to(self.device), item["simp_b2"].to(self.device),
                                    ph_features=ph)
                    total_val_loss += criterion(lgt.view(-1), target).item()

            avg_val_loss = total_val_loss / max(1, len(val_items))
            if not np.isfinite(avg_val_loss):
                raise ValueError("Validation loss is non-finite; refusing to select a checkpoint")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= 5:
                    break

        if best_model_weights is None or best_epoch is None:
            raise ValueError("No valid validation checkpoint was produced")
        model.load_state_dict(best_model_weights)

        val_y = []
        val_probs = []
        model.eval()
        with torch.no_grad():
            for item in val_items:
                val_y.append(item["label"])
                fam = model_family.lower()
                if fam in ["gcn", "gat", "sage"]:
                    x = item["x_node"].to(self.device)
                    b = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                    out = model(x, item["edge_index"].to(self.device), batch=b, ph_features=item["ph_tensor"].to(self.device), edge_attr=item["edge_attr"].to(self.device))
                elif fam in ["cwn", "topotune"]:
                    out = model(item["x_node"].to(self.device), item["cell_x1"].to(self.device), item["x2"].to(self.device),
                                item["b1"].to(self.device), item["b2"].to(self.device), item["l0"].to(self.device),
                                item["l1"].to(self.device), item["l2"].to(self.device), ph_features=item["ph_tensor"].to(self.device))
                else:
                    out = model(item["x_node"].to(self.device), item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                item["simp_b1"].to(self.device), item["simp_b2"].to(self.device), ph_features=item["ph_tensor"].to(self.device))
                val_probs.append(float(torch.sigmoid(out.view(-1))[0].item()))
        try:
            from source.src.models.train_harness import compute_optimal_threshold
        except ModuleNotFoundError:
            from src.models.train_harness import compute_optimal_threshold
        threshold = compute_optimal_threshold(np.array(val_y), np.array(val_probs))
                
        # Test evaluation & prediction serialization
        model.eval()
        pred_records = []
        y_true_list = []
        y_prob_list = []
        
        with torch.no_grad():
            for item in test_items:
                x_node = item["x_node"].to(self.device)
                ph = item["ph_tensor"].to(self.device)
                target = item["label"]
                
                fam = model_family.lower()
                if fam in ["gcn", "gat", "sage"]:
                    ei = item["edge_index"].to(self.device)
                    b = torch.zeros(x_node.size(0), dtype=torch.long, device=self.device)
                    logits = model(x_node, ei, batch=b, ph_features=ph, edge_attr=item["edge_attr"].to(self.device))
                elif fam in ["cwn", "topotune"]:
                    logits = model(x_node, item["cell_x1"].to(self.device), item["x2"].to(self.device),
                                   item["b1"].to(self.device), item["b2"].to(self.device),
                                   item["l0"].to(self.device), item["l1"].to(self.device),
                                   item["l2"].to(self.device), ph_features=ph)
                elif fam == "mpsn":
                    logits = model(x_node, item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                   item["simp_b1"].to(self.device), item["simp_b2"].to(self.device),
                                   ph_features=ph)
                                   
                prob = float(torch.sigmoid(logits.view(-1))[0].item())
                y_prob_list.append(prob)
                y_true_list.append(target)
                
                pred_records.append({
                    "candidate_id": item["cand"].candidate_id,
                    "model_family": model_family,
                    "seed": seed,
                    "fold": fold,
                    "true_label": target,
                    "predicted_prob": prob,
                    "predicted_label": 1 if prob >= threshold else 0,
                    "pattern_id": item["pattern_id"]
                })
                
        metrics = compute_classification_metrics(np.array(y_true_list), np.array(y_prob_list), val_threshold=threshold)
        
        # Save run directory and artifacts
        run_identity = {
            "model_family": model_family,
            "seed": seed,
            "fold": fold,
            "epochs": epochs,
            "learning_rate": lr,
            "training_protocol": TRAINING_PROTOCOL,
            "model_configuration": {
                "baseline_hidden_dim": self.baseline_hidden_dim,
                "baseline_heads": self.baseline_heads,
            },
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "data_provenance": data_provenance,
        }
        run_digest = hashlib.sha256(
            json.dumps(run_identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:12]
        run_id = f"run_amlworld_{model_family}_seed{seed}_fold{fold}_{run_digest}"
        run_path = self.runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(pred_records)
        parquet_path = run_path / "predictions.parquet"
        df.to_parquet(parquet_path, index=False)
        parquet_sha = compute_file_sha256(parquet_path)
        
        manifest_data = {
            "run_id": run_id,
            "model_family": model_family,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "seed": seed,
            "fold": fold,
            "epochs": epochs,
            "actual_epochs_run": epoch + 1,
            "selected_checkpoint_epoch": best_epoch,
            "learning_rate": lr,
            "training_protocol": TRAINING_PROTOCOL,
            "model_configuration": {
                "baseline_hidden_dim": self.baseline_hidden_dim,
                "baseline_heads": self.baseline_heads,
            },
            "data_provenance": data_provenance,
            "ph_backend": "gudhi" if self.ph_engine.has_gudhi else "algebraic_reduction",
            "decision_threshold_source": "validation_fold",
            "num_test_samples": len(pred_records),
            "metrics": metrics,
            "predictions_parquet": "predictions.parquet",
            "predictions_parquet_sha256": parquet_sha,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        manifest_file = run_path / "run_manifest.json"
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        
        return manifest_data


def run_amlworld_primary_benchmark(
    seeds: Optional[List[int]] = None,
    families: Optional[List[str]] = None,
    runs_dir: str = "runs",
    epochs: Optional[int] = None,
    baseline_hidden_dim: Optional[int] = None,
    baseline_heads: Optional[int] = None,
    max_transactions: Optional[int] = None,
    max_cycles: Optional[int] = None,
    min_cycle_length: Optional[int] = None,
    max_cycle_length: Optional[int] = None,
    candidate_hop_radius: Optional[int] = None,
    trans_csv_path: Optional[str] = None,
    patterns_txt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs a specified AML CSV prefix cohort; scope must be reported with results."""
    if any(value is None for value in (
        max_transactions, max_cycles, min_cycle_length, max_cycle_length,
        candidate_hop_radius, trans_csv_path, patterns_txt_path, epochs,
        baseline_hidden_dim, baseline_heads,
    )):
        raise ValueError(
            "Declare raw input paths, cohort limits, cycle bounds, hop radius, epochs, and baseline widths "
            "before running the AML cohort"
        )
    if epochs <= 0 or baseline_hidden_dim <= 0 or baseline_heads <= 0:
        raise ValueError("Epoch count and baseline dimensions must be positive")
    if not seeds or not families:
        raise ValueError("Pass an explicit non-empty seed list and model-family list")
    if len(set(seeds)) != len(seeds) or len(set(families)) != len(families):
        raise ValueError("Seed and model-family lists must not contain duplicates")
    target_seeds = list(seeds)
    target_families = list(families)
    
    candidates, cleanup_fn, provenance = build_candidate_pool(
        trans_csv_path=trans_csv_path,
        patterns_txt_path=patterns_txt_path,
        max_tx=max_transactions,
        max_cycles=max_cycles,
        min_cycle_length=min_cycle_length,
        max_cycle_length=max_cycle_length,
        hop_radius=candidate_hop_radius,
    )
    provenance.update(compute_engine_provenance())

    # ── INV-001: Assert data provenance is derived, not assumed ──
    assert provenance.get("data_source") == "ibm_amlworld_synthetic_csv", (
        f"[INV-001 Violation] Data source is '{provenance.get('data_source')}', "
        f"expected 'ibm_amlworld_synthetic_csv'. Aborting benchmark."
    )
    assert provenance.get("trans_csv_sha256"), (
        "[INV-001 Violation] Missing SHA-256 hash for transaction CSV."
    )

    try:
        runner = BenchmarkRunner(
            candidates,
            runs_dir=runs_dir,
            baseline_hidden_dim=baseline_hidden_dim,
            baseline_heads=baseline_heads,
        )
        n_folds = 5
        runs_output = []
        # Collect per-model pooled predictions for statistical comparison
        model_pooled_scores: Dict[str, List[float]] = {fam: [] for fam in target_families}
        model_pooled_labels: Dict[str, List[int]] = {fam: [] for fam in target_families}
        model_cluster_ids: Dict[str, List[str]] = {fam: [] for fam in target_families}

        for fam in target_families:
            for s in target_seeds:
                for fold_idx in range(n_folds):
                    res = runner.evaluate_model_seed(
                        model_family=fam, seed=s, fold=fold_idx, epochs=epochs,
                        data_provenance=provenance,
                    )
                    runs_output.append(res)
                    print(f"Completed run: {res['run_id']} | AUPRC: {res['metrics']['auprc']:.4f}")

        # Fold scores reuse the same candidate groups across seeds and overlapping
        # training sets. They are not independent bootstrap clusters. Publish paired
        # descriptive differences only until a group-level inference protocol is fixed.
        statistical_results = {"status": "not_estimated",
                               "reason": "Repeated cross-validation fold scores are dependent; no valid independent bootstrap units supplied.",
                               "paired_fold_ap_differences": {}}
        if len(target_families) >= 2:
            baseline = target_families[0]
            baseline_runs = {(r["seed"], r["fold"]): r for r in runs_output if r["model_family"] == baseline}
            for fam in target_families[1:]:
                differences = []
                for run in runs_output:
                    if run["model_family"] == fam:
                        other = baseline_runs[(run["seed"], run["fold"])]
                        differences.append(run["metrics"]["auprc"] - other["metrics"]["auprc"])
                statistical_results["paired_fold_ap_differences"][f"{fam}_vs_{baseline}"] = differences

        summary = {
            "benchmark": "AMLworld_Primary_Track",
            "claim_scope": "exploratory_until_cohort_and_controls_are_validated",
            "cohort_configuration": {
                "max_transactions": max_transactions,
                "max_cycles": max_cycles,
                "cycle_length_bounds": [min_cycle_length, max_cycle_length],
                "candidate_hop_radius": candidate_hop_radius,
                "baseline_hidden_dim": baseline_hidden_dim,
                "baseline_heads": baseline_heads,
                "epochs": epochs,
            },
            "declared_seeds": target_seeds,
            "executed_seeds": target_seeds,
            "models": target_families,
            "n_folds": n_folds,
            "total_runs": len(runs_output),
            "runs": runs_output,
            "statistical_inference": statistical_results,
            "data_provenance": provenance,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return summary
    finally:
        if cleanup_fn:
            cleanup_fn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMLworld cohort benchmark runner")
    parser.add_argument("--runs-dir", default="runs", help="Directory to store runs")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Declared evaluation seeds")
    parser.add_argument("--families", nargs="+", type=str, required=True, help="Target model families, e.g. gcn gat")
    parser.add_argument("--epochs", type=int, required=True, help="Training epochs")
    parser.add_argument("--baseline-hidden-dim", type=int, required=True,
                        help="Hidden dim for the intentionally widened graph-baseline stress test")
    parser.add_argument("--baseline-heads", type=int, required=True, help="Attention heads for GAT baseline")
    parser.add_argument("--max-transactions", type=int, required=True, help="Predeclared CSV prefix size")
    parser.add_argument("--max-cycles", type=int, required=True, help="Predeclared label-blind cycle enumeration cap")
    parser.add_argument("--min-cycle-length", type=int, required=True)
    parser.add_argument("--max-cycle-length", type=int, required=True)
    parser.add_argument("--candidate-hop-radius", type=int, required=True)
    parser.add_argument("--transactions", required=True, help="Explicit AML transaction CSV path")
    parser.add_argument("--patterns", required=True, help="Explicit paired AML patterns path")
    parser.add_argument("--summary-output", required=True, help="Path for the full study summary JSON")
    args = parser.parse_args()
    res = run_amlworld_primary_benchmark(
        seeds=args.seeds,
        families=args.families,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        baseline_hidden_dim=args.baseline_hidden_dim,
        baseline_heads=args.baseline_heads,
        max_transactions=args.max_transactions,
        max_cycles=args.max_cycles,
        min_cycle_length=args.min_cycle_length,
        max_cycle_length=args.max_cycle_length,
        candidate_hop_radius=args.candidate_hop_radius,
        trans_csv_path=args.transactions,
        patterns_txt_path=args.patterns,
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Benchmark completed successfully: {res['total_runs']} runs executed.")
