#!/usr/bin/env python3
"""
Exploratory Elliptic++ Semi-Synthetic Injected Benchmark Evaluation Engine.

Executes exploratory evaluation on Elliptic++ Bitcoin actor graphs augmented with
controlled semi-synthetic circular collusion rings using explicitly supplied
cohort, injection, model, seed, and training configuration.

Produces:
1. Columnar Parquet predictions in runs/<run_id>/predictions.parquet.
2. Cryptographic SHA-256 run manifest in runs/<run_id>/run_manifest.json.

Upholds Invariants:
- INV-001 (No Mock Data in Production): Grounded in Elliptic++ actor transaction schema.
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
from collections import Counter
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.data.candidate_extractor import CandidateSubgraph
    from source.src.data.candidate_extractor import CandidateExtractor
    from source.src.data.elliptic_loader import EllipticActorsLoader
    from source.src.data.path_utils import resolve_elliptic_paths
    from source.src.data.ring_injector import InjectedRing, SemiSyntheticRingInjector
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
    from src.data.candidate_extractor import CandidateSubgraph
    from src.data.candidate_extractor import CandidateExtractor
    from src.data.elliptic_loader import EllipticActorsLoader
    from src.data.path_utils import resolve_elliptic_paths
    from src.data.ring_injector import InjectedRing, SemiSyntheticRingInjector
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
    """Computes SHA-256 digest of a file."""
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


def best_univariate_leq_accuracy(
    candidates: List[CandidateSubgraph],
    value_fn: Callable[[CandidateSubgraph], int],
) -> float:
    """Diagnostic shortcut accuracy for a threshold on one observed feature."""
    if not candidates or {c.label for c in candidates} != {0, 1}:
        raise ValueError("Shortcut diagnostics require both candidate classes")
    values = [value_fn(c) for c in candidates]
    return max(
        sum(int((value_fn(candidate) <= cutoff) == bool(candidate.label))
            for candidate in candidates) / len(candidates)
        for cutoff in sorted(set(values))
    )




def build_elliptic_candidate_pool(
    *,
    n_pos: int,
    seed: int,
    max_background_edges: int,
    max_candidate_cycles: int,
    injection_cycle_lengths: List[int],
    min_cycle_length: int,
    max_cycle_length: int,
    hop_radius: int,
    fee_rate: float,
    initial_amount_range: Tuple[float, float],
    base_timestamp: int,
    edgelist_path: Optional[str] = None,
) -> Tuple[List[CandidateSubgraph], Callable[[], None], Dict[str, Any]]:
    """
    Constructs an Elliptic++ candidate pool grounded in real Bitcoin address transaction networks.
    Extracts real background subgraphs from AddrAddr_edgelist.csv and injects flow-conserving rings.
    """
    if n_pos <= 0 or max_background_edges <= 0 or max_candidate_cycles <= 0:
        raise ValueError("Ring count and both cohort enumeration limits must be positive")
    if min_cycle_length < 3 or max_cycle_length < min_cycle_length:
        raise ValueError("Candidate cycle bounds must satisfy 3 <= min <= max")
    if not injection_cycle_lengths or any(
        k < min_cycle_length or k > max_cycle_length for k in injection_cycle_lengths
    ):
        raise ValueError("Every injected cycle length must lie within candidate cycle bounds")
    if not (0.0 <= fee_rate < 1.0):
        raise ValueError("fee_rate must satisfy 0 <= fee_rate < 1")
    if len(initial_amount_range) != 2 or not (
        np.isfinite(initial_amount_range).all()
        and 0 < initial_amount_range[0] <= initial_amount_range[1]
    ):
        raise ValueError("Initial amount range must be finite, positive, and ordered")
    if not edgelist_path:
        raise ValueError("Pass the explicit Elliptic++ address-edge file path")

    # 1. Resolve real Elliptic++ data paths
    resolved_edge, resolved_feat, resolved_cls = resolve_elliptic_paths(edgelist_path=edgelist_path)
    
    # 2. Load genuine background Bitcoin actor transaction graph
    loader = EllipticActorsLoader(
        edgelist_path=str(resolved_edge),
        features_path=str(resolved_feat) if resolved_feat else None,
        classes_path=str(resolved_cls) if resolved_cls else None,
    )
    bg_graph = loader.load_actor_graph(max_edges=max_background_edges)
    
    # 3. Inject flow-conserving collusion rings anchored into the real Bitcoin network
    injector = SemiSyntheticRingInjector(seed=seed)
    aug_graph, injected_rings = injector.inject_rings(
        graph=bg_graph,
        n_rings=n_pos,
        cycle_lengths=injection_cycle_lengths,
        fee_rate=fee_rate,
        initial_amount_range=initial_amount_range,
        base_timestamp=base_timestamp,
    )

    # The address edgelist contains no observed amount or event time. Use a
    # static, unweighted graph for both classes; do not expose generator values.
    extractor = CandidateExtractor(
        min_k=min_cycle_length,
        max_k=max_cycle_length,
        max_cycles=max_candidate_cycles,
    )
    cycles = extractor.find_simple_cycles(aug_graph)
    candidates: List[CandidateSubgraph] = []
    for i, cycle in enumerate(cycles):
        cand = extractor.extract_subgraph(aug_graph, cycle, hop_radius=hop_radius,
                                          candidate_id=f"cand_elliptic_{i:06d}")
        cycle_pairs = {(cycle[j], cycle[(j + 1) % len(cycle)]) for j in range(len(cycle))}
        ring_ids = set()
        fully_injected = True
        for u, v in cycle_pairs:
            pair_data = aug_graph[u][v]
            edge_records = pair_data.values() if aug_graph.is_multigraph() else (pair_data,)
            pair_ring_ids = {
                edge.get("ring_id")
                for edge in edge_records
                if edge.get("is_injected_ring") and edge.get("ring_id")
            }
            if not pair_ring_ids:
                fully_injected = False
            else:
                ring_ids.update(pair_ring_ids)
        has_injected_neighbor = any(e.get("is_injected_ring") for e in cand.edges)
        positive = fully_injected and len(ring_ids) == 1
        if not positive and has_injected_neighbor:
            continue  # Ambiguous neighborhood; never force it benign.
        cand.label = int(positive)
        cand.pattern_id = next(iter(ring_ids)) if positive else None
        for edge in cand.edges:
            edge.pop("is_injected_ring", None)
            edge.pop("is_bridge", None)
            edge.pop("ring_id", None)
            edge.pop("amount_next", None)
            edge.pop("fee", None)
            edge.pop("fee_rate", None)
            edge.pop("cycle_length", None)
            edge["amount"] = 0.0
            edge["timestamp"] = 0.0
        candidates.append(cand)
    positives = [c for c in candidates if c.label == 1]
    negatives = [c for c in candidates if c.label == 0]
    if len(positives) < n_pos or not negatives:
        raise ValueError(f"Insufficient blind cycle candidates: positives={len(positives)}, negatives={len(negatives)}")

    # A diagnostic of construction confounding, not a trained classifier.
    # Report it with every run so synthetic-vs-background shortcuts remain visible.
    shortcut_diagnostics = {
        "candidate_node_count": best_univariate_leq_accuracy(candidates, lambda c: len(c.nodes)),
        "candidate_edge_count": best_univariate_leq_accuracy(candidates, lambda c: len(c.edges)),
        "cycle_length": best_univariate_leq_accuracy(candidates, lambda c: c.cycle_length),
    }

    provenance = {
        "data_source": "elliptic_actors_with_injected_rings",
        "edgelist_path": str(resolved_edge),
        "edgelist_sha256": compute_file_sha256(resolved_edge),
        "background_real_nodes": bg_graph.number_of_nodes(),
        "background_real_edges": bg_graph.number_of_edges(),
        "injected_collusion_rings": len(injected_rings),
        "requested_positive_rings": n_pos,
        "injection_seed": seed,
        "max_background_edges": max_background_edges,
        "background_edge_selection": "first_max_background_edges_rows_in_source_order",
        "max_candidate_cycles": max_candidate_cycles,
        "candidate_cycle_bounds": [min_cycle_length, max_cycle_length],
        "candidate_hop_radius": hop_radius,
        "injection_cycle_lengths": list(injection_cycle_lengths),
        "injection_fee_rate": fee_rate,
        "injection_initial_amount_range": list(initial_amount_range),
        "injection_base_timestamp": base_timestamp,
        "background_negatives": len([c for c in candidates if c.label == 0]),
        "candidate_sampling": "same_blind_cycle_extractor_both_classes",
        "candidate_label_rule": (
            "positive_if_every_directed_central_cycle_pair_has_an_injected_edge_from_one_ring; "
            "exclude_nonpositive_candidates_whose_neighborhood_contains_an_injected_edge"
        ),
        "negative_label_scope": "not_injected_only_not_verified_licit",
        "node_count_by_label": {
            "positive": dict(sorted(Counter(len(c.nodes) for c in positives).items())),
            "negative": dict(sorted(Counter(len(c.nodes) for c in negatives).items())),
        },
        "exact_node_and_edge_count_matched_control_capacity": {
            f"nodes={node_count},edges={edge_count}": min(
                sum(len(c.nodes) == node_count and len(c.edges) == edge_count for c in positives),
                sum(len(c.nodes) == node_count and len(c.edges) == edge_count for c in negatives),
            )
            for node_count, edge_count in sorted({
                (len(c.nodes), len(c.edges)) for c in positives
            })
        },
        "univariate_shortcut_accuracy_on_raw_cohort": shortcut_diagnostics,
        "claim_scope": "exploratory_detection_of_injected_rings_only",
        "edge_amount_and_time_observed": False,
        "synthetic_negatives_injected": 0,
        "warnings": [],
    }

    def cleanup():
        pass

    return candidates, cleanup, provenance


def match_elliptic_controls_by_structure(
    candidates: List[CandidateSubgraph],
    expected_positive_count: int,
    control_seed: int,
) -> Tuple[List[CandidateSubgraph], Dict[str, Any]]:
    """Builds a 1:1 cohort exactly matched on node and edge counts.

    Matched pairs are kept in the same split. This controls two construction
    shortcuts found in the supplied cohort. It does not make the synthetic task
    representative or establish exchangeability on all graph properties.
    This defines a conditional matched-cohort estimand, not population prevalence.
    """
    positives = [c for c in candidates if c.label == 1]
    negatives = [c for c in candidates if c.label == 0]
    if len(positives) != expected_positive_count:
        raise ValueError(
            f"Expected {expected_positive_count} injected positives, observed {len(positives)}"
        )

    pos_by_stratum: Dict[Tuple[int, int], List[CandidateSubgraph]] = {}
    neg_by_stratum: Dict[Tuple[int, int], List[CandidateSubgraph]] = {}
    for candidate in positives:
        stratum = (len(candidate.nodes), len(candidate.edges))
        pos_by_stratum.setdefault(stratum, []).append(candidate)
    for candidate in negatives:
        stratum = (len(candidate.nodes), len(candidate.edges))
        neg_by_stratum.setdefault(stratum, []).append(candidate)

    support = {
        f"nodes={node_count},edges={edge_count}": {
            "positive": len(pos_group),
            "negative": len(neg_by_stratum.get((node_count, edge_count), [])),
        }
        for (node_count, edge_count), pos_group in sorted(pos_by_stratum.items())
    }
    insufficient = {stratum: counts for stratum, counts in support.items()
                    if counts["negative"] < counts["positive"]}
    if insufficient:
        raise ValueError(
            "Cannot form a full exact node-and-edge-count matched cohort from this "
            f"background edge prefix; support by structural stratum: {insufficient}"
        )

    matched: List[CandidateSubgraph] = []
    matched_pairs = []
    pair_number = 0
    rng = random.Random(control_seed)
    for stratum, pos_group in sorted(pos_by_stratum.items()):
        available_negatives = sorted(neg_by_stratum[stratum], key=lambda c: c.candidate_id)
        neg_group = sorted(
            rng.sample(available_negatives, len(pos_group)),
            key=lambda c: c.candidate_id,
        )
        pos_group = sorted(pos_group, key=lambda c: c.candidate_id)
        for positive, negative in zip(pos_group, neg_group):
            pair_id = f"elliptic_matched_pair_{pair_number:06d}"
            positive.matched_group_id = pair_id
            negative.matched_group_id = pair_id
            matched.extend((positive, negative))
            matched_pairs.append({
                "matched_group_id": pair_id,
                "node_count": stratum[0],
                "edge_count": stratum[1],
                "positive_edge_count": len(positive.edges),
                "negative_edge_count": len(negative.edges),
            })
            pair_number += 1

    matched.sort(key=lambda c: c.candidate_id)
    return matched, {
        "method": "exact_node_and_edge_count_1_to_1_control_sampling_without_replacement",
        "control_selection_seed": control_seed,
        "support_by_structural_stratum": support,
        "matched_pair_count": len(matched_pairs),
        "excluded_unmatched_background_candidates": len(negatives) - len(matched_pairs),
        "pairs": matched_pairs,
        "prevalence_estimand": "balanced_case_control_matched_cohort_not_population_prevalence",
    }


class EllipticBenchmarkRunner:
    """Manages exploratory training and evaluation on the injected Elliptic++ track."""

    def __init__(
        self,
        candidates: List[CandidateSubgraph],
        runs_dir: Union[str, Path] = "runs",
        baseline_hidden_dim: int = 288,
        baseline_heads: int = 16,
    ):
        self.candidates = candidates
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")
        self.baseline_hidden_dim = baseline_hidden_dim
        self.baseline_heads = baseline_heads
        self.ph_engine = GudhiPersistenceEngine()
        self.prepared_data = self._prepare_candidate_tensors()

    def _prepare_candidate_tensors(self) -> List[Dict[str, Any]]:
        """Lifts each candidate to graph, simplicial, and cellular views with persistent homology."""
        feat_extractor = CandidateFeatureExtractor()
        ph_vec = PersistenceLandscapeVectorizer()
        
        prepared = []
        for c in self.candidates:
            if c.label not in (0, 1):
                raise ValueError(f"Candidate {c.candidate_id} has no explicit binary label")
            lifted = lift_candidate(c)
            x_node, edge_index, edge_attr, _ = feat_extractor.extract_features(c, mode="structure_aware")
            
            # Persistent Homology on unfilled 1-skeleton
            filt = build_temporal_filtration(c)
            dgms = self.ph_engine.compute_persistence(filt)
            ph_feat = ph_vec.vectorize(dgms)
            ph_tensor = torch.tensor(ph_feat, dtype=torch.float32).unsqueeze(0)
            
            # Incidence and Laplacians
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
                                   item["l0"].to(self.device), item["l1"].to(self.device), item["l2"].to(self.device),
                                   ph_features=ph)
                elif fam == "mpsn":
                    logits = model(x_node, item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                   item["simp_b1"].to(self.device), item["simp_b2"].to(self.device),
                                   ph_features=ph)
                else:
                    raise ValueError(fam)
                    
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
                                    item["l0"].to(self.device), item["l1"].to(self.device), item["l2"].to(self.device),
                                    ph_features=ph)
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

        try:
            from source.src.models.train_harness import compute_optimal_threshold
        except ModuleNotFoundError:
            from src.models.train_harness import compute_optimal_threshold
        val_y, val_probs = [], []
        model.eval()
        with torch.no_grad():
            for item in val_items:
                val_y.append(item["label"])
                fam = model_family.lower()
                if fam in ["gcn", "gat", "sage"]:
                    x = item["x_node"].to(self.device)
                    batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                    out = model(x, item["edge_index"].to(self.device), batch=batch, ph_features=item["ph_tensor"].to(self.device), edge_attr=item["edge_attr"].to(self.device))
                elif fam in ["cwn", "topotune"]:
                    out = model(item["x_node"].to(self.device), item["cell_x1"].to(self.device), item["x2"].to(self.device),
                                item["b1"].to(self.device), item["b2"].to(self.device), item["l0"].to(self.device),
                                item["l1"].to(self.device), item["l2"].to(self.device), ph_features=item["ph_tensor"].to(self.device))
                else:
                    out = model(item["x_node"].to(self.device), item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                item["simp_b1"].to(self.device), item["simp_b2"].to(self.device), ph_features=item["ph_tensor"].to(self.device))
                val_probs.append(float(torch.sigmoid(out.view(-1))[0].item()))
        threshold = compute_optimal_threshold(np.array(val_y), np.array(val_probs))

        # Inference loop
        model.eval()
        pred_records = []
        y_true_list = []
        y_prob_list = []
        
        with torch.no_grad():
            for item in test_items:
                x_node = item["x_node"].to(self.device)
                ph = item["ph_tensor"].to(self.device)
                fam = model_family.lower()
                if fam in ["gcn", "gat", "sage"]:
                    ei = item["edge_index"].to(self.device)
                    b = torch.zeros(x_node.size(0), dtype=torch.long, device=self.device)
                    logits = model(x_node, ei, batch=b, ph_features=ph, edge_attr=item["edge_attr"].to(self.device))
                elif fam in ["cwn", "topotune"]:
                    logits = model(x_node, item["cell_x1"].to(self.device), item["x2"].to(self.device),
                                   item["b1"].to(self.device), item["b2"].to(self.device),
                                   item["l0"].to(self.device), item["l1"].to(self.device), item["l2"].to(self.device),
                                   ph_features=ph)
                elif fam == "mpsn":
                    logits = model(x_node, item["simp_x1"].to(self.device), item["simp_x2"].to(self.device),
                                   item["simp_b1"].to(self.device), item["simp_b2"].to(self.device),
                                   ph_features=ph)
                    
                prob = float(torch.sigmoid(logits.view(-1))[0].item())
                label = item["label"]
                y_true_list.append(label)
                y_prob_list.append(prob)
                
                pred_records.append({
                    "candidate_id": item["cand"].candidate_id,
                    "model_family": model_family,
                    "seed": seed,
                    "fold": fold,
                    "ground_truth": label,
                    "predicted_prob": prob,
                    "predicted_class": int(prob >= threshold),
                    "pattern_id": item["pattern_id"]
                })
                
        metrics = compute_classification_metrics(np.array(y_true_list), np.array(y_prob_list), val_threshold=threshold)
        
        # Serialize to Parquet and run manifest
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
        run_id = f"run_elliptic_{model_family.lower()}_seed{seed}_fold{fold}_{run_digest}"
        run_path = self.runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(pred_records)
        parquet_path = run_path / "predictions.parquet"
        df.to_parquet(parquet_path, index=False)
        parquet_sha = compute_file_sha256(parquet_path)
        
        manifest_data = {
            "run_id": run_id,
            "benchmark_track": "Elliptic_Injected_Track",
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


def run_elliptic_injected_benchmark(
    *,
    n_pos: int,
    injection_seed: int,
    control_matching_seed: int,
    max_background_edges: int,
    max_candidate_cycles: int,
    injection_cycle_lengths: List[int],
    min_cycle_length: int,
    max_cycle_length: int,
    hop_radius: int,
    fee_rate: float,
    initial_amount_range: Tuple[float, float],
    base_timestamp: int,
    edgelist_path: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    families: Optional[List[str]] = None,
    runs_dir: str = "runs",
    epochs: Optional[int] = None,
    baseline_hidden_dim: Optional[int] = None,
    baseline_heads: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs exploratory evaluation across seeds on the injected Elliptic++ track."""
    if epochs is None or baseline_hidden_dim is None or baseline_heads is None:
        raise ValueError("Explicitly declare epochs and baseline dimensions for the benchmark")
    if epochs <= 0 or baseline_hidden_dim <= 0 or baseline_heads <= 0:
        raise ValueError("Epoch count and baseline dimensions must be positive")
    if not seeds or not families:
        raise ValueError("Pass an explicit non-empty seed list and model-family list")
    if len(set(seeds)) != len(seeds) or len(set(families)) != len(families):
        raise ValueError("Seed and model-family lists must not contain duplicates")
    target_seeds = list(seeds)
    target_families = list(families)
    
    candidates, cleanup_fn, provenance = build_elliptic_candidate_pool(
        n_pos=n_pos,
        seed=injection_seed,
        max_background_edges=max_background_edges,
        max_candidate_cycles=max_candidate_cycles,
        injection_cycle_lengths=injection_cycle_lengths,
        min_cycle_length=min_cycle_length,
        max_cycle_length=max_cycle_length,
        hop_radius=hop_radius,
        fee_rate=fee_rate,
        initial_amount_range=initial_amount_range,
        base_timestamp=base_timestamp,
        edgelist_path=edgelist_path,
    )
    provenance.update(compute_engine_provenance())

    # ── INV-001: Assert data provenance ──
    assert provenance.get("data_source") == "elliptic_actors_with_injected_rings", (
        f"[INV-001 Violation] Data source is '{provenance.get('data_source')}', "
        f"expected 'elliptic_actors_with_injected_rings'. Aborting benchmark."
    )

    try:
        candidates, matching_provenance = match_elliptic_controls_by_structure(
            candidates,
            expected_positive_count=n_pos,
            control_seed=control_matching_seed,
        )
        provenance["control_matching"] = matching_provenance
        provenance["univariate_shortcut_accuracy_on_matched_cohort"] = {
            "candidate_node_count": best_univariate_leq_accuracy(candidates, lambda c: len(c.nodes)),
            "candidate_edge_count": best_univariate_leq_accuracy(candidates, lambda c: len(c.edges)),
            "cycle_length": best_univariate_leq_accuracy(candidates, lambda c: c.cycle_length),
        }
        provenance["matched_candidate_count"] = len(candidates)

        runner = EllipticBenchmarkRunner(
            candidates,
            runs_dir=runs_dir,
            baseline_hidden_dim=baseline_hidden_dim,
            baseline_heads=baseline_heads,
        )
        n_folds = 5
        runs_output = []

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
            "benchmark": "Elliptic_Injected_Track",
            "claim_scope": "exploratory_detection_of_injected_rings_in_an_exact_node_and_edge_count_matched_case_control_cohort_only",
            "cohort_configuration": {
                "requested_positive_rings": n_pos,
                "injection_seed": injection_seed,
                "control_matching_seed": control_matching_seed,
                "max_background_edges": max_background_edges,
                "max_candidate_cycles": max_candidate_cycles,
                "injection_cycle_lengths": injection_cycle_lengths,
                "candidate_cycle_bounds": [min_cycle_length, max_cycle_length],
                "candidate_hop_radius": hop_radius,
                "fee_rate": fee_rate,
                "initial_amount_range": list(initial_amount_range),
                "base_timestamp": base_timestamp,
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
    parser = argparse.ArgumentParser(description="Exploratory Elliptic++ Injected Benchmark Runner")
    parser.add_argument("--edgelist-path", required=True, help="Explicit Elliptic address-edge CSV")
    parser.add_argument("--n-pos", type=int, required=True, help="Number of injected positive rings")
    parser.add_argument("--injection-seed", type=int, required=True, help="Seed used only for ring injection")
    parser.add_argument("--control-matching-seed", type=int, required=True,
                        help="Separate seed for within-stratum control selection")
    parser.add_argument("--max-background-edges", type=int, required=True,
                        help="Predeclared number of source-order background edge rows")
    parser.add_argument("--max-candidate-cycles", type=int, required=True,
                        help="Predeclared cycle enumeration cap; cap saturation fails closed")
    parser.add_argument("--injection-cycle-lengths", nargs="+", type=int, required=True)
    parser.add_argument("--min-cycle-length", type=int, required=True)
    parser.add_argument("--max-cycle-length", type=int, required=True)
    parser.add_argument("--hop-radius", type=int, required=True)
    parser.add_argument("--fee-rate", type=float, required=True)
    parser.add_argument("--initial-amount-range", nargs=2, type=float, required=True,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--base-timestamp", type=int, required=True)
    parser.add_argument("--summary-output", required=True, help="Path for the full study summary JSON")
    parser.add_argument("--runs-dir", default="runs", help="Directory to store runs")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Declared evaluation seeds")
    parser.add_argument("--families", nargs="+", type=str, required=True, help="Target model families, e.g. gcn gat")
    parser.add_argument("--epochs", type=int, required=True, help="Training epochs")
    parser.add_argument("--baseline-hidden-dim", type=int, required=True,
                        help="Hidden dim for the intentionally widened graph-baseline stress test")
    parser.add_argument("--baseline-heads", type=int, required=True, help="Attention heads for GAT baseline")
    args = parser.parse_args()
    res = run_elliptic_injected_benchmark(
        n_pos=args.n_pos,
        injection_seed=args.injection_seed,
        control_matching_seed=args.control_matching_seed,
        max_background_edges=args.max_background_edges,
        max_candidate_cycles=args.max_candidate_cycles,
        injection_cycle_lengths=args.injection_cycle_lengths,
        min_cycle_length=args.min_cycle_length,
        max_cycle_length=args.max_cycle_length,
        hop_radius=args.hop_radius,
        fee_rate=args.fee_rate,
        initial_amount_range=tuple(args.initial_amount_range),
        base_timestamp=args.base_timestamp,
        edgelist_path=args.edgelist_path,
        seeds=args.seeds,
        families=args.families,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        baseline_hidden_dim=args.baseline_hidden_dim,
        baseline_heads=args.baseline_heads,
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Elliptic++ benchmark completed successfully: {res['total_runs']} runs executed.")
