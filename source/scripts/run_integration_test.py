#!/usr/bin/env python3
"""
End-to-End Real-Data Integration Pipeline Engine.

Executes the mandatory real-data integration trace (§10.1, INV-012):
raw file -> parser -> candidate -> lifting -> PH -> forward -> loss -> backward -> step -> parquet -> manifest

Upholds Invariants:
- INV-001 (No Mock Data in Production): Operates strictly on real file schema.
- INV-002 (Simplex vs Polygonal Cell Separation): Operates on k >= 4 cycle cells.
- INV-003 (Unfilled 1-Skeleton for Persistent Homology): H1 on 1-skeleton.
- INV-004 (Exact Boundary Nilpotency): B1 @ B2 = 0.
- INV-006 (Cryptographic Lineage Tracking): Generates SHA-256 run manifest.
- INV-008 (Self-Contained Verification Scripts).
- INV-009 (Model Parameter Matching Protocol).
- INV-012 (Real File Schema Grounding).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

# Resolve imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.data.amlworld_loader import AMLWorldLoader, AMLWorldTransaction
    from source.src.data.candidate_extractor import CandidateExtractor, CandidateSubgraph
    from source.src.data.path_utils import resolve_amlworld_paths
    from source.src.topology.lift_dataset import lift_candidate, MultiDomainExample
    from source.src.ph.graph_filtration import build_temporal_filtration
    from source.src.ph.gudhi_backend import GudhiPersistenceEngine
    from source.src.ph.vectorize import PersistenceLandscapeVectorizer
    from source.src.models.feature_encoders import CandidateFeatureExtractor
    from source.src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from source.src.models.tdl_simplicial import SimplicialComplexNetwork
    from source.src.models.tdl_cellular import CellularComplexNetwork
    from source.src.models.topotune_adapter import TopoTuneInspiredAdapter, HybridTopologicalReadout
except ModuleNotFoundError:
    from src.data.amlworld_loader import AMLWorldLoader, AMLWorldTransaction
    from src.data.candidate_extractor import CandidateExtractor, CandidateSubgraph
    from src.data.path_utils import resolve_amlworld_paths
    from src.topology.lift_dataset import lift_candidate, MultiDomainExample
    from src.ph.graph_filtration import build_temporal_filtration
    from src.ph.gudhi_backend import GudhiPersistenceEngine
    from src.ph.vectorize import PersistenceLandscapeVectorizer
    from src.models.feature_encoders import CandidateFeatureExtractor
    from src.models.graph_baselines import GCNBaseline, GATBaseline, GraphSAGEBaseline
    from src.models.tdl_simplicial import SimplicialComplexNetwork
    from src.models.tdl_cellular import CellularComplexNetwork
    from src.models.topotune_adapter import TopoTuneInspiredAdapter, HybridTopologicalReadout


class TopoTuneInspiredModel(nn.Module):
    """TopoTune-inspired model combining adapter layer and hybrid readout."""
    def __init__(
        self,
        in_dim0: int = 13,
        in_dim1: int = 2,
        in_dim2: int = 4,
        hidden_dim: int = 27,
        ph_dim: Optional[int] = None,
    ):
        super().__init__()
        self.adapter = TopoTuneInspiredAdapter(
            in_dims={0: in_dim0, 1: in_dim1, 2: in_dim2},
            out_dim=hidden_dim,
        )
        self.readout = HybridTopologicalReadout(
            cell_hidden_dim=hidden_dim,
            num_ranks=3,
            ph_dim=ph_dim,
        )

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        l0: Optional[torch.Tensor] = None,
        l1: Optional[torch.Tensor] = None,
        l2: Optional[torch.Tensor] = None,
        ph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = {0: x0, 1: x1, 2: x2}
        incidences = {(1, 0): b1, (2, 1): b2}
        h = self.adapter(features, incidences)
        return self.readout([h[0], h[1], h[2]], ph_features=ph_features)


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_timestamp_to_float(ts_str: Any) -> float:
    """Parse an observed event time; malformed values invalidate the trace."""
    if isinstance(ts_str, (int, float)):
        value = float(ts_str)
        if not math.isfinite(value):
            raise ValueError(f"Invalid transaction timestamp: {ts_str!r}")
        return value
    s = str(ts_str).strip()
    try:
        return float(s)
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(s)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        value = parsed.timestamp()
        if not math.isfinite(value):
            raise ValueError("timestamp is not finite")
        return value
    except ValueError as exc:
        raise ValueError(f"Invalid transaction timestamp: {ts_str!r}") from exc


class IntegrationPipelineEngine:
    """Orchestrates the complete real-data pipeline trace from raw file to optimization and artifact generation."""

    def __init__(
        self,
        trans_csv_path: str,
        patterns_txt_path: Optional[str] = None,
        model_family: str = "cwn",
        learning_rate: float = 0.001,
        seed: int = 42,
    ):
        # Resolve data paths robustly
        resolved_trans, resolved_pat = resolve_amlworld_paths(trans_csv_path, patterns_txt_path)
        self.trans_csv_path = str(resolved_trans)
        self.patterns_txt_path = str(resolved_pat) if resolved_pat else None
        self.model_family = model_family.lower()
        self.learning_rate = learning_rate
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Initialize Loader
        self.loader = AMLWorldLoader(
            trans_csv_path=self.trans_csv_path,
            patterns_txt_path=self.patterns_txt_path,
        )

        # 2. Initialize Extractor & Topological Tools
        self.extractor = CandidateExtractor(min_k=3, max_k=6, max_cycles=100)
        self.ph_engine = GudhiPersistenceEngine()
        self.ph_vectorizer = PersistenceLandscapeVectorizer()

    def build_transaction_graph(self, max_transactions: int = 5000) -> nx.MultiDiGraph:
        """Parses real stream transactions into a directed multigraph / digraph."""
        graph = nx.MultiDiGraph()
        count = 0
        t_base: Optional[float] = None

        for batch in self.loader.stream_transactions(chunk_size=1000):
            for tx in batch:
                graph.add_node(
                    tx.sender_key,
                    bank=tx.from_bank,
                    account=tx.from_account,
                )
                graph.add_node(
                    tx.receiver_key,
                    bank=tx.to_bank,
                    account=tx.to_account,
                )
                ts_num = parse_timestamp_to_float(tx.timestamp)
                if t_base is None:
                    t_base = ts_num
                # Normalize large epoch seconds to relative hours
                rel_ts = (ts_num - t_base) / 3600.0
                graph.add_edge(
                    tx.sender_key,
                    tx.receiver_key,
                    tx_id=tx.tx_id or f"tx_{count}",
                    timestamp=rel_ts,
                    amount=tx.amount_paid,
                    currency=tx.payment_currency,
                    format=tx.payment_format,
                    is_laundering=tx.is_laundering,
                    pattern_id=tx.pattern_id,
                )
                count += 1
                if count >= max_transactions:
                    return graph
        return graph

    def instantiate_model(self) -> Tuple[nn.Module, str]:
        """Instantiates capacity-matched model family."""
        if self.model_family == "gcn":
            model = GCNBaseline(in_dim=13, hidden_dim=64, ph_dim=310)
            name = "GCN"
        elif self.model_family == "gat":
            model = GATBaseline(in_dim=13, hidden_dim=64, ph_dim=310)
            name = "GAT"
        elif self.model_family == "sage":
            model = GraphSAGEBaseline(in_dim=13, hidden_dim=64, ph_dim=310)
            name = "GraphSAGE"
        elif self.model_family == "mpsn":
            model = SimplicialComplexNetwork(in_dim0=13, in_dim1=3, in_dim2=4, hidden_dim=27, ph_dim=310)
            name = "SimplicialMPSN"
        elif self.model_family == "cwn":
            model = CellularComplexNetwork(in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=310)
            name = "CellularCWN"
        elif self.model_family == "topotune":
            model = TopoTuneInspiredModel(in_dim0=13, in_dim1=2, in_dim2=4, hidden_dim=27, ph_dim=310)
            name = "TopoTune-inspired"
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")
        return model, name

    def execute_trace(
        self,
        max_transactions: int = 5000,
        epochs: int = 2,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the mandatory end-to-end integration trace.
        """
        t0 = time.perf_counter()

        # Step 1: Raw Parser & Graph Construction
        graph = self.build_transaction_graph(max_transactions=max_transactions)
        num_raw_nodes = graph.number_of_nodes()
        num_raw_edges = graph.number_of_edges()
        if num_raw_nodes == 0 or num_raw_edges == 0:
            raise ValueError("No transaction nodes or edges loaded from raw source.")

        # Step 2: Label-Blind Candidate Extraction
        raw_cycles = self.extractor.find_simple_cycles(graph)
        candidates: List[CandidateSubgraph] = []
        for idx, cycle in enumerate(raw_cycles[:10]):
            cand = self.extractor.extract_subgraph(
                graph=graph,
                cycle_nodes=cycle,
                hop_radius=1,
                candidate_id=f"integ_cand_{idx:03d}",
            )
            candidates.append(cand)

        for c in candidates:
            cycle_pairs = {(c.cycle_nodes[i], c.cycle_nodes[(i + 1) % len(c.cycle_nodes)])
                           for i in range(len(c.cycle_nodes))}
            c.label = int(all(any(e["source"] == u and e["target"] == v and
                                  int(e.get("is_laundering", 0)) == 1 for e in c.edges)
                              for u, v in cycle_pairs))
            for e in c.edges:
                e.pop("is_laundering", None)
                e.pop("pattern_id", None)
            c.pattern_id = None

        if not candidates:
            raise ValueError("Candidate extraction yielded zero candidates from raw transaction graph.")

        primary_cand = candidates[0]

        # Step 3: Multi-Domain Topological Lifting
        multi_domain = lift_candidate(primary_cand)

        # Invariant INV-004 verification on lifted cell complex
        b1_mat = multi_domain.cell.B1.toarray()
        b2_mat = multi_domain.cell.B2.toarray()
        nilpotency_error = float(np.max(np.abs(b1_mat @ b2_mat))) if (b1_mat.size > 0 and b2_mat.size > 0) else 0.0
        if nilpotency_error > 1e-10:
            raise ValueError(f"INV-004 Violation: B1 @ B2 = {nilpotency_error} > 1e-10")

        # Step 4: Persistent Homology on Unfilled 1-Skeleton (INV-003)
        ph_graph = build_temporal_filtration(primary_cand)
        persistence_diagram = self.ph_engine.compute_persistence(ph_graph)
        ph_vector = self.ph_vectorizer.vectorize(persistence_diagram)
        ph_tensor = torch.tensor(ph_vector, dtype=torch.float32).unsqueeze(0)

        # Step 5: Instantiate Model & Optimizer
        model, model_name = self.instantiate_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        if primary_cand.label not in (0, 1):
            raise ValueError("Integration candidate has no explicit binary label")
        target_label = float(primary_cand.label)
        target_tensor = torch.tensor([target_label], dtype=torch.float32)

        # Step 6: Multi-Epoch Optimization & Backward Step Trace
        loss_history = []
        model.train()

        # Extract 13-dim structure-aware features for node / 0-cell representations
        cand_feat_ext = CandidateFeatureExtractor()
        x_node, edge_index_cand, edge_attr_cand, _ = cand_feat_ext.extract_features(
            primary_cand, mode="structure_aware"
        )

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Execute forward pass based on model architecture
            if self.model_family in ["gcn", "gat", "sage"]:
                batch_vec = torch.zeros(x_node.size(0), dtype=torch.long)
                logits = model(x_node, edge_index_cand, batch=batch_vec, ph_features=ph_tensor, edge_attr=edge_attr_cand)
            elif self.model_family in ["cwn", "topotune"]:
                x0 = x_node
                x1 = torch.tensor(multi_domain.cell.x_1, dtype=torch.float32)
                raw_x2 = multi_domain.cell.x_2
                if raw_x2.shape[0] > 0:
                    if raw_x2.shape[1] < 4:
                        x2 = torch.cat([
                            torch.tensor(raw_x2, dtype=torch.float32),
                            torch.zeros((raw_x2.shape[0], 4 - raw_x2.shape[1]), dtype=torch.float32),
                        ], dim=-1)
                    else:
                        x2 = torch.tensor(raw_x2[:, :4], dtype=torch.float32)
                else:
                    x2 = torch.empty((0, 4), dtype=torch.float32)

                b1 = torch.tensor(b1_mat, dtype=torch.float32)
                b2 = torch.tensor(b2_mat, dtype=torch.float32)
                l0 = torch.tensor(multi_domain.cell.L0.toarray(), dtype=torch.float32)
                l1 = torch.tensor(multi_domain.cell.L1.toarray(), dtype=torch.float32)
                l2 = torch.tensor(multi_domain.cell.L2.toarray(), dtype=torch.float32)
                logits = model(x0, x1, x2, b1, b2, l0, l1, l2, ph_features=ph_tensor)
            elif self.model_family == "mpsn":
                x0 = x_node
                x1 = multi_domain.simplicial.x_1
                num_triangles = multi_domain.simplicial.x_2.shape[0]
                if num_triangles > 0:
                    simp_x2 = multi_domain.simplicial.x_2
                    if simp_x2.shape[1] < 4:
                        x2 = torch.cat([
                            simp_x2,
                            torch.zeros((num_triangles, 4 - simp_x2.shape[1])),
                        ], dim=-1)
                    else:
                        x2 = simp_x2[:, :4]
                else:
                    x2 = torch.empty((0, 4), dtype=torch.float32)

                b1 = torch.tensor(multi_domain.simplicial.B1.toarray(), dtype=torch.float32)
                b2 = torch.tensor(multi_domain.simplicial.B2.toarray(), dtype=torch.float32)
                logits = model(x0, x1, x2, b1, b2, ph_features=ph_tensor)

            loss = criterion(logits.view(-1), target_tensor)
            loss_history.append(loss.item())

            # Autograd Backpropagation
            loss.backward()

            # Gradient Integrity Verification
            grad_norm = 0.0
            has_nan = False
            active_params = 0
            for param_name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan = True
                        break
                    grad_norm += param.grad.norm().item() ** 2
                    active_params += 1
            grad_norm = float(np.sqrt(grad_norm))

            if has_nan or active_params == 0 or (grad_norm == 0.0 and loss.item() > 1e-7):
                raise RuntimeError(
                    f"Autograd failed: invalid gradients (norm={grad_norm}, active_params={active_params}, loss={loss.item()})"
                )

            # Optimizer step
            optimizer.step()

        # Step 7: Prediction Serialization & Artifact Writing
        pred_prob = float(torch.sigmoid(logits).item())
        pred_label = 1 if pred_prob >= 0.5 else 0
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        records = [
            {
                "candidate_id": primary_cand.candidate_id,
                "fold_id": 0,
                "model_name": model_name,
                "true_label": int(target_label),
                "predicted_prob": pred_prob,
                "predicted_label": pred_label,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
                "latency_ms": total_time_ms,
            }
        ]

        # Serialization to Parquet and Run Manifest
        parquet_sha256 = ""
        manifest_path_str = ""
        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)

            # Write Parquet
            df = pd.DataFrame(records)
            parquet_path = out_p / "predictions.parquet"
            df.to_parquet(parquet_path, index=False)
            parquet_sha256 = compute_file_sha256(parquet_path)

            # Write Run Manifest
            raw_sha = compute_file_sha256(self.trans_csv_path) if os.path.exists(self.trans_csv_path) else "NA"
            patterns_sha = (
                compute_file_sha256(self.patterns_txt_path)
                if self.patterns_txt_path and os.path.exists(self.patterns_txt_path)
                else None
            )
            manifest = {
                "run_id": f"RUN_INTEG_{model_name}_{int(time.time())}",
                "claim_scope": "execution_smoke_trace_only_not_model_evaluation",
                "model_family": model_name,
                "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "raw_transactions_path": self.trans_csv_path,
                "raw_transactions_sha256": raw_sha,
                "patterns_path": self.patterns_txt_path,
                "patterns_sha256": patterns_sha,
                "candidate_selection": "first_candidate_from_sorted_label_blind_cycle_enumeration",
                "predictions_parquet": str(parquet_path),
                "predictions_parquet_sha256": parquet_sha256,
                "nilpotency_error": nilpotency_error,
                "initial_loss": loss_history[0],
                "final_loss": loss_history[-1],
                "gradient_norm": grad_norm,
                "total_time_ms": total_time_ms,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_file = out_p / "run_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as fp:
                json.dump(manifest, fp, indent=2)
            manifest_path_str = str(manifest_file)

        return {
            "model_name": model_name,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "nilpotency_error": nilpotency_error,
            "initial_loss": loss_history[0],
            "final_loss": loss_history[-1],
            "gradient_norm": grad_norm,
            "predicted_prob": pred_prob,
            "predictions_parquet_sha256": parquet_sha256,
            "manifest_path": manifest_path_str,
            "total_time_ms": total_time_ms,
            "success": True,
        }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Real-Data Integration Pipeline CLI")
    parser.add_argument(
        "--raw-trans-csv",
        type=str,
        default="data/raw/ibm_amlworld/HI-Small_Trans.csv",
        help="Path to real transaction CSV file",
    )
    parser.add_argument(
        "--raw-patterns-txt",
        type=str,
        default=None,
        help="Optional path to real laundering patterns file",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["cwn", "mpsn", "gcn", "gat", "sage", "topotune"],
        default="cwn",
        help="Target model architecture family",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="results/integration_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resolved_trans, resolved_pat = resolve_amlworld_paths(args.raw_trans_csv, args.raw_patterns_txt)
    print("=" * 60)
    print("STARTING END-TO-END REAL-DATA INTEGRATION TEST (§10.1, INV-012)")
    print(f"Dataset Target: {resolved_trans}")
    print(f"Patterns Target: {resolved_pat}")
    print(f"Model Family: {args.model_family}")
    print("=" * 60)

    engine = IntegrationPipelineEngine(
        trans_csv_path=str(resolved_trans),
        patterns_txt_path=str(resolved_pat) if resolved_pat else None,
        model_family=args.model_family,
        seed=args.seed,
    )
    res = engine.execute_trace(
        epochs=args.epochs,
        output_dir=args.output_dir,
    )

    print("INTEGRATION TRACE SUCCESSFUL:")
    print(f"  Model:            {res['model_name']} ({res['parameter_count']} params)")
    print(f"  Nilpotency Error: {res['nilpotency_error']:.2e} (INV-004)")
    print(f"  Initial Loss:     {res['initial_loss']:.4f}")
    print(f"  Final Loss:       {res['final_loss']:.4f}")
    print(f"  Gradient Norm:    {res['gradient_norm']:.4f}")
    print(f"  Parquet SHA256:   {res['predictions_parquet_sha256']}")
    print(f"  Execution Time:   {res['total_time_ms']:.2f} ms")
    sys.exit(0)


if __name__ == "__main__":
    main()
