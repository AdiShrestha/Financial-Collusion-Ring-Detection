"""
Candidate Dataset Builder and Serialization Pipeline.

Orchestrates candidate extraction, ground-truth pattern labeling,
group-safe 5-fold stratified splitting, and cryptographic manifest generation.

Traces to Contract C02-05, FR-004, INV-001, INV-006, INV-007, INV-008.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import networkx as nx

# Resolve import paths
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.data.candidate_extractor import (
        CandidateExtractor,
        CandidateSubgraph,
    )
    from source.src.data.splits import GroupSafeSplitter, FoldAuditSummary
except ModuleNotFoundError:
    from src.data.candidate_extractor import (
        CandidateExtractor,
        CandidateSubgraph,
    )
    from src.data.splits import GroupSafeSplitter, FoldAuditSummary


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class CandidateDatasetBuilder:
    """Builds and serializes candidate subgraphs into group-safe cross-validation folds."""

    def __init__(
        self,
        n_splits: int = 5,
        min_k: int = 3,
        max_k: int = 6,
        max_cycles: int = 1000,
        hop_radius: int = 1,
        seed: int = 42,
    ):
        self.n_splits = n_splits
        self.min_k = min_k
        self.max_k = max_k
        self.max_cycles = max_cycles
        self.hop_radius = hop_radius
        self.seed = seed
        self.extractor = CandidateExtractor(
            min_k=self.min_k,
            max_k=self.max_k,
            max_cycles=self.max_cycles,
        )
        self.splitter = GroupSafeSplitter(n_splits=self.n_splits, seed=self.seed)

    def extract_candidates(
        self,
        graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
    ) -> List[CandidateSubgraph]:
        """Extracts cycles and applies observed transaction ground truth."""
        raw_cycles = self.extractor.find_simple_cycles(graph)
        candidates: List[CandidateSubgraph] = []

        for idx, cycle_nodes in enumerate(raw_cycles):
            cand_id = f"cand_{idx:06d}"
            cand = self.extractor.extract_subgraph(
                graph=graph,
                cycle_nodes=cycle_nodes,
                hop_radius=self.hop_radius,
                candidate_id=cand_id,
            )
            candidates.append(cand)

        if candidates:
            candidates = self.extractor.label_candidates(candidates)

        return candidates

    def build_and_split(
        self,
        graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
    ) -> Tuple[List[CandidateSubgraph], Dict[str, int], FoldAuditSummary]:
        """Extracts candidates and computes group-safe stratified fold assignments."""
        candidates = self.extract_candidates(graph)
        if not candidates:
            raise ValueError("No candidates extracted from graph.")

        assignments = self.splitter.split(candidates)
        summary = self.splitter.verify_zero_leakage(candidates, assignments)
        return candidates, assignments, summary

    def save_dataset(
        self,
        candidates: List[CandidateSubgraph],
        split_assignments: Dict[str, int],
        audit_summary: FoldAuditSummary,
        output_dir: Union[str, Path],
        dataset_name: str = "candidates",
        source_provenance: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serializes fold partitions and cryptographic manifest to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Partition candidates by fold
        fold_partitions: Dict[int, List[Dict[str, Any]]] = {
            f: [] for f in range(self.n_splits)
        }
        for cand in candidates:
            f = split_assignments[cand.candidate_id]
            fold_partitions[f].append(cand.to_dict())

        file_hashes: Dict[str, str] = {}
        for f in range(self.n_splits):
            fold_file = out_path / f"fold_{f}.json"
            with open(fold_file, "w", encoding="utf-8") as fp:
                json.dump(fold_partitions[f], fp, indent=2)
            file_hashes[f"fold_{f}.json"] = compute_file_sha256(fold_file)

        # Build cryptographic manifest
        manifest_data = {
            "dataset_name": dataset_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_splits": self.n_splits,
            "total_candidates": len(candidates),
            "fold_candidate_counts": audit_summary.fold_candidate_counts,
            "fold_positive_counts": audit_summary.fold_positive_counts,
            "fold_positive_ratios": audit_summary.fold_positive_ratios,
            "zero_node_leakage": audit_summary.zero_node_leakage,
            "zero_edge_leakage": audit_summary.zero_edge_leakage,
            "file_hashes": file_hashes,
            "source_provenance": source_provenance or {},
        }

        manifest_file = out_path / "candidates_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest_data, fp, indent=2)

        return manifest_file



def build_candidate_pool(
    trans_csv_path: Optional[str] = None,
    patterns_txt_path: Optional[str] = None,
    max_tx: Optional[int] = None,
    max_cycles: Optional[int] = None,
    min_cycle_length: Optional[int] = None,
    max_cycle_length: Optional[int] = None,
    hop_radius: Optional[int] = None,
) -> Tuple[List[CandidateSubgraph], Optional[Callable[[], None]], Dict[str, Any]]:
    """Extracts uniformly defined cycle candidates from an explicit AMLworld CSV cohort.

    Args:
        trans_csv_path: Explicit path to transaction CSV. If None, auto-resolved.
        patterns_txt_path: Explicit path to patterns file. If None, auto-resolved.
        max_tx: Maximum number of transactions to ingest from the CSV.
        max_cycles: Maximum number of cycles to enumerate, regardless of label.

    Returns:
        (candidates, cleanup_fn, provenance) where provenance is a dict containing:
        - data_source: Explicit IBM AMLworld synthetic CSV provenance tag.
        - trans_csv_sha256: SHA-256 of the ingested transaction CSV.
        - trans_csv_path: resolved path to the transaction CSV
        - synthetic_negatives_injected: Always zero in this CSV-derived path.
        - warnings: list of warning strings
    """
    try:
        from source.src.data.amlworld_loader import AMLWorldLoader
        from source.src.data.path_utils import resolve_amlworld_paths
    except ModuleNotFoundError:
        from src.data.amlworld_loader import AMLWorldLoader
        from src.data.path_utils import resolve_amlworld_paths

    cleanup_fn = None
    if not trans_csv_path or not patterns_txt_path:
        raise ValueError("Pass explicit AML transaction and paired pattern file paths")
    resolved_trans_path, resolved_patterns_path = resolve_amlworld_paths(
        trans_csv_path=trans_csv_path,
        patterns_txt_path=patterns_txt_path,
    )
    if resolved_patterns_path is None:
        raise FileNotFoundError("An explicit matching pattern file is required for the AML proxy labels")

    if max_tx is None or max_tx <= 0:
        raise ValueError("Declare a positive max_tx cohort size")
    if max_cycles is None or max_cycles <= 0:
        raise ValueError("Declare a positive max_cycles enumeration limit")
    if min_cycle_length is None or max_cycle_length is None or hop_radius is None:
        raise ValueError("Declare cycle-length bounds and candidate hop_radius before extraction")
    if min_cycle_length < 3 or max_cycle_length < min_cycle_length or hop_radius < 0:
        raise ValueError("Require 3 <= min_cycle_length <= max_cycle_length and hop_radius >= 0")

    trans_hash = compute_file_sha256(resolved_trans_path)
    provenance: Dict[str, Any] = {
        "data_source": "ibm_amlworld_synthetic_csv",
        "trans_csv_sha256": trans_hash,
        "trans_csv_path": str(resolved_trans_path),
        "patterns_txt_path": str(resolved_patterns_path) if resolved_patterns_path else None,
        "patterns_txt_sha256": compute_file_sha256(resolved_patterns_path) if resolved_patterns_path else None,
        "max_transactions": max_tx,
        "max_cycles": max_cycles,
        "candidate_cycle_bounds": [min_cycle_length, max_cycle_length],
        "candidate_hop_radius": hop_radius,
        "candidate_label_rule": (
            "positive_if_each_directed_central_cycle_pair_has_at_least_one_laundering_transfer; "
            "negative_if_all_neighborhood_transfers_are_unlabelled; exclude_mixed_neighborhoods"
        ),
        "target_scope": "transaction-flag-derived cycle proxy, not adjudicated ring truth",
        "timestamp_format": "%Y/%m/%d %H:%M",
        "timestamp_timezone_assumption": "UTC",
        "timestamp_origin": "first_ingested_transaction",
        "timestamp_unit": "hours",
        "synthetic_negatives_injected": 0,
        "warnings": [],
    }

    loader = AMLWorldLoader(
        trans_csv_path=str(resolved_trans_path),
        patterns_txt_path=str(resolved_patterns_path) if resolved_patterns_path else None,
    )

    from datetime import datetime, timezone
    graph = nx.MultiDiGraph()
    t_base = None
    stream_count = 0
    for batch in loader.stream_transactions(chunk_size=1000):
        for tx in batch:
            graph.add_node(tx.sender_key, bank=tx.from_bank, account=tx.from_account)
            graph.add_node(tx.receiver_key, bank=tx.to_bank, account=tx.to_account)
            try:
                ts = datetime.strptime(tx.timestamp, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
            except ValueError as exc:
                raise ValueError(f"Invalid AMLworld timestamp in {tx.tx_id}: {tx.timestamp}") from exc
            if t_base is None:
                t_base = ts
            rel_ts = (ts - t_base) / 3600.0
            graph.add_edge(
                tx.sender_key,
                tx.receiver_key,
                tx_id=tx.tx_id or f"tx_{stream_count}",
                timestamp=rel_ts,
                amount=tx.amount_paid,
                currency=tx.payment_currency,
                format=tx.payment_format,
                is_laundering=tx.is_laundering,
                pattern_id=tx.pattern_id,
            )
            stream_count += 1
            if stream_count >= max_tx:
                break
        if stream_count >= max_tx:
            break

    extractor = CandidateExtractor(
        min_k=min_cycle_length,
        max_k=max_cycle_length,
        max_cycles=max_cycles,
    )
    raw_cycles = extractor.find_simple_cycles(graph)
    candidates: List[CandidateSubgraph] = []
    for idx, cycle in enumerate(raw_cycles):
        c = extractor.extract_subgraph(
            graph=graph,
            cycle_nodes=cycle,
            hop_radius=hop_radius,
        candidate_id=f"aml_cycle_{idx:06d}"
        )
        cycle_pairs = {(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}
        on_cycle = [e for e in c.edges if (e["source"], e["target"]) in cycle_pairs]
        # A positive is a fully labelled cycle. Mixed/adjacent laundering evidence is
        # ambiguous for this binary target and is excluded, never relabelled benign.
        positive = all(any(int(e.get("is_laundering", 0)) == 1 for e in on_cycle if
                               e["source"] == u and e["target"] == v) for u, v in cycle_pairs)
        negative = all(int(e.get("is_laundering", 0)) == 0 for e in c.edges)
        if not positive and not negative:
            continue
        c.label = int(positive)
        c.pattern_id = None
        for e in c.edges:
            e.pop("is_laundering", None)
            e.pop("pattern_id", None)
        candidates.append(c)

    provenance["transactions_ingested"] = stream_count
    provenance["candidate_count"] = len(candidates)
    provenance["candidate_labels"] = {"positive": sum(c.label == 1 for c in candidates), "negative": sum(c.label == 0 for c in candidates)}
    if set(c.label for c in candidates) != {0, 1}:
        raise ValueError(f"Cohort lacks both classes after label-blind extraction: {provenance['candidate_labels']}")
    return candidates, cleanup_fn, provenance



def main():
    parser = argparse.ArgumentParser(description="Candidate Subgraph Serialization Pipeline")
    parser.add_argument("--transactions", required=True, help="Explicit AML transaction CSV path")
    parser.add_argument("--patterns", required=True, help="Explicit matching patterns file path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, required=True)
    parser.add_argument("--min-k", type=int, required=True)
    parser.add_argument("--max-k", type=int, required=True)
    parser.add_argument("--max-transactions", type=int, required=True)
    parser.add_argument("--max-cycles", type=int, required=True)
    parser.add_argument("--hop-radius", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    candidates, cleanup, provenance = build_candidate_pool(
        trans_csv_path=args.transactions,
        patterns_txt_path=args.patterns,
        max_tx=args.max_transactions,
        max_cycles=args.max_cycles,
        min_cycle_length=args.min_k,
        max_cycle_length=args.max_k,
        hop_radius=args.hop_radius,
    )
    try:
        splitter = GroupSafeSplitter(n_splits=args.n_splits, seed=args.seed)
        assignments = splitter.split(candidates)
        summary = splitter.verify_zero_leakage(candidates, assignments)
        builder = CandidateDatasetBuilder(
            n_splits=args.n_splits,
            min_k=args.min_k,
            max_k=args.max_k,
            max_cycles=args.max_cycles,
            hop_radius=args.hop_radius,
            seed=args.seed,
        )
        manifest = builder.save_dataset(
            candidates,
            assignments,
            summary,
            args.output_dir,
            dataset_name=args.dataset_name,
            source_provenance=provenance,
        )
    finally:
        if cleanup:
            cleanup()
    print(f"Wrote {len(candidates)} candidates to {manifest}")


if __name__ == "__main__":
    main()
