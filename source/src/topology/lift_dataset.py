"""
Multi-Domain Lifting Dataset Converter and Batch Collator.

Converts candidate subgraphs into unified (GraphView, SimplicialView, CellView)
multi-domain representations and provides disk caching and dataset interfaces.

Traces to Contract C03-04, FR-003, FR-004, INV-001, INV-006, INV-007, INV-008.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Union

# Resolve imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.data.candidate_extractor import CandidateSubgraph
    from source.src.topology.graph_view import GraphData, lift_to_graph
    from source.src.topology.cell_view import CellComplex, lift_to_cell_complex
    from source.src.topology.simplicial_view import (
        SimplicialComplex,
        lift_to_simplicial_complex,
    )
except ModuleNotFoundError:
    from src.data.candidate_extractor import CandidateSubgraph
    from src.topology.graph_view import GraphData, lift_to_graph
    from src.topology.cell_view import CellComplex, lift_to_cell_complex
    from src.topology.simplicial_view import (
        SimplicialComplex,
        lift_to_simplicial_complex,
    )


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class MultiDomainExample:
    """Unified container holding multi-domain topological representations."""
    candidate_id: str
    label: int
    graph: GraphData
    simplicial: SimplicialComplex
    cell: CellComplex


def lift_candidate(candidate: Union[CandidateSubgraph, Dict[str, Any]]) -> MultiDomainExample:
    """Lifts a candidate subgraph simultaneously into Graph, Simplicial, and Cell views."""
    # Ensure CandidateSubgraph instance
    if isinstance(candidate, dict):
        cand_obj = CandidateSubgraph.from_dict(candidate)
    else:
        cand_obj = candidate

    cand_id = cand_obj.candidate_id
    if cand_obj.label not in (0, 1):
        raise ValueError(f"Candidate {cand_id} requires an explicit binary label before lifting")
    label = int(cand_obj.label)

    # 1. Lift to GraphView
    graph_view = lift_to_graph(cand_obj)

    # 2. Lift to SimplicialView
    simplicial_view = lift_to_simplicial_complex(cand_obj)

    # 3. Lift to CellView
    cell_view = lift_to_cell_complex(cand_obj)

    # Invariant assertion: label consistency across all domains
    assert int(graph_view.y.item()) == label, f"Graph label mismatch: {graph_view.y.item()} != {label}"
    assert simplicial_view.label == label, f"Simplicial label mismatch: {simplicial_view.label} != {label}"
    assert cell_view.label == label, f"Cell label mismatch: {cell_view.label} != {label}"

    return MultiDomainExample(
        candidate_id=cand_id,
        label=label,
        graph=graph_view,
        simplicial=simplicial_view,
        cell=cell_view,
    )


class MultiDomainDataset:
    """Dataset class providing access to MultiDomainExample objects."""

    def __init__(self, examples: List[MultiDomainExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MultiDomainExample:
        return self.examples[idx]

    @classmethod
    def load_from_pickle(
        cls, filepath: Union[str, Path], manifest_path: Optional[Union[str, Path]] = None
    ) -> "MultiDomainDataset":
        """Load a trusted pickle only after checking its companion manifest hash."""
        p = Path(filepath)
        manifest = Path(manifest_path) if manifest_path else p.parent / "lifted_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Lifted data manifest not found: {manifest}")
        with open(manifest, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        expected = payload.get("file_hashes", {}).get(p.name)
        if not expected or compute_file_sha256(p) != expected:
            raise ValueError(f"Lifted pickle hash does not match manifest: {p}")
        # Pickle can execute code; the adjacent digest is integrity checking,
        # not an authenticity guarantee. Load artifacts only from trusted sources.
        with open(p, "rb") as f:
            examples = pickle.load(f)
        return cls(examples)


def convert_and_cache_folds(
    candidates_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_splits: int = 5,
) -> Path:
    """
    Reads serialized fold files (fold_0.json ... fold_{n-1}.json),
    lifts each candidate to (Graph, Simplicial, Cell) views,
    and caches the lifted structures to disk with a cryptographic manifest.
    """
    cand_path = Path(candidates_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_hashes: Dict[str, str] = {}
    input_file_hashes: Dict[str, str] = {}
    fold_counts: Dict[int, int] = {}
    fold_pos_counts: Dict[int, int] = {}
    total_lifted = 0

    for f in range(n_splits):
        fold_json = cand_path / f"fold_{f}.json"
        if not fold_json.exists():
            raise FileNotFoundError(f"Required candidate fold file not found: {fold_json}")
        input_file_hashes[fold_json.name] = compute_file_sha256(fold_json)

        with open(fold_json, "r", encoding="utf-8") as fp:
            candidates_data = json.load(fp)

        lifted_examples: List[MultiDomainExample] = []
        pos_count = 0

        for cand_dict in candidates_data:
            example = lift_candidate(cand_dict)
            lifted_examples.append(example)
            if example.label == 1:
                pos_count += 1

        fold_file = out_path / f"fold_{f}_lifted.pkl"
        with open(fold_file, "wb") as fp:
            pickle.dump(lifted_examples, fp, protocol=pickle.HIGHEST_PROTOCOL)

        file_hashes[f"fold_{f}_lifted.pkl"] = compute_file_sha256(fold_file)
        fold_counts[f] = len(lifted_examples)
        fold_pos_counts[f] = pos_count
        total_lifted += len(lifted_examples)

    # Write cryptographic manifest
    manifest_data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_splits": n_splits,
        "total_lifted": total_lifted,
        "fold_counts": fold_counts,
        "fold_positive_counts": fold_pos_counts,
        "file_hashes": file_hashes,
        "input_file_hashes": input_file_hashes,
    }

    manifest_path = out_path / "lifted_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, indent=2)

    return manifest_path
