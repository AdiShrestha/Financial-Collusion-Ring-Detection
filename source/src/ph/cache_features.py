"""
Persistent Homology Feature Caching Pipeline & Fold Dataset Integration.

Precomputes fixed-length topological persistence vectors (persistence landscapes
and summary statistics) for candidate subgraphs across cross-validation folds.
Serializes cached arrays to compressed NPZ files with cryptographic SHA-256
manifest lineage tracking.

Upholds Invariants:
- INV-001 (No Mock Data in Production): Operates on genuine candidate subgraphs.
- INV-006 (Cryptographic Lineage Tracking): Generates SHA-256 manifest of cached features.
- INV-007 (Leakage-Free Group-Safe Splitting): Preserves fold boundaries strictly.
- INV-008 (Self-Contained Verification Scripts).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Resolve imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.ph.graph_filtration import build_temporal_filtration
    from source.src.ph.gudhi_backend import GudhiPersistenceEngine
    from source.src.ph.vectorize import PersistenceLandscapeVectorizer
except ModuleNotFoundError:
    from src.ph.graph_filtration import build_temporal_filtration
    from src.ph.gudhi_backend import GudhiPersistenceEngine
    from src.ph.vectorize import PersistenceLandscapeVectorizer


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class PHFeatureCache:
    """Manages precomputation, serialization, and cryptographic validation of PH features."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        vectorizer: Optional[PersistenceLandscapeVectorizer] = None,
        persistence_engine: Optional[GudhiPersistenceEngine] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = vectorizer or PersistenceLandscapeVectorizer()
        self.persistence_engine = persistence_engine or GudhiPersistenceEngine()
        self.cached_files: Dict[str, Dict[str, Any]] = {}
        self.source_dataset_manifest_sha256: Optional[str] = None
        self.source_provenance: Optional[Dict[str, Any]] = None

    def extract_vector(self, candidate: Any) -> np.ndarray:
        """Computes a single fixed-length topological vector for a candidate subgraph."""
        ph_view = build_temporal_filtration(candidate)
        diagram = self.persistence_engine.compute_persistence(ph_view)
        return self.vectorizer.vectorize(diagram)

    def cache_fold(
        self,
        candidates: List[Any],
        fold_id: Union[int, str],
        source_fold_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extracts topological feature vectors for all candidates in a fold,
        serializes to compressed NPZ, and registers cryptographic hash.
        """
        cand_ids: List[str] = []
        labels: List[int] = []
        feature_list: List[np.ndarray] = []
        seen_ids = set()

        for cand in candidates:
            if hasattr(cand, "candidate_id"):
                cid = str(cand.candidate_id)
                lbl = getattr(cand, "label", None)
            elif isinstance(cand, dict):
                cid = str(cand["candidate_id"])
                lbl = cand.get("label")
            else:
                raise TypeError(f"Unsupported candidate type: {type(cand)}")
            if cid in seen_ids:
                raise ValueError(f"Duplicate candidate ID in fold {fold_id}: {cid}")
            seen_ids.add(cid)
            if lbl not in (0, 1):
                raise ValueError(f"Candidate {cid} requires an explicit binary label")

            vec = self.extract_vector(cand)
            if vec.shape != (self.vectorizer.feature_dim,) or not np.isfinite(vec).all():
                raise ValueError(f"Invalid PH feature vector for candidate {cid}")
            cand_ids.append(cid)
            labels.append(lbl)
            feature_list.append(vec)

        if feature_list:
            feature_matrix = np.stack(feature_list, axis=0).astype(np.float32)
        else:
            feature_matrix = np.empty((0, self.vectorizer.feature_dim), dtype=np.float32)

        out_filename = f"fold_{fold_id}_ph.npz"
        out_filepath = self.output_dir / out_filename

        np.savez_compressed(
            out_filepath,
            features=feature_matrix,
            candidate_ids=np.array(cand_ids, dtype=str),
            labels=np.array(labels, dtype=np.int64),
        )

        file_hash = compute_file_sha256(out_filepath)
        file_size = out_filepath.stat().st_size

        metadata = {
            "fold_id": str(fold_id),
            "filename": out_filename,
            "filepath": str(out_filepath),
            "sha256": file_hash,
            "num_candidates": len(cand_ids),
            "feature_dim": self.vectorizer.feature_dim,
            "size_bytes": file_size,
            "source_fold_sha256": source_fold_sha256,
        }
        self.cached_files[str(fold_id)] = metadata
        return metadata

    def load_fold(self, fold_id: Union[int, str]) -> Dict[str, Any]:
        """Loads cached fold features from disk and verifies integrity."""
        out_filename = f"fold_{fold_id}_ph.npz"
        out_filepath = self.output_dir / out_filename

        if not out_filepath.exists():
            raise FileNotFoundError(f"Cached fold file not found: {out_filepath}")

        manifest_file = self.output_dir / "ph_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"PH cache manifest is required: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        metadata = manifest.get("folds", {}).get(str(fold_id))
        if not metadata or metadata.get("sha256") != compute_file_sha256(out_filepath):
            raise ValueError(f"PH cache fold {fold_id} failed manifest integrity verification")
        with np.load(out_filepath, allow_pickle=False) as data:
            features = data["features"]
            candidate_ids = [str(x) for x in data["candidate_ids"]]
            labels = data["labels"]
        if (
            features.ndim != 2
            or features.shape[1] != self.vectorizer.feature_dim
            or len(candidate_ids) != len(labels)
            or len(set(candidate_ids)) != len(candidate_ids)
            or not np.isfinite(features).all()
            or not set(np.unique(labels).tolist()).issubset({0, 1})
        ):
            raise ValueError(f"Cached fold {fold_id} has invalid dimensions, IDs, labels, or values")

        return {
            "fold_id": str(fold_id),
            "features": features,
            "candidate_ids": candidate_ids,
            "labels": labels,
            "num_candidates": len(candidate_ids),
            "feature_dim": features.shape[1] if len(features.shape) > 1 else 0,
        }

    def write_manifest(
        self,
        manifest_filename: str = "ph_manifest.json",
        source_dataset_manifest_sha256: Optional[str] = None,
        source_provenance: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Writes cryptographic manifest recording all cached fold files."""
        manifest_path = self.output_dir / manifest_filename
        payload = {
            "manifest_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_dim": self.vectorizer.feature_dim,
            "num_landscapes": self.vectorizer.num_landscapes,
            "resolution": self.vectorizer.resolution,
            "persistence_backend": (
                "gudhi" if self.persistence_engine.has_gudhi else "algebraic_reduction"
            ),
            "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
            "source_provenance": source_provenance,
            "folds": self.cached_files,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Homology Feature Caching CLI")
    parser.add_argument("--candidates-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, required=True)
    parser.add_argument("--num-landscapes", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cand_dir = Path(args.candidates_dir)
    out_dir = Path(args.output_dir)

    vectorizer = PersistenceLandscapeVectorizer(
        num_landscapes=args.num_landscapes,
        resolution=args.resolution,
    )
    cache = PHFeatureCache(output_dir=out_dir, vectorizer=vectorizer)

    dataset_manifest_file = cand_dir / "candidates_manifest.json"
    if not dataset_manifest_file.exists():
        raise FileNotFoundError(f"Required candidate dataset manifest missing: {dataset_manifest_file}")
    dataset_manifest_sha = compute_file_sha256(dataset_manifest_file)
    dataset_manifest = json.loads(dataset_manifest_file.read_text(encoding="utf-8"))
    if dataset_manifest.get("n_splits") != args.n_splits:
        raise ValueError("Requested fold count does not match the candidate dataset manifest")
    cache.source_dataset_manifest_sha256 = dataset_manifest_sha
    cache.source_provenance = dataset_manifest.get("source_provenance")

    print(f"Starting Persistent Homology feature caching across {args.n_splits} folds...")
    print(f"Candidates dir: {cand_dir}")
    print(f"Output dir: {out_dir}")

    total_candidates = 0
    for fold in range(args.n_splits):
        fold_cand_file = cand_dir / f"fold_{fold}.json"
        if not fold_cand_file.exists():
            raise FileNotFoundError(f"Required fold file {fold_cand_file} not found")
        expected_sha = dataset_manifest.get("file_hashes", {}).get(f"fold_{fold}.json")
        if not expected_sha or compute_file_sha256(fold_cand_file) != expected_sha:
            raise ValueError(f"Candidate fold {fold} failed its dataset-manifest integrity check")

        with open(fold_cand_file, "r", encoding="utf-8") as f:
            fold_data = json.load(f)

        if not isinstance(fold_data, list):
            raise ValueError(f"Expected a list of candidates in {fold_cand_file}")
        candidates = fold_data
        expected_count = dataset_manifest.get("fold_candidate_counts", {}).get(str(fold))
        if expected_count is not None and len(candidates) != expected_count:
            raise ValueError(f"Candidate fold {fold} count differs from its dataset manifest")
        meta = cache.cache_fold(candidates, fold_id=fold, source_fold_sha256=expected_sha)
        print(f"Fold {fold}: Cached {meta['num_candidates']} candidates -> {meta['sha256'][:16]}...")
        total_candidates += meta["num_candidates"]

    manifest_file = cache.write_manifest(
        source_dataset_manifest_sha256=dataset_manifest_sha,
        source_provenance=cache.source_provenance,
    )
    print(f"Caching complete. Manifest written to {manifest_file}. Total candidates: {total_candidates}")


if __name__ == "__main__":
    main()
