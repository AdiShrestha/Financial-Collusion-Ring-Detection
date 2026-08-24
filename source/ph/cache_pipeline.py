"""Persistent homology feature caching pipeline with cryptographic verification and group-safe split audits."""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from source.data.candidate_extractor import CandidateExample
from source.ph.persistence_extractor import PersistenceExtractor
from source.ph.ph_graph_view import PHGraphView
from source.ph.vectorizer import PersistenceVectorizer


class TopologicalFeatureCache:
    """Extracts, caches, and verifies persistent homology representations on disk."""

    def __init__(self, vectorizer: Optional[PersistenceVectorizer] = None):
        self.vectorizer = vectorizer or PersistenceVectorizer()

    def build_and_save_cache(
        self,
        candidates: Sequence[CandidateExample],
        split_manifest_path: Optional[str] = None,
        cache_dir: str = "data/cache",
        filtration_type: str = "temporal",
    ) -> Dict[str, Any]:
        """Extract topological vectors, verify disjointness, and save .npz archive and manifest."""
        os.makedirs(cache_dir, exist_ok=True)

        # 1. Load split assignments if split_manifest_path provided
        split_map: Dict[str, str] = {}
        if split_manifest_path and os.path.exists(split_manifest_path):
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                splits_dict = manifest_data.get("splits", {})
                for split_name, c_ids in splits_dict.items():
                    for cid in c_ids:
                        split_map[str(cid)] = split_name

        # 2. Extract features for each candidate
        x_topo_list: List[np.ndarray] = []
        y_list: List[int] = []
        candidate_ids: List[str] = []
        split_assignments: List[str] = []

        # Track accounts per split for INV-006 disjointness check
        split_accounts: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

        for cand in candidates:
            ph_view = PHGraphView.from_candidate_example(cand, filtration_type=filtration_type)
            diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)
            vec = self.vectorizer.vectorize(diag)

            cid = str(cand.candidate_id)
            split_label = split_map.get(cid, cand.metadata.get("split", "train"))

            x_topo_list.append(vec)
            y_list.append(int(cand.target_y))
            candidate_ids.append(cid)
            split_assignments.append(split_label)

            if split_label in split_accounts:
                for p in cand.participant_ids:
                    split_accounts[split_label].add(str(p))

        # 3. Verify group-safe disjointness (INV-006)
        train_val_overlap = split_accounts["train"].intersection(split_accounts["val"])
        train_test_overlap = split_accounts["train"].intersection(split_accounts["test"])
        val_test_overlap = split_accounts["val"].intersection(split_accounts["test"])

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(
                f"Split leakage detected during caching (INV-006 violation)! "
                f"Train/Val overlap: {len(train_val_overlap)}, "
                f"Train/Test overlap: {len(train_test_overlap)}, "
                f"Val/Test overlap: {len(val_test_overlap)}"
            )

        # 4. Serialize to .npz
        x_topo = np.array(x_topo_list, dtype=np.float32) if x_topo_list else np.zeros((0, self.vectorizer.output_dim), dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.int64) if y_list else np.zeros(0, dtype=np.int64)
        cids_arr = np.array(candidate_ids, dtype=object)
        splits_arr = np.array(split_assignments, dtype=object)

        npz_filename = "topological_features.npz"
        npz_path = os.path.join(cache_dir, npz_filename)

        np.savez_compressed(
            npz_path,
            x_topo=x_topo,
            y=y_arr,
            candidate_ids=cids_arr,
            split_assignments=splits_arr,
        )

        # 5. Compute SHA-256 checksum of npz
        with open(npz_path, "rb") as f:
            npz_hash = hashlib.sha256(f.read()).hexdigest()

        # 6. Generate cache manifest
        split_counts = {
            "train": int(np.sum(splits_arr == "train")),
            "val": int(np.sum(splits_arr == "val")),
            "test": int(np.sum(splits_arr == "test")),
        }

        manifest = {
            "num_candidates": len(candidates),
            "feature_dim": int(self.vectorizer.output_dim),
            "filtration_type": filtration_type,
            "split_counts": split_counts,
            "npz_filename": npz_filename,
            "npz_checksum_sha256": npz_hash,
            "split_manifest_path": split_manifest_path,
            "disjointness_verified": True,
        }

        manifest_path = os.path.join(cache_dir, "cache_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    @classmethod
    def load_cache(cls, cache_dir: str = "data/cache") -> Dict[str, Any]:
        """Load cached topological features and verify SHA-256 integrity against manifest."""
        manifest_path = os.path.join(cache_dir, "cache_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Cache manifest not found at: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        npz_filename = manifest.get("npz_filename", "topological_features.npz")
        npz_path = os.path.join(cache_dir, npz_filename)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Cache archive not found at: {npz_path}")

        # Checksum verification
        with open(npz_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        expected_hash = manifest.get("npz_checksum_sha256")
        if actual_hash != expected_hash:
            raise ValueError(
                f"Cache checksum mismatch! Expected {expected_hash}, got {actual_hash}. "
                f"Cache file may be corrupted."
            )

        data = np.load(npz_path, allow_pickle=True)
        return {
            "x_topo": data["x_topo"],
            "y": data["y"],
            "candidate_ids": data["candidate_ids"],
            "split_assignments": data["split_assignments"],
            "manifest": manifest,
        }
