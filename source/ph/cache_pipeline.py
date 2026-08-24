"""Persistent homology feature caching pipeline with cryptographic verification and group-safe split audits.

Contract C12-02 (T-DESC): Generates data/cache/topological_features.npz containing 372-dimensional
normalized persistent homology representations for all candidates in data/processed/candidates.jsonl,
with verified split assignments and archive checksums.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.candidate_extractor import CandidateExample
from source.ph.normalized_filtration import NormalizedPersistenceVectorizer
from source.ph.persistence_extractor import PersistenceExtractor
from source.ph.ph_graph_view import PHGraphView
from source.ph.vectorizer import PersistenceVectorizer


class TopologicalFeatureCache:
    """Extracts, caches, and verifies persistent homology representations on disk."""

    def __init__(
        self,
        vectorizer: Optional[Union[NormalizedPersistenceVectorizer, PersistenceVectorizer]] = None,
    ):
        self.vectorizer = vectorizer or NormalizedPersistenceVectorizer()

    def build_cache_from_jsonl(
        self,
        candidates_jsonl_path: str = "data/processed/candidates.jsonl",
        split_manifest_path: str = "data/manifests/split_manifest.json",
        output_npz_path: str = "data/cache/topological_features.npz",
    ) -> Dict[str, Any]:
        """Generate and save 372-dim topological features archive from JSONL and manifest."""
        if not os.path.exists(candidates_jsonl_path):
            raise FileNotFoundError(f"Missing candidates file: {candidates_jsonl_path}")

        split_map: Dict[str, str] = {}
        if os.path.exists(split_manifest_path):
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            splits_data = manifest_data.get("splits", {})
            for split_name, s_val in splits_data.items():
                canon_name = "val" if split_name in ("val", "validation") else split_name
                if isinstance(s_val, dict):
                    for cid in s_val.get("candidate_ids", []):
                        split_map[str(cid)] = canon_name
                elif isinstance(s_val, list):
                    for item in s_val:
                        if isinstance(item, dict):
                            cid = item.get("candidate_id")
                            if cid:
                                split_map[str(cid)] = canon_name
                        elif isinstance(item, str):
                            split_map[str(item)] = canon_name

        candidates = []
        with open(candidates_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        features_list = []
        candidate_ids = []
        split_assignments = []
        y_list = []

        norm_vec = self.vectorizer if isinstance(self.vectorizer, NormalizedPersistenceVectorizer) else NormalizedPersistenceVectorizer()

        for cand in candidates:
            cid = str(cand.get("candidate_id", ""))
            vec = norm_vec.vectorize_candidate(cand)
            split_tag = split_map.get(cid, "train")

            features_list.append(vec)
            candidate_ids.append(cid)
            split_assignments.append(split_tag)
            y_list.append(int(cand.get("label", cand.get("is_laundering", 0))))

        features_arr = np.array(features_list, dtype=np.float32)
        cand_ids_arr = np.array(candidate_ids, dtype=object)
        splits_arr = np.array(split_assignments, dtype=object)
        y_arr = np.array(y_list, dtype=np.int64)

        os.makedirs(os.path.dirname(os.path.abspath(output_npz_path)), exist_ok=True)
        np.savez_compressed(
            output_npz_path,
            features=features_arr,
            x_topo=features_arr,
            candidate_ids=cand_ids_arr,
            split_assignments=splits_arr,
            y=y_arr,
            feature_dim=np.array(372, dtype=np.int32),
        )

        with open(output_npz_path, "rb") as f:
            archive_sha256 = hashlib.sha256(f.read()).hexdigest()

        sparsity = float(np.mean(features_arr == 0.0))
        mean_norm = float(np.mean(np.linalg.norm(features_arr, axis=1))) if len(features_arr) > 0 else 0.0

        split_counts = {
            "train": int(np.sum(splits_arr == "train")),
            "val": int(np.sum(splits_arr == "val")),
            "test": int(np.sum(splits_arr == "test")),
        }

        # Save metadata JSON beside npz
        meta_path = output_npz_path.replace(".npz", "_manifest.json")
        meta_data = {
            "num_candidates": len(candidates),
            "total_candidates": len(candidates),
            "feature_dim": 372,
            "archive_path": output_npz_path,
            "archive_sha256": archive_sha256,
            "sha256": archive_sha256,
            "split_counts": split_counts,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        return meta_data

    def build_and_save_cache(
        self,
        candidates: Sequence[CandidateExample],
        split_manifest_path: Optional[str] = None,
        cache_dir: str = "data/cache",
        filtration_type: str = "temporal",
    ) -> Dict[str, Any]:
        """Extract topological vectors, verify disjointness, and save .npz archive and manifest."""
        os.makedirs(cache_dir, exist_ok=True)

        split_map: Dict[str, str] = {}
        if split_manifest_path and os.path.exists(split_manifest_path):
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                splits_dict = manifest_data.get("splits", {})
                for split_name, c_ids in splits_dict.items():
                    if isinstance(c_ids, list):
                        for cid in c_ids:
                            split_map[str(cid)] = split_name
                    elif isinstance(c_ids, dict):
                        for cid in c_ids.get("candidate_ids", []):
                            split_map[str(cid)] = split_name

        x_topo_list: List[np.ndarray] = []
        y_list: List[int] = []
        candidate_ids: List[str] = []
        split_assignments: List[str] = []

        split_accounts: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

        fallback_vec = PersistenceVectorizer() if isinstance(self.vectorizer, NormalizedPersistenceVectorizer) else self.vectorizer

        for cand in candidates:
            ph_view = PHGraphView.from_candidate_example(cand, filtration_type=filtration_type)
            diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)
            vec = fallback_vec.vectorize(diag)

            cid = str(cand.candidate_id)
            split_label = split_map.get(cid, cand.metadata.get("split", "train"))

            x_topo_list.append(vec)
            y_list.append(int(cand.target_y))
            candidate_ids.append(cid)
            split_assignments.append(split_label)

            if split_label in split_accounts:
                for p in cand.participant_ids:
                    split_accounts[split_label].add(str(p))

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

        archive_path = os.path.join(cache_dir, "topological_features.npz")
        features_arr = np.array(x_topo_list, dtype=np.float32)
        cand_ids_arr = np.array(candidate_ids, dtype=object)
        splits_arr = np.array(split_assignments, dtype=object)
        y_arr = np.array(y_list, dtype=np.int64)

        np.savez_compressed(
            archive_path,
            features=features_arr,
            x_topo=features_arr,
            candidate_ids=cand_ids_arr,
            split_assignments=splits_arr,
            y=y_arr,
            feature_dim=np.array(features_arr.shape[1] if len(features_arr) > 0 else 372, dtype=np.int32),
        )

        with open(archive_path, "rb") as f:
            archive_sha256 = hashlib.sha256(f.read()).hexdigest()

        split_counts = {
            "train": int(np.sum(splits_arr == "train")),
            "val": int(np.sum(splits_arr == "val")),
            "test": int(np.sum(splits_arr == "test")),
        }

        meta_path = os.path.join(cache_dir, "cache_manifest.json")
        meta_data = {
            "num_candidates": len(candidates),
            "total_candidates": len(candidates),
            "feature_dim": features_arr.shape[1] if len(features_arr) > 0 else 372,
            "archive_path": archive_path,
            "archive_sha256": archive_sha256,
            "sha256": archive_sha256,
            "split_counts": split_counts,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        return meta_data

    @classmethod
    def load_cache(
        cls,
        npz_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Load cached topological features archive from disk with integrity check."""
        if npz_path is None:
            if cache_dir is not None:
                npz_path = os.path.join(cache_dir, "topological_features.npz")
            else:
                npz_path = "data/cache/topological_features.npz"

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Topological features archive not found: {npz_path}")

        # Check checksum against manifest if manifest exists
        target_dir = os.path.dirname(os.path.abspath(npz_path))
        for m_name in ("cache_manifest.json", "topological_features_manifest.json"):
            m_path = os.path.join(target_dir, m_name)
            if os.path.exists(m_path):
                with open(m_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                expected_sha = m_data.get("sha256", m_data.get("archive_sha256"))
                if expected_sha:
                    with open(npz_path, "rb") as f:
                        actual_sha = hashlib.sha256(f.read()).hexdigest()
                    if actual_sha != expected_sha:
                        raise ValueError(f"Cache checksum mismatch! Expected {expected_sha}, got {actual_sha}")
                break

        data = np.load(npz_path, allow_pickle=True)
        feats = data["features"] if "features" in data else data["x_topo"]
        return {
            "features": feats,
            "x_topo": feats,
            "candidate_ids": data["candidate_ids"],
            "split_assignments": data["split_assignments"],
            "y": data["y"],
        }


if __name__ == "__main__":
    cache = TopologicalFeatureCache()
    stats = cache.build_cache_from_jsonl()
    print("Topological feature cache generated:")
    print(json.dumps(stats, indent=2))
