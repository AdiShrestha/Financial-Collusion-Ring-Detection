"""Loader for Elliptic++ Actor (wallet) graph datasets."""

import csv
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class EllipticActorsLoader:
    """Loader for Elliptic++ Actors wallet features, class labels, and transaction edgelists."""

    CLASS_MAP = {
        "1": "illicit",
        "2": "licit",
        "3": "unknown",
        "illicit": "illicit",
        "licit": "licit",
        "unknown": "unknown",
    }

    def __init__(self, expected_num_features: int = 56):
        self.expected_num_features = expected_num_features

    def load_features(
        self,
        features_source: Union[str, io.StringIO, io.TextIOBase],
    ) -> Dict[str, Dict[str, Any]]:
        """Load wallet features table (wallet_id -> {timestep, features: list[float]})."""
        if isinstance(features_source, str):
            if os.path.isfile(features_source):
                with open(features_source, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                content = features_source
        else:
            content = features_source.read()

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return {}

        start_idx = 0
        header = [c.strip().lower() for c in rows[0]]
        if "wallet_id" in header or "address" in header or "timestep" in header:
            start_idx = 1

        wallet_features: Dict[str, Dict[str, Any]] = {}
        for line_num, parts in enumerate(rows[start_idx:], start=start_idx + 1):
            if not parts or not any(parts):
                continue
            parts = [p.strip() for p in parts]
            if len(parts) < 2:
                continue

            wallet_id = parts[0]
            try:
                timestep = int(parts[1])
            except ValueError:
                timestep = 1

            raw_feats = parts[2:]
            features = []
            for feat_str in raw_feats:
                try:
                    features.append(float(feat_str))
                except ValueError:
                    features.append(0.0)

            # Pad or truncate to expected feature count if needed
            if len(features) < self.expected_num_features:
                features.extend([0.0] * (self.expected_num_features - len(features)))
            elif len(features) > self.expected_num_features:
                features = features[: self.expected_num_features]

            wallet_features[wallet_id] = {
                "wallet_id": wallet_id,
                "timestep": timestep,
                "features": features,
            }

        return wallet_features

    def load_classes(
        self,
        classes_source: Union[str, io.StringIO, io.TextIOBase],
    ) -> Dict[str, str]:
        """Load wallet class annotations (wallet_id -> class_label)."""
        if isinstance(classes_source, str):
            if os.path.isfile(classes_source):
                with open(classes_source, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                content = classes_source
        else:
            content = classes_source.read()

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return {}

        start_idx = 0
        header = [c.strip().lower() for c in rows[0]]
        if "wallet_id" in header or "address" in header or "class" in header:
            start_idx = 1

        wallet_classes: Dict[str, str] = {}
        for parts in rows[start_idx:]:
            if not parts or not any(parts):
                continue
            parts = [p.strip() for p in parts]
            if len(parts) < 2:
                continue

            wallet_id = parts[0]
            raw_class = parts[1].strip()
            norm_class = self.CLASS_MAP.get(raw_class, "unknown")
            wallet_classes[wallet_id] = norm_class

        return wallet_classes

    def load_edgelist(
        self,
        edgelist_source: Union[str, io.StringIO, io.TextIOBase],
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Load Address-Address transaction edgelist."""
        if isinstance(edgelist_source, str):
            if os.path.isfile(edgelist_source):
                with open(edgelist_source, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                content = edgelist_source
        else:
            content = edgelist_source.read()

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return []

        start_idx = 0
        header = [c.strip().lower() for c in rows[0]]
        if "input_address" in header or "source" in header or "from" in header:
            start_idx = 1

        edgelist: List[Tuple[str, str, Dict[str, Any]]] = []
        for parts in rows[start_idx:]:
            if not parts or not any(parts):
                continue
            parts = [p.strip() for p in parts]
            if len(parts) < 2:
                continue

            src = parts[0]
            dst = parts[1]
            tx_attr = {}
            if len(parts) > 2:
                tx_attr["tx_hash"] = parts[2]
            if len(parts) > 3:
                try:
                    tx_attr["timestep"] = int(parts[3])
                except ValueError:
                    pass

            edgelist.append((src, dst, tx_attr))

        return edgelist
