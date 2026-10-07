"""Elliptic++ Actors dataset loader.

Traces to Contract C02-04, Foundation Document 01, and Invariants INV-001/INV-012.
"""

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import networkx as nx


class EllipticActorsLoader:
    """Loads Elliptic++ actor-level transaction networks and wallet features."""

    def __init__(
        self,
        edgelist_path: str = "data/raw/elliptic_actors/AddrAddr_edgelist.csv",
        features_path: Optional[str] = "data/raw/elliptic_actors/wallets_features.csv",
        classes_path: Optional[str] = "data/raw/elliptic_actors/wallets_classes.csv",
    ):
        self.edgelist_path = edgelist_path
        self.features_path = features_path
        self.classes_path = classes_path
        self.features: Dict[str, List[float]] = {}
        self.classes: Dict[str, str] = {}
        self._features_loaded = False
        self._classes_loaded = False

    def load_wallet_features(self) -> Dict[str, List[float]]:
        """Loads wallet feature vectors from wallets_features.csv."""
        if not self.features_path or not os.path.isfile(self.features_path):
            raise FileNotFoundError(f"Wallet features file not found: {self.features_path}")

        features: Dict[str, List[float]] = {}
        with open(self.features_path, "r", encoding="utf-8", errors="strict") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or header[0] != "address" or len(header) < 3:
                raise ValueError(f"Unexpected Elliptic wallet-feature schema: {header!r}")
            for row in reader:
                if not row:
                    continue
                wallet_id = row[0].strip()
                try:
                    if len(row) != len(header) or not wallet_id:
                        raise ValueError("feature row width or wallet key is invalid")
                    feat_vec = [float(val.strip()) for val in row[1:]]
                    if not all(math.isfinite(value) for value in feat_vec):
                        raise ValueError("feature row contains a non-finite value")
                    if wallet_id in features:
                        raise ValueError(f"Repeated wallet {wallet_id}: feature rows require a time-indexed loader")
                    features[wallet_id] = feat_vec
                except ValueError as exc:
                    raise ValueError(f"Invalid wallet feature row for {wallet_id}") from exc

        self.features = features
        self._features_loaded = True
        return self.features

    def load_wallet_classes(self) -> Dict[str, str]:
        """Loads wallet classes (illicit, licit, unknown) from wallets_classes.csv."""
        if not self.classes_path or not os.path.isfile(self.classes_path):
            raise FileNotFoundError(f"Wallet classes file not found: {self.classes_path}")

        classes: Dict[str, str] = {}
        with open(self.classes_path, "r", encoding="utf-8", errors="strict") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != ["address", "class"]:
                raise ValueError(f"Unexpected Elliptic wallet-class schema: {header!r}")
            for row_idx, row in enumerate(reader, start=2):
                if len(row) != 2 or not row[0].strip() or row[1].strip() not in {"1", "2", "3"}:
                    raise ValueError(f"Invalid Elliptic wallet-class row {row_idx}: {row!r}")
                wallet_id, label = row[0].strip(), row[1].strip()
                if wallet_id in classes:
                    raise ValueError(f"Duplicate Elliptic wallet-class key at row {row_idx}: {wallet_id}")
                classes[wallet_id] = label

        self.classes = classes
        self._classes_loaded = True
        return self.classes

    def load_actor_graph(self, max_edges: Optional[int] = None) -> nx.MultiDiGraph:
        """Constructs a directed multigraph, preserving each address transaction row."""
        if not os.path.isfile(self.edgelist_path):
            raise FileNotFoundError(f"Edgelist file not found: {self.edgelist_path}")
        if max_edges is not None and max_edges <= 0:
            raise ValueError("max_edges must be positive when a row limit is supplied")

        graph = nx.MultiDiGraph()
        edge_count = 0

        with open(self.edgelist_path, "r", encoding="utf-8", errors="strict") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != ["input_address", "output_address"]:
                raise ValueError(f"Unexpected Elliptic actor edge schema: {header!r}")
            for row in reader:
                if len(row) != 2 or not row[0].strip() or not row[1].strip():
                    raise ValueError(f"Invalid two-column edgelist row {edge_count + 2}: {row!r}")
                src = row[0].strip()
                dst = row[1].strip()

                # The supplied actor edgelist provides no observed amount or time.
                edge_attrs: Dict[str, Any] = {
                    "source": src,
                    "target": dst,
                }

                graph.add_edge(src, dst, key=f"source_row_{edge_count}", **edge_attrs)
                edge_count += 1
                if max_edges and edge_count >= max_edges:
                    break

        return graph

    def get_wallet_features(self, wallet_id: str) -> Optional[List[float]]:
        """Retrieves feature vector for a specific wallet address."""
        if not self._features_loaded and self.features_path and os.path.isfile(self.features_path):
            self.load_wallet_features()
        return self.features.get(wallet_id)
