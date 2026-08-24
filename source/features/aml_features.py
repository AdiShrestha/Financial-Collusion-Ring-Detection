"""Specialized label-blind domain feature builder for AML transaction networks.

Contract C11-01: Constructs 16-dimensional node feature vectors and 8-dimensional
edge feature vectors from transaction dynamics, enforcing train-only scaler fitting
and strict label-blindness.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch


PAYMENT_FORMAT_MAP = {
    "ach": 0,
    "cheque": 1,
    "credit card": 2,
    "credit_card": 2,
    "wire": 3,
    "other": 4,
}


class AMLFeatureBuilder:
    """Label-blind feature constructor and normalizer for AML graphs."""

    def __init__(self, use_scaler: bool = True):
        self.use_scaler = use_scaler
        self.node_scaler = RobustScaler()
        self.edge_scaler = RobustScaler()
        self.is_fitted: bool = False

    def _extract_raw_node_features(
        self,
        transactions: List[Dict[str, Any]],
        node_ids: Sequence[str],
    ) -> np.ndarray:
        """Extract unscaled 16-dimensional node features from transaction list for specific nodes."""
        node_str_list = [str(n) for n in node_ids]
        node_map = {n: i for i, n in enumerate(node_str_list)}
        num_nodes = len(node_str_list)

        if num_nodes == 0:
            return np.zeros((0, 16), dtype=np.float32)

        # Accumulators per node
        in_counts = np.zeros(num_nodes, dtype=np.float32)
        out_counts = np.zeros(num_nodes, dtype=np.float32)
        in_amounts = np.zeros(num_nodes, dtype=np.float32)
        out_amounts = np.zeros(num_nodes, dtype=np.float32)
        all_amounts: List[List[float]] = [[] for _ in range(num_nodes)]
        in_counterparties: List[Set[str]] = [set() for _ in range(num_nodes)]
        out_counterparties: List[Set[str]] = [set() for _ in range(num_nodes)]
        timestamps: List[List[float]] = [[] for _ in range(num_nodes)]
        orig_banks: List[Set[str]] = [set() for _ in range(num_nodes)]
        dest_banks: List[Set[str]] = [set() for _ in range(num_nodes)]
        cross_bank_counts = np.zeros(num_nodes, dtype=np.float32)

        for tx in transactions:
            u = str(tx.get("from_account", ""))
            v = str(tx.get("to_account", ""))
            amt_paid = float(tx.get("amount_paid", tx.get("amount", 0.0)))
            amt_rec = float(tx.get("amount_received", amt_paid))
            ts = float(tx.get("timestamp_epoch", tx.get("timestamp", 0.0)))
            f_bank = str(tx.get("from_bank", ""))
            t_bank = str(tx.get("to_bank", ""))
            is_cross = 1.0 if f_bank and t_bank and f_bank != t_bank else 0.0

            if u in node_map:
                idx = node_map[u]
                out_counts[idx] += 1.0
                out_amounts[idx] += max(0.0, amt_paid)
                all_amounts[idx].append(amt_paid)
                if v:
                    out_counterparties[idx].add(v)
                if ts > 0:
                    timestamps[idx].append(ts)
                if f_bank:
                    orig_banks[idx].add(f_bank)
                if is_cross > 0:
                    cross_bank_counts[idx] += 1.0

            if v in node_map:
                idx = node_map[v]
                in_counts[idx] += 1.0
                in_amounts[idx] += max(0.0, amt_rec)
                all_amounts[idx].append(amt_rec)
                if u:
                    in_counterparties[idx].add(u)
                if ts > 0:
                    timestamps[idx].append(ts)
                if t_bank:
                    dest_banks[idx].add(t_bank)

        # Assemble 16 dimensions
        feats = np.zeros((num_nodes, 16), dtype=np.float32)
        eps = 1e-6

        for i in range(num_nodes):
            in_c = in_counts[i]
            out_c = out_counts[i]
            tot_c = in_c + out_c

            in_amt = in_amounts[i]
            out_amt = out_amounts[i]

            feats[i, 0] = in_c
            feats[i, 1] = out_c
            feats[i, 2] = math.log1p(in_amt)
            feats[i, 3] = math.log1p(out_amt)
            feats[i, 4] = (in_amt - out_amt) / (in_amt + out_amt + eps)

            amts = all_amounts[i]
            if amts:
                feats[i, 5] = float(np.mean(amts))
                feats[i, 6] = float(np.max(amts))
                feats[i, 7] = float(np.std(amts)) if len(amts) > 1 else 0.0
            else:
                feats[i, 5] = 0.0
                feats[i, 6] = 0.0
                feats[i, 7] = 0.0

            uniq_in = len(in_counterparties[i])
            uniq_out = len(out_counterparties[i])
            feats[i, 8] = float(uniq_in)
            feats[i, 9] = float(uniq_out)
            feats[i, 10] = (uniq_in + uniq_out) / (tot_c + eps)

            ts_list = timestamps[i]
            if len(ts_list) > 1:
                dur = float(max(ts_list) - min(ts_list))
                feats[i, 11] = math.log1p(max(0.0, dur))
                feats[i, 12] = dur / (tot_c + eps)
            else:
                feats[i, 11] = 0.0
                feats[i, 12] = 0.0

            feats[i, 13] = float(len(orig_banks[i]))
            feats[i, 14] = float(len(dest_banks[i]))
            feats[i, 15] = cross_bank_counts[i] / (tot_c + eps)

        return feats

    def _extract_raw_edge_features(
        self,
        transactions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Extract unscaled 8-dimensional edge features from transaction list."""
        num_edges = len(transactions)
        if num_edges == 0:
            return np.zeros((0, 8), dtype=np.float32)

        # Get temporal bounds for normalization
        all_epochs = [float(tx.get("timestamp_epoch", tx.get("timestamp", 0.0))) for tx in transactions]
        valid_epochs = [t for t in all_epochs if t > 0]
        min_ts = min(valid_epochs) if valid_epochs else 0.0
        max_ts = max(valid_epochs) if valid_epochs else 1.0
        ts_span = max(1.0, max_ts - min_ts)

        feats = np.zeros((num_edges, 8), dtype=np.float32)

        for i, tx in enumerate(transactions):
            amt = float(tx.get("amount_paid", tx.get("amount", 0.0)))
            ts = float(tx.get("timestamp_epoch", tx.get("timestamp", 0.0)))

            feats[i, 0] = math.log1p(max(0.0, amt))
            feats[i, 1] = max(0.0, min(1.0, (ts - min_ts) / ts_span)) if ts > 0 else 0.0

            # One-hot payment format (indices 2..6: ACH, Cheque, Credit Card, Wire, Other)
            fmt = str(tx.get("payment_format", "other")).strip().lower()
            fmt_idx = PAYMENT_FORMAT_MAP.get(fmt, 4)
            feats[i, 2 + fmt_idx] = 1.0

            # Currency match (index 7)
            rec_curr = str(tx.get("receiving_currency", "")).upper()
            pay_curr = str(tx.get("payment_currency", "")).upper()
            if rec_curr and pay_curr and rec_curr == pay_curr:
                feats[i, 7] = 1.0
            else:
                feats[i, 7] = 0.0

        return feats

    def fit(self, training_transactions: List[Dict[str, Any]]) -> "AMLFeatureBuilder":
        """Fit scalers exclusively on training data to prevent split leakage."""
        if not training_transactions:
            self.is_fitted = True
            return self

        # Collect unique accounts in training transactions
        accs = set()
        for tx in training_transactions:
            if tx.get("from_account"):
                accs.add(str(tx["from_account"]))
            if tx.get("to_account"):
                accs.add(str(tx["to_account"]))

        raw_nodes = self._extract_raw_node_features(training_transactions, list(accs))
        raw_edges = self._extract_raw_edge_features(training_transactions)

        if len(raw_nodes) > 0:
            self.node_scaler.fit(raw_nodes)
        if len(raw_edges) > 0:
            self.edge_scaler.fit(raw_edges)

        self.is_fitted = True
        return self

    def transform_nodes(
        self,
        transactions: List[Dict[str, Any]],
        node_ids: Sequence[str],
    ) -> torch.Tensor:
        """Transform node features into scaled torch tensor of shape [num_nodes, 16]."""
        raw = self._extract_raw_node_features(transactions, node_ids)
        if self.use_scaler and self.is_fitted and len(raw) > 0:
            try:
                scaled = self.node_scaler.transform(raw)
            except Exception:
                scaled = raw
        else:
            scaled = raw
        return torch.tensor(scaled, dtype=torch.float32)

    def transform_edges(
        self,
        transactions: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Transform edge features into scaled torch tensor of shape [num_edges, 8]."""
        raw = self._extract_raw_edge_features(transactions)
        if self.use_scaler and self.is_fitted and len(raw) > 0:
            try:
                scaled = self.edge_scaler.transform(raw)
            except Exception:
                scaled = raw
        else:
            scaled = raw
        return torch.tensor(scaled, dtype=torch.float32)

    def build_candidate_features(
        self,
        candidate: Dict[str, Any],
        is_train: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Construct full tensor dictionary for candidate example."""
        txs = candidate.get("transactions", [])
        nodes = candidate.get("participants", candidate.get("nodes", []))
        if not nodes:
            # Infer from transactions
            n_set = set()
            for tx in txs:
                if tx.get("from_account"):
                    n_set.add(str(tx["from_account"]))
                if tx.get("to_account"):
                    n_set.add(str(tx["to_account"]))
            nodes = sorted(list(n_set))

        nodes = [str(n) for n in nodes]
        node_map = {n: i for i, n in enumerate(nodes)}

        # Fit scalers on first training candidate if not yet fitted
        if is_train and not self.is_fitted:
            self.fit(txs)

        x = self.transform_nodes(txs, nodes)
        edge_attr = self.transform_edges(txs)

        # Build edge_index [2, E]
        src_indices = []
        dst_indices = []
        for tx in txs:
            u = str(tx.get("from_account", ""))
            v = str(tx.get("to_account", ""))
            if u in node_map and v in node_map:
                src_indices.append(node_map[u])
                dst_indices.append(node_map[v])

        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        label_val = candidate.get("label", candidate.get("is_laundering", 0))
        y = torch.tensor([int(label_val)], dtype=torch.long)

        return {
            "x": x,
            "edge_attr": edge_attr,
            "edge_index": edge_index,
            "y": y,
            "num_nodes": torch.tensor(len(nodes), dtype=torch.long),
        }
