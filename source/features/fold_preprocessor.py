"""Fold-isolated AML feature engineering and preprocessor pipeline.

Contract C17-01 (T-COMP): Implements AMLFeatureBuilder with strict fold isolation.
Fits feature scalers and format encoders exclusively on training partitions of each outer fold,
preventing test leakage (INV-006) and serializing preprocessors to artifacts/preprocessors/.
"""

import json
import math
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


class AMLFeatureBuilder:
    """Extracts and scales AML graph features with strict training fold isolation."""

    FORMAT_VOCAB = ["ACH", "Wire", "Cheque", "Credit Card", "Reinvestment", "Cash"]
    CURRENCY_VOCAB = ["US Dollar", "Euro", "Yen", "Yuan", "Rupee", "Canadian Dollar", "Swiss Franc", "Shekel", "Australian Dollar", "Saudi Riyal"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names: List[str] = []

    def extract_raw_candidate_features(
        self,
        candidate_record: Dict[str, Any],
        transactions: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Extract unscaled numerical and categorical feature vector for a candidate."""
        k = candidate_record["cycle_length"]
        epochs = [tx["timestamp_epoch"] for tx in transactions if "timestamp_epoch" in tx]
        amounts = [float(tx["amount_paid"]) for tx in transactions if "amount_paid" in tx]
        
        # 1. Structural features
        num_nodes = float(k)
        num_edges = float(len(transactions))
        
        # 2. Temporal dynamics
        duration = (max(epochs) - min(epochs)) if len(epochs) > 1 else 0.0
        log_duration = math.log1p(max(0.0, duration))
        
        # Inter-transaction time intervals
        sorted_epochs = sorted(epochs)
        intervals = [sorted_epochs[i+1] - sorted_epochs[i] for i in range(len(sorted_epochs)-1)]
        mean_interval = float(np.mean(intervals)) if intervals else 0.0
        std_interval = float(np.std(intervals)) if intervals else 0.0

        # 3. Monetary flow statistics
        med_amt = float(np.median(amounts)) if amounts else 0.0
        mean_amt = float(np.mean(amounts)) if amounts else 0.0
        std_amt = float(np.std(amounts)) if amounts else 0.0
        total_amt = float(np.sum(amounts)) if amounts else 0.0
        log_med_amt = math.log1p(max(0.0, med_amt))
        log_total_amt = math.log1p(max(0.0, total_amt))
        
        # Amount conservation ratio (ratio of min to max amount along cycle)
        amt_ratio = (min(amounts) / (max(amounts) + 1e-9)) if amounts else 1.0

        # 4. Bank and Payment Format Distributions
        from_banks = [tx.get("from_bank", 0) for tx in transactions]
        to_banks = [tx.get("to_bank", 0) for tx in transactions]
        cross_bank_count = sum(1 for fb, tb in zip(from_banks, to_banks) if fb != tb)
        cross_bank_ratio = (cross_bank_count / len(transactions)) if transactions else 0.0

        format_counts = defaultdict(int)
        for tx in transactions:
            fmt = tx.get("payment_format", "ACH")
            format_counts[fmt] += 1
        format_feats = [(format_counts[fmt] / len(transactions)) if transactions else 0.0 for fmt in self.FORMAT_VOCAB]

        feats = [
            num_nodes,
            num_edges,
            log_duration,
            mean_interval,
            std_interval,
            log_med_amt,
            mean_amt,
            std_amt,
            log_total_amt,
            amt_ratio,
            cross_bank_ratio,
        ] + format_feats

        return np.array(feats, dtype=np.float32)

    def fit(self, X_train: np.ndarray) -> "AMLFeatureBuilder":
        """Fit scaler exclusively on training partition."""
        self.scaler.fit(X_train)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise RuntimeError("AMLFeatureBuilder must be fitted before transforming.")
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit scaler and transform training features."""
        return self.fit(X_train).transform(X_train)


def build_fold_preprocessors(
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    candidate_txs_parquet_path: str = "artifacts/candidates/candidate_transactions.parquet",
    fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    output_dir: str = "artifacts/preprocessors",
) -> Dict[str, Any]:
    """Fit and serialize fold-safe AMLFeatureBuilder instances for each outer fold."""
    if not os.path.exists(candidates_parquet_path):
        raise FileNotFoundError(f"Missing candidates: {candidates_parquet_path}")
    if not os.path.exists(candidate_txs_parquet_path):
        raise FileNotFoundError(f"Missing candidate txs: {candidate_txs_parquet_path}")
    if not os.path.exists(fold_manifest_path):
        raise FileNotFoundError(f"Missing fold manifest: {fold_manifest_path}")

    c_table = pq.read_table(candidates_parquet_path)
    tx_table = pq.read_table(candidate_txs_parquet_path)

    candidates = c_table.to_pylist()
    tx_records = tx_table.to_pylist()

    cand_by_id = {c["candidate_id"]: c for c in candidates}
    txs_by_cand: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tx in tx_records:
        txs_by_cand[tx["candidate_id"]].append(tx)

    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for of in manifest["outer_folds"]:
        f_id = of["outer_fold_id"]
        train_ids = of["train_candidate_ids"]
        test_ids = of["test_candidate_ids"]

        builder = AMLFeatureBuilder()

        # Extract raw features
        X_train_raw = np.array([
            builder.extract_raw_candidate_features(cand_by_id[cid], txs_by_cand[cid])
            for cid in train_ids
        ], dtype=np.float32)

        X_test_raw = np.array([
            builder.extract_raw_candidate_features(cand_by_id[cid], txs_by_cand[cid])
            for cid in test_ids
        ], dtype=np.float32)

        # Fit exclusively on training data
        builder.fit(X_train_raw)

        # Save preprocessor
        out_pkl = os.path.join(output_dir, f"fold_{f_id}_preprocessor.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(builder, f)

        results[f"fold_{f_id}"] = {
            "preprocessor_path": out_pkl,
            "train_samples": len(train_ids),
            "test_samples": len(test_ids),
            "feature_dim": X_train_raw.shape[1],
        }

    return {
        "status": "PREPROCESSORS_BUILT",
        "n_outer_folds": len(manifest["outer_folds"]),
        "folds": results,
    }


if __name__ == "__main__":
    res = build_fold_preprocessors()
    print(json.dumps(res, indent=2))
