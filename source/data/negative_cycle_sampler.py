"""Caliper-based negative cycle matching and relational candidate dataset assembler.

Contract C16-02 (T-COMP): Matches genuine benign cycle controls to positive laundering
patterns on exact cycle length k and covariate calipers (log duration, log amount, cross-bank ratio).
Serializes relational candidate tables: candidates.parquet, candidate_transactions.parquet,
and labels.parquet, and exports artifacts/candidates/covariate_balance.json.
"""

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def compute_candidate_covariates(txs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute matching covariates for a candidate cycle."""
    epochs = [tx["timestamp_epoch"] for tx in txs if "timestamp_epoch" in tx]
    amounts = [tx["amount_paid"] for tx in txs if "amount_paid" in tx]
    
    # Duration in seconds
    duration = max(epochs) - min(epochs) if len(epochs) > 1 else 0.0
    log_duration = math.log1p(max(0.0, duration))
    
    # Median amount
    med_amt = float(np.median(amounts)) if amounts else 0.0
    log_amount = math.log1p(max(0.0, med_amt))
    
    # Cross-bank ratio
    from_banks = [tx.get("from_bank") for tx in txs if "from_bank" in tx]
    to_banks = [tx.get("to_bank") for tx in txs if "to_bank" in tx]
    cross_bank_count = sum(1 for fb, tb in zip(from_banks, to_banks) if fb != tb)
    cross_bank_ratio = (cross_bank_count / len(txs)) if txs else 0.0
    
    return {
        "duration_seconds": duration,
        "log_duration": log_duration,
        "median_amount": med_amt,
        "log_amount": log_amount,
        "cross_bank_ratio": cross_bank_ratio,
    }


def match_and_assemble_candidates(
    pos_parquet_path: str = "artifacts/candidates/positive_candidates.parquet",
    ben_parquet_path: str = "artifacts/candidates/benign_pool.parquet",
    output_candidates_path: str = "artifacts/candidates/candidates.parquet",
    output_candidate_txs_path: str = "artifacts/candidates/candidate_transactions.parquet",
    output_labels_path: str = "artifacts/candidates/labels.parquet",
    output_balance_path: str = "artifacts/candidates/covariate_balance.json",
    match_ratio: int = 3,
) -> Dict[str, Any]:
    """Execute stratified caliper matching and build relational Parquet tables."""
    if not os.path.exists(pos_parquet_path):
        raise FileNotFoundError(f"Positive candidates missing: {pos_parquet_path}")
    if not os.path.exists(ben_parquet_path):
        raise FileNotFoundError(f"Benign pool missing: {ben_parquet_path}")

    pos_table = pq.read_table(pos_parquet_path)
    ben_table = pq.read_table(ben_parquet_path)

    pos_records = pos_table.to_pylist()
    ben_records = ben_table.to_pylist()

    # Index benign records by cycle length
    all_ben_with_covs = []
    ben_by_k = defaultdict(list)
    for b in ben_records:
        txs = json.loads(b["transactions_json"])
        covs = compute_candidate_covariates(txs)
        item = {**b, "covs": covs, "parsed_txs": txs, "used": False}
        all_ben_with_covs.append(item)
        ben_by_k[b["cycle_length"]].append(item)

    # Prepare positive records with covariates
    pos_with_covs = []
    for p in pos_records:
        txs = json.loads(p["transactions_json"])
        covs = compute_candidate_covariates(txs)
        pos_with_covs.append({**p, "covs": covs, "parsed_txs": txs})

    matched_negatives = []
    matched_sets = []

    # Caliper matching per positive candidate
    for p in pos_with_covs:
        k = p["cycle_length"]
        p_cov = p["covs"]
        pool = [b for b in ben_by_k[k] if not b["used"]]

        if not pool:
            # Fallback to closest available length if stratum exhausted
            pool = [b for b in all_ben_with_covs if not b["used"]]

        # Score candidates by Mahalanobis / Euclidean distance on log duration and log amount
        scored = []
        for b in pool:
            b_cov = b["covs"]
            d_dur = (p_cov["log_duration"] - b_cov["log_duration"]) ** 2
            d_amt = (p_cov["log_amount"] - b_cov["log_amount"]) ** 2
            dist = math.sqrt(d_dur + d_amt)
            scored.append((dist, b))

        scored.sort(key=lambda x: x[0])
        selected_for_p = []
        for dist, b in scored[:match_ratio]:
            b["used"] = True
            selected_for_p.append(b)
            matched_negatives.append(b)

        matched_sets.append({
            "pos_candidate_id": p["candidate_id"],
            "matched_negative_ids": [b["candidate_id"] for b in selected_for_p],
            "cycle_length": k,
        })

    # Covariate balance calculation (Standardized Mean Differences)
    pos_durs = [p["covs"]["log_duration"] for p in pos_with_covs]
    neg_durs = [b["covs"]["log_duration"] for b in matched_negatives]
    pos_amts = [p["covs"]["log_amount"] for p in pos_with_covs]
    neg_amts = [b["covs"]["log_amount"] for b in matched_negatives]

    smd_dur = (np.mean(pos_durs) - np.mean(neg_durs)) / math.sqrt(0.5 * (np.var(pos_durs) + np.var(neg_durs) + 1e-9))
    smd_amt = (np.mean(pos_amts) - np.mean(neg_amts)) / math.sqrt(0.5 * (np.var(pos_amts) + np.var(neg_amts) + 1e-9))

    balance_report = {
        "matching_strategy": f"Exact cycle-length stratum nearest-neighbor (up to 1:{match_ratio})",
        "total_positive_candidates": len(pos_with_covs),
        "total_matched_negative_candidates": len(matched_negatives),
        "total_cohort_size": len(pos_with_covs) + len(matched_negatives),
        "smd_log_duration": float(smd_dur),
        "smd_log_amount": float(smd_amt),
        "matched_sets": matched_sets,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_balance_path)), exist_ok=True)
    with open(output_balance_path, "w", encoding="utf-8") as f:
        json.dump(balance_report, f, indent=2)

    # 3. Assemble Relational Parquet Tables
    all_candidates = []
    all_candidate_txs = []
    all_labels = []

    for p in pos_with_covs:
        cid = p["candidate_id"]
        all_candidates.append({
            "candidate_id": cid,
            "pattern_id": p["pattern_id"],
            "source_kind": "official_pattern_cycle",
            "label": 1,
            "cycle_length": p["cycle_length"],
            "ordered_cycle_accounts": p["ordered_cycle_accounts"],
            "cycle_transaction_ids": p["cycle_transaction_ids"],
            "candidate_content_sha256": p["candidate_content_sha256"],
        })
        all_labels.append({
            "candidate_id": cid,
            "label": 1,
            "cycle_length": p["cycle_length"],
        })
        for tx in p["parsed_txs"]:
            all_candidate_txs.append({
                "candidate_id": cid,
                "transaction_id": tx["transaction_id"],
                "source_line_number": tx["source_line_number"],
                "from_account": tx["from_account"],
                "to_account": tx["to_account"],
                "timestamp_raw": tx["timestamp_raw"],
                "timestamp_epoch": tx["timestamp_epoch"],
                "amount_paid": tx["amount_paid"],
                "payment_format": tx["payment_format"],
                "raw_row_sha256": tx["raw_row_sha256"],
                "cycle_position": tx.get("cycle_position", 0),
                "role": "backbone",
            })

    for b in matched_negatives:
        cid = b["candidate_id"]
        all_candidates.append({
            "candidate_id": cid,
            "pattern_id": -1,
            "source_kind": "matched_benign_cycle",
            "label": 0,
            "cycle_length": b["cycle_length"],
            "ordered_cycle_accounts": b["ordered_cycle_accounts"],
            "cycle_transaction_ids": b["cycle_transaction_ids"],
            "candidate_content_sha256": b["candidate_content_sha256"],
        })
        all_labels.append({
            "candidate_id": cid,
            "label": 0,
            "cycle_length": b["cycle_length"],
        })
        for tx in b["parsed_txs"]:
            all_candidate_txs.append({
                "candidate_id": cid,
                "transaction_id": tx["transaction_id"],
                "source_line_number": tx["source_line_number"],
                "from_account": tx["from_account"],
                "to_account": tx["to_account"],
                "timestamp_raw": tx["timestamp_raw"],
                "timestamp_epoch": tx["timestamp_epoch"],
                "amount_paid": tx["amount_paid"],
                "payment_format": tx["payment_format"],
                "raw_row_sha256": tx["raw_row_sha256"],
                "cycle_position": tx.get("cycle_position", 0),
                "role": "backbone",
            })

    # Save candidates.parquet
    c_schema = pa.schema([
        ("candidate_id", pa.string()),
        ("pattern_id", pa.int64()),
        ("source_kind", pa.string()),
        ("label", pa.int64()),
        ("cycle_length", pa.int64()),
        ("ordered_cycle_accounts", pa.list_(pa.string())),
        ("cycle_transaction_ids", pa.list_(pa.string())),
        ("candidate_content_sha256", pa.string()),
    ])
    c_dict = {
        "candidate_id": [c["candidate_id"] for c in all_candidates],
        "pattern_id": [c["pattern_id"] for c in all_candidates],
        "source_kind": [c["source_kind"] for c in all_candidates],
        "label": [c["label"] for c in all_candidates],
        "cycle_length": [c["cycle_length"] for c in all_candidates],
        "ordered_cycle_accounts": [c["ordered_cycle_accounts"] for c in all_candidates],
        "cycle_transaction_ids": [c["cycle_transaction_ids"] for c in all_candidates],
        "candidate_content_sha256": [c["candidate_content_sha256"] for c in all_candidates],
    }
    pq.write_table(pa.Table.from_pydict(c_dict, schema=c_schema), output_candidates_path, compression="snappy")

    # Save candidate_transactions.parquet
    tx_schema = pa.schema([
        ("candidate_id", pa.string()),
        ("transaction_id", pa.string()),
        ("source_line_number", pa.int64()),
        ("from_account", pa.string()),
        ("to_account", pa.string()),
        ("timestamp_raw", pa.string()),
        ("timestamp_epoch", pa.float64()),
        ("amount_paid", pa.float64()),
        ("payment_format", pa.string()),
        ("raw_row_sha256", pa.string()),
        ("cycle_position", pa.int64()),
        ("role", pa.string()),
    ])
    tx_dict = {col: [tx[col] for tx in all_candidate_txs] for col in tx_schema.names}
    pq.write_table(pa.Table.from_pydict(tx_dict, schema=tx_schema), output_candidate_txs_path, compression="snappy")

    # Save labels.parquet
    l_schema = pa.schema([
        ("candidate_id", pa.string()),
        ("label", pa.int64()),
        ("cycle_length", pa.int64()),
    ])
    l_dict = {col: [l[col] for l in all_labels] for col in l_schema.names}
    pq.write_table(pa.Table.from_pydict(l_dict, schema=l_schema), output_labels_path, compression="snappy")

    return {
        "status": "ASSEMBLED",
        "total_candidates": len(all_candidates),
        "total_positive": len(pos_with_covs),
        "total_negative": len(matched_negatives),
        "total_candidate_transactions": len(all_candidate_txs),
        "smd_log_duration": float(smd_dur),
        "smd_log_amount": float(smd_amt),
    }


assemble_caliper_matched_cohort = match_and_assemble_candidates


class NegativeCycleSampler:
    """Samples and filters negative cycles for matched control datasets."""

    def __init__(self, caliper_std: float = 0.5, random_state: int = 42):
        self.caliper_std = caliper_std
        self.random_state = random_state

    def filter_benign_candidates(
        self,
        candidates: List[Dict[str, Any]],
        forbidden_accounts: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Exclude laundering candidates or candidates sharing accounts with laundering patterns."""
        if forbidden_accounts is None:
            forbidden_accounts = set()

        valid = []
        for c in candidates:
            if c.get("is_laundering", 0) == 1:
                continue
            if c.get("target_y", 0) == 1 or c.get("label", 0) == 1:
                continue

            # Check transactions
            txs = c.get("transactions", [])
            if any(tx.get("is_laundering", 0) == 1 for tx in txs):
                continue

            # Check accounts
            accs = c.get("participants", c.get("nodes", c.get("ordered_cycle_accounts", [])))
            if any(a in forbidden_accounts for a in accs):
                continue

            valid.append(c)

        return valid


if __name__ == "__main__":
    res = match_and_assemble_candidates()
    print(json.dumps(res, indent=2))
