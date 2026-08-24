"""Exact positive pattern cycle reconstruction and master row multiset joiner.

Contract C15-03 (T-COMP): Parses official IBM AMLworld HI-Small pattern blocks, joins 100%
of pattern transactions to master Parquet rows with source line numbers and row hashes,
and reconstructs the 40 genuine positive cycle candidates (k in [3..12]) preserving directed
endpoint chaining.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.audit_data import parse_amlworld_pattern_blocks


def reconstruct_positive_candidates(
    patterns_path: str = "data/raw/HI-Small_Patterns.txt",
    transactions_parquet_path: str = "artifacts/raw/transactions.parquet",
    output_parquet_path: str = "artifacts/candidates/positive_candidates.parquet",
    min_length: int = 3,
    max_length: int = 12,
) -> Dict[str, Any]:
    """Reconstruct positive cycle candidates with exact directed chaining and master row lineage."""
    if not os.path.exists(patterns_path):
        raise FileNotFoundError(f"Patterns file missing: {patterns_path}")
    if not os.path.exists(transactions_parquet_path):
        raise FileNotFoundError(f"Master transactions parquet missing: {transactions_parquet_path}")

    # Load pattern blocks
    blocks = parse_amlworld_pattern_blocks(patterns_path)
    cycle_blocks = [
        b for b in blocks
        if b.get("typology", "").upper() == "CYCLE"
        and min_length <= len(b.get("participants", [])) <= max_length
    ]

    if len(cycle_blocks) != 40:
        raise RuntimeError(f"Expected exactly 40 cycle patterns with length {min_length}..{max_length}, found {len(cycle_blocks)}")

    # Load master transactions for multiset joining
    # Read relevant columns into dict of lists
    table = pq.read_table(
        transactions_parquet_path,
        columns=[
            "transaction_id",
            "source_line_number",
            "from_account",
            "to_account",
            "timestamp_raw",
            "timestamp_epoch",
            "amount_paid",
            "payment_format",
            "raw_row_sha256",
        ],
    )

    # Build multiset lookup index: (from_acc, to_acc, timestamp_raw, amount_paid, payment_format) -> list of tx metadata
    master_index: Dict[Tuple[str, str, str, float, str], List[Dict[str, Any]]] = {}
    
    p_from = table.column("from_account").to_pylist()
    p_to = table.column("to_account").to_pylist()
    p_ts = table.column("timestamp_raw").to_pylist()
    p_epoch = table.column("timestamp_epoch").to_pylist()
    p_amt = table.column("amount_paid").to_pylist()
    p_fmt = table.column("payment_format").to_pylist()
    p_txid = table.column("transaction_id").to_pylist()
    p_line = table.column("source_line_number").to_pylist()
    p_sha = table.column("raw_row_sha256").to_pylist()

    for idx in range(len(p_txid)):
        key = (p_from[idx], p_to[idx], p_ts[idx], round(float(p_amt[idx]), 2), p_fmt[idx])
        tx_meta = {
            "transaction_id": p_txid[idx],
            "source_line_number": p_line[idx],
            "timestamp_epoch": p_epoch[idx],
            "raw_row_sha256": p_sha[idx],
            "used": False,
        }
        if key not in master_index:
            master_index[key] = []
        master_index[key].append(tx_meta)

    # Match each pattern cycle
    candidates = []
    total_matched_txs = 0
    total_expected_txs = 0

    for b in cycle_blocks:
        pattern_id = b["pattern_id"]
        txs = b.get("transactions", [])
        k = len(txs)
        total_expected_txs += k

        ordered_accounts = []
        ordered_tx_ids = []
        ordered_tx_meta = []

        # Verify directed chaining
        for step in range(k):
            u = txs[step]["from_account"]
            v = txs[step]["to_account"]
            next_u = txs[(step + 1) % k]["from_account"]

            if v != next_u:
                raise RuntimeError(
                    f"Pattern {pattern_id} cycle broken at step {step}: to_account {v} != next from_account {next_u}"
                )

            ordered_accounts.append(u)

            # Match against master index
            amt = round(float(txs[step].get("amount_paid", 0.0)), 2)
            ts_raw = txs[step].get("timestamp", "")
            p_fmt = txs[step].get("payment_format", "")
            key = (u, v, ts_raw, amt, p_fmt)

            matching_rows = master_index.get(key, [])
            # Find first unused row
            matched_row = None
            for r in matching_rows:
                if not r["used"]:
                    r["used"] = True
                    matched_row = r
                    break

            if matched_row is None:
                raise RuntimeError(
                    f"Pattern {pattern_id} tx step {step} ({u} -> {v}, {ts_raw}, {amt}, {p_fmt}) not found in master Parquet table!"
                )

            total_matched_txs += 1
            ordered_tx_ids.append(matched_row["transaction_id"])
            ordered_tx_meta.append({
                "transaction_id": matched_row["transaction_id"],
                "source_line_number": matched_row["source_line_number"],
                "from_account": u,
                "to_account": v,
                "timestamp_raw": ts_raw,
                "timestamp_epoch": matched_row["timestamp_epoch"],
                "amount_paid": amt,
                "payment_format": p_fmt,
                "raw_row_sha256": matched_row["raw_row_sha256"],
                "role": "backbone",
                "cycle_position": step,
            })

        candidate_id = f"pos_cand_{pattern_id:04d}"
        content_hash = hashlib.sha256(
            json.dumps({"id": candidate_id, "accounts": ordered_accounts, "txs": ordered_tx_ids}).encode("utf-8")
        ).hexdigest()

        candidates.append({
            "candidate_id": candidate_id,
            "pattern_id": pattern_id,
            "source_kind": "official_pattern_cycle",
            "label": 1,
            "cycle_length": k,
            "ordered_cycle_accounts": ordered_accounts,
            "cycle_transaction_ids": ordered_tx_ids,
            "transactions_json": json.dumps(ordered_tx_meta),
            "candidate_content_sha256": content_hash,
        })

    # Save to Parquet
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    
    arrow_schema = pa.schema([
        ("candidate_id", pa.string()),
        ("pattern_id", pa.int64()),
        ("source_kind", pa.string()),
        ("label", pa.int64()),
        ("cycle_length", pa.int64()),
        ("ordered_cycle_accounts", pa.list_(pa.string())),
        ("cycle_transaction_ids", pa.list_(pa.string())),
        ("transactions_json", pa.string()),
        ("candidate_content_sha256", pa.string()),
    ])

    batch_dict = {
        "candidate_id": [c["candidate_id"] for c in candidates],
        "pattern_id": [c["pattern_id"] for c in candidates],
        "source_kind": [c["source_kind"] for c in candidates],
        "label": [c["label"] for c in candidates],
        "cycle_length": [c["cycle_length"] for c in candidates],
        "ordered_cycle_accounts": [c["ordered_cycle_accounts"] for c in candidates],
        "cycle_transaction_ids": [c["cycle_transaction_ids"] for c in candidates],
        "transactions_json": [c["transactions_json"] for c in candidates],
        "candidate_content_sha256": [c["candidate_content_sha256"] for c in candidates],
    }

    out_table = pa.Table.from_pydict(batch_dict, schema=arrow_schema)
    pq.write_table(out_table, output_parquet_path, compression="snappy")

    return {
        "status": "RECONSTRUCTED",
        "total_positive_candidates": len(candidates),
        "total_matched_transactions": total_matched_txs,
        "total_expected_transactions": total_expected_txs,
        "output_path": output_parquet_path,
    }


if __name__ == "__main__":
    res = reconstruct_positive_candidates()
    print(json.dumps(res, indent=2))
