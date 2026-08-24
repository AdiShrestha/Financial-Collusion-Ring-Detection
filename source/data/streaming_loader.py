"""Streaming and validated CSV-to-Parquet ingestion engine for IBM AMLworld datasets.

Contract C15-01 (T-DESC): Validates raw transaction logs, extracts source line numbers,
computes stable deterministic transaction IDs and raw row hashes, parses absolute epoch
timestamps, logs rejected records, and writes partitioned immutable Parquet tables.
"""

import csv
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple, Union
import pyarrow as pa
import pyarrow.parquet as pq


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a physical file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_timestamp_to_epoch(ts_str: str) -> float:
    """Parse transaction timestamp string to float Unix epoch seconds."""
    ts_clean = str(ts_str).strip()
    if not ts_clean:
        raise ValueError("Empty timestamp string")

    for fmt in (
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            dt = datetime.strptime(ts_clean, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    # Fallback to direct float conversion if already epoch
    try:
        return float(ts_clean)
    except ValueError:
        raise ValueError(f"Unrecognized timestamp format: {ts_str}")


class StreamingTransactionLoader:
    """Memory-bounded streaming CSV loader and validator for AML transaction data."""

    EXPECTED_HEADER_COUNT = 11

    def __init__(
        self,
        chunk_size: int = 100000,
        reject_threshold: int = 0,
    ):
        self.chunk_size = chunk_size
        self.reject_threshold = reject_threshold
        self.total_rows_read: int = 0
        self.valid_rows_emitted: int = 0
        self.rejected_rows_count: int = 0
        self.reject_ledger: List[Dict[str, Any]] = []

    def stream_csv_chunks(
        self,
        csv_source: Union[str, TextIO],
        source_sha256: Optional[str] = None,
    ) -> Iterator[Dict[str, List[Any]]]:
        """Stream validated row batches formatted as column dictionaries for PyArrow conversion."""
        self.total_rows_read = 0
        self.valid_rows_emitted = 0
        self.rejected_rows_count = 0
        self.reject_ledger = []

        is_file_path = isinstance(csv_source, str) and os.path.exists(csv_source)
        if is_file_path and source_sha256 is None:
            source_sha256 = compute_file_sha256(csv_source)
        elif source_sha256 is None:
            source_sha256 = "stream_unknown_hash"

        file_prefix = source_sha256[:8]

        f = open(csv_source, "r", encoding="utf-8", newline="") if is_file_path else csv_source
        try:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("Empty CSV source")

            header = [h.strip() for h in header]
            if len(header) != self.EXPECTED_HEADER_COUNT:
                raise ValueError(
                    f"Header column count mismatch: expected {self.EXPECTED_HEADER_COUNT}, got {len(header)}"
                )

            batch_data: Dict[str, List[Any]] = {
                "transaction_id": [],
                "source_line_number": [],
                "timestamp_raw": [],
                "timestamp_epoch": [],
                "from_bank": [],
                "from_account": [],
                "to_bank": [],
                "to_account": [],
                "amount_received": [],
                "receiving_currency": [],
                "amount_paid": [],
                "payment_currency": [],
                "payment_format": [],
                "is_laundering": [],
                "raw_row_sha256": [],
            }

            line_number = 1  # 1-indexed (line 1 was header)

            for row in reader:
                line_number += 1
                self.total_rows_read += 1

                if len(row) < self.EXPECTED_HEADER_COUNT:
                    self.rejected_rows_count += 1
                    self.reject_ledger.append({
                        "line_number": line_number,
                        "reason": f"Column count mismatch: expected {self.EXPECTED_HEADER_COUNT}, got {len(row)}",
                        "raw_row": row,
                    })
                    continue

                try:
                    ts_raw = row[0].strip()
                    ts_epoch = parse_timestamp_to_epoch(ts_raw)

                    from_bank = int(row[1].strip())
                    from_account = str(row[2].strip())
                    to_bank = int(row[3].strip())
                    to_account = str(row[4].strip())

                    amt_received = float(row[5].strip())
                    rec_currency = str(row[6].strip())
                    amt_paid = float(row[7].strip())
                    pay_currency = str(row[8].strip())
                    pay_format = str(row[9].strip())
                    is_laundering = int(row[10].strip())

                    if amt_received < 0 or amt_paid < 0:
                        raise ValueError(f"Negative amount: received={amt_received}, paid={amt_paid}")

                    if is_laundering not in (0, 1):
                        raise ValueError(f"Invalid is_laundering flag: {is_laundering}")

                    tx_id = f"tx_{file_prefix}_{line_number:08d}"
                    raw_row_bytes = ",".join(row).encode("utf-8")
                    raw_row_sha = hashlib.sha256(raw_row_bytes).hexdigest()

                    batch_data["transaction_id"].append(tx_id)
                    batch_data["source_line_number"].append(line_number)
                    batch_data["timestamp_raw"].append(ts_raw)
                    batch_data["timestamp_epoch"].append(ts_epoch)
                    batch_data["from_bank"].append(from_bank)
                    batch_data["from_account"].append(from_account)
                    batch_data["to_bank"].append(to_bank)
                    batch_data["to_account"].append(to_account)
                    batch_data["amount_received"].append(amt_received)
                    batch_data["receiving_currency"].append(rec_currency)
                    batch_data["amount_paid"].append(amt_paid)
                    batch_data["payment_currency"].append(pay_currency)
                    batch_data["payment_format"].append(pay_format)
                    batch_data["is_laundering"].append(is_laundering)
                    batch_data["raw_row_sha256"].append(raw_row_sha)

                    self.valid_rows_emitted += 1

                except Exception as ex:
                    self.rejected_rows_count += 1
                    self.reject_ledger.append({
                        "line_number": line_number,
                        "reason": str(ex),
                        "raw_row": row,
                    })

                if len(batch_data["transaction_id"]) >= self.chunk_size:
                    yield batch_data
                    batch_data = {k: [] for k in batch_data}

            if batch_data["transaction_id"]:
                yield batch_data

        finally:
            if is_file_path:
                f.close()

        if self.rejected_rows_count > self.reject_threshold:
            raise RuntimeError(
                f"Ingestion reject threshold exceeded: {self.rejected_rows_count} rejected rows (threshold: {self.reject_threshold})"
            )


def convert_csv_to_parquet(
    csv_path: str = "data/raw/HI-Small_Trans.csv",
    output_parquet: str = "artifacts/raw/transactions.parquet",
    reject_ledger_path: str = "artifacts/raw/reject_ledger.json",
    chunk_size: int = 100000,
) -> Dict[str, Any]:
    """Execute complete validated streaming conversion from raw CSV to master Parquet table."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV file missing: {csv_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(reject_ledger_path)), exist_ok=True)

    file_sha256 = compute_file_sha256(csv_path)
    loader = StreamingTransactionLoader(chunk_size=chunk_size, reject_threshold=0)

    arrow_schema = pa.schema([
        ("transaction_id", pa.string()),
        ("source_line_number", pa.int64()),
        ("timestamp_raw", pa.string()),
        ("timestamp_epoch", pa.float64()),
        ("from_bank", pa.int64()),
        ("from_account", pa.string()),
        ("to_bank", pa.int64()),
        ("to_account", pa.string()),
        ("amount_received", pa.float64()),
        ("receiving_currency", pa.string()),
        ("amount_paid", pa.float64()),
        ("payment_currency", pa.string()),
        ("payment_format", pa.string()),
        ("is_laundering", pa.int64()),
        ("raw_row_sha256", pa.string()),
    ])

    writer = None
    try:
        for chunk_data in loader.stream_csv_chunks(csv_path, source_sha256=file_sha256):
            table = pa.Table.from_pydict(chunk_data, schema=arrow_schema)
            if writer is None:
                writer = pq.ParquetWriter(output_parquet, arrow_schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    # Save reject ledger
    with open(reject_ledger_path, "w", encoding="utf-8") as f:
        json.dump({
            "csv_path": csv_path,
            "source_sha256": file_sha256,
            "total_rows_read": loader.total_rows_read,
            "valid_rows_emitted": loader.valid_rows_emitted,
            "rejected_rows_count": loader.rejected_rows_count,
            "reject_samples": loader.reject_ledger[:100],
        }, f, indent=2)

    return {
        "status": "CONVERTED",
        "csv_path": csv_path,
        "source_sha256": file_sha256,
        "parquet_path": output_parquet,
        "reject_ledger_path": reject_ledger_path,
        "total_rows_read": loader.total_rows_read,
        "valid_rows_emitted": loader.valid_rows_emitted,
        "rejected_rows_count": loader.rejected_rows_count,
    }


if __name__ == "__main__":
    res = convert_csv_to_parquet()
    print("CSV to Parquet Conversion Result:")
    print(json.dumps(res, indent=2))
