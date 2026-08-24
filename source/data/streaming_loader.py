"""Streaming and chunked CSV ingestion engine for IBM AMLworld transaction datasets.

Contract C10-01: Provides memory-bounded streaming iterator yielding chunked
batches of multigraph transactions with deterministic transaction IDs, malformed
row logging, timestamp normalization, and log1p amount transforms.
"""

import csv
import io
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, TextIO, Union


def parse_timestamp_to_epoch(ts_str: str) -> float:
    """Parse transaction timestamp string to float epoch seconds."""
    ts_clean = str(ts_str).strip()
    if not ts_clean:
        return 0.0
    try:
        return float(ts_clean)
    except ValueError:
        pass

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

    return 0.0


class StreamingTransactionLoader:
    """Memory-bounded streaming CSV loader for AML transaction data."""

    def __init__(
        self,
        chunk_size: int = 50000,
        max_malformed_tolerance: float = 0.001,
    ):
        self.chunk_size = chunk_size
        self.max_malformed_tolerance = max_malformed_tolerance
        self.total_rows_read: int = 0
        self.valid_rows_emitted: int = 0
        self.malformed_rows_count: int = 0
        self.malformed_row_samples: List[Dict[str, Any]] = []
        self._min_timestamp_epoch: Optional[float] = None

    def reset_stats(self) -> None:
        """Reset internal counter statistics."""
        self.total_rows_read = 0
        self.valid_rows_emitted = 0
        self.malformed_rows_count = 0
        self.malformed_row_samples.clear()
        self._min_timestamp_epoch = None

    def _resolve_column_indices(self, header: List[str]) -> Dict[str, int]:
        """Resolve column indices from header names supporting duplicate 'Account' headers."""
        indices: Dict[str, int] = {}
        clean_header = [h.strip().lower() for h in header]

        # Find timestamps
        for i, h in enumerate(clean_header):
            if "timestamp" in h or "time" in h or "date" in h:
                indices["timestamp"] = i
                break

        # Find From Bank & To Bank
        for i, h in enumerate(clean_header):
            if "from bank" in h or (h.startswith("from") and "bank" in h):
                indices["from_bank"] = i
            elif "to bank" in h or (h.startswith("to") and "bank" in h):
                indices["to_bank"] = i

        # Find Account columns (AMLworld standard has 'Account' then 'Account.1' or repeated 'Account')
        account_indices = [
            i for i, h in enumerate(clean_header)
            if "account" in h and "bank" not in h
        ]

        if len(account_indices) >= 2:
            indices["from_account"] = account_indices[0]
            indices["to_account"] = account_indices[1]
        elif len(account_indices) == 1:
            indices["from_account"] = account_indices[0]
            indices["to_account"] = account_indices[0]

        # Specific alias overrides if present
        for i, h in enumerate(clean_header):
            if "from_account" in h or "from account" in h:
                indices["from_account"] = i
            elif "to_account" in h or "to account" in h:
                indices["to_account"] = i
            elif "amount received" in h or "amount_received" in h or "received" in h:
                indices["amount_received"] = i
            elif "receiving currency" in h or "receiving_currency" in h:
                indices["receiving_currency"] = i
            elif "amount paid" in h or "amount_paid" in h or "paid" in h:
                indices["amount_paid"] = i
            elif "payment currency" in h or "payment_currency" in h:
                indices["payment_currency"] = i
            elif "payment format" in h or "payment_format" in h or "format" in h:
                indices["payment_format"] = i
            elif "is laundering" in h or "is_laundering" in h or "laundering" in h:
                indices["is_laundering"] = i

        # Fallback to positional mapping if key columns missing
        defaults = {
            "timestamp": 0,
            "from_bank": 1,
            "from_account": 2,
            "to_bank": 3,
            "to_account": 4,
            "amount_received": 5,
            "receiving_currency": 6,
            "amount_paid": 7,
            "payment_currency": 8,
            "payment_format": 9,
            "is_laundering": 10,
        }
        for k, v in defaults.items():
            if k not in indices:
                indices[k] = v

        return indices

    def _parse_row(
        self,
        row: List[str],
        col_idx: Dict[str, int],
        tx_id: int,
        line_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Parse single CSV row into normalized transaction dictionary."""
        max_req_idx = max(
            col_idx.get("from_account", 2),
            col_idx.get("to_account", 4),
            col_idx.get("timestamp", 0),
        )

        if len(row) <= max_req_idx:
            self.malformed_rows_count += 1
            if len(self.malformed_row_samples) < 50:
                self.malformed_row_samples.append({
                    "line_number": line_num,
                    "reason": f"Insufficient columns ({len(row)} <= {max_req_idx})",
                    "raw": ",".join(row),
                })
            return None

        # Extract values
        ts_raw = row[col_idx["timestamp"]].strip() if col_idx["timestamp"] < len(row) else ""
        ts_epoch = parse_timestamp_to_epoch(ts_raw)

        if self._min_timestamp_epoch is None or ts_epoch < self._min_timestamp_epoch:
            if ts_epoch > 0:
                self._min_timestamp_epoch = ts_epoch

        t_min = self._min_timestamp_epoch or 0.0
        ts_rel = max(0.0, ts_epoch - t_min)

        from_bank = row[col_idx["from_bank"]].strip() if col_idx["from_bank"] < len(row) else ""
        from_account = row[col_idx["from_account"]].strip() if col_idx["from_account"] < len(row) else ""
        to_bank = row[col_idx["to_bank"]].strip() if col_idx["to_bank"] < len(row) else ""
        to_account = row[col_idx["to_account"]].strip() if col_idx["to_account"] < len(row) else ""

        if not from_account or not to_account:
            self.malformed_rows_count += 1
            if len(self.malformed_row_samples) < 50:
                self.malformed_row_samples.append({
                    "line_number": line_num,
                    "reason": "Missing account identifiers",
                    "raw": ",".join(row),
                })
            return None

        # Parse numeric amounts
        amt_rec_raw = row[col_idx["amount_received"]].strip() if col_idx["amount_received"] < len(row) else "0"
        try:
            amt_received = float(amt_rec_raw) if amt_rec_raw else 0.0
        except ValueError:
            amt_received = 0.0

        amt_paid_raw = row[col_idx["amount_paid"]].strip() if col_idx["amount_paid"] < len(row) else "0"
        try:
            amt_paid = float(amt_paid_raw) if amt_paid_raw else 0.0
        except ValueError:
            amt_paid = 0.0

        rec_curr = row[col_idx["receiving_currency"]].strip() if col_idx["receiving_currency"] < len(row) else "USD"
        pay_curr = row[col_idx["payment_currency"]].strip() if col_idx["payment_currency"] < len(row) else "USD"
        pay_format = row[col_idx["payment_format"]].strip() if col_idx["payment_format"] < len(row) else "Cheque"

        # Laundering flag
        is_laundering_raw = row[col_idx["is_laundering"]].strip() if col_idx["is_laundering"] < len(row) else "0"
        try:
            is_laundering = int(is_laundering_raw) if is_laundering_raw in ("0", "1") else (1 if is_laundering_raw.lower() in ("true", "yes", "1") else 0)
        except ValueError:
            is_laundering = 0

        log_amount = math.log1p(max(0.0, amt_paid))

        return {
            "tx_id": tx_id,
            "timestamp_raw": ts_raw,
            "timestamp_epoch": ts_epoch,
            "timestamp_rel": ts_rel,
            "from_bank": from_bank,
            "from_account": from_account,
            "to_bank": to_bank,
            "to_account": to_account,
            "amount_received": amt_received,
            "receiving_currency": rec_curr,
            "amount_paid": amt_paid,
            "payment_currency": pay_curr,
            "payment_format": pay_format,
            "is_laundering": is_laundering,
            "log_amount_paid": log_amount,
            "log_amount": log_amount,
        }

    def stream_transactions(
        self,
        filepath_or_buffer: Union[str, io.TextIOBase, io.StringIO],
    ) -> Iterator[List[Dict[str, Any]]]:
        """Yield transaction batches of size `chunk_size` from CSV input."""
        self.reset_stats()

        if isinstance(filepath_or_buffer, str):
            if not os.path.exists(filepath_or_buffer):
                raise FileNotFoundError(f"File not found: {filepath_or_buffer}")
            file_handle = open(filepath_or_buffer, "r", encoding="utf-8", errors="replace")
            should_close = True
        else:
            file_handle = filepath_or_buffer
            should_close = False

        try:
            reader = csv.reader(file_handle)
            first_row = next(reader, None)
            if first_row is None:
                return

            # Check if first row is header
            first_row_clean = [c.strip().lower() for c in first_row]
            has_header = any(
                k in first_row_clean
                for k in ("timestamp", "from bank", "account", "to bank", "amount paid")
            )

            if has_header:
                col_idx = self._resolve_column_indices(first_row)
                line_offset = 2
            else:
                col_idx = self._resolve_column_indices([
                    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                    "Amount Received", "Receiving Currency", "Amount Paid",
                    "Payment Currency", "Payment Format", "Is Laundering"
                ])
                line_offset = 1
                # Process first row as data
                self.total_rows_read += 1
                rec = self._parse_row(first_row, col_idx, tx_id=1, line_num=1)
                if rec is not None:
                    self.valid_rows_emitted += 1
                    current_chunk = [rec]
                else:
                    current_chunk = []

            if not has_header:
                current_chunk = current_chunk if 'current_chunk' in locals() else []
            else:
                current_chunk = []

            tx_counter = len(current_chunk) + 1

            for row_idx, row in enumerate(reader, start=line_offset):
                if not row or not any(row):
                    continue

                self.total_rows_read += 1
                tx_record = self._parse_row(row, col_idx, tx_id=tx_counter, line_num=row_idx)

                if tx_record is not None:
                    self.valid_rows_emitted += 1
                    current_chunk.append(tx_record)
                    tx_counter += 1

                if len(current_chunk) >= self.chunk_size:
                    yield current_chunk
                    current_chunk = []

            if current_chunk:
                yield current_chunk

            # Check error tolerance threshold
            if self.total_rows_read > 0:
                malformed_ratio = self.malformed_rows_count / self.total_rows_read
                if malformed_ratio > self.max_malformed_tolerance and self.malformed_rows_count > 10:
                    raise ValueError(
                        f"Malformed row ratio {malformed_ratio:.4f} exceeded "
                        f"tolerance threshold {self.max_malformed_tolerance:.4f} "
                        f"({self.malformed_rows_count}/{self.total_rows_read} malformed rows)."
                    )

        finally:
            if should_close:
                file_handle.close()
