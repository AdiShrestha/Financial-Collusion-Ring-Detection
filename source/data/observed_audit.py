"""Observed raw data auditor and lineage parser for AMLworld transaction datasets.

Contract C10-02 (T-DESC): Computes physical file integrity metrics (exact byte size,
SHA-256 hash), distinct account counts, currency/format distributions, and quality checks.
"""

import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from source.data.streaming_loader import StreamingTransactionLoader


def compute_file_sha256_and_size(
    source: Union[str, io.TextIOBase, io.StringIO, bytes],
) -> Dict[str, Any]:
    """Compute exact SHA-256 hash, byte size, and line count for a file or stream."""
    if isinstance(source, bytes):
        byte_data = source
        sha = hashlib.sha256(byte_data).hexdigest()
        size = len(byte_data)
        line_count = len(byte_data.splitlines())
        return {"sha256": sha, "byte_size": size, "line_count": line_count}

    if isinstance(source, str) and os.path.isfile(source):
        hasher = hashlib.sha256()
        size = 0
        line_count = 0
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
                size += len(chunk)
                line_count += chunk.count(b"\n")
        return {"sha256": hasher.hexdigest(), "byte_size": size, "line_count": line_count}

    if isinstance(source, str):
        byte_data = source.encode("utf-8")
        sha = hashlib.sha256(byte_data).hexdigest()
        size = len(byte_data)
        line_count = len(source.splitlines())
        return {"sha256": sha, "byte_size": size, "line_count": line_count}

    # Stream / TextIO
    content = source.read()
    if isinstance(content, str):
        byte_data = content.encode("utf-8")
    else:
        byte_data = content
    sha = hashlib.sha256(byte_data).hexdigest()
    size = len(byte_data)
    line_count = len(byte_data.splitlines())
    return {"sha256": sha, "byte_size": size, "line_count": line_count}


class ObservedDataAuditor:
    """Audits physical file metrics, record schemas, and distributions of AML datasets."""

    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        self.loader = StreamingTransactionLoader(chunk_size=chunk_size)

    def audit_transactions(
        self,
        trans_source: Union[str, io.TextIOBase, io.StringIO],
    ) -> Dict[str, Any]:
        """Audit transaction stream for distributions and data quality."""
        # Calculate file integrity metadata
        file_meta = compute_file_sha256_and_size(trans_source)

        # Reset stream pointer if seekable
        if hasattr(trans_source, "seek"):
            trans_source.seek(0)

        total_tx = 0
        from_accounts: Set[str] = set()
        to_accounts: Set[str] = set()
        all_accounts: Set[str] = set()
        unique_banks: Set[str] = set()
        currencies: Dict[str, int] = {}
        payment_formats: Dict[str, int] = {}
        laundering_count = 0
        benign_count = 0
        seen_keys: Set[Tuple[str, str, str, float]] = set()
        duplicate_count = 0
        out_of_order_count = 0
        last_epoch = -1.0

        for chunk in self.loader.stream_transactions(trans_source):
            for tx in chunk:
                total_tx += 1
                f_acc = str(tx["from_account"])
                t_acc = str(tx["to_account"])
                f_bank = str(tx["from_bank"])
                t_bank = str(tx["to_bank"])

                from_accounts.add(f_acc)
                to_accounts.add(t_acc)
                all_accounts.add(f_acc)
                all_accounts.add(t_acc)

                if f_bank:
                    unique_banks.add(f_bank)
                if t_bank:
                    unique_banks.add(t_bank)

                # Currencies
                p_curr = tx.get("payment_currency", "USD")
                currencies[p_curr] = currencies.get(p_curr, 0) + 1

                # Formats
                p_fmt = tx.get("payment_format", "Cheque")
                payment_formats[p_fmt] = payment_formats.get(p_fmt, 0) + 1

                # Laundering label
                if tx.get("is_laundering", 0) == 1:
                    laundering_count += 1
                else:
                    benign_count += 1

                # Duplicate detection: (from_acc, to_acc, timestamp_raw, amount_paid)
                key = (f_acc, t_acc, tx.get("timestamp_raw", ""), tx.get("amount_paid", 0.0))
                if key in seen_keys:
                    duplicate_count += 1
                else:
                    seen_keys.add(key)

                # Timestamp monotonicity
                epoch = tx.get("timestamp_epoch", 0.0)
                if epoch < last_epoch and epoch > 0:
                    out_of_order_count += 1
                if epoch > 0:
                    last_epoch = epoch

        laundering_pct = (laundering_count / total_tx * 100.0) if total_tx > 0 else 0.0

        return {
            "file_metadata": file_meta,
            "transaction_stats": {
                "total_transactions": total_tx,
                "unique_accounts_count": len(all_accounts),
                "unique_from_accounts": len(from_accounts),
                "unique_to_accounts": len(to_accounts),
                "unique_banks_count": len(unique_banks),
                "currency_distribution": currencies,
                "payment_format_distribution": payment_formats,
            },
            "data_quality": {
                "missing_values_count": self.loader.malformed_rows_count,
                "malformed_rows_count": self.loader.malformed_rows_count,
                "duplicate_transactions_count": duplicate_count,
                "out_of_order_timestamps_count": out_of_order_count,
                "valid_rows_emitted": self.loader.valid_rows_emitted,
            },
            "label_distribution": {
                "laundering_count": laundering_count,
                "benign_count": benign_count,
                "laundering_percentage": laundering_pct,
            },
        }

    def audit_raw_dataset(
        self,
        trans_source: Union[str, io.TextIOBase, io.StringIO],
        patterns_source: Optional[Union[str, io.TextIOBase, io.StringIO]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive raw dataset audit across transactions and patterns."""
        tx_audit = self.audit_transactions(trans_source)

        patterns_meta = {}
        if patterns_source is not None:
            patterns_meta = compute_file_sha256_and_size(patterns_source)

        return {
            "transactions_audit": tx_audit,
            "patterns_file_metadata": patterns_meta,
            "audit_status": "COMPLETED",
        }
