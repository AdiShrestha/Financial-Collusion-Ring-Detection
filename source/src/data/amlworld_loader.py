"""Chunked streaming loader and pattern linker for IBM AMLworld HI-Small dataset.

Traces to Contract C02-03, FR-001, INV-001, INV-012.
"""

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import networkx as nx

try:
    from source.src.data.amlworld_patterns import (
        LaunderingPatternBlock,
        PatternTransaction,
        parse_amlworld_patterns,
    )
except ModuleNotFoundError:
    from src.data.amlworld_patterns import (
        LaunderingPatternBlock,
        PatternTransaction,
        parse_amlworld_patterns,
    )


@dataclass
class AMLWorldTransaction:
    timestamp: str
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: float
    receiving_currency: str
    amount_paid: float
    payment_currency: str
    payment_format: str
    is_laundering: int
    tx_id: Optional[str] = None
    pattern_id: Optional[str] = None

    @property
    def sender_key(self) -> str:
        return f"{self.from_bank}_{self.from_account}"

    @property
    def receiver_key(self) -> str:
        return f"{self.to_bank}_{self.to_account}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "from_bank": self.from_bank,
            "from_account": self.from_account,
            "to_bank": self.to_bank,
            "to_account": self.to_account,
            "amount_received": self.amount_received,
            "receiving_currency": self.receiving_currency,
            "amount_paid": self.amount_paid,
            "payment_currency": self.payment_currency,
            "payment_format": self.payment_format,
            "is_laundering": self.is_laundering,
            "tx_id": self.tx_id,
            "pattern_id": self.pattern_id,
            "sender_key": self.sender_key,
            "receiver_key": self.receiver_key,
        }


class AMLWorldLoader:
    """Out-of-core chunked streaming loader for IBM AMLworld transactions."""

    def __init__(
        self,
        trans_csv_path: str,
        patterns_txt_path: Optional[str] = None,
    ):
        self.trans_csv_path = trans_csv_path
        self.patterns_txt_path = patterns_txt_path
        self.patterns: List[LaunderingPatternBlock] = []
        self._pattern_tx_index: Dict[Tuple[str, str, str, float], str] = {}

        if patterns_txt_path:
            if not os.path.isfile(patterns_txt_path):
                raise FileNotFoundError(f"Pattern file not found: {patterns_txt_path}")
            self._load_patterns(patterns_txt_path)

    def _load_patterns(self, path: str) -> None:
        p = Path(path)
        blocks, audit_summary = parse_amlworld_patterns(p)
        if audit_summary["malformed_lines_count"]:
            raise ValueError(f"Malformed AMLworld pattern file: {audit_summary['malformed_examples']}")
        self.patterns = blocks
        self.pattern_audit = audit_summary
        for block in self.patterns:
            for ptx in block.transactions:
                sig = (ptx.sender_key, ptx.receiver_key, ptx.timestamp, round(ptx.amount_paid, 2))
                if sig in self._pattern_tx_index and self._pattern_tx_index[sig] != block.pattern_id:
                    raise ValueError(f"Ambiguous pattern transaction signature: {sig}")
                self._pattern_tx_index[sig] = block.pattern_id

    def stream_transactions(
        self,
        chunk_size: int = 100000,
    ) -> Generator[List[AMLWorldTransaction], None, None]:
        """Streams transactions in memory-bounded batches.

        Yields:
            List of parsed AMLWorldTransaction objects.
        """
        if not os.path.isfile(self.trans_csv_path):
            raise FileNotFoundError(f"Raw transaction dataset not found at {self.trans_csv_path}")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        with open(self.trans_csv_path, "r", encoding="utf-8", errors="strict") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError("Transaction CSV is empty")
            expected = ["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
            if header != expected:
                raise ValueError(f"Unexpected AMLworld CSV header: {header}")

            batch: List[AMLWorldTransaction] = []
            row_idx = 0

            for row in reader:
                if len(row) != 11:
                    raise ValueError(f"Malformed AMLworld CSV row {row_idx + 2}: expected 11 columns, got {len(row)}")
                if any(not row[i].strip() for i in (0, 1, 2, 3, 4, 6, 8, 9)):
                    raise ValueError(f"Missing required AMLworld field at CSV row {row_idx + 2}")

                try:
                    # Parse numerical fields robustly
                    amt_rec = float(row[5].replace("$", "").replace(",", "").strip())
                    amt_paid = float(row[7].replace("$", "").replace(",", "").strip())
                    is_launder = int(row[10].strip())
                    if is_launder not in (0, 1) or not (math.isfinite(amt_rec) and math.isfinite(amt_paid)) or amt_rec < 0 or amt_paid < 0:
                        raise ValueError("nonbinary label, nonfinite amount, or negative amount")
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid AMLworld numeric field at CSV row {row_idx + 2}: {row}") from exc

                tx_id = f"tx_{row_idx:08d}"
                sender = f"{row[1].strip()}_{row[2].strip()}"
                receiver = f"{row[3].strip()}_{row[4].strip()}"
                timestamp = row[0].strip()

                sig = (sender, receiver, timestamp, round(amt_paid, 2))
                pattern_id = self._pattern_tx_index.get(sig)

                tx = AMLWorldTransaction(
                    timestamp=timestamp,
                    from_bank=row[1].strip(),
                    from_account=row[2].strip(),
                    to_bank=row[3].strip(),
                    to_account=row[4].strip(),
                    amount_received=amt_rec,
                    receiving_currency=row[6].strip(),
                    amount_paid=amt_paid,
                    payment_currency=row[8].strip(),
                    payment_format=row[9].strip(),
                    is_laundering=is_launder,
                    tx_id=tx_id,
                    pattern_id=pattern_id,
                )
                batch.append(tx)
                row_idx += 1

                if len(batch) >= chunk_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def build_transaction_index(
        self,
        max_rows: Optional[int] = None,
    ) -> Dict[Tuple[str, str, str, float], int]:
        """Maps transaction signatures (sender, receiver, timestamp, amount_paid) to is_laundering label."""
        index: Dict[Tuple[str, str, str, float], int] = {}
        count = 0
        for batch in self.stream_transactions(chunk_size=50000):
            for tx in batch:
                sig = (tx.sender_key, tx.receiver_key, tx.timestamp, round(tx.amount_paid, 2))
                if sig in index and index[sig] != tx.is_laundering:
                    raise ValueError(f"Conflicting labels for duplicate transaction signature: {sig}")
                index[sig] = tx.is_laundering
                count += 1
                if max_rows and count >= max_rows:
                    return index
        return index

    def load_account_neighborhood(
        self,
        target_account_key: str,
        max_transactions: int = 1000,
    ) -> nx.MultiDiGraph:
        """Retrieves local transaction graph around target account."""
        if max_transactions <= 0:
            raise ValueError("max_transactions must be positive")
        graph = nx.MultiDiGraph()
        count = 0

        for batch in self.stream_transactions(chunk_size=50000):
            for tx in batch:
                if tx.sender_key == target_account_key or tx.receiver_key == target_account_key:
                    graph.add_node(tx.sender_key, bank=tx.from_bank, account=tx.from_account)
                    graph.add_node(tx.receiver_key, bank=tx.to_bank, account=tx.to_account)
                    graph.add_edge(
                        tx.sender_key,
                        tx.receiver_key,
                        tx_id=tx.tx_id,
                        timestamp=tx.timestamp,
                        amount=tx.amount_paid,
                        currency=tx.payment_currency,
                        format=tx.payment_format,
                        is_laundering=tx.is_laundering,
                        pattern_id=tx.pattern_id,
                    )
                    count += 1
                    if count >= max_transactions:
                        return graph

        return graph
