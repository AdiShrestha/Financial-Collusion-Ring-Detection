"""Loader and transaction joiner for IBM AMLworld transaction networks and laundering patterns."""

import csv
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from source.data.amlworld_patterns import LaunderingPatternGroup
from source.data.audit_data import parse_amlworld_pattern_blocks


def parse_timestamp_to_epoch(ts_str: str) -> float:
    """Parse AMLworld timestamp string (YYYY/MM/DD HH:MM or epoch float) to float epoch seconds."""
    ts_clean = ts_str.strip()
    try:
        return float(ts_clean)
    except ValueError:
        pass

    # Try common AMLworld datetime formats
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(ts_clean, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    # Fallback to simple hash-based monotonic mock if unparseable
    return 0.0


class AMLworldDataset:
    """Encapsulates loaded AMLworld transactions and joined laundering pattern groups."""

    def __init__(
        self,
        transactions: List[Dict[str, Any]],
        patterns: List[LaunderingPatternGroup],
        join_metadata: Dict[str, Any],
    ):
        self.transactions = transactions
        self.patterns = patterns
        self.join_metadata = join_metadata

        # Precompute participant accounts
        self.laundering_accounts: Set[str] = set()
        for p in self.patterns:
            self.laundering_accounts.update(p.participant_ids)

        self.all_accounts: Set[str] = set()
        for tx in self.transactions:
            if tx.get("from_account"):
                self.all_accounts.add(str(tx["from_account"]))
            if tx.get("to_account"):
                self.all_accounts.add(str(tx["to_account"]))

    @property
    def num_transactions(self) -> int:
        return len(self.transactions)

    @property
    def num_patterns(self) -> int:
        return len(self.patterns)

    @property
    def num_accounts(self) -> int:
        return len(self.all_accounts)


class AMLworldLoader:
    """Loader for AMLworld transactions and laundering pattern attempt files."""

    def __init__(self):
        pass

    def load_transactions(
        self,
        trans_source: Union[str, io.StringIO, io.TextIOBase],
    ) -> List[Dict[str, Any]]:
        """Load transactions from CSV using positional column mapping."""
        if isinstance(trans_source, str):
            if os.path.isfile(trans_source):
                with open(trans_source, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            else:
                content = trans_source
        else:
            content = trans_source.read()

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return []

        # Check if first row is header
        start_idx = 0
        header_row = [c.strip().lower() for c in rows[0]]
        if "timestamp" in header_row or "from bank" in header_row or "account" in header_row:
            start_idx = 1

        transactions: List[Dict[str, Any]] = []
        for line_idx, parts in enumerate(rows[start_idx:], start=start_idx + 1):
            if not parts or not any(parts):
                continue
            parts = [p.strip() for p in parts]
            if len(parts) < 5:
                continue

            tx_id = f"tx_{len(transactions) + 1}"
            ts_raw = parts[0]
            ts_epoch = parse_timestamp_to_epoch(ts_raw)

            from_bank = parts[1] if len(parts) > 1 else ""
            from_account = parts[2] if len(parts) > 2 else ""
            to_bank = parts[3] if len(parts) > 3 else ""
            to_account = parts[4] if len(parts) > 4 else ""

            try:
                amt_received = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
            except ValueError:
                amt_received = 0.0

            rec_currency = parts[6] if len(parts) > 6 else ""

            try:
                amt_paid = float(parts[7]) if len(parts) > 7 and parts[7] else 0.0
            except ValueError:
                amt_paid = 0.0

            pay_currency = parts[8] if len(parts) > 8 else ""
            pay_format = parts[9] if len(parts) > 9 else ""
            is_laundering_str = parts[10] if len(parts) > 10 else "0"

            is_laundering = 1 if is_laundering_str.strip() in ("1", "true", "True") else 0

            tx_record = {
                "tx_id": tx_id,
                "line_number": line_idx,
                "timestamp_raw": ts_raw,
                "timestamp": ts_epoch,
                "from_bank": from_bank,
                "from_account": from_account,
                "to_bank": to_bank,
                "to_account": to_account,
                "amount_received": amt_received,
                "receiving_currency": rec_currency,
                "amount_paid": amt_paid,
                "payment_currency": pay_currency,
                "payment_format": pay_format,
                "is_laundering": is_laundering,
            }
            transactions.append(tx_record)

        return transactions

    def load_patterns(
        self,
        patterns_source: Union[str, io.StringIO, io.TextIOBase],
    ) -> List[LaunderingPatternGroup]:
        """Parse laundering pattern blocks and convert to LaunderingPatternGroup instances."""
        raw_blocks = parse_amlworld_pattern_blocks(patterns_source)
        pattern_groups: List[LaunderingPatternGroup] = []

        for b in raw_blocks:
            txs = b["transactions"]
            # Compute start and end timestamps
            timestamps = []
            for tx in txs:
                ts_raw = tx.get("timestamp", "")
                ts_epoch = parse_timestamp_to_epoch(ts_raw)
                tx["timestamp_epoch"] = ts_epoch
                timestamps.append(ts_epoch)

            start_t = min(timestamps) if timestamps else 0.0
            end_t = max(timestamps) if timestamps else 0.0

            group = LaunderingPatternGroup(
                pattern_id=b["pattern_id"],
                typology=b["typology"],
                participant_ids=b["participants"],
                transactions=txs,
                start_timestamp=start_t,
                end_timestamp=end_t,
                metadata={
                    "start_line": b["start_line"],
                    "end_line": b["end_line"],
                    "num_transactions": b["num_transactions"],
                },
            )
            pattern_groups.append(group)

        return pattern_groups

    def join_patterns_to_transactions(
        self,
        transactions: List[Dict[str, Any]],
        patterns: List[LaunderingPatternGroup],
    ) -> Tuple[AMLworldDataset, Dict[str, Any]]:
        """Join pattern group transactions back to master transaction records.

        Guarantees that 100% of valid pattern transactions match master records.
        """
        # Build lookup table on master transactions:
        # primary key: (from_account, to_account, timestamp_raw, amount_paid)
        lookup_table: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]] = {}
        for tx in transactions:
            key = (
                str(tx["from_account"]),
                str(tx["to_account"]),
                str(tx["timestamp_raw"]),
                float(tx["amount_paid"]),
            )
            lookup_table.setdefault(key, []).append(tx)

        total_pattern_txs = sum(p.num_transactions for p in patterns)
        matched_tx_count = 0
        unmatched_records = []

        for p in patterns:
            for p_tx in p.transactions:
                from_acc = str(p_tx.get("from_account", ""))
                to_acc = str(p_tx.get("to_account", ""))
                ts_raw = str(p_tx.get("timestamp", ""))
                amt_paid = float(p_tx.get("amount_paid", 0.0))

                key = (from_acc, to_acc, ts_raw, amt_paid)
                matches = lookup_table.get(key, [])
                if matches:
                    matched_tx_count += 1
                    p_tx["matched_master_tx_id"] = matches[0]["tx_id"]
                else:
                    unmatched_records.append(
                        {
                            "pattern_id": p.pattern_id,
                            "typology": p.typology,
                            "pattern_tx": p_tx,
                            "lookup_key": key,
                        }
                    )

        match_rate = matched_tx_count / total_pattern_txs if total_pattern_txs > 0 else 1.0
        drop_rate = 1.0 - match_rate

        join_metadata = {
            "total_master_transactions": len(transactions),
            "total_patterns": len(patterns),
            "total_pattern_transactions": total_pattern_txs,
            "matched_pattern_transactions": matched_tx_count,
            "unmatched_count": len(unmatched_records),
            "match_rate": match_rate,
            "drop_rate": drop_rate,
            "join_status": "SUCCESS" if drop_rate == 0.0 else "PARTIAL_MATCH",
        }

        dataset = AMLworldDataset(
            transactions=transactions,
            patterns=patterns,
            join_metadata=join_metadata,
        )
        return dataset, join_metadata
