"""
Robust Block-Oriented Parser for IBM AMLworld HI-Small Laundering Patterns.
Implements exact parsing of BEGIN/END LAUNDERING ATTEMPT blocks without schema invention.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import csv
import json
import math
import re


@dataclass(frozen=True)
class PatternTransaction:
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
    raw_line: str

    @property
    def sender_key(self) -> str:
        return f"{self.from_bank}_{self.from_account}"

    @property
    def receiver_key(self) -> str:
        return f"{self.to_bank}_{self.to_account}"


@dataclass
class LaunderingPatternBlock:
    pattern_id: str
    sequence_num: int
    typology_raw: str
    typology: str
    transactions: List[PatternTransaction] = field(default_factory=list)

    @property
    def participant_accounts(self) -> Set[str]:
        participants = set()
        for tx in self.transactions:
            participants.add(tx.sender_key)
            participants.add(tx.receiver_key)
        return participants

    @property
    def transaction_count(self) -> int:
        return len(self.transactions)

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "sequence_num": self.sequence_num,
            "typology": self.typology,
            "typology_raw": self.typology_raw,
            "transaction_count": self.transaction_count,
            "participant_count": len(self.participant_accounts),
            "participants": sorted(list(self.participant_accounts)),
        }


def normalize_typology(raw_name: str) -> str:
    """Normalize raw typology string into canonical token."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name.strip()).strip("_").lower()
    return cleaned


def parse_transaction_line(line: str) -> Optional[PatternTransaction]:
    """Parse a single comma-delimited transaction line from AMLworld pattern file."""
    try:
        rows = list(csv.reader([line], strict=True))
    except csv.Error:
        return None
    if len(rows) != 1 or len(rows[0]) != 11:
        return None
    parts = [p.strip() for p in rows[0]]
    if any(not parts[index] for index in (0, 1, 2, 3, 4, 6, 8, 9)):
        return None
    try:
        amount_received = float(parts[5])
        amount_paid = float(parts[7])
        label = int(parts[10])
        if (
            not math.isfinite(amount_received)
            or not math.isfinite(amount_paid)
            or amount_received < 0
            or amount_paid < 0
            or label not in (0, 1)
        ):
            return None
        return PatternTransaction(
            timestamp=parts[0],
            from_bank=parts[1],
            from_account=parts[2],
            to_bank=parts[3],
            to_account=parts[4],
            amount_received=amount_received,
            receiving_currency=parts[6],
            amount_paid=amount_paid,
            payment_currency=parts[8],
            payment_format=parts[9],
            is_laundering=label,
            raw_line=line.strip(),
        )
    except (ValueError, IndexError):
        return None


def parse_amlworld_patterns(
    file_path: Path,
) -> Tuple[List[LaunderingPatternBlock], Dict[str, Any]]:
    """
    Parse a physical HI-Small_Patterns.txt file into a list of pattern blocks.
    Returns (blocks, audit_summary).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Patterns file not found at: {file_path}")

    blocks: List[LaunderingPatternBlock] = []
    current_block: Optional[LaunderingPatternBlock] = None
    seq_num = 0

    total_lines = 0
    tx_lines = 0
    malformed_lines = []
    typology_counts: Dict[str, int] = {}

    begin_pattern = re.compile(r"^BEGIN\s+LAUNDERING\s+ATTEMPT\s*-\s*(.+)$", re.IGNORECASE)
    end_pattern = re.compile(r"^END\s+LAUNDERING\s+ATTEMPT\s*-\s*(.+)$", re.IGNORECASE)

    with open(file_path, "r", encoding="utf-8", errors="strict") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            total_lines += 1
            line = raw_line.strip()
            if not line:
                continue

            begin_match = begin_pattern.match(line)
            if begin_match:
                if current_block is not None:
                    malformed_lines.append(
                        f"Line {line_idx}: Unexpected BEGIN without preceding END for {current_block.pattern_id}"
                    )
                    blocks.append(current_block)
                seq_num += 1
                raw_type = begin_match.group(1).strip()
                norm_type = normalize_typology(raw_type)
                current_block = LaunderingPatternBlock(
                    pattern_id=f"AML_PATTERN_{seq_num:05d}_{norm_type}",
                    sequence_num=seq_num,
                    typology_raw=raw_type,
                    typology=norm_type,
                )
                typology_counts[norm_type] = typology_counts.get(norm_type, 0) + 1
                continue

            end_match = end_pattern.match(line)
            if end_match:
                if current_block is None:
                    malformed_lines.append(f"Line {line_idx}: Unexpected END without active block")
                else:
                    end_typology = normalize_typology(end_match.group(1))
                    begin_family = normalize_typology(current_block.typology_raw.split(":", 1)[0])
                    if end_typology != begin_family:
                        malformed_lines.append(
                            f"Line {line_idx}: END typology {end_match.group(1)!r} does not match "
                            f"BEGIN typology {current_block.typology_raw!r}"
                        )
                    blocks.append(current_block)
                    current_block = None
                continue

            # Within an active block: parse transaction line
            if current_block is not None:
                tx = parse_transaction_line(line)
                if tx is not None:
                    current_block.transactions.append(tx)
                    tx_lines += 1
                else:
                    malformed_lines.append(f"Line {line_idx}: Could not parse transaction: {line}")
            else:
                # Outside any block
                malformed_lines.append(f"Line {line_idx}: Line outside block: {line}")

    if current_block is not None:
        malformed_lines.append(f"EOF: Unterminated block {current_block.pattern_id}")
        blocks.append(current_block)

    audit_summary = {
        "source_file": str(file_path),
        "total_lines_read": total_lines,
        "total_blocks_parsed": len(blocks),
        "total_transactions_parsed": tx_lines,
        "typology_counts": typology_counts,
        "malformed_lines_count": len(malformed_lines),
        "malformed_examples": malformed_lines[:10],
    }

    return blocks, audit_summary
