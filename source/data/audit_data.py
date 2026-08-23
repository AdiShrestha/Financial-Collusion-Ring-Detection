"""Dataset raw byte schema inspection and block parsing utilities for AMLworld and Elliptic++."""

import csv
import hashlib
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


def parse_amlworld_pattern_blocks(
    source: Union[str, io.StringIO, io.TextIOBase],
) -> List[Dict[str, Any]]:
    """Parse IBM AMLworld pattern text file into structured pattern block records.

    Format specification:
        Blocks begin with: `BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>`
        Followed by optional header and transaction records.
        Blocks end with: `END LAUNDERING ATTEMPT`

    Returns:
        List of dictionaries with fields:
            pattern_id: int (1-based ordinal)
            typology: str (e.g. "FAN-OUT", "CYCLE", "GATHER-SCATTER")
            transactions: list of dicts with parsed transaction fields
            participants: list of distinct account IDs involved in this pattern
            start_line: int
            end_line: int
    """
    if isinstance(source, str):
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        else:
            lines = source.splitlines(keepends=True)
    else:
        lines = source.readlines()

    pattern_blocks: List[Dict[str, Any]] = []
    in_block = False
    current_typology: Optional[str] = None
    current_txs: List[Dict[str, Any]] = []
    current_participants: Set[str] = set()
    current_start_line = 0
    pattern_ordinal = 0
    header_fields: Optional[List[str]] = None

    begin_regex = re.compile(r"^\s*BEGIN\s+LAUNDERING\s+ATTEMPT\s*-\s*([A-Za-z0-9_\-\s]+)", re.IGNORECASE)
    end_regex = re.compile(r"^\s*END\s+LAUNDERING\s+ATTEMPT", re.IGNORECASE)

    for line_num, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        begin_match = begin_regex.match(line)
        if begin_match:
            if in_block:
                raise ValueError(
                    f"Line {line_num}: Nested or unclosed pattern block detected. "
                    f"Previous block started at line {current_start_line}."
                )
            in_block = True
            current_typology = begin_match.group(1).strip().upper()
            current_txs = []
            current_participants = set()
            current_start_line = line_num
            header_fields = None
            pattern_ordinal += 1
            continue

        if end_regex.match(line):
            if not in_block:
                raise ValueError(f"Line {line_num}: Encountered END LAUNDERING ATTEMPT without matching BEGIN.")
            in_block = False
            pattern_blocks.append(
                {
                    "pattern_id": pattern_ordinal,
                    "typology": current_typology,
                    "transactions": current_txs,
                    "participants": sorted(list(current_participants)),
                    "num_transactions": len(current_txs),
                    "start_line": current_start_line,
                    "end_line": line_num,
                }
            )
            current_typology = None
            current_txs = []
            current_participants = set()
            header_fields = None
            continue

        if in_block:
            # Check if this line is a CSV header inside the block
            parts = [p.strip() for p in line.split(",")]
            lower_parts = [p.lower() for p in parts]
            if "timestamp" in lower_parts or "from bank" in lower_parts or "account" in lower_parts:
                header_fields = parts
                continue

            # Parse transaction record
            # Standard AMLworld format: Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering
            if len(parts) >= 5:
                # Extract accounts and timestamp
                tx_dict: Dict[str, Any] = {
                    "raw_line": line,
                    "line_number": line_num,
                }
                # Standard AMLworld positional mapping:
                # 0: Timestamp, 1: From Bank, 2: From Account, 3: To Bank, 4: To Account,
                # 5: Amount Received, 6: Receiving Currency, 7: Amount Paid, 8: Payment Currency,
                # 9: Payment Format, 10: Is Laundering
                tx_dict["timestamp"] = parts[0]
                tx_dict["from_bank"] = parts[1] if len(parts) > 1 else ""
                tx_dict["from_account"] = parts[2] if len(parts) > 2 else ""
                tx_dict["to_bank"] = parts[3] if len(parts) > 3 else ""
                tx_dict["to_account"] = parts[4] if len(parts) > 4 else ""
                try:
                    tx_dict["amount_received"] = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
                except ValueError:
                    tx_dict["amount_received"] = parts[5] if len(parts) > 5 else 0.0

                try:
                    tx_dict["amount_paid"] = float(parts[7]) if len(parts) > 7 and parts[7] else 0.0
                except ValueError:
                    tx_dict["amount_paid"] = parts[7] if len(parts) > 7 else 0.0

                if len(parts) > 6:
                    tx_dict["receiving_currency"] = parts[6]
                if len(parts) > 8:
                    tx_dict["payment_currency"] = parts[8]
                if len(parts) > 9:
                    tx_dict["payment_format"] = parts[9]
                if len(parts) > 10:
                    tx_dict["is_laundering"] = parts[10]

                from_acc = tx_dict.get("from_account")
                to_acc = tx_dict.get("to_account")

                if from_acc:
                    current_participants.add(str(from_acc))
                if to_acc:
                    current_participants.add(str(to_acc))

                current_txs.append(tx_dict)

    if in_block:
        raise ValueError(
            f"Unclosed pattern block at end of file. Block started at line {current_start_line} "
            f"with typology {current_typology}."
        )

    return pattern_blocks


def audit_csv_schema(
    file_path_or_content: Union[str, io.StringIO],
    expected_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Audit CSV schema structure, column presence, null statistics, and SHA-256 hash."""
    if isinstance(file_path_or_content, str) and os.path.isfile(file_path_or_content):
        with open(file_path_or_content, "rb") as f_raw:
            file_bytes = f_raw.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        text_content = file_bytes.decode("utf-8", errors="replace")
        file_name = os.path.basename(file_path_or_content)
        file_size = len(file_bytes)
    elif isinstance(file_path_or_content, str):
        text_content = file_path_or_content
        sha256_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
        file_name = "<in_memory_string>"
        file_size = len(text_content.encode("utf-8"))
    else:
        text_content = file_path_or_content.read()
        sha256_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
        file_name = "<stream>"
        file_size = len(text_content.encode("utf-8"))

    reader = csv.reader(io.StringIO(text_content))
    rows = list(reader)

    if not rows:
        return {
            "file_name": file_name,
            "sha256": sha256_hash,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "status": "EMPTY",
        }

    header = rows[0]
    data_rows = rows[1:]
    num_cols = len(header)

    col_null_counts = {col: 0 for col in header}
    for row in data_rows:
        for idx, col in enumerate(header):
            if idx >= len(row) or not row[idx].strip():
                col_null_counts[col] += 1

    missing_expected = []
    if expected_columns:
        header_set = set(header)
        for exp in expected_columns:
            if exp not in header_set:
                missing_expected.append(exp)

    return {
        "file_name": file_name,
        "size_bytes": file_size,
        "sha256": sha256_hash,
        "row_count": len(data_rows),
        "column_count": num_cols,
        "columns": header,
        "null_counts": col_null_counts,
        "missing_expected_columns": missing_expected,
        "schema_valid": len(missing_expected) == 0,
    }


def generate_audit_manifest(
    amlworld_files: Sequence[str],
    elliptic_files: Sequence[str],
    output_path: str = "data/manifests/raw_source_audit.json",
) -> Dict[str, Any]:
    """Generate structured raw schema audit manifest."""
    manifest = {
        "schema_version": "1.0.0",
        "factory_version": "2.2.0",
        "amlworld_sources": {},
        "elliptic_sources": {},
        "audit_summary": {
            "total_files_audited": 0,
            "all_schemas_valid": True,
        },
    }

    for path in amlworld_files:
        if os.path.exists(path):
            res = audit_csv_schema(path)
            manifest["amlworld_sources"][os.path.basename(path)] = res
            manifest["audit_summary"]["total_files_audited"] += 1
            if not res["schema_valid"]:
                manifest["audit_summary"]["all_schemas_valid"] = False

    for path in elliptic_files:
        if os.path.exists(path):
            res = audit_csv_schema(path)
            manifest["elliptic_sources"][os.path.basename(path)] = res
            manifest["audit_summary"]["total_files_audited"] += 1
            if not res["schema_valid"]:
                manifest["audit_summary"]["all_schemas_valid"] = False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
