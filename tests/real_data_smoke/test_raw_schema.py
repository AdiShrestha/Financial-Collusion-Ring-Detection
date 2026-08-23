"""Smoke and schema tests for IBM AMLworld pattern block parsing and Elliptic++ schema auditing."""

import io
import pytest

from source.data.audit_data import (
    audit_csv_schema,
    generate_audit_manifest,
    parse_amlworld_pattern_blocks,
)


SAMPLE_AMLWORLD_PATTERNS = """
# IBM AMLworld Pattern Export Sample
BEGIN LAUNDERING ATTEMPT - FAN-OUT
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,012,1001,012,2001,500.0,US Dollar,500.0,US Dollar,Reinvestment,1
2022/09/01 00:25,012,1001,012,2002,450.0,US Dollar,450.0,US Dollar,Reinvestment,1
2022/09/01 00:30,012,1001,012,2003,600.0,US Dollar,600.0,US Dollar,Reinvestment,1
END LAUNDERING ATTEMPT

BEGIN LAUNDERING ATTEMPT - CYCLE
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/02 10:00,001,ACC_A,001,ACC_B,1000.0,Euro,1000.0,Euro,Cheque,1
2022/09/02 11:00,001,ACC_B,001,ACC_C,980.0,Euro,980.0,Euro,Cheque,1
2022/09/02 12:00,001,ACC_C,001,ACC_A,950.0,Euro,950.0,Euro,Cheque,1
END LAUNDERING ATTEMPT
"""

SAMPLE_UNCLOSED_PATTERN = """
BEGIN LAUNDERING ATTEMPT - FAN-IN
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,012,1001,012,2001,500.0,US Dollar,500.0,US Dollar,Reinvestment,1
# Missing END LAUNDERING ATTEMPT
"""

SAMPLE_ORPHAN_END_PATTERN = """
2022/09/01 00:20,012,1001,012,2001,500.0,US Dollar,500.0,US Dollar,Reinvestment,1
END LAUNDERING ATTEMPT
"""


def test_amlworld_pattern_block_parser():
    """Verify parsing of valid AMLworld pattern attempt blocks."""
    blocks = parse_amlworld_pattern_blocks(SAMPLE_AMLWORLD_PATTERNS)
    assert len(blocks) == 2

    # Block 1: FAN-OUT
    b1 = blocks[0]
    assert b1["pattern_id"] == 1
    assert b1["typology"] == "FAN-OUT"
    assert b1["num_transactions"] == 3
    assert b1["participants"] == ["1001", "2001", "2002", "2003"]
    assert b1["transactions"][0]["from_account"] == "1001"
    assert b1["transactions"][0]["to_account"] == "2001"
    assert b1["transactions"][0]["amount_received"] == 500.0

    # Block 2: CYCLE
    b2 = blocks[1]
    assert b2["pattern_id"] == 2
    assert b2["typology"] == "CYCLE"
    assert b2["num_transactions"] == 3
    assert b2["participants"] == ["ACC_A", "ACC_B", "ACC_C"]


def test_malformed_pattern_block_detection():
    """Verify parser raises descriptive ValueError on unclosed or orphan blocks."""
    with pytest.raises(ValueError, match="Unclosed pattern block"):
        parse_amlworld_pattern_blocks(SAMPLE_UNCLOSED_PATTERN)

    with pytest.raises(ValueError, match="without matching BEGIN"):
        parse_amlworld_pattern_blocks(SAMPLE_ORPHAN_END_PATTERN)


def test_elliptic_actors_schema_checker():
    """Verify CSV schema audit on simulated Elliptic++ actor feature header."""
    # Construct simulated 56-feature header
    feature_cols = [f"feat_{i}" for i in range(56)]
    header = ["wallet_id", "timestep"] + feature_cols
    row = ["addr_xyz123", "1"] + ["0.123"] * 56
    csv_content = ",".join(header) + "\n" + ",".join(row) + "\n"

    audit = audit_csv_schema(csv_content, expected_columns=["wallet_id", "timestep", "feat_0", "feat_55"])
    assert audit["schema_valid"] is True
    assert audit["column_count"] == 58
    assert audit["row_count"] == 1
    assert len(audit["missing_expected_columns"]) == 0
    assert audit["sha256"] is not None


def test_missing_column_detection():
    """Verify that audit_csv_schema flags missing required columns."""
    csv_content = "wallet_id,timestep\naddr_1,1\n"
    audit = audit_csv_schema(csv_content, expected_columns=["wallet_id", "timestep", "class_label"])
    assert audit["schema_valid"] is False
    assert "class_label" in audit["missing_expected_columns"]


def test_manifest_generation(tmp_path):
    """Verify generate_audit_manifest writes valid JSON manifest."""
    out_file = tmp_path / "test_audit_manifest.json"
    manifest = generate_audit_manifest([], [], output_path=str(out_file))
    assert out_file.exists()
    assert manifest["factory_version"] == "2.2.0"
    assert "amlworld_sources" in manifest
    assert "elliptic_sources" in manifest
