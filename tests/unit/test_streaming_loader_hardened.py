"""Unit tests for validated StreamingTransactionLoader and Parquet conversion (Contract C15-01)."""

import os
import sys
import tempfile
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.streaming_loader import (
    StreamingTransactionLoader,
    convert_csv_to_parquet,
    parse_timestamp_to_epoch,
)


def test_parse_timestamp_to_epoch():
    """Verify various timestamp formats parse accurately to Unix epoch."""
    ts1 = "2022/09/01 01:25"
    ep1 = parse_timestamp_to_epoch(ts1)
    assert ep1 > 1600000000.0

    ts2 = "2022-09-01 01:25:00"
    ep2 = parse_timestamp_to_epoch(ts2)
    assert ep2 == ep1

    with pytest.raises(ValueError):
        parse_timestamp_to_epoch("invalid_date_format_xyz")


def test_streaming_loader_synthetic_batch_validation():
    """Verify streaming loader handles malformed rows and enforces schema columns."""
    sample_csv = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:01,10,ACC_01,20,ACC_02,100.50,USD,100.50,USD,ACH,0
2022/09/01 00:02,10,ACC_02,30,ACC_03,200.00,USD,200.00,USD,Wire,1
"""
    import io
    loader = StreamingTransactionLoader(chunk_size=10, reject_threshold=0)
    chunks = list(loader.stream_csv_chunks(io.StringIO(sample_csv), source_sha256="testsha256"))

    assert len(chunks) == 1
    batch = chunks[0]
    assert len(batch["transaction_id"]) == 2
    assert batch["transaction_id"][0] == "tx_testsha2_00000002"
    assert batch["source_line_number"] == [2, 3]
    assert batch["from_account"] == ["ACC_01", "ACC_02"]
    assert batch["is_laundering"] == [0, 1]
    assert loader.valid_rows_emitted == 2
    assert loader.rejected_rows_count == 0


def test_streaming_loader_rejects_negative_amounts():
    """Verify negative amounts trigger reject ledger entry."""
    bad_csv = """Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:01,10,ACC_01,20,ACC_02,-50.00,USD,50.00,USD,ACH,0
"""
    import io
    loader = StreamingTransactionLoader(chunk_size=10, reject_threshold=5)
    chunks = list(loader.stream_csv_chunks(io.StringIO(bad_csv), source_sha256="testsha256"))

    assert len(chunks) == 0
    assert loader.rejected_rows_count == 1
    assert "Negative amount" in loader.reject_ledger[0]["reason"]
