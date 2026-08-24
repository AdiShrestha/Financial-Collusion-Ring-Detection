"""Unit tests for StreamingTransactionLoader (Contract C10-01)."""

import io
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.streaming_loader import StreamingTransactionLoader, parse_timestamp_to_epoch


def test_parse_timestamp_formats():
    """Test parsing various timestamp representations to epoch seconds."""
    ts1 = parse_timestamp_to_epoch("2022/09/01 01:20")
    assert ts1 > 0.0

    ts2 = parse_timestamp_to_epoch("2022-09-01 01:20:00")
    assert ts2 > 0.0

    ts3 = parse_timestamp_to_epoch("1661990400.0")
    assert ts3 == 1661990400.0

    ts_empty = parse_timestamp_to_epoch("")
    assert ts_empty == 0.0


def test_streaming_loader_chunking():
    """Test that stream_transactions yields records in expected chunk sizes."""
    csv_data = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:01,BankA,ACC1,BankB,ACC2,100.0,USD,100.0,USD,Wire,0\n"
        "2022/09/01 00:02,BankA,ACC1,BankB,ACC2,200.0,USD,200.0,USD,Wire,0\n"
        "2022/09/01 00:03,BankB,ACC2,BankC,ACC3,300.0,USD,300.0,USD,Wire,1\n"
        "2022/09/01 00:04,BankC,ACC3,BankA,ACC1,400.0,USD,400.0,USD,Wire,1\n"
        "2022/09/01 00:05,BankA,ACC1,BankD,ACC4,500.0,USD,500.0,USD,Wire,0\n"
    )

    loader = StreamingTransactionLoader(chunk_size=2)
    chunks = list(loader.stream_transactions(io.StringIO(csv_data)))

    assert len(chunks) == 3
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert len(chunks[2]) == 1

    assert loader.total_rows_read == 5
    assert loader.valid_rows_emitted == 5
    assert loader.malformed_rows_count == 0


def test_multigraph_edge_preservation():
    """Verify that multiple transactions between the same pair retain distinct tx_ids and records."""
    csv_data = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:01,BankA,ACC_A,BankB,ACC_B,50.0,USD,50.0,USD,ACH,0\n"
        "2022/09/01 00:02,BankA,ACC_A,BankB,ACC_B,75.0,USD,75.0,USD,ACH,0\n"
        "2022/09/01 00:03,BankA,ACC_A,BankB,ACC_B,120.0,USD,120.0,USD,ACH,1\n"
    )

    loader = StreamingTransactionLoader(chunk_size=10)
    chunks = list(loader.stream_transactions(io.StringIO(csv_data)))
    records = chunks[0]

    assert len(records) == 3
    tx_ids = [r["tx_id"] for r in records]
    assert len(set(tx_ids)) == 3  # All distinct
    assert tx_ids == [1, 2, 3]

    # Check amounts and log transforms
    assert records[0]["amount_paid"] == 50.0
    assert records[0]["log_amount_paid"] == pytest.approx(math.log1p(50.0))
    assert records[1]["amount_paid"] == 75.0
    assert records[2]["is_laundering"] == 1


def test_relative_timestamp_normalization():
    """Verify timestamps are normalized relative to min timestamp."""
    csv_data = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:00,BankA,ACC1,BankB,ACC2,10.0,USD,10.0,USD,Cheque,0\n"
        "2022/09/01 01:00,BankA,ACC1,BankB,ACC2,20.0,USD,20.0,USD,Cheque,0\n"
    )

    loader = StreamingTransactionLoader(chunk_size=10)
    chunks = list(loader.stream_transactions(io.StringIO(csv_data)))
    records = chunks[0]

    assert records[0]["timestamp_rel"] == 0.0
    assert records[1]["timestamp_rel"] == pytest.approx(3600.0)


def test_malformed_row_handling_and_logging():
    """Test that malformed lines are logged and counted without crashing unless exceeding threshold."""
    csv_data = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:00,BankA,ACC1,BankB,ACC2,10.0,USD,10.0,USD,Cheque,0\n"
        "broken_row_with_few_cols\n"
        "2022/09/01 01:00,BankA,ACC1,BankB,ACC2,20.0,USD,20.0,USD,Cheque,0\n"
    )

    loader = StreamingTransactionLoader(chunk_size=10, max_malformed_tolerance=0.5)
    chunks = list(loader.stream_transactions(io.StringIO(csv_data)))
    records = chunks[0]

    assert len(records) == 2
    assert loader.total_rows_read == 3
    assert loader.malformed_rows_count == 1
    assert len(loader.malformed_row_samples) == 1
    assert loader.malformed_row_samples[0]["line_number"] == 3
