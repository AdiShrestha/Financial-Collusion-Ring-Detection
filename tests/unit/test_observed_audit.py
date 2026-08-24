"""Unit tests for ObservedDataAuditor (Contract C10-02)."""

import io
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.observed_audit import ObservedDataAuditor, compute_file_sha256_and_size


def test_compute_file_sha256_and_size():
    """Verify SHA-256 and size computation on in-memory and string sources."""
    content = "Hello, AMLworld!\nSecond line.\n"
    res = compute_file_sha256_and_size(content)

    assert len(res["sha256"]) == 64
    assert res["byte_size"] == len(content.encode("utf-8"))
    assert res["line_count"] == 2


def test_observed_data_auditor_transactions():
    """Verify full audit metrics from transaction CSV data."""
    csv_data = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:01,BankA,ACC1,BankB,ACC2,100.0,USD,100.0,USD,Wire,0\n"
        "2022/09/01 00:02,BankA,ACC1,BankB,ACC2,200.0,USD,200.0,USD,Wire,0\n"
        "2022/09/01 00:03,BankB,ACC2,BankC,ACC3,300.0,EUR,300.0,EUR,ACH,1\n"
        "2022/09/01 00:04,BankC,ACC3,BankA,ACC1,400.0,USD,400.0,USD,Cheque,1\n"
    )

    auditor = ObservedDataAuditor(chunk_size=2)
    report = auditor.audit_transactions(io.StringIO(csv_data))

    assert "file_metadata" in report
    assert len(report["file_metadata"]["sha256"]) == 64

    stats = report["transaction_stats"]
    assert stats["total_transactions"] == 4
    assert stats["unique_accounts_count"] == 3
    assert stats["unique_banks_count"] == 3
    assert stats["currency_distribution"]["USD"] == 3
    assert stats["currency_distribution"]["EUR"] == 1
    assert stats["payment_format_distribution"]["Wire"] == 2

    quality = report["data_quality"]
    assert quality["malformed_rows_count"] == 0
    assert quality["duplicate_transactions_count"] == 0

    labels = report["label_distribution"]
    assert labels["laundering_count"] == 2
    assert labels["benign_count"] == 2
    assert labels["laundering_percentage"] == pytest.approx(50.0)


def test_observed_data_auditor_raw_dataset_summary():
    """Verify audit_raw_dataset combines transactions and pattern file metadata."""
    trans_csv = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:01,BankA,ACC1,BankB,ACC2,100.0,USD,100.0,USD,Wire,0\n"
    )
    patterns_txt = "BEGIN LAUNDERING ATTEMPT - CYCLE\nEND LAUNDERING ATTEMPT\n"

    auditor = ObservedDataAuditor()
    summary = auditor.audit_raw_dataset(
        trans_source=io.StringIO(trans_csv),
        patterns_source=io.StringIO(patterns_txt),
    )

    assert summary["audit_status"] == "COMPLETED"
    assert "transactions_audit" in summary
    assert "patterns_file_metadata" in summary
    assert len(summary["patterns_file_metadata"]["sha256"]) == 64
