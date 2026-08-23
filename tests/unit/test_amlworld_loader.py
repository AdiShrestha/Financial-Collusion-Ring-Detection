"""Unit tests for IBM AMLworld data loader, pattern parser, and transaction joiner."""

import pytest

from source.data.amlworld_loader import (
    AMLworldLoader,
    AMLworldDataset,
    parse_timestamp_to_epoch,
)
from source.data.amlworld_patterns import LaunderingPatternGroup


MOCK_TRANS_CSV = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,012,1001,012,2001,500.0,US Dollar,500.0,US Dollar,Reinvestment,1
2022/09/01 00:25,012,1001,012,2002,450.0,US Dollar,450.0,US Dollar,Reinvestment,1
2022/09/01 00:30,012,1001,012,2003,600.0,US Dollar,600.0,US Dollar,Reinvestment,1
2022/09/01 01:00,099,9991,099,9992,100.0,US Dollar,100.0,US Dollar,Cheque,0
2022/09/01 02:00,001,ACC_A,001,ACC_B,1000.0,Euro,1000.0,Euro,Wire,1
2022/09/01 03:00,001,ACC_B,001,ACC_C,980.0,Euro,980.0,Euro,Wire,1
2022/09/01 04:00,001,ACC_C,001,ACC_A,950.0,Euro,950.0,Euro,Wire,1
"""

MOCK_PATTERNS_TXT = """BEGIN LAUNDERING ATTEMPT - FAN-OUT
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,012,1001,012,2001,500.0,US Dollar,500.0,US Dollar,Reinvestment,1
2022/09/01 00:25,012,1001,012,2002,450.0,US Dollar,450.0,US Dollar,Reinvestment,1
2022/09/01 00:30,012,1001,012,2003,600.0,US Dollar,600.0,US Dollar,Reinvestment,1
END LAUNDERING ATTEMPT

BEGIN LAUNDERING ATTEMPT - CYCLE
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 02:00,001,ACC_A,001,ACC_B,1000.0,Euro,1000.0,Euro,Wire,1
2022/09/01 03:00,001,ACC_B,001,ACC_C,980.0,Euro,980.0,Euro,Wire,1
2022/09/01 04:00,001,ACC_C,001,ACC_A,950.0,Euro,950.0,Euro,Wire,1
END LAUNDERING ATTEMPT
"""


def test_load_and_join_synthetic_amlworld():
    """Verify loading and joining AMLworld transactions and patterns achieves 100% match."""
    loader = AMLworldLoader()
    transactions = loader.load_transactions(MOCK_TRANS_CSV)
    assert len(transactions) == 7

    patterns = loader.load_patterns(MOCK_PATTERNS_TXT)
    assert len(patterns) == 2

    # Verify Pattern 1: FAN-OUT
    p1 = patterns[0]
    assert p1.typology == "FAN-OUT"
    assert p1.num_transactions == 3
    assert p1.participant_ids == ["1001", "2001", "2002", "2003"]

    # Verify Pattern 2: CYCLE
    p2 = patterns[1]
    assert p2.typology == "CYCLE"
    assert p2.num_transactions == 3
    assert p2.participant_ids == ["ACC_A", "ACC_B", "ACC_C"]

    # Join
    dataset, meta = loader.join_patterns_to_transactions(transactions, patterns)
    assert meta["match_rate"] == 1.0
    assert meta["drop_rate"] == 0.0
    assert meta["unmatched_count"] == 0
    assert meta["matched_pattern_transactions"] == 6

    assert dataset.num_transactions == 7
    assert dataset.num_patterns == 2
    assert "1001" in dataset.laundering_accounts
    assert "ACC_A" in dataset.laundering_accounts
    assert "9991" in dataset.all_accounts
    assert "9991" not in dataset.laundering_accounts


def test_unmatched_transaction_handling():
    """Verify that an unmatched pattern transaction is reported cleanly in join metadata."""
    loader = AMLworldLoader()
    # Transaction dataset without the cycle transactions
    partial_csv = """Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,012,1001,012,2001,500.0,US Dollar,500.0,US Dollar,Reinvestment,1
"""
    transactions = loader.load_transactions(partial_csv)
    patterns = loader.load_patterns(MOCK_PATTERNS_TXT)

    dataset, meta = loader.join_patterns_to_transactions(transactions, patterns)
    assert meta["matched_pattern_transactions"] == 1
    assert meta["unmatched_count"] == 5
    assert meta["match_rate"] < 1.0
    assert meta["join_status"] == "PARTIAL_MATCH"


def test_timestamp_epoch_parsing():
    """Verify epoch and date string conversions."""
    t1 = parse_timestamp_to_epoch("2022/09/01 00:20")
    t2 = parse_timestamp_to_epoch("2022/09/01 00:25")
    assert t2 > t1
    assert parse_timestamp_to_epoch("1662000000.0") == 1662000000.0


def test_laundering_pattern_group_dataclass():
    """Verify LaunderingPatternGroup serialization."""
    g = LaunderingPatternGroup(
        pattern_id=1,
        typology="CYCLE",
        participant_ids=["a", "b"],
        transactions=[{"from_account": "a", "to_account": "b"}],
        start_timestamp=100.0,
        end_timestamp=200.0,
    )
    d = g.to_dict()
    assert d["pattern_id"] == 1
    assert d["typology"] == "CYCLE"
    assert g.num_participants == 2
    assert g.num_transactions == 1
