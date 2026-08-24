"""Unit tests for PatternGroupAuditor (Contract C10-03)."""

import io
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.pattern_audit import PatternGroupAuditor


def test_pattern_audit_cycle_and_typology_breakdown():
    """Verify cycle length distribution and typology breakdown on pattern blocks."""
    patterns_txt = (
        "BEGIN LAUNDERING ATTEMPT - CYCLE\n"
        "2022/09/01 00:00,BankA,ACC1,BankB,ACC2,100,USD,100,USD,Wire,1\n"
        "2022/09/01 01:00,BankB,ACC2,BankC,ACC3,100,USD,100,USD,Wire,1\n"
        "2022/09/01 02:00,BankC,ACC3,BankA,ACC1,100,USD,100,USD,Wire,1\n"
        "END LAUNDERING ATTEMPT\n\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE\n"
        "2022/09/01 00:00,BankA,ACC4,BankB,ACC5,50,USD,50,USD,Wire,1\n"
        "2022/09/01 01:00,BankB,ACC5,BankC,ACC6,50,USD,50,USD,Wire,1\n"
        "2022/09/01 02:00,BankC,ACC6,BankD,ACC7,50,USD,50,USD,Wire,1\n"
        "2022/09/01 03:00,BankD,ACC7,BankA,ACC4,50,USD,50,USD,Wire,1\n"
        "END LAUNDERING ATTEMPT\n\n"
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT\n"
        "2022/09/01 00:00,BankA,ACC8,BankB,ACC9,20,USD,20,USD,Wire,1\n"
        "2022/09/01 00:00,BankA,ACC8,BankC,ACC10,20,USD,20,USD,Wire,1\n"
        "END LAUNDERING ATTEMPT\n"
    )

    auditor = PatternGroupAuditor()
    report = auditor.audit_patterns(io.StringIO(patterns_txt))

    assert report["total_pattern_blocks"] == 3
    assert report["total_patterns_by_typology"]["CYCLE"] == 2
    assert report["total_patterns_by_typology"]["FAN-OUT"] == 1

    cycle_stats = report["cycle_pattern_stats"]
    assert cycle_stats["total_cycle_patterns"] == 2
    assert cycle_stats["length_distribution"][3] == 1  # ACC1->ACC2->ACC3 (k=3)
    assert cycle_stats["length_distribution"][4] == 1  # ACC4->ACC5->ACC6->ACC7 (k=4)
    assert cycle_stats["total_cycle_accounts"] == 7  # 3 + 4

    cluster_stats = report["cluster_independence_stats"]
    # All 3 patterns have disjoint accounts -> 3 independent components
    assert cluster_stats["independent_groups_count"] == 3
    assert cluster_stats["largest_component_size"] == 1


def test_pattern_audit_with_overlapping_groups_and_join_rate():
    """Verify group overlap merges into components and computes join rate."""
    patterns_txt = (
        "BEGIN LAUNDERING ATTEMPT - CYCLE\n"
        "2022/09/01 00:00,BankA,ACC1,BankB,ACC2,100,USD,100,USD,Wire,1\n"
        "2022/09/01 01:00,BankB,ACC2,BankA,ACC1,100,USD,100,USD,Wire,1\n"
        "END LAUNDERING ATTEMPT\n\n"
        "BEGIN LAUNDERING ATTEMPT - FAN-IN\n"
        "2022/09/01 00:00,BankC,ACC3,BankA,ACC1,50,USD,50,USD,Wire,1\n"
        "END LAUNDERING ATTEMPT\n"
    )

    trans_csv = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:00,BankA,ACC1,BankB,ACC2,100,USD,100,USD,Wire,1\n"
        "2022/09/01 01:00,BankB,ACC2,BankA,ACC1,100,USD,100,USD,Wire,1\n"
        "2022/09/01 00:00,BankC,ACC3,BankA,ACC1,50,USD,50,USD,Wire,1\n"
    )

    auditor = PatternGroupAuditor()
    report = auditor.audit_patterns(
        patterns_source=io.StringIO(patterns_txt),
        trans_source=io.StringIO(trans_csv),
    )

    # Patterns 1 and 2 share ACC1 -> connected into 1 cluster
    assert report["cluster_independence_stats"]["independent_groups_count"] == 1
    assert report["cluster_independence_stats"]["largest_component_size"] == 2

    # All 3 pattern transactions match trans_csv
    assert report["transaction_join_rate"] == 1.0
    assert report["total_pattern_transactions"] == 3
    assert report["matched_pattern_transactions"] == 3
