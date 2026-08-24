"""Unit tests for hardened observed audit and true provenance exporter (Contract C15-02)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.audit_exporter import export_observed_audit


def test_hardened_audit_raises_filenotfound_on_missing_sources():
    """Verify hard exception when required benchmark files are absent."""
    with pytest.raises(FileNotFoundError):
        export_observed_audit(
            trans_source="/tmp/non_existent_path_transactions.csv",
            patterns_source="data/raw/HI-Small_Patterns.txt",
        )

    with pytest.raises(FileNotFoundError):
        export_observed_audit(
            trans_source="data/raw/HI-Small_Trans.csv",
            patterns_source="/tmp/non_existent_path_patterns.txt",
        )


def test_hardened_audit_report_artifact_structure_and_counts():
    """Verify generated audit report records true physical hashes and row counts."""
    report = export_observed_audit(
        trans_source="data/raw/HI-Small_Trans.csv",
        patterns_source="data/raw/HI-Small_Patterns.txt",
        output_path="artifacts/audit/observed_data_audit.json",
    )

    assert os.path.exists("artifacts/audit/observed_data_audit.json")
    assert report["files"]["transactions_csv"]["sha256"] == "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"
    assert report["files"]["patterns_txt"]["sha256"] == "2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b"
    assert report["observed_summary"]["total_transactions"] == 5078345
    assert report["observed_summary"]["unique_active_accounts"] == 515080
    assert report["observed_summary"]["laundering_transactions"] == 5177
    assert report["cycle_typology_breakdown"]["total_cycles_length_3_to_12"] == 40
