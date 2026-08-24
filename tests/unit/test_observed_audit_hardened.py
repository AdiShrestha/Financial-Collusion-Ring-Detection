"""Unit tests for Provenance-Hardened Observed Data Audit (Contract C14-01)."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.audit_exporter import export_observed_audit


def test_hardened_audit_raises_filenotfound_on_missing_sources():
    """Verify export_observed_audit raises hard FileNotFoundError without silent mocks."""
    with pytest.raises(FileNotFoundError, match="not found"):
        export_observed_audit(
            trans_source="/nonexistent/path/to/trans.csv",
            patterns_source="data/raw/HI-Small_Patterns.txt",
        )

    with pytest.raises(FileNotFoundError, match="not found"):
        export_observed_audit(
            trans_source="data/raw/HI-Small_Trans.csv",
            patterns_source="/nonexistent/path/to/patterns.txt",
        )


def test_hardened_audit_report_artifact_structure_and_counts():
    """Verify data/observed_audit_report.json exists and reflects >5,000,000 transactions."""
    report_path = "data/observed_audit_report.json"
    assert os.path.exists(report_path), "Missing observed_audit_report.json"

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_name"] == "IBM AMLworld HI-Small"
    assert "files" in data
    assert "observed_summary" in data
    assert data["observed_summary"]["total_transactions"] > 5000000
    assert data["observed_summary"]["total_accounts"] > 500000

    assert "cycle_typology_breakdown" in data
    assert data["cycle_typology_breakdown"]["total_cycle_patterns"] > 0

    assert "quality_checks" in data
    assert data["quality_checks"]["no_critical_malformed_rows"] is True
