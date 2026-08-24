"""Unit tests for Observed Audit Exporter (Contract C10-04)."""

import json
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.audit_exporter import export_observed_audit


def test_export_observed_audit_structure_and_types():
    """Verify export_observed_audit produces schema-compliant JSON artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_observed_audit_report.json")
        report = export_observed_audit(output_path=out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["audit_version"] == "2.2.0"
        assert data["dataset_name"] == "IBM AMLworld HI-Small"

        # Check file integrity hashes
        files = data["files"]
        assert "transactions_csv" in files
        assert "patterns_txt" in files
        assert len(files["transactions_csv"]["sha256"]) == 64
        assert len(files["patterns_txt"]["sha256"]) == 64
        assert files["transactions_csv"]["byte_size"] > 0
        assert files["patterns_txt"]["byte_size"] > 0

        # Check cycle breakdown
        cycle_info = data["cycle_typology_breakdown"]
        assert cycle_info["total_cycles"] > 0
        for k in ["3", "4", "5", "6"]:
            assert k in cycle_info["length_distribution"]
            assert cycle_info["length_distribution"][k] > 0

        # Check cluster independence
        assert data["cluster_independence"]["independent_groups_count"] >= 10

        # Check data quality
        quality = data["data_quality_summary"]
        assert quality["join_rate"] >= 0.999
        assert quality["malformed_rows"] == 0
