"""Unit tests for Confirmatory Statistical Results & Claim Registry Compilation."""

import json
import os
import pytest

from source.evidence.claim_registry import ClaimRecord, ClaimRegistry


def test_claim_registry_serialization(tmp_path):
    """Verify claim registry registration and export."""
    registry = ClaimRegistry()

    rec = ClaimRecord(
        claim_id="CLM-001",
        hypothesis_id="H1",
        statement="TopoRingNet outperforms baseline GNNs on PR-AUC.",
        primary_metric="pr_auc",
        p_value_raw=0.001,
        p_value_adj=0.004,
        effect_size_delta=0.65,
        effect_ci_low=0.45,
        effect_ci_high=0.85,
        verdict="SUPPORTED",
        sample_size=28,
        reproduction_command="python3 -m pytest tests/unit/test_hypothesis_tester.py",
    )
    registry.register_claim(rec)

    retrieved = registry.get_claim("CLM-001")
    assert retrieved.claim_id == "CLM-001"
    assert retrieved.verdict == "SUPPORTED"

    all_c = registry.all_claims()
    assert "CLM-001" in all_c
    assert all_c["CLM-001"]["effect_size_delta"] == 0.65


def test_claim_registry_reproduction_commands():
    """Verify that empty reproduction commands are rejected."""
    registry = ClaimRegistry()

    with pytest.raises(ValueError, match="missing mandatory reproduction command"):
        registry.register_claim(ClaimRecord(
            claim_id="CLM-ERR",
            hypothesis_id="H1",
            statement="Invalid claim without reproduction command.",
            primary_metric="pr_auc",
            p_value_raw=0.01,
            p_value_adj=0.04,
            effect_size_delta=0.5,
            effect_ci_low=0.2,
            effect_ci_high=0.8,
            verdict="SUPPORTED",
            sample_size=28,
            reproduction_command="",
        ))


def test_confirmatory_stats_json_exists_and_valid():
    """Verify that results/confirmatory_stats.json exists and conforms to schema."""
    stats_path = "results/confirmatory_stats.json"
    assert os.path.exists(stats_path), f"{stats_path} does not exist"

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "model_benchmarks" in data
    assert "hypothesis_testing" in data
    assert "subgroup_analysis" in data
    assert "claim_registry" in data

    claims = data["claim_registry"]
    for cid in ["CLM-001", "CLM-002", "CLM-003", "CLM-004"]:
        assert cid in claims
        c_entry = claims[cid]
        assert "verdict" in c_entry
        assert "reproduction_command" in c_entry
        assert len(c_entry["reproduction_command"]) > 0
