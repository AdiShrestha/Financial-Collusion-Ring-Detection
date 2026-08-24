"""Unit tests for Track Robustness & Subgroup Stratification Analysis."""

import json
import os
import pytest

from source.evidence.subgroup_analysis import SubgroupAnalyzer


def _load_sample_predictions():
    pred_path = "runs/confirmatory/predictions.json"
    assert os.path.exists(pred_path)
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_subgroup_stratification_by_typology():
    """Verify that subgroup analyzer stratifies across all typologies."""
    data = _load_sample_predictions()
    analyzer = SubgroupAnalyzer()
    res = analyzer.analyze_subgroups(data)

    assert "by_typology" in res
    assert "by_track" in res

    by_typ = res["by_typology"]
    for typ in ["CYCLE", "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "BIPARTITE", "NEGATIVE_CANDIDATE"]:
        assert typ in by_typ
        for model in ["GCNBaseline", "CellularComplexNet", "TopoRingNet"]:
            assert model in by_typ[typ]
            m_res = by_typ[typ][model]
            assert "mean_pr_auc" in m_res
            assert "mean_f1_macro" in m_res


def test_track_robustness_comparison():
    """Verify that metrics are computed across both dataset tracks."""
    data = _load_sample_predictions()
    analyzer = SubgroupAnalyzer()
    res = analyzer.analyze_subgroups(data)

    by_track = res["by_track"]
    assert "amlworld" in by_track
    assert "elliptic_actors" in by_track

    for track in ["amlworld", "elliptic_actors"]:
        assert "TopoRingNet" in by_track[track]
        assert by_track[track]["TopoRingNet"]["count"] > 0
        assert by_track[track]["TopoRingNet"]["mean_pr_auc"] >= 0.0
