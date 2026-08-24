"""Gate C Architecture Realizability Oracle Tests."""

import json
import os
import pytest

from source.models.gate_c_verifier import GateCVerifier, generate_gate_c_synthetic_batch


def test_gate_c_all_models_overfit_smoke_test():
    """Execute full Gate C verification across all 6 model architectures on 10-candidate batch."""
    report_path = "project/gate_c_report.json"
    verifier = GateCVerifier(output_report_path=report_path)

    batch_items = generate_gate_c_synthetic_batch(size=10)
    report = verifier.run_memorization_smoke_test(batch_items=batch_items, max_epochs=100, lr=0.02)

    assert report["gate_c_status"] == "GATE_C_PASS"
    assert report["num_models_tested"] == 6
    assert report["batch_size"] == 10
    assert report["all_models_achieved_100_percent_accuracy"] is True
    assert report["all_models_converged"] is True
    assert report["all_gradients_healthy"] is True

    for model_name, res in report["models"].items():
        assert res["passed"] is True, f"Model {model_name} failed Gate C smoke test"
        assert res["final_accuracy"] == 1.0, f"Model {model_name} did not achieve 100% accuracy"
        assert res["final_loss"] < 0.05, f"Model {model_name} final loss too high: {res['final_loss']}"
        assert res["has_dead_parameters"] is False, f"Model {model_name} has dead parameters"


def test_gate_c_gradient_flow_audit():
    """Verify that all 6 architectures have positive, healthy gradient norms across all layers."""
    report_path = "project/gate_c_report.json"
    assert os.path.exists(report_path), "gate_c_report.json must exist"

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    for model_name, res in report["models"].items():
        summary = res["gradient_norm_summary"]
        assert summary["min_grad_norm"] > 0.0, f"Model {model_name} min gradient norm is 0"
        assert summary["max_grad_norm"] > summary["min_grad_norm"]


def test_gate_c_report_json_schema():
    """Verify schema integrity of the Gate C certification report."""
    report_path = "project/gate_c_report.json"
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "gate_c_status",
        "num_models_tested",
        "batch_size",
        "max_epochs_allowed",
        "models",
        "all_models_achieved_100_percent_accuracy",
        "all_models_converged",
        "all_gradients_healthy",
    ]
    for k in required_keys:
        assert k in data, f"Missing required key '{k}' in gate_c_report.json"

    expected_models = [
        "GCNBaseline",
        "GATBaseline",
        "GraphSAGEBaseline",
        "SimplicialComplexNet",
        "CellularComplexNet",
        "TopoRingNet",
    ]
    for m in expected_models:
        assert m in data["models"], f"Missing model '{m}' in gate_c_report.json models dictionary"
