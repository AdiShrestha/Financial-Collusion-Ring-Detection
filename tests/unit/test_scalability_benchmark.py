"""Unit tests for Computational Complexity & Scalability Benchmark."""

import pytest
import torch

from source.experiments.scalability_benchmark import (
    ScalabilityProfiler,
    count_parameters,
    generate_scale_synthetic_candidate,
)
from source.models.cell_net import CellularComplexNet
from source.models.gnn_baselines import GCNBaseline


def test_scalability_profiler_timing():
    """Verify that profiled latency is strictly positive and finite."""
    profiler = ScalabilityProfiler(in_dim_node=56, in_dim_edge=2, in_dim_cell=2, hidden_dim=16)
    cand = generate_scale_synthetic_candidate(num_nodes=10, node_dim=56)
    model = GCNBaseline(in_dim=56, hidden_dim=16, out_dim=2, num_layers=2)

    profile = profiler.profile_model(model, cand, num_warmup=3, num_repeats=10)

    assert profile["mean_latency_ms"] > 0.0
    assert profile["p95_latency_ms"] >= profile["mean_latency_ms"]
    assert profile["throughput_qps"] > 0.0
    assert profile["trainable_parameters"] > 0
    assert profile["param_memory_mb"] > 0.0


def test_parameter_count_consistency():
    """Verify parameter count helper accurately computes trainable parameters."""
    model = CellularComplexNet(in_dim_0=56, in_dim_1=2, in_dim_2=2, hidden_dim=16, num_layers=2)
    expected_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count_parameters(model) == expected_params


def test_scaling_sweep_execution():
    """Verify scaling sweep across models and node scales |V|."""
    profiler = ScalabilityProfiler(in_dim_node=56, in_dim_edge=2, in_dim_cell=2, hidden_dim=16)
    results = profiler.run_scalability_suite(
        scales=[10, 25],
        model_names=["GCNBaseline", "CellularComplexNet", "TopoRingNet"],
        num_warmup=2,
        num_repeats=5,
    )

    assert "results" in results
    assert "GCNBaseline" in results["results"]
    assert 10 in results["results"]["GCNBaseline"]
    assert 25 in results["results"]["CellularComplexNet"]
    assert results["results"]["TopoRingNet"][25]["mean_latency_ms"] > 0.0
