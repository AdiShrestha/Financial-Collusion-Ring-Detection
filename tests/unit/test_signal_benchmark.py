"""Unit tests for topological signal separation and discriminability benchmark."""

import pytest
import numpy as np

from source.data.candidate_extractor import CandidateExample
from source.ph.signal_benchmark import TopologicalSignalBenchmark


def create_benchmark_cohort(n_pos: int = 20, n_neg: int = 20) -> list[CandidateExample]:
    cohort = []

    # Positive examples: true collusion cycles with closed loop (H1 birth)
    for i in range(n_pos):
        k = 4
        nodes = [f"pos_{i}_n_{j}" for j in range(k)]
        edges = [
            (nodes[0], nodes[1], {"amount": 100.0, "timestamp": 10.0}),
            (nodes[1], nodes[2], {"amount": 95.0, "timestamp": 20.0}),
            (nodes[2], nodes[3], {"amount": 90.0, "timestamp": 30.0}),
            (nodes[3], nodes[0], {"amount": 85.0, "timestamp": 40.0}),  # Cycle closing edge
        ]
        cand = CandidateExample(
            candidate_id=f"pos_{i:03d}",
            dataset_track="amlworld",
            temporal_bounds=(10.0, 40.0),
            participant_ids=nodes,
            edges=edges,
            node_features={n: [0.0] * 56 for n in nodes},
            target_y=1,
            typology_label="CYCLE",
            group_id=f"pos_group_{i}",
        )
        cohort.append(cand)

    # Negative examples: benign open paths without cycle (0 H1 features)
    for i in range(n_neg):
        k = 4
        nodes = [f"neg_{i}_n_{j}" for j in range(k)]
        edges = [
            (nodes[0], nodes[1], {"amount": 50.0, "timestamp": 10.0}),
            (nodes[1], nodes[2], {"amount": 50.0, "timestamp": 20.0}),
            (nodes[2], nodes[3], {"amount": 50.0, "timestamp": 30.0}),
            # No closing edge
        ]
        cand = CandidateExample(
            candidate_id=f"neg_{i:03d}",
            dataset_track="amlworld",
            temporal_bounds=(10.0, 30.0),
            participant_ids=nodes,
            edges=edges,
            node_features={n: [0.0] * 56 for n in nodes},
            target_y=0,
            typology_label="NEGATIVE_CANDIDATE",
            group_id=f"neg_group_{i}",
        )
        cohort.append(cand)

    return cohort


def test_topological_signal_separation():
    """Verify that topological features alone yield ROC-AUC > 0.50 above uniform chance."""
    train_cohort = create_benchmark_cohort(n_pos=20, n_neg=20)
    val_cohort = create_benchmark_cohort(n_pos=15, n_neg=15)

    benchmark = TopologicalSignalBenchmark(random_state=42)
    metrics = benchmark.fit_and_evaluate(train_cohort, val_cohort)

    assert "val_roc_auc" in metrics
    assert "val_pr_auc" in metrics
    assert metrics["val_roc_auc"] > 0.50  # Must beat random chance
    assert metrics["val_roc_auc"] >= 0.80  # Persistent homology easily separates cycles from paths


def test_deterministic_benchmark_results():
    """Verify identical evaluation metrics under fixed random seed."""
    train_cohort = create_benchmark_cohort(n_pos=10, n_neg=10)
    val_cohort = create_benchmark_cohort(n_pos=10, n_neg=10)

    b1 = TopologicalSignalBenchmark(random_state=123)
    m1 = b1.fit_and_evaluate(train_cohort, val_cohort)

    b2 = TopologicalSignalBenchmark(random_state=123)
    m2 = b2.fit_and_evaluate(train_cohort, val_cohort)

    assert m1["val_roc_auc"] == m2["val_roc_auc"]
    assert m1["val_pr_auc"] == m2["val_pr_auc"]
    assert m1["val_accuracy"] == m2["val_accuracy"]


def test_semantic_cycle_discrimination_power():
    """Explicit semantic test proving that persistent homology yields near-perfect AUC on cycle detection."""
    train_cohort = create_benchmark_cohort(n_pos=25, n_neg=25)
    val_cohort = create_benchmark_cohort(n_pos=20, n_neg=20)

    benchmark = TopologicalSignalBenchmark(random_state=42)
    metrics = benchmark.fit_and_evaluate(train_cohort, val_cohort)

    # In our controlled test between closed cycles and open paths, H1 presence gives ROC-AUC >= 0.95
    assert metrics["val_roc_auc"] >= 0.95
    assert metrics["val_pr_auc"] >= 0.95
    assert metrics["val_accuracy"] >= 0.90
