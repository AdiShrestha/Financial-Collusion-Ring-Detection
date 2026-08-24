"""Unit tests for persistence diagram and barcode extractor."""

import numpy as np
import pytest

from source.data.candidate_extractor import CandidateExample
from source.ph.persistence_extractor import PersistenceDiagram, PersistenceExtractor
from source.ph.ph_graph_view import PHGraphView


def create_4cycle_candidate() -> CandidateExample:
    return CandidateExample(
        candidate_id="cand_diag_001",
        dataset_track="amlworld",
        temporal_bounds=(10.0, 40.0),
        participant_ids=["N1", "N2", "N3", "N4"],
        edges=[
            ("N1", "N2", {"amount": 100.0, "timestamp": 10.0}),
            ("N2", "N3", {"amount": 100.0, "timestamp": 20.0}),
            ("N3", "N4", {"amount": 100.0, "timestamp": 30.0}),
            ("N4", "N1", {"amount": 100.0, "timestamp": 40.0}),  # Closing edge
        ],
        node_features={n: [0.0] * 56 for n in ["N1", "N2", "N3", "N4"]},
        target_y=1,
        typology_label="CYCLE",
        group_id="group_diag",
    )


def test_persistence_diagram_h0_and_h1():
    """Verify H0 connected component merging and H1 cycle birth at closing edge timestamp."""
    cand = create_4cycle_candidate()
    ph_view = PHGraphView.from_candidate_example(cand, filtration_type="temporal")
    diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)

    assert diag.candidate_id == "cand_diag_001"
    assert diag.h0_intervals.shape[0] >= 1
    assert diag.has_h1_cycle is True
    assert diag.h1_intervals.shape[0] == 1

    # In a 4-cycle with edges at 10, 20, 30, 40, the 1-cycle is born at t=40.0
    h1_birth, h1_death = diag.h1_intervals[0]
    assert h1_birth == 40.0
    assert h1_death == 41.0  # Capped at max_filtration (40.0) + 1.0


def test_persistence_lifetimes_calculation():
    """Verify persistence lifetime computation (death - birth >= 0)."""
    cand = create_4cycle_candidate()
    ph_view = PHGraphView.from_candidate_example(cand)
    diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)

    h0_life = diag.h0_lifetimes
    h1_life = diag.h1_lifetimes

    assert len(h0_life) == len(diag.h0_intervals)
    assert len(h1_life) == len(diag.h1_intervals)
    assert np.all(h0_life >= 0.0)
    assert np.all(h1_life >= 0.0)


def test_essential_feature_capping():
    """Verify finite values when cap_infinity=True vs inf when False."""
    cand = create_4cycle_candidate()
    ph_view = PHGraphView.from_candidate_example(cand)

    diag_capped = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)
    assert np.all(np.isfinite(diag_capped.h0_intervals))
    assert np.all(np.isfinite(diag_capped.h1_intervals))

    diag_uncapped = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=False)
    # The essential H0 class (and unfilled H1 class) should have inf death
    has_inf = np.isinf(diag_uncapped.h0_intervals[:, 1]).any() or np.isinf(diag_uncapped.h1_intervals[:, 1]).any()
    assert bool(has_inf) is True


def test_semantic_known_answer_cycle_persistence():
    """Explicit semantic operator test verifying exact persistence values against known graph topology."""
    # Construct a 3-cycle triangle where edges arrive at t=5.0, t=10.0, t=15.0
    cand = CandidateExample(
        candidate_id="cand_known_tri",
        dataset_track="amlworld",
        temporal_bounds=(5.0, 15.0),
        participant_ids=["A", "B", "C"],
        edges=[
            ("A", "B", {"amount": 10.0, "timestamp": 5.0}),
            ("B", "C", {"amount": 10.0, "timestamp": 10.0}),
            ("C", "A", {"amount": 10.0, "timestamp": 15.0}),
        ],
        node_features={"A": [0.0] * 56, "B": [0.0] * 56, "C": [0.0] * 56},
        target_y=1,
        typology_label="CYCLE",
        group_id="group_tri",
    )
    ph_view = PHGraphView.from_candidate_example(cand)
    diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True, infinity_cap_margin=2.0)

    assert diag.has_h1_cycle is True
    # H1 cycle born at exactly t=15.0 and persists to infinity capped at 15.0 + 2.0 = 17.0
    assert diag.h1_intervals[0, 0] == 15.0
    assert diag.h1_intervals[0, 1] == 17.0
