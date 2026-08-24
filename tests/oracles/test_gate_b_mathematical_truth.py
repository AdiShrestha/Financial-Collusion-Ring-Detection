"""Gate B (Mathematical Truth) formal oracle test suite."""

import json
from pathlib import Path
import numpy as np
import pytest

from source.data.candidate_extractor import CandidateExample
from source.ph.graph_filtration import build_temporal_ph_graph
from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.gate_b_verifier import GateBVerifier
from source.topology.oracles import (
    compute_betti_numbers,
    verify_boundary_nilpotence,
    verify_hodge_laplacian_properties,
)


def generate_candidate_cohort(n: int = 50) -> list[CandidateExample]:
    cohort = []
    for i in range(n):
        k = 3 + (i % 4)  # Cycle length in {3, 4, 5, 6}
        nodes = [f"cand_{i}_node_{j}" for j in range(k)]
        edges = [
            (nodes[j], nodes[(j + 1) % k], {"amount": float(10 * (j + 1)), "timestamp": float(j + 1)})
            for j in range(k)
        ]
        # Add random chords or context nodes
        if i % 3 == 0 and k >= 4:
            edges.append((nodes[0], nodes[2], {"amount": 5.0, "timestamp": 1.5}))

        cand = CandidateExample(
            candidate_id=f"cand_gateb_{i:03d}",
            dataset_track="amlworld" if i % 2 == 0 else "elliptic_plus",
            temporal_bounds=(0.0, float(k + 2)),
            participant_ids=nodes,
            edges=edges,
            node_features={n: [float(i)] * 56 for n in nodes},
            target_y=1 if i % 2 == 0 else 0,
            typology_label="CYCLE" if i % 2 == 0 else "NEGATIVE_CANDIDATE",
            group_id=f"group_gateb_{i // 5}",
            metadata={"cycle_nodes": nodes, "cycle_length": k},
        )
        cohort.append(cand)
    return cohort


def test_gate_b_nilpotence_on_candidate_cohort():
    """Prove B1 @ B2 == 0 holds identically on 100% of candidate complexes (INV-004)."""
    cohort = generate_candidate_cohort(50)
    for cand in cohort:
        # Simplicial complex
        sv = CliqueSimplicialView.from_candidate_example(cand)
        assert verify_boundary_nilpotence(sv.B1, sv.B2) is True
        prod_s = np.dot(sv.B1, sv.B2)
        if prod_s.size > 0:
            assert np.max(np.abs(prod_s)) == 0.0

        # Cellular complex
        cv = CycleCellView.from_candidate_example(cand)
        assert verify_boundary_nilpotence(cv.B1, cv.B2) is True
        prod_c = np.dot(cv.B1, cv.B2)
        if prod_c.size > 0:
            assert np.max(np.abs(prod_c)) == 0.0


def test_gate_b_simplex_vs_cell_boundary_distinction():
    """Verify that k-cycles (k >= 4) are polygonal rank-2 cells and NOT 2-simplices (INV-002)."""
    for k in (4, 5, 6):
        nodes = [f"u_{i}" for i in range(k)]
        edges = [(nodes[i], nodes[(i + 1) % k], {"amount": 1.0, "timestamp": float(i)}) for i in range(k)]
        cand = CandidateExample(
            candidate_id=f"cand_pure_{k}",
            dataset_track="amlworld",
            temporal_bounds=(0.0, float(k)),
            participant_ids=nodes,
            edges=edges,
            node_features={n: [0.0] * 56 for n in nodes},
            target_y=1,
            typology_label="CYCLE",
            group_id="group_pure",
            metadata={"cycle_nodes": nodes, "cycle_length": k},
        )

        sv = CliqueSimplicialView.from_candidate_example(cand)
        cv = CycleCellView.from_candidate_example(cand)

        # Simplicial: 0 2-simplices because k >= 4 has no 3-cliques
        assert sv.num_faces_2 == 0
        b_simp_0, b_simp_1 = compute_betti_numbers(sv.B1, sv.B2)
        assert b_simp_1 == 1  # Unfilled 1-hole

        # Cellular: 1 polygonal 2-cell with k boundary edges
        assert cv.num_cells_2 == 1
        assert len(cv.cells_2[0]) == k
        b_cell_0, b_cell_1 = compute_betti_numbers(cv.B1, cv.B2)
        assert b_cell_1 == 0  # Filled polygonal cell


def test_gate_b_ph_domain_unfilled_integrity():
    """Verify that Persistent Homology graph filtration operates strictly on 1-skeletons (INV-003)."""
    cohort = generate_candidate_cohort(10)
    for cand in cohort:
        nodes_map = {n: 0.0 for n in cand.participant_ids}
        edges_list = [(u, v, float(attr.get("timestamp", 0.0))) for u, v, attr in cand.edges]
        filt = build_temporal_ph_graph(nodes_map, edges_list, disallow_higher_simplices=True)
        # Filtration must contain only 0-simplices and 1-simplices
        for simplex, val in filt:
            assert len(simplex) in (1, 2)  # Dimension 0 (1 node) or Dimension 1 (2 nodes)
            assert len(simplex) <= 2


def test_gate_b_full_verification_engine_and_report(tmp_path):
    """Run full GateBVerifier engine and confirm GATE_B_PASS report."""
    cohort = generate_candidate_cohort(30)
    simp_list = [CliqueSimplicialView.from_candidate_example(c) for c in cohort]
    cell_list = [CycleCellView.from_candidate_example(c) for c in cohort]

    report_file = tmp_path / "gate_b_report.json"
    report = GateBVerifier.generate_and_save_report(
        simplicial_complexes=simp_list,
        cellular_complexes=cell_list,
        output_path=str(report_file),
    )

    assert report["gate_b_status"] == "GATE_B_PASS"
    assert len(report["violations"]) == 0
    assert report["simplicial_audit"]["nilpotence_passed"] == 30
    assert report["cellular_audit"]["nilpotence_passed"] == 30
    assert report["laplacian_audit"]["l0_psd"] == 60
    assert report["laplacian_audit"]["l1_psd"] == 60

    # Also save to canonical project/gate_b_report.json
    GateBVerifier.generate_and_save_report(
        simplicial_complexes=simp_list,
        cellular_complexes=cell_list,
        output_path="project/gate_b_report.json",
    )
    assert Path("project/gate_b_report.json").exists()
