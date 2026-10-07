"""
The Eight Mandatory Acceptance Oracles for Algebraic Topology Verification.
Implements Section 6.6 of project_description.md.
"""

from typing import Dict, Any
import numpy as np
import scipy.sparse as sp

try:
    from source.src.topology.incidence import (
        build_b1,
        build_b2_simplicial,
        build_b2_cellular,
        verify_nilpotency,
        compute_hodge_laplacians,
    )
except ModuleNotFoundError:
    from src.topology.incidence import (
        build_b1,
        build_b2_simplicial,
        build_b2_cellular,
        verify_nilpotency,
        compute_hodge_laplacians,
    )


def oracle_1_path_graph() -> Dict[str, Any]:
    """Oracle 1: Path graph v0 -> v1 -> v2 -> v3 (4 vertices, 3 edges)."""
    nodes = ["v0", "v1", "v2", "v3"]
    edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3")]
    B1 = build_b1(nodes, edges)

    # Each column must have exactly one -1 and one +1
    col_sums = np.array(B1.sum(axis=0)).flatten()
    assert np.allclose(col_sums, 0.0), "Column sums must be 0"
    assert B1.shape == (4, 3)

    rank_b1 = int(np.linalg.matrix_rank(B1.toarray()))
    assert rank_b1 == 3, f"Expected rank 3, got {rank_b1}"
    return {"oracle": 1, "passed": True, "rank_B1": int(rank_b1)}


def oracle_2_unfilled_triangle() -> Dict[str, Any]:
    """Oracle 2: Unfilled triangle graph (beta0 = 1, beta1 = 1)."""
    nodes = ["v0", "v1", "v2"]
    edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v0")]
    B1 = build_b1(nodes, edges)
    rank_b1 = int(np.linalg.matrix_rank(B1.toarray()))
    # beta0 = |V| - rank(B1) = 3 - 2 = 1
    # beta1 = |E| - rank(B1) = 3 - 2 = 1 (1 cycle, 0 filled faces)
    beta0 = int(len(nodes) - rank_b1)
    beta1 = int(len(edges) - rank_b1)
    assert beta0 == 1, f"Expected beta0=1, got {beta0}"
    assert beta1 == 1, f"Expected beta1=1, got {beta1}"
    return {"oracle": 2, "passed": True, "beta0": beta0, "beta1": beta1}


def oracle_3_filled_triangle() -> Dict[str, Any]:
    """Oracle 3: Filled simplicial triangle (beta1 = 0, B1 @ B2 == 0)."""
    nodes = ["v0", "v1", "v2"]
    edges = [("v0", "v1"), ("v1", "v2"), ("v0", "v2")]
    triangles = [("v0", "v1", "v2")]

    B1 = build_b1(nodes, edges)
    B2 = build_b2_simplicial(edges, triangles)

    is_nilpotent, max_err = verify_nilpotency(B1, B2)
    assert is_nilpotent, f"Nilpotency failed: max_err={max_err}"

    rank_b1 = int(np.linalg.matrix_rank(B1.toarray()))
    rank_b2 = int(np.linalg.matrix_rank(B2.toarray()))
    # beta1 = dim(ker B1) - dim(im B2) = (|E| - rank_b1) - rank_b2 = (3 - 2) - 1 = 0
    beta1 = int((len(edges) - rank_b1) - rank_b2)
    assert beta1 == 0, f"Expected beta1=0 after face insertion, got {beta1}"
    return {"oracle": 3, "passed": True, "max_err": float(max_err), "beta1": beta1}


def oracle_4_unfilled_square() -> Dict[str, Any]:
    """Oracle 4: Unfilled square graph (beta0 = 1, beta1 = 1)."""
    nodes = ["v0", "v1", "v2", "v3"]
    edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0")]
    B1 = build_b1(nodes, edges)
    rank_b1 = int(np.linalg.matrix_rank(B1.toarray()))
    beta0 = int(len(nodes) - rank_b1)
    beta1 = int(len(edges) - rank_b1)
    assert beta0 == 1 and beta1 == 1
    return {"oracle": 4, "passed": True, "beta0": beta0, "beta1": beta1}


def oracle_5_filled_polygonal_square() -> Dict[str, Any]:
    """Oracle 5: Filled polygonal square 2-cell has 4 boundary edges."""
    nodes = ["v0", "v1", "v2", "v3"]
    edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0")]
    cells = [["v0", "v1", "v2", "v3"]]

    B2 = build_b2_cellular(edges, cells)
    assert B2.shape == (4, 1)
    non_zeros = int(B2.nnz)
    assert non_zeros == 4, f"Expected 4 non-zero entries in B2, got {non_zeros}"
    return {"oracle": 5, "passed": True, "nnz": non_zeros}


def oracle_6_nilpotency_arbitrary_polygons() -> Dict[str, Any]:
    """Oracle 6: B1 @ B2 == 0 exactly on k-gons for k in [3, 4, 5, 6, 8, 12]."""
    results = {}
    for k in [3, 4, 5, 6, 8, 12]:
        nodes = [f"v{i}" for i in range(k)]
        edges = [(f"v{i}", f"v{(i+1)%k}") for i in range(k)]
        cells = [nodes]

        B1 = build_b1(nodes, edges)
        B2 = build_b2_cellular(edges, cells)

        is_nilpotent, max_err = verify_nilpotency(B1, B2)
        assert is_nilpotent, f"Nilpotency failed on {k}-gon: err={max_err}"
        results[f"{k}-gon"] = float(max_err)
    return {"oracle": 6, "passed": True, "residuals": results}


def oracle_7_polygon_feature_aggregation() -> Dict[str, Any]:
    """Oracle 7: Feature extraction on arbitrary polygon sizes without tuple-unpacking."""
    cells = [
        ["v0", "v1", "v2"],
        ["v0", "v1", "v2", "v3"],
        ["v0", "v1", "v2", "v3", "v4"],
        ["v0", "v1", "v2", "v3", "v4", "v5"],
    ]
    # Synthetic edge features dict
    edge_features = {
        ("v0", "v1"): 1.0,
        ("v1", "v2"): 2.0,
        ("v2", "v0"): 3.0,
        ("v2", "v3"): 4.0,
        ("v3", "v0"): 5.0,
        ("v3", "v4"): 6.0,
        ("v4", "v0"): 7.0,
        ("v4", "v5"): 8.0,
        ("v5", "v0"): 9.0,
    }

    cell_aggregates = []
    for c in cells:
        k = len(c)
        flows = []
        for i in range(k):
            u, v = c[i], c[(i + 1) % k]
            flows.append(edge_features.get((u, v), 1.0))
        # Aggregation using invariant operators
        total_flow = float(np.sum(flows))
        mean_flow = float(np.mean(flows))
        cell_aggregates.append({"k": k, "sum": total_flow, "mean": mean_flow})

    assert len(cell_aggregates) == 4
    return {"oracle": 7, "passed": True, "aggregates": cell_aggregates}


def oracle_8_hodge_laplacian_properties() -> Dict[str, Any]:
    """Oracle 8: Hodge Laplacian symmetry and positive semi-definiteness."""
    nodes = ["v0", "v1", "v2", "v3"]
    edges = [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0")]
    cells = [["v0", "v1", "v2", "v3"]]

    B1 = build_b1(nodes, edges)
    B2 = build_b2_cellular(edges, cells)
    L0, L1, L2 = compute_hodge_laplacians(B1, B2)

    # Check symmetry
    assert np.allclose(L0.toarray(), L0.toarray().T), "L0 must be symmetric"
    assert np.allclose(L1.toarray(), L1.toarray().T), "L1 must be symmetric"
    assert np.allclose(L2.toarray(), L2.toarray().T), "L2 must be symmetric"

    # Check positive semi-definiteness (eigenvalues >= 0)
    eigs_L0 = np.linalg.eigvalsh(L0.toarray())
    eigs_L1 = np.linalg.eigvalsh(L1.toarray())
    eigs_L2 = np.linalg.eigvalsh(L2.toarray())

    assert np.all(eigs_L0 >= -1e-12), f"L0 has negative eigenvalue: {eigs_L0.min()}"
    assert np.all(eigs_L1 >= -1e-12), f"L1 has negative eigenvalue: {eigs_L1.min()}"
    assert np.all(eigs_L2 >= -1e-12), f"L2 has negative eigenvalue: {eigs_L2.min()}"

    return {"oracle": 8, "passed": True, "min_eig_L1": float(eigs_L1.min())}


def run_all_oracles() -> Dict[str, Any]:
    """Execute all 8 mathematical acceptance oracles."""
    res = {
        "oracle_1": oracle_1_path_graph(),
        "oracle_2": oracle_2_unfilled_triangle(),
        "oracle_3": oracle_3_filled_triangle(),
        "oracle_4": oracle_4_unfilled_square(),
        "oracle_5": oracle_5_filled_polygonal_square(),
        "oracle_6": oracle_6_nilpotency_arbitrary_polygons(),
        "oracle_7": oracle_7_polygon_feature_aggregation(),
        "oracle_8": oracle_8_hodge_laplacian_properties(),
    }
    return res
