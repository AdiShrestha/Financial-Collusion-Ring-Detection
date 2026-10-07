"""
Combinatorial Boundary Operators and Hodge Laplacians for Simplicial and Cellular Complexes.
Implements exact signed boundary matrices B1, B2 and verifies the nilpotency condition B1 @ B2 == 0.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp


def build_b1(nodes: List[str], edges: List[Tuple[str, str]]) -> sp.csr_matrix:
    """
    Construct signed vertex-edge incidence matrix B1 (nodes x edges).
    For edge e = (u, v), (B1)_{u, e} = -1 and (B1)_{v, e} = +1.
    """
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_v = len(nodes)
    n_e = len(edges)

    row_indices = []
    col_indices = []
    data = []

    for e_idx, (u, v) in enumerate(edges):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        # u is source (-1), v is target (+1)
        row_indices.extend([u_idx, v_idx])
        col_indices.extend([e_idx, e_idx])
        data.extend([-1.0, 1.0])

    B1 = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_v, n_e), dtype=np.float64)
    return B1


def build_b2_simplicial(
    edges: List[Tuple[str, str]], triangles: List[Tuple[str, str, str]]
) -> sp.csr_matrix:
    """
    Construct boundary matrix B2 for a 2-simplicial complex (edges x triangles).
    Assumes canonical sorted vertex order v0 < v1 < v2 for each triangle [v0, v1, v2].
    Boundary: [v1, v2] - [v0, v2] + [v0, v1].
    """
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        # normalize undirected key with canonical orientation
        key = (min(u, v), max(u, v))
        if u == v or key in edge_to_idx:
            raise ValueError(f"Simplicial edges must be unique, non-loop pairs: {(u, v)}")
        edge_to_idx[key] = (idx, 1.0 if u < v else -1.0)

    n_e = len(edges)
    n_f = len(triangles)

    row_indices = []
    col_indices = []
    data = []

    for f_idx, (v0, v1, v2) in enumerate(triangles):
        # Sort triangle vertices canonically
        verts = sorted([v0, v1, v2])
        e_opp0 = (verts[1], verts[2])  # + [v1, v2]
        e_opp1 = (verts[0], verts[2])  # - [v0, v2]
        e_opp2 = (verts[0], verts[1])  # + [v0, v1]

        for e_key, sign in [(e_opp0, 1.0), (e_opp1, -1.0), (e_opp2, 1.0)]:
            if e_key not in edge_to_idx:
                raise ValueError(f"Triangle {verts} is missing boundary edge {e_key}")
            e_idx, dir_mult = edge_to_idx[e_key]
            row_indices.append(e_idx)
            col_indices.append(f_idx)
            data.append(sign * dir_mult)

    B2 = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_e, n_f), dtype=np.float64)
    return B2


def build_b2_cellular(
    edges: List[Tuple[str, str]], cycle_cells: List[List[str]]
) -> sp.csr_matrix:
    """
    Construct boundary matrix B2 for a rank-2 cell complex (edges x polygonal cells).
    Each cell is an ordered cycle of vertices [v0, v1, ..., v_{k-1}].
    Edges are assigned +1 if aligned with cycle traversal direction, -1 if opposite.
    """
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        if u == v:
            raise ValueError("A self-loop cannot be a regular oriented 1-cell")
        if (u, v) in edge_to_idx:
            raise ValueError(f"Duplicate cellular edge requires an explicit transaction identity: {(u, v)}")
        edge_to_idx[(u, v)] = idx

    n_e = len(edges)
    n_c = len(cycle_cells)

    row_indices = []
    col_indices = []
    data = []

    for c_idx, cycle_verts in enumerate(cycle_cells):
        k = len(cycle_verts)
        for i in range(k):
            u = cycle_verts[i]
            v = cycle_verts[(i + 1) % k]
            if (u, v) in edge_to_idx:
                e_idx, sign = edge_to_idx[(u, v)], 1.0
            elif (v, u) in edge_to_idx:
                e_idx, sign = edge_to_idx[(v, u)], -1.0
            else:
                raise ValueError(f"Cycle boundary edge ({u}, {v}) not found in edge set!")
            row_indices.append(e_idx)
            col_indices.append(c_idx)
            data.append(sign)

    B2 = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_e, n_c), dtype=np.float64)
    return B2


def verify_nilpotency(B1: sp.csr_matrix, B2: sp.csr_matrix, tol: float = 1e-12) -> Tuple[bool, float]:
    """
    Assert B1 @ B2 == 0 within numeric tolerance.
    Returns (is_valid, max_absolute_residual).
    """
    prod = B1.dot(B2)
    max_err = float(np.abs(prod.data).max()) if prod.nnz > 0 else 0.0
    return (max_err <= tol, max_err)


def compute_hodge_laplacians(
    B1: sp.csr_matrix, B2: sp.csr_matrix
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """
    Compute combinatorial Hodge Laplacians:
      L0 = B1 @ B1.T
      L1 = B1.T @ B1 + B2 @ B2.T
      L2 = B2.T @ B2
    """
    B1_t = B1.T
    B2_t = B2.T

    L0 = B1.dot(B1_t)
    L1_down = B1_t.dot(B1)
    L1_up = B2.dot(B2_t)
    L1 = L1_down + L1_up
    L2 = B2_t.dot(B2)

    return L0, L1, L2
