"""Topological incidence and boundary matrix calculation utilities."""

from typing import List, Sequence, Tuple, Union
import numpy as np


def compute_boundary_matrix_b1(
    nodes: Sequence[Union[int, str]],
    edges: Sequence[Tuple[Union[int, str], Union[int, str]]],
) -> np.ndarray:
    """Compute signed boundary matrix B1 (nodes x edges).

    Convention:
        If edge e_j = (u, v) is directed from u to v:
        B1[u, j] = -1  (source)
        B1[v, j] = +1  (target)
    """
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    num_edges = len(edges)

    b1 = np.zeros((num_nodes, num_edges), dtype=np.float64)
    for j, (u, v) in enumerate(edges):
        if u not in node_to_idx or v not in node_to_idx:
            raise ValueError(f"Edge ({u}, {v}) references unknown node")
        b1[node_to_idx[u], j] = -1.0
        b1[node_to_idx[v], j] = 1.0

    return b1


def compute_boundary_matrix_b2(
    edges: Sequence[Tuple[Union[int, str], Union[int, str]]],
    faces_or_cells: Sequence[Sequence[Union[int, str]]],
) -> np.ndarray:
    """Compute signed boundary matrix B2 (edges x faces/cells).

    Each face or cell is an ordered sequence of vertices defining the cycle (v0, v1, ..., vk-1).
    The boundary of the cell consists of the oriented step edges (v0, v1), (v1, v2), ..., (vk-1, v0).

    Convention:
        If cell c contains the directed step (u, v):
            If edge e_i = (u, v), then B2[i, c] = +1 (aligned)
            If edge e_i = (v, u), then B2[i, c] = -1 (oppositely oriented)
    """
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_map[(u, v)] = (idx, 1.0)
        edge_map[(v, u)] = (idx, -1.0)

    num_edges = len(edges)
    num_cells = len(faces_or_cells)

    b2 = np.zeros((num_edges, num_cells), dtype=np.float64)

    for cell_idx, cell_nodes in enumerate(faces_or_cells):
        k = len(cell_nodes)
        if k < 3:
            raise ValueError(f"Cell must have at least 3 vertices, got {k}")

        for step in range(k):
            u = cell_nodes[step]
            v = cell_nodes[(step + 1) % k]

            if (u, v) not in edge_map:
                raise ValueError(f"Cell boundary segment ({u}, {v}) not found in edge list")

            edge_idx, orientation = edge_map[(u, v)]
            b2[edge_idx, cell_idx] += orientation

    return b2


def compute_hodge_laplacian_0(b1: np.ndarray) -> np.ndarray:
    """Compute 0-Hodge Laplacian (node Laplacian): L0 = B1 @ B1.T."""
    return b1 @ b1.T


def compute_hodge_laplacian_1(b1: np.ndarray, b2: np.ndarray = None) -> np.ndarray:
    """Compute 1-Hodge Laplacian (edge Laplacian): L1 = B1.T @ B1 + B2 @ B2.T."""
    l1 = b1.T @ b1
    if b2 is not None and b2.size > 0:
        if b2.shape[0] != b1.shape[1]:
            raise ValueError(
                f"Dimension mismatch: B2 has {b2.shape[0]} rows, but B1 has {b1.shape[1]} columns"
            )
        l1 = l1 + b2 @ b2.T
    return l1
