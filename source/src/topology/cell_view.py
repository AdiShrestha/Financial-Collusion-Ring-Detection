"""
Regular Cell Complex Domain Lifter for Financial Collusion Rings.

Upholds Invariants:
- INV-002 (Simplex vs Polygonal Cell Separation): k-cycles for k >= 4 are
  retained as rank-2 polygonal cells and never triangulated into simplices.
- INV-004 (Boundary Operator Nilpotency): Exactly satisfies B1 @ B2 = 0.
"""

from dataclasses import dataclass
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.sparse as sp

# Resolve imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.topology.incidence import (
        build_b1,
        build_b2_cellular,
        compute_hodge_laplacians,
        verify_nilpotency,
    )
except ModuleNotFoundError:
    from src.topology.incidence import (
        build_b1,
        build_b2_cellular,
        compute_hodge_laplacians,
        verify_nilpotency,
    )


class NilpotencyViolationError(Exception):
    """Raised when boundary operator composition B1 @ B2 != 0."""
    pass


@dataclass
class CellComplex:
    """Represents a regular 2-dimensional cell complex with Hodge linear operators."""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    cells: List[Tuple[str, ...]]
    x_0: np.ndarray
    x_1: np.ndarray
    x_2: np.ndarray
    B1: sp.csr_matrix
    B2: sp.csr_matrix
    L0: sp.csr_matrix
    L1_down: sp.csr_matrix
    L1_up: sp.csr_matrix
    L1: sp.csr_matrix
    L2: sp.csr_matrix
    label: int


class CellComplexLifter:
    """Lifts candidate subgraphs into regular cell complexes."""

    def __init__(self, tolerance: float = 1e-12):
        self.tolerance = tolerance

    def lift(self, candidate: Any) -> CellComplex:
        """
        Converts CandidateSubgraph into a CellComplex:
        1. 0-cells: Unique nodes in the candidate.
        2. 1-cells: Directed edges in the candidate.
        3. 2-cells: Polygonal cycle attached along candidate.cycle_nodes (k >= 3).
        4. Verifies B1 @ B2 = 0.
        5. Computes Hodge Laplacians L0, L1, L2.
        """
        if hasattr(candidate, "nodes"):
            raw_nodes = list(candidate.nodes)
            raw_edges = list(candidate.edges)
            cycle_nodes = list(getattr(candidate, "cycle_nodes", []))
            cycle_length = int(getattr(candidate, "cycle_length", len(cycle_nodes)))
            label = getattr(candidate, "label", None)
        elif isinstance(candidate, dict):
            raw_nodes = list(candidate["nodes"])
            raw_edges = list(candidate["edges"])
            cycle_nodes = list(candidate.get("cycle_nodes", []))
            cycle_length = int(candidate.get("cycle_length", len(cycle_nodes)))
            label = candidate.get("label")
        else:
            raise TypeError(f"Unsupported candidate type: {type(candidate)}")
        if label not in (0, 1):
            raise ValueError("Cell lifting requires an explicit binary label")
        label = int(label)

        # 1. Canonical Node Ordering
        nodes = sorted(list(set(str(n) for n in raw_nodes)))

        # 2. Extract 1-cells (Edges)
        edge_list: List[Tuple[str, str]] = []
        edge_attrs: List[List[float]] = []
        seen_edges = set()
        edge_position: Dict[Tuple[str, str], int] = {}

        for e in raw_edges:
            if isinstance(e, dict):
                u = str(e.get("source", ""))
                v = str(e.get("target", ""))
                amt = float(e.get("amount", 0.0))
                ts = float(e.get("timestamp", 0.0))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                u = str(e[0])
                v = str(e[1])
                amt = 1.0
                ts = 0.0
            else:
                continue

            # A self-transfer is observed data, but it is not a regular
            # oriented 1-cell with distinct endpoints. Other views retain it.
            if u == v:
                continue
            pair = (u, v)
            if pair not in seen_edges:
                seen_edges.add(pair)
                edge_position[pair] = len(edge_list)
                edge_list.append(pair)
                edge_attrs.append([1.0, ts])
            else:
                i = edge_position[pair]
                edge_attrs[i][0] += 1.0
                edge_attrs[i][1] = min(edge_attrs[i][1], ts)

        # A missing observed transaction cannot be repaired by inventing an edge.
        if len(cycle_nodes) >= 3:
            k = len(cycle_nodes)
            for i in range(k):
                u = str(cycle_nodes[i])
                v = str(cycle_nodes[(i + 1) % k])
                if (u, v) not in seen_edges and (v, u) not in seen_edges:
                    raise ValueError(f"Cycle boundary edge ({u}, {v}) is absent from observed transactions")

        # 3. Form 2-cells: Preserve k-cycle as a polygonal 2-cell (INV-002)
        cells: List[Tuple[str, ...]] = []
        cycle_cells_list: List[List[str]] = []
        cell_attrs: List[List[float]] = []

        if len(cycle_nodes) >= 3:
            cycle_tuple = tuple(str(n) for n in cycle_nodes)
            cells.append(cycle_tuple)
            cycle_cells_list.append(list(cycle_tuple))
            cell_attrs.append([float(len(cycle_tuple)), 1.0])

        # 4. Compute Boundary Operators
        B1 = build_b1(nodes, edge_list)
        B2 = build_b2_cellular(edge_list, cycle_cells_list)

        # 5. Verify Invariant INV-004: Boundary Nilpotency B1 @ B2 = 0
        valid, max_res = verify_nilpotency(B1, B2, tol=self.tolerance)
        if not valid:
            raise NilpotencyViolationError(
                f"Invariant INV-004 Violated: ||B1 @ B2||_inf = {max_res} > {self.tolerance}"
            )

        # 6. Compute Hodge Laplacians
        L0, L1, L2 = compute_hodge_laplacians(B1, B2)
        B1_t = B1.T
        B2_t = B2.T
        L1_down = B1_t.dot(B1)
        L1_up = B2.dot(B2_t)

        # 7. Construct Feature Tensors
        cycle_set = set(str(n) for n in cycle_nodes)
        x_0 = np.zeros((len(nodes), 2), dtype=np.float32)
        for i, n in enumerate(nodes):
            x_0[i, 0] = 1.0 if n in cycle_set else 0.0
            x_0[i, 1] = float(len(cycle_nodes)) if n in cycle_set else 0.0

        if edge_attrs:
            origin = min(row[1] for row in edge_attrs)
            x_1 = np.array([[np.log1p(row[0]), row[1] - origin] for row in edge_attrs], dtype=np.float32)
        else:
            x_1 = np.zeros((len(edge_list), 2), dtype=np.float32)
        x_2 = np.array(cell_attrs, dtype=np.float32) if cell_attrs else np.zeros((len(cells), 2), dtype=np.float32)

        return CellComplex(
            nodes=nodes,
            edges=edge_list,
            cells=cells,
            x_0=x_0,
            x_1=x_1,
            x_2=x_2,
            B1=B1,
            B2=B2,
            L0=L0,
            L1_down=L1_down,
            L1_up=L1_up,
            L1=L1,
            L2=L2,
            label=label,
        )


def lift_to_cell_complex(candidate: Any) -> CellComplex:
    """Convenience helper to lift a candidate subgraph to a CellComplex."""
    lifter = CellComplexLifter()
    return lifter.lift(candidate)
