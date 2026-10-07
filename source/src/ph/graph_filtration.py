"""
Unfilled 1-Skeleton Temporal Graph Filtration Engine for Financial Networks.

Upholds Invariants:
- INV-003 (Unfilled 1-Skeleton for H1): Primary persistent homology is
  strictly computed on an unfilled 1-skeleton (dim <= 1). No 2-simplices
  or filled cycle cells are permitted.
- Monotonic Event-Time Filtration: filt(e) >= max(filt(u), filt(v)).
"""

from dataclasses import dataclass
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# Resolve imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))


class FilledSkeletonViolationError(Exception):
    """Raised when a simplex of dimension >= 2 is inserted into PHGraphView (INV-003)."""
    pass


class FiltrationMonotonicityError(Exception):
    """Raised when an edge enters before either of its incident vertices."""
    pass


@dataclass(frozen=True)
class FilteredSimplex:
    """Represents an individual simplex with assigned filtration value."""
    simplex: Tuple[str, ...]
    dimension: int
    filtration_val: float

    def __post_init__(self):
        if self.dimension > 1:
            raise FilledSkeletonViolationError(
                f"Invariant INV-003 Violated: Simplex {self.simplex} has dimension "
                f"{self.dimension} > 1. Persistent homology requires an unfilled 1-skeleton."
            )


@dataclass
class PHGraphView:
    """Dedicated 1-dimensional simplicial complex skeleton with temporal filtration."""
    candidate_id: str
    vertices: List[FilteredSimplex]
    edges: List[FilteredSimplex]
    label: int
    node_map: Dict[str, int]

    @property
    def max_dimension(self) -> int:
        return 1

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_gudhi_simplex_tree(self) -> Any:
        """Constructs and populates a GUDHI SimplexTree with the filtered 1-skeleton."""
        try:
            import gudhi
        except ImportError:
            raise ImportError("GUDHI library is required for SimplexTree conversion.")

        st = gudhi.SimplexTree()

        # Insert 0-simplices (vertices)
        for v in self.vertices:
            v_int = self.node_map[v.simplex[0]]
            st.insert([v_int], filtration=float(v.filtration_val))

        # Insert 1-simplices (edges)
        for e in self.edges:
            u_int = self.node_map[e.simplex[0]]
            v_int = self.node_map[e.simplex[1]]
            st.insert([u_int, v_int], filtration=float(e.filtration_val))

        # Verify filtration is non-decreasing with respect to faces
        st.make_filtration_non_decreasing()
        return st

    def compute_persistence(
        self, infinity_replacement: Optional[float] = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Computes exact H0 and H1 persistence diagrams on the filtered 1-skeleton.
        If GUDHI is available, uses GUDHI SimplexTree; otherwise uses exact DSU reduction.
        """
        try:
            import gudhi
            st = self.to_gudhi_simplex_tree()
            st.compute_persistence(persistence_dim_max=True)
            h0_raw = st.persistence_intervals_in_dimension(0)
            h1_raw = st.persistence_intervals_in_dimension(1)
            h0 = [(float(b), float(d)) for b, d in h0_raw]
            h1 = [(float(b), float(d)) for b, d in h1_raw]
            if infinity_replacement is not None:
                max_t = max([v.filtration_val for v in self.vertices] + [e.filtration_val for e in self.edges] + [0.0])
                rep_val = max_t + infinity_replacement
                h0 = [(b, rep_val if np.isinf(d) else d) for b, d in h0]
                h1 = [(b, rep_val if np.isinf(d) else d) for b, d in h1]
            return h0, h1
        except ImportError:
            pass

        # Exact algebraic DSU reduction for 1-skeleton persistence
        parent: Dict[str, str] = {}
        birth: Dict[str, float] = {}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        for v in self.vertices:
            node = v.simplex[0]
            parent[node] = node
            birth[node] = v.filtration_val

        h0_intervals: List[Tuple[float, float]] = []
        h1_intervals: List[Tuple[float, float]] = []

        # Sort edges by filtration time
        sorted_edges = sorted(self.edges, key=lambda e: (e.filtration_val, e.simplex))

        for e in sorted_edges:
            u, v = e.simplex
            t_e = e.filtration_val
            r_u = find(u)
            r_v = find(v)

            if r_u != r_v:
                # Two components merge: younger component dies
                b_u = birth[r_u]
                b_v = birth[r_v]
                if b_u <= b_v:
                    older, younger = r_u, r_v
                else:
                    older, younger = r_v, r_u

                h0_intervals.append((birth[younger], t_e))
                parent[younger] = older
            else:
                # Both endpoints already in same component: 1-cycle is born
                h1_intervals.append((t_e, float("inf")))

        # Record infinite H0 bars for remaining roots
        unique_roots = set(find(v.simplex[0]) for v in self.vertices)
        for r in sorted(unique_roots, key=lambda root: birth[root]):
            h0_intervals.append((birth[r], float("inf")))

        if infinity_replacement is not None:
            max_t = max([v.filtration_val for v in self.vertices] + [e.filtration_val for e in self.edges] + [0.0])
            rep_val = max_t + infinity_replacement
            h0_intervals = [(b, rep_val if np.isinf(d) else d) for b, d in h0_intervals]
            h1_intervals = [(b, rep_val if np.isinf(d) else d) for b, d in h1_intervals]

        return h0_intervals, h1_intervals


class TemporalFiltrationBuilder:
    """Builds monotonic event-time filtrations over candidate subgraphs."""

    def __init__(self, time_scale: float = 1.0):
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive and finite")
        self.time_scale = time_scale

    def build(self, candidate: Any) -> PHGraphView:
        """
        Extracts an unfilled 1-skeleton from candidate:
        1. Vertices enter at min(incident_transaction_timestamps).
        2. Edges enter at min(transaction_timestamps_between_endpoints).
        3. Enforces filt(e) >= max(filt(u), filt(v)).
        4. Asserts no 2-simplices exist (INV-003).
        """
        if hasattr(candidate, "candidate_id"):
            cand_id = str(candidate.candidate_id)
            raw_nodes = list(candidate.nodes)
            raw_edges = list(candidate.edges)
            label = getattr(candidate, "label", None)
        elif isinstance(candidate, dict):
            cand_id = str(candidate["candidate_id"])
            raw_nodes = list(candidate["nodes"])
            raw_edges = list(candidate["edges"])
            label = candidate.get("label")
        else:
            raise TypeError(f"Unsupported candidate object type: {type(candidate)}")
        if label not in (0, 1):
            raise ValueError(f"Candidate {cand_id} requires an explicit binary label for PH caching")
        label = int(label)

        # Collect unique node identifiers
        normalized_nodes = [str(n) for n in raw_nodes]
        if not normalized_nodes or any(not n for n in normalized_nodes):
            raise ValueError(f"Candidate {cand_id} requires nonempty node identifiers")
        if len(set(normalized_nodes)) != len(normalized_nodes):
            raise ValueError(f"Candidate {cand_id} contains duplicate node identifiers")
        nodes = sorted(normalized_nodes)
        node_map = {n: i for i, n in enumerate(nodes)}

        # Aggregate transactions by undirected edge pair
        edge_events: Dict[Tuple[str, str], List[float]] = {}
        node_events: Dict[str, List[float]] = {n: [] for n in nodes}

        for e in raw_edges:
            if isinstance(e, dict):
                if not {"source", "target", "timestamp"}.issubset(e):
                    raise ValueError(f"Candidate {cand_id} edge lacks source, target, or timestamp: {e!r}")
                u = str(e["source"])
                v = str(e["target"])
                ts = float(e["timestamp"])
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                u = str(e[0])
                v = str(e[1])
                ts = float(e[2])
            else:
                raise ValueError(f"Candidate {cand_id} contains a malformed temporal edge: {e!r}")

            if u not in node_map or v not in node_map:
                raise ValueError(f"Candidate {cand_id} edge references an undeclared node: {(u, v)}")
            if u == v:
                continue  # Skip self-loops in 1-skeleton
            if not math.isfinite(ts):
                raise ValueError(f"Candidate {cand_id} edge has non-finite timestamp: {ts!r}")

            pair = (min(u, v), max(u, v))
            if pair not in edge_events:
                edge_events[pair] = []
            edge_events[pair].append(ts)
            node_events[u].append(ts)
            node_events[v].append(ts)

        # 1. Compute vertex filtration values: first observed transaction time.
        vertex_simplices: List[FilteredSimplex] = []
        node_birth: Dict[str, float] = {}

        for n in nodes:
            events = node_events[n]
            if not events:
                raise ValueError(f"Candidate {cand_id} node {n!r} has no observed transaction time")
            t_birth = min(events) * self.time_scale
            if not math.isfinite(t_birth):
                raise ValueError(f"Candidate {cand_id} node {n!r} has an invalid filtration time")
            node_birth[n] = t_birth
            vertex_simplices.append(
                FilteredSimplex(simplex=(n,), dimension=0, filtration_val=t_birth)
            )

        # 2. Compute edge filtration values: first transaction time between endpoints
        edge_simplices: List[FilteredSimplex] = []
        for (u, v), events in sorted(edge_events.items()):
            e_birth = min(events) * self.time_scale
            if not math.isfinite(e_birth):
                raise ValueError(f"Candidate {cand_id} edge {(u, v)} has an invalid filtration time")

            # Monotonicity check
            u_birth = node_birth[u]
            v_birth = node_birth[v]
            if e_birth < max(u_birth, v_birth) - 1e-12:
                raise FiltrationMonotonicityError(
                    f"Filtration non-monotonic: Edge ({u}, {v}) birth {e_birth} < "
                    f"max(u_birth={u_birth}, v_birth={v_birth})"
                )

            edge_simplices.append(
                FilteredSimplex(simplex=(u, v), dimension=1, filtration_val=e_birth)
            )

        return PHGraphView(
            candidate_id=cand_id,
            vertices=vertex_simplices,
            edges=edge_simplices,
            label=label,
            node_map=node_map,
        )


def build_temporal_filtration(candidate: Any) -> PHGraphView:
    """Convenience helper to build an unfilled 1-skeleton PHGraphView."""
    builder = TemporalFiltrationBuilder()
    return builder.build(candidate)
