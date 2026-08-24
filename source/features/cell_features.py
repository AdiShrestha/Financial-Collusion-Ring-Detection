"""Polygonal 2-cell feature extraction supporting arbitrary cycle lengths."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


def extract_cell_features(
    cell_nodes: Sequence[str],
    candidate_edges: Sequence[Tuple[str, str, Dict[str, Any]]],
    node_features_map: Optional[Dict[str, Sequence[float]]] = None,
) -> List[float]:
    """Extract higher-order topological features for a polygonal 2-cell of arbitrary length k >= 3.

    Calculates:
    - cycle_length (k)
    - total_flow
    - mean_flow
    - min_flow
    - max_flow
    - flow_std
    - temporal_span
    - direction_consistency (fraction of steps whose forward direction exists in edges)

    Returns:
        List of 8 float features.
    """
    k = len(cell_nodes)
    if k < 3:
        raise ValueError(f"Polygonal cell must have at least 3 nodes, got {k}")

    # Build directed edge lookup
    dir_edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    undir_edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for u, v, attr in candidate_edges:
        u_s, v_s = str(u), str(v)
        dir_edge_map[(u_s, v_s)] = dict(attr)
        undir_edge_map[(min(u_s, v_s), max(u_s, v_s))] = dict(attr)

    step_amounts: List[float] = []
    step_timestamps: List[float] = []
    forward_aligned_count = 0

    for step in range(k):
        u = str(cell_nodes[step])
        v = str(cell_nodes[(step + 1) % k])

        # Check forward direction
        if (u, v) in dir_edge_map:
            forward_aligned_count += 1
            attr = dir_edge_map[(u, v)]
        elif (v, u) in dir_edge_map:
            attr = dir_edge_map[(v, u)]
        else:
            attr = undir_edge_map.get((min(u, v), max(u, v)), {})

        amt = float(attr.get("amount", attr.get("amount_paid", attr.get("amount_received", 0.0))))
        ts = float(attr.get("timestamp", 0.0))

        step_amounts.append(amt)
        step_timestamps.append(ts)

    cycle_length = float(k)
    total_flow = float(np.sum(step_amounts))
    mean_flow = float(np.mean(step_amounts)) if step_amounts else 0.0
    min_flow = float(np.min(step_amounts)) if step_amounts else 0.0
    max_flow = float(np.max(step_amounts)) if step_amounts else 0.0
    flow_std = float(np.std(step_amounts)) if step_amounts else 0.0
    temporal_span = float(max(step_timestamps) - min(step_timestamps)) if step_timestamps else 0.0
    direction_consistency = float(forward_aligned_count / k)

    return [
        cycle_length,
        total_flow,
        mean_flow,
        min_flow,
        max_flow,
        flow_std,
        temporal_span,
        direction_consistency,
    ]


def extract_batch_cell_features(
    cells: Sequence[Sequence[str]],
    candidate_edges: Sequence[Tuple[str, str, Dict[str, Any]]],
) -> np.ndarray:
    """Extract cell feature matrix for all rank-2 cells in a complex."""
    feature_rows = []
    for c in cells:
        feats = extract_cell_features(c, candidate_edges)
        feature_rows.append(feats)

    if feature_rows:
        return np.array(feature_rows, dtype=np.float32)
    return np.zeros((0, 8), dtype=np.float32)
