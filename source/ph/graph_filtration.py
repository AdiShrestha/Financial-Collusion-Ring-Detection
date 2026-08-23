"""Persistent homology graph filtration builders with temporal monotonicity guarantees."""

from typing import Dict, List, Sequence, Tuple, Union
import numpy as np


def build_temporal_ph_graph(
    nodes_with_times: Dict[Union[int, str], float],
    edges_with_times: Sequence[Tuple[Union[int, str], Union[int, str], float]],
    disallow_higher_simplices: bool = True,
) -> List[Tuple[List[Union[int, str]], float]]:
    """Construct a temporally filtered 1-skeleton (PHGraphView) simplicial complex.

    Guarantees:
    1. Monotonicity: for every edge e = (u, v), filtration(e) >= max(filtration(u), filtration(v)).
    2. Strict 1-skeleton: Contains only 0-simplices (vertices) and 1-simplices (edges).
       If any element of dimension >= 2 is provided, raises ValueError (INV-003).

    Args:
        nodes_with_times: Mapping of node identifier to discovery/first_seen timestamp.
        edges_with_times: Sequence of (u, v, timestamp) tuples.
        disallow_higher_simplices: If True, enforces strict 1-skeleton invariant (INV-003).

    Returns:
        List of (simplex_nodes, filtration_value) pairs sorted monotonically.
    """
    filtered_simplices: List[Tuple[List[Union[int, str]], float]] = []

    # 1. Add 0-simplices (vertices)
    for node, t_v in nodes_with_times.items():
        filtered_simplices.append(([node], float(t_v)))

    # 2. Add 1-simplices (edges) with monotonic clamping
    for edge_item in edges_with_times:
        if len(edge_item) != 3:
            raise ValueError(
                f"Edge entry must be exactly (u, v, timestamp) [len 3], got length {len(edge_item)} ({edge_item}). "
                f"2-simplices and 2-cells are strictly prohibited in PHGraphView per INV-003."
            )

        u, v, raw_t_e = edge_item[0], edge_item[1], edge_item[2]

        if u not in nodes_with_times:
            raise ValueError(f"Edge endpoint {u} not found in node timestamp map")
        if v not in nodes_with_times:
            raise ValueError(f"Edge endpoint {v} not found in node timestamp map")

        t_u = float(nodes_with_times[u])
        t_v = float(nodes_with_times[v])

        # Monotonicity rule: edge filtration value must be >= max(t_u, t_v, raw_t_e)
        t_e = max(t_u, t_v, float(raw_t_e))

        simplex = sorted([u, v], key=lambda x: str(x))
        filtered_simplices.append((simplex, t_e))

    # 3. Sort simplices by (filtration_value, dimension)
    filtered_simplices.sort(key=lambda s: (s[1], len(s[0])))

    return filtered_simplices


def validate_filtration_monotonicity(
    filtered_simplices: Sequence[Tuple[Sequence[Union[int, str]], float]],
) -> bool:
    """Verify that all face inclusions have non-decreasing filtration values.

    Returns True if monotonic, raises ValueError otherwise.
    """
    simplex_map = {tuple(sorted(map(str, s[0]))): s[1] for s in filtered_simplices}

    for s_nodes, s_time in filtered_simplices:
        nodes = list(s_nodes)
        dim = len(nodes) - 1
        if dim == 1:
            # Check endpoint vertices
            u_key = (str(nodes[0]),)
            v_key = (str(nodes[1]),)
            t_u = simplex_map.get(u_key)
            t_v = simplex_map.get(v_key)

            if t_u is None or v_key not in simplex_map:
                raise ValueError(f"Edge {nodes} has missing endpoint in simplex complex")

            if s_time < t_u - 1e-9 or s_time < t_v - 1e-9:
                raise ValueError(
                    f"Monotonicity violation: edge {nodes} enters at t={s_time}, "
                    f"earlier than endpoints (t_u={t_u}, t_v={t_v})"
                )

    return True
