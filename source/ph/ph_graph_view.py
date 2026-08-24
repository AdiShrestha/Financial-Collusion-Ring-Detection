"""PHGraphView 1-skeleton persistent homology filtration constructor."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from source.data.candidate_extractor import CandidateExample
from source.ph.graph_filtration import build_temporal_ph_graph, validate_filtration_monotonicity


class PHGraphView:
    """Constructs and validates a 1-skeleton simplicial filtration for persistent homology.

    In accordance with INV-003:
    - Contains ONLY 0-simplices (vertices) and 1-simplices (edges).
    - 2-simplices, triangles, and polygonal 2-cells are STRICTLY PROHIBITED.
    - Preserves 1-dimensional cycle persistence barcodes across temporal filtrations.
    """

    def __init__(
        self,
        candidate_id: str,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        filtration_type: str,
        simplices: List[Tuple[List[str], float]],
        target_y: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.candidate_id = candidate_id
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.filtration_type = filtration_type
        self.simplices = list(simplices)
        self.target_y = int(target_y)
        self.metadata = dict(metadata or {})

        # Verify strict 1-skeleton invariant (INV-003)
        for s_nodes, _ in self.simplices:
            if len(s_nodes) > 2:
                raise ValueError(
                    f"2-simplex or higher-dimensional cell found in PHGraphView for candidate "
                    f"{self.candidate_id}: {s_nodes}. Violates INV-003."
                )

        # Verify monotonicity
        validate_filtration_monotonicity(self.simplices)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_simplices(self) -> int:
        return len(self.simplices)

    @classmethod
    def from_candidate_example(
        cls,
        candidate: CandidateExample,
        filtration_type: str = "temporal",
    ) -> "PHGraphView":
        """Construct PHGraphView from CandidateExample under specified filtration."""
        nodes = list(candidate.participant_ids)
        t_start, _ = candidate.temporal_bounds

        # 1. Compute node and edge filtration values based on type
        nodes_with_times: Dict[str, float] = {}
        edges_with_times: List[Tuple[str, str, float]] = []

        if filtration_type == "temporal":
            # Node discovery time = minimum transaction timestamp touching the node
            node_min_time: Dict[str, float] = {n: float("inf") for n in nodes}
            raw_edge_tuples = []

            for u, v, attr in candidate.edges:
                u_s, v_s = str(u), str(v)
                ts = float(attr.get("timestamp", t_start))
                raw_edge_tuples.append((u_s, v_s, ts))
                node_min_time[u_s] = min(node_min_time[u_s], ts)
                node_min_time[v_s] = min(node_min_time[v_s], ts)

            for n in nodes:
                nodes_with_times[n] = node_min_time[n] if node_min_time[n] != float("inf") else t_start

            for u_s, v_s, ts in raw_edge_tuples:
                edges_with_times.append((u_s, v_s, ts))

        elif filtration_type == "amount":
            # Filtration ordered by transaction amount
            node_max_amt: Dict[str, float] = {n: 0.0 for n in nodes}
            raw_edge_tuples = []

            for u, v, attr in candidate.edges:
                u_s, v_s = str(u), str(v)
                amt = float(attr.get("amount", attr.get("amount_paid", 0.0)))
                raw_edge_tuples.append((u_s, v_s, amt))
                node_max_amt[u_s] = max(node_max_amt[u_s], amt)
                node_max_amt[v_s] = max(node_max_amt[v_s], amt)

            for n in nodes:
                # Nodes enter at 0.0 (or min amount), edges at their amount
                nodes_with_times[n] = 0.0

            for u_s, v_s, amt in raw_edge_tuples:
                edges_with_times.append((u_s, v_s, amt))

        elif filtration_type == "degree":
            # Filtration ordered by node degree
            degrees: Dict[str, int] = {n: 0 for n in nodes}
            for u, v, _ in candidate.edges:
                degrees[str(u)] = degrees.get(str(u), 0) + 1
                degrees[str(v)] = degrees.get(str(v), 0) + 1

            for n in nodes:
                nodes_with_times[n] = float(degrees[n])

            for u, v, _ in candidate.edges:
                u_s, v_s = str(u), str(v)
                # Edge enters at max degree of its endpoints
                deg_e = float(max(degrees[u_s], degrees[v_s]))
                edges_with_times.append((u_s, v_s, deg_e))
        else:
            raise ValueError(f"Unsupported filtration_type: '{filtration_type}'")

        # 2. Build monotonically ordered 1-skeleton filtration
        simplices = build_temporal_ph_graph(
            nodes_with_times=nodes_with_times,
            edges_with_times=edges_with_times,
            disallow_higher_simplices=True,
        )

        return cls(
            candidate_id=candidate.candidate_id,
            nodes=nodes,
            edges=edges_with_times,
            filtration_type=filtration_type,
            simplices=simplices,
            target_y=candidate.target_y,
            metadata={
                **candidate.metadata,
                "typology_label": candidate.typology_label,
                "dataset_track": candidate.dataset_track,
                "filtration_type": filtration_type,
            },
        )
