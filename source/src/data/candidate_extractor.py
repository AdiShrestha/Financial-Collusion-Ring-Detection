"""Label-Blind Candidate Cycle and Motif Subgraph Extractor.

Discovers simple cycles and flow circulation motifs in financial transaction networks,
extracts induced h-hop neighborhoods, and structures candidates into CandidateSubgraph records
without label peeking.

Traces to Contract C02-01, FR-001, FR-002, INV-002, INV-007.
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx


@dataclass
class CandidateSubgraph:
    candidate_id: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    cycle_length: int
    cycle_nodes: List[str]
    label: Optional[int] = None
    pattern_id: Optional[str] = None
    overlap_score: Optional[float] = None
    matched_group_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "nodes": list(self.nodes),
            "edges": [dict(e) for e in self.edges],
            "cycle_length": self.cycle_length,
            "cycle_nodes": list(self.cycle_nodes),
            "label": self.label,
            "pattern_id": self.pattern_id,
            "overlap_score": self.overlap_score,
            "matched_group_id": self.matched_group_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateSubgraph":
        return cls(
            candidate_id=data["candidate_id"],
            nodes=list(data["nodes"]),
            edges=[dict(e) for e in data["edges"]],
            cycle_length=int(data["cycle_length"]),
            cycle_nodes=list(data["cycle_nodes"]),
            label=data.get("label"),
            pattern_id=data.get("pattern_id"),
            overlap_score=data.get("overlap_score"),
            matched_group_id=data.get("matched_group_id"),
        )


class CandidateExtractor:
    """Label-blind cycle finder and subgraph neighborhood extractor."""

    def __init__(self, min_k: int = 3, max_k: int = 6, max_cycles: int = 1000):
        if min_k < 1 or max_k < min_k:
            raise ValueError("Cycle length bounds must satisfy 1 <= min_k <= max_k")
        if max_cycles <= 0:
            raise ValueError("max_cycles must be a positive, predeclared limit")
        self.min_k = min_k
        self.max_k = max_k
        self.max_cycles = max_cycles

    def find_simple_cycles(
        self,
        graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
        max_cycles: Optional[int] = None,
    ) -> List[List[str]]:
        """Finds simple cycles of length k in [min_k, max_k] using bounded DFS.

        Cycles are returned in canonical form: ordered sequence of node IDs
        beginning at the lexicographically minimum node in the cycle.
        Operates purely on graph topology without accessing label attributes.
        """
        min_len = min_k if min_k is not None else self.min_k
        max_len = max_k if max_k is not None else self.max_k
        limit = max_cycles if max_cycles is not None else self.max_cycles
        if min_len < 1 or max_len < min_len:
            raise ValueError("Cycle length bounds must satisfy 1 <= min_k <= max_k")
        if limit <= 0:
            raise ValueError("max_cycles must be a positive, predeclared limit")

        is_directed = graph.is_directed()
        nodes = sorted(list(graph.nodes()))
        node_order = {node: i for i, node in enumerate(nodes)}

        found_cycles: List[List[str]] = []
        seen_cycles: Set[Tuple[str, ...]] = set()
        visited_in_path: Set[str] = set()

        def is_over_limit() -> bool:
            # Retain one extra cycle so callers can distinguish a complete
            # enumeration of exactly `limit` cycles from a truncated result.
            return len(found_cycles) > limit

        def dfs(start_node: str, current_node: str, depth: int, path: List[str]):
            if is_over_limit():
                return

            start_idx = node_order[start_node]

            # Check outbound neighbors
            neighbors = graph.neighbors(current_node) if not is_directed else graph.successors(current_node)
            for nbr in sorted(neighbors, key=node_order.__getitem__):
                if is_over_limit():
                    return

                if nbr == start_node:
                    if min_len <= depth <= max_len:
                        cycle = list(path)
                        if is_directed:
                            key = tuple(cycle)
                        else:
                            reverse = (cycle[0], *reversed(cycle[1:]))
                            key = min(tuple(cycle), reverse)
                        if key not in seen_cycles:
                            seen_cycles.add(key)
                            found_cycles.append(list(key))
                elif depth < max_len:
                    # Enforce ordering to avoid permutation duplicates: only visit nodes > start_node
                    if node_order[nbr] > start_idx:
                        # For undirected graphs, avoid immediate backtrack along the same edge
                        if not is_directed and len(path) >= 2 and nbr == path[-2]:
                            continue
                        if nbr not in visited_in_path:
                            visited_in_path.add(nbr)
                            path.append(nbr)
                            dfs(start_node, nbr, depth + 1, path)
                            path.pop()
                            visited_in_path.remove(nbr)

        for n in nodes:
            if is_over_limit():
                break
            visited_in_path.add(n)
            dfs(n, n, 1, [n])
            visited_in_path.remove(n)

        if is_over_limit():
            raise ValueError(
                f"Cycle enumeration exceeded the predeclared cap ({limit}); "
                "the candidate population is incomplete"
            )
        return found_cycles

    def extract_subgraph(
        self,
        graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
        cycle_nodes: List[str],
        hop_radius: int = 1,
        candidate_id: str = "cand_000",
    ) -> CandidateSubgraph:
        """Extracts the induced subgraph of nodes within hop_radius of cycle_nodes."""
        # 1. Multi-source BFS to find all nodes within hop_radius
        subgraph_nodes: Set[str] = set(cycle_nodes)
        queue = deque([(node, 0) for node in cycle_nodes])
        visited_hops = {node: 0 for node in cycle_nodes}

        while queue:
            curr, dist = queue.popleft()
            if dist < hop_radius:
                # In financial graphs, edges can be incoming or outgoing
                if graph.is_directed():
                    nbrs = set(graph.successors(curr)) | set(graph.predecessors(curr))
                else:
                    nbrs = set(graph.neighbors(curr))

                for nbr in nbrs:
                    if nbr not in visited_hops or visited_hops[nbr] > dist + 1:
                        visited_hops[nbr] = dist + 1
                        subgraph_nodes.add(nbr)
                        queue.append((nbr, dist + 1))

        # 2. Extract edges incident between subgraph_nodes
        ordered_nodes = sorted(list(subgraph_nodes))
        extracted_edges: List[Dict[str, Any]] = []

        if graph.is_multigraph():
            for u in ordered_nodes:
                for v in graph[u]:
                    if v in subgraph_nodes:
                        for key, edge_data in graph[u][v].items():
                            edge_record = {
                                "source": u,
                                "target": v,
                                "tx_id": edge_data.get("tx_id", f"{u}_{v}_{key}"),
                                "timestamp": edge_data.get("timestamp", ""),
                                "amount": edge_data.get("amount", 0.0),
                            }
                            # Copy any additional attributes
                            for k, val in edge_data.items():
                                if k not in edge_record:
                                    edge_record[k] = val
                            extracted_edges.append(edge_record)
        else:
            for u in ordered_nodes:
                for v in graph[u]:
                    if v in subgraph_nodes:
                        edge_data = graph[u][v]
                        edge_record = {
                            "source": u,
                            "target": v,
                            "tx_id": edge_data.get("tx_id", f"{u}_{v}"),
                            "timestamp": edge_data.get("timestamp", ""),
                            "amount": edge_data.get("amount", 0.0),
                        }
                        for k, val in edge_data.items():
                            if k not in edge_record:
                                edge_record[k] = val
                        extracted_edges.append(edge_record)

        return CandidateSubgraph(
            candidate_id=candidate_id,
            nodes=ordered_nodes,
            edges=extracted_edges,
            cycle_length=len(cycle_nodes),
            cycle_nodes=list(cycle_nodes),
        )

    def label_candidates(
        self,
        candidates: List[CandidateSubgraph],
    ) -> List[CandidateSubgraph]:
        """Apply the same strict central-cycle target used by the AML benchmark.

        The source CSV's transaction labels are required. Pattern blocks are
        metadata only; partial pair overlap cannot establish a positive ring.
        Ambiguous neighborhoods are excluded rather than declared benign.
        """
        labelled: List[CandidateSubgraph] = []
        for cand in candidates:
            if not cand.cycle_nodes or any("is_laundering" not in e for e in cand.edges):
                raise ValueError("Cycle and observed transaction labels are required for ground truth")
            pairs = {(cand.cycle_nodes[i], cand.cycle_nodes[(i + 1) % len(cand.cycle_nodes)])
                     for i in range(len(cand.cycle_nodes))}
            positive = all(any(e["source"] == u and e["target"] == v and
                               int(e["is_laundering"]) == 1 for e in cand.edges)
                           for u, v in pairs)
            negative = all(int(e["is_laundering"]) == 0 for e in cand.edges)
            if not positive and not negative:
                continue
            cand.label = int(positive)
            ids = {e.get("pattern_id") for e in cand.edges if
                   (e["source"], e["target"]) in pairs and e.get("is_laundering") == 1 and e.get("pattern_id")}
            cand.pattern_id = next(iter(ids)) if positive and len(ids) == 1 else None
            cand.overlap_score = None
            for e in cand.edges:
                e.pop("is_laundering", None)
                e.pop("pattern_id", None)
            labelled.append(cand)
        return labelled
