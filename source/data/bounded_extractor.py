"""Bounded directed cycle candidate extractor for AML transaction networks.

Contract C11-02 (T-DESC): Enumerates simple directed cycles with bounded lengths
k in {3,4,5,6}, extracts candidate subgraphs, and evaluates extraction recall against
ground-truth laundering patterns.
"""

from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx


def canonical_cycle(cycle_nodes: Sequence[str]) -> Tuple[str, ...]:
    """Return the lexicographically smallest cyclic permutation of a cycle."""
    n = len(cycle_nodes)
    if n == 0:
        return tuple()
    nodes = [str(x) for x in cycle_nodes]
    min_val = min(nodes)
    min_indices = [i for i, x in enumerate(nodes) if x == min_val]

    best = tuple(nodes)
    for idx in min_indices:
        perm = tuple(nodes[idx:] + nodes[:idx])
        if perm < best:
            best = perm
    return best


class BoundedCycleExtractor:
    """Extracts bounded directed cycle subgraphs (k in {3,4,5,6}) from transaction streams."""

    def __init__(
        self,
        min_cycle_len: int = 3,
        max_cycle_len: int = 6,
        window_size_seconds: float = 86400.0 * 30.0,
        max_candidates: int = 10000,
    ):
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.window_size_seconds = window_size_seconds
        self.max_candidates = max_candidates

    def find_bounded_directed_cycles(
        self,
        G: nx.DiGraph,
    ) -> List[Tuple[str, ...]]:
        """Find all simple directed cycles with length in [min_cycle_len, max_cycle_len]."""
        cycles: Set[Tuple[str, ...]] = set()
        nodes = sorted(list(G.nodes()))
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Bounded depth-first search
        for start_node in nodes:
            start_idx = node_to_idx[start_node]

            def dfs(curr_node: str, path: List[str], visited: Set[str]):
                if len(cycles) >= self.max_candidates:
                    return

                depth = len(path)
                for neighbor in G.successors(curr_node):
                    if neighbor == start_node:
                        if self.min_cycle_len <= depth <= self.max_cycle_len:
                            cycles.add(canonical_cycle(path))
                    elif neighbor not in visited and node_to_idx.get(neighbor, -1) > start_idx:
                        if depth < self.max_cycle_len:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor, path, visited)
                            path.pop()
                            visited.remove(neighbor)

            visited_set = {start_node}
            dfs(start_node, [start_node], visited_set)

        return sorted(list(cycles))

    def extract_candidates(
        self,
        transactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract candidate subgraphs corresponding to bounded directed cycles."""
        G = nx.DiGraph()
        tx_by_edge: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

        for tx in transactions:
            u = str(tx.get("from_account", ""))
            v = str(tx.get("to_account", ""))
            if not u or not v:
                continue

            G.add_edge(u, v)
            edge_key = (u, v)
            if edge_key not in tx_by_edge:
                tx_by_edge[edge_key] = []
            tx_by_edge[edge_key].append(tx)

        detected_cycles = self.find_bounded_directed_cycles(G)
        candidates: List[Dict[str, Any]] = []

        for idx, cyc in enumerate(detected_cycles, start=1):
            k = len(cyc)
            cyc_nodes = list(cyc)
            cyc_node_set = set(cyc_nodes)

            # Extract cycle edges and all internal transactions between cycle nodes
            cyc_edges = [(cyc[i], cyc[(i + 1) % k]) for i in range(k)]
            cand_txs: List[Dict[str, Any]] = []
            for u, v in cyc_edges:
                cand_txs.extend(tx_by_edge.get((u, v), []))

            # Temporal metrics
            epochs = [float(tx.get("timestamp_epoch", tx.get("timestamp", 0.0))) for tx in cand_txs]
            valid_epochs = [t for t in epochs if t > 0]
            start_t = min(valid_epochs) if valid_epochs else 0.0
            end_t = max(valid_epochs) if valid_epochs else 0.0
            dur = max(0.0, end_t - start_t)

            # Determine laundering label (1 if any tx is laundering)
            is_laundering = 1 if any(int(tx.get("is_laundering", 0)) == 1 for tx in cand_txs) else 0

            cand_record: Dict[str, Any] = {
                "candidate_id": f"cand_cyc_{idx:04d}",
                "cycle_length": k,
                "nodes": cyc_nodes,
                "participants": cyc_nodes,
                "edges": cyc_edges,
                "transactions": cand_txs,
                "start_time": start_t,
                "end_time": end_t,
                "duration_seconds": dur,
                "is_laundering": is_laundering,
                "label": is_laundering,
            }
            candidates.append(cand_record)

        return candidates

    def evaluate_pattern_recall(
        self,
        candidates: List[Dict[str, Any]],
        pattern_blocks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute candidate extraction recall against ground-truth CYCLE pattern blocks."""
        cycle_patterns = [b for b in pattern_blocks if str(b.get("typology", "")).upper() == "CYCLE"]
        total_cycle_patterns = len(cycle_patterns)

        if total_cycle_patterns == 0:
            return {
                "recall": 1.0,
                "total_cycle_patterns": 0,
                "matched_cycle_patterns": 0,
                "unmatched_cycle_patterns": [],
            }

        candidate_node_sets = [set(c["nodes"]) for c in candidates]
        matched = 0
        unmatched = []

        for p in cycle_patterns:
            p_nodes = set(str(x) for x in p.get("participants", []))
            if any(p_nodes == c_set for c_set in candidate_node_sets):
                matched += 1
            else:
                unmatched.append(p.get("pattern_id", "unknown"))

        recall = matched / total_cycle_patterns

        return {
            "recall": recall,
            "total_cycle_patterns": total_cycle_patterns,
            "matched_cycle_patterns": matched,
            "unmatched_cycle_patterns": unmatched,
        }
