"""Controlled Semi-Synthetic Collusion Ring Injection Engine.

Injects circular transaction motifs with strict flow conservation and monotonic timestamp progression.
Traces to Contract C02-04, FR-002, INV-001, INV-002, INV-008.
"""

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx


@dataclass
class InjectedRing:
    ring_id: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    cycle_length: int
    initial_volume: float
    final_volume: float
    fee_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ring_id": self.ring_id,
            "nodes": list(self.nodes),
            "edges": [dict(e) for e in self.edges],
            "cycle_length": self.cycle_length,
            "initial_volume": self.initial_volume,
            "final_volume": self.final_volume,
            "fee_rate": self.fee_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InjectedRing":
        return cls(
            ring_id=data["ring_id"],
            nodes=list(data["nodes"]),
            edges=[dict(e) for e in data["edges"]],
            cycle_length=int(data["cycle_length"]),
            initial_volume=float(data["initial_volume"]),
            final_volume=float(data["final_volume"]),
            fee_rate=float(data["fee_rate"]),
        )


class SemiSyntheticRingInjector:
    """Injects flow-conserving circular transaction motifs into transaction graphs."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject_rings(
        self,
        graph: Union[nx.DiGraph, nx.MultiDiGraph],
        n_rings: int = 10,
        cycle_lengths: Optional[List[int]] = None,
        fee_rate: float = 0.005,
        initial_amount_range: Tuple[float, float] = (100.0, 1000.0),
        base_timestamp: int = 1000000,
    ) -> Tuple[nx.DiGraph, List[InjectedRing]]:
        """Injects n_rings circular collusion motifs into graph.

        Args:
            graph: Target directed transaction graph, preserving parallel transfers when multigraph.
            n_rings: Total number of rings to inject.
            cycle_lengths: Allowed cycle lengths k in [3, 6].
            fee_rate: Per-hop dissipation rate epsilon (e.g. 0.005 = 0.5%).
            initial_amount_range: (min_amt, max_amt) for ring input.
            base_timestamp: Starting integer timestamp.

        Returns:
            Tuple of (augmented_graph, list_of_injected_rings).
        """
        if cycle_lengths is None:
            cycle_lengths = [3, 4, 5, 6]

        augmented_graph = graph.copy()
        existing_nodes = list(augmented_graph.nodes())
        injected_rings: List[InjectedRing] = []

        current_timestamp = base_timestamp

        for ring_idx in range(n_rings):
            k = self.rng.choice(cycle_lengths)
            ring_id = f"INJECTED_RING_{ring_idx:04d}_k{k}"

            # Select or create k nodes for the ring
            ring_nodes = [f"{ring_id}_node_{i}" for i in range(k)]
            for node in ring_nodes:
                augmented_graph.add_node(node, is_injected=True, ring_id=ring_id)

            # Compute flow conservation amounts — declare initial volume first
            # so the bridge edge uses the ring's actual initial volume
            v0 = round(self.rng.uniform(initial_amount_range[0], initial_amount_range[1]), 2)

            # Optionally bridge into background graph if background nodes exist
            if existing_nodes:
                anchor = self.rng.choice(existing_nodes)
                # Bridge edge uses declared ring initial volume (not a hardcoded constant)
                augmented_graph.add_edge(
                    anchor,
                    ring_nodes[0],
                    is_bridge=True,
                    ring_id=ring_id,
                    amount=v0,
                    timestamp=current_timestamp,
                )
                current_timestamp += 10

            # Compute per-hop flow with fee dissipation
            v_curr = v0
            ring_edges = []

            for i in range(k):
                src = ring_nodes[i]
                dst = ring_nodes[(i + 1) % k]
                v_next = round(v_curr * (1.0 - fee_rate), 6)
                current_timestamp += self.rng.randint(60, 3600)  # Monotonic timestamp advance

                edge_data = {
                    "source": src,
                    "target": dst,
                    "tx_id": f"{ring_id}_tx_{i}",
                    "amount": v_curr,
                    "amount_next": v_next,
                    "fee": round(v_curr - v_next, 6),
                    "fee_rate": fee_rate,
                    "timestamp": current_timestamp,
                    "is_injected_ring": True,
                    "ring_id": ring_id,
                    "cycle_length": k,
                }
                augmented_graph.add_edge(src, dst, **edge_data)
                ring_edges.append(edge_data)
                v_curr = v_next

            # Final volume after k hops
            expected_final = v0 * ((1.0 - fee_rate) ** k)

            ring_obj = InjectedRing(
                ring_id=ring_id,
                nodes=ring_nodes,
                edges=ring_edges,
                cycle_length=k,
                initial_volume=v0,
                final_volume=v_curr,
                fee_rate=fee_rate,
            )
            # Mandatory flow conservation gate — raises on violation
            if not self.verify_flow_conservation(ring_obj):
                raise ValueError(
                    f"[INV-003 Violation] Flow conservation check FAILED for {ring_id}. "
                    f"initial_volume={v0}, final_volume={v_curr}, "
                    f"expected={expected_final}, fee_rate={fee_rate}"
                )
            injected_rings.append(ring_obj)

        return augmented_graph, injected_rings

    @staticmethod
    def verify_flow_conservation(
        ring: InjectedRing,
        max_fee_tolerance: float = 0.02,
    ) -> bool:
        """Verifies flow conservation, bounded fee dissipation, and monotonic timestamp progression.

        Args:
            ring: InjectedRing to verify.
            max_fee_tolerance: Maximum allowable fee rate per hop.

        Returns:
            True if flow conservation and timestamp progression strictly hold.
        """
        if ring.fee_rate > max_fee_tolerance or ring.fee_rate < 0.0:
            return False

        k = ring.cycle_length
        if len(ring.edges) != k or len(ring.nodes) != k or len(set(ring.nodes)) != k:
            return False

        if not all(math.isfinite(float(v)) and float(v) >= 0 for v in
                   [ring.initial_volume, ring.final_volume, ring.fee_rate] +
                   [e["amount"] for e in ring.edges] +
                   [e["amount_next"] for e in ring.edges]):
            return False

        # 1. Verify monotonic timestamps
        for i in range(k - 1):
            t_curr = ring.edges[i]["timestamp"]
            t_next = ring.edges[i + 1]["timestamp"]
            if t_next <= t_curr:
                return False

        for i, edge in enumerate(ring.edges):
            if edge["source"] != ring.nodes[i] or edge["target"] != ring.nodes[(i + 1) % k]:
                return False
            if i + 1 < k and abs(edge["amount_next"] - ring.edges[i + 1]["amount"]) > 1e-4:
                return False
        if abs(ring.edges[0]["amount"] - ring.initial_volume) > 1e-4:
            return False
        if abs(ring.edges[-1]["amount_next"] - ring.final_volume) > 1e-4:
            return False

        # 2. Verify per-hop flow balance: V_next == V_curr * (1 - fee_rate)
        for edge in ring.edges:
            amt = edge["amount"]
            amt_next = edge["amount_next"]
            expected_next = amt * (1.0 - ring.fee_rate)
            if abs(amt_next - expected_next) > 1e-4:
                return False

        # 3. Verify total cycle flow conservation
        v0 = ring.initial_volume
        vk = ring.final_volume
        theoretical_vk = v0 * ((1.0 - ring.fee_rate) ** k)
        if abs(vk - theoretical_vk) > 1e-4:
            return False

        return True
