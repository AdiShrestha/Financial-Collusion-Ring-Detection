"""Deterministic semi-synthetic collusion ring injector for Elliptic++ Actor graphs."""

import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from source.data.candidate_extractor import CandidateExample


class RingInjector:
    """Injects deterministic directed collusion cycle rings into transaction networks.

    Guarantees:
    1. Deterministic generation given seed.
    2. Strict preservation of split partition boundaries (zero cross-partition edges).
    3. Production of immutable injection manifests recording every inserted edge and node.
    """

    def __init__(
        self,
        num_injections: int = 10,
        cycle_lengths: Sequence[int] = (3, 4, 5, 6),
        seed: int = 42,
    ):
        self.num_injections = num_injections
        self.cycle_lengths = list(cycle_lengths)
        self.seed = seed

    def inject_rings(
        self,
        wallet_ids: Sequence[str],
        base_edgelist: Sequence[Tuple[str, str, Dict[str, Any]]],
        node_timesteps: Optional[Dict[str, int]] = None,
        split_partitions: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[CandidateExample], List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
        """Inject semi-synthetic cycle rings into the network."""
        rng = random.Random(self.seed)
        unique_wallets = sorted(list(set(wallet_ids)))

        # Group wallets by (partition, timestep) to prevent boundary crossing
        buckets: Dict[Tuple[str, int], List[str]] = {}
        for w in unique_wallets:
            partition = split_partitions.get(w, "all") if split_partitions else "all"
            ts = node_timesteps.get(w, 1) if node_timesteps else 1
            buckets.setdefault((partition, ts), []).append(w)

        # Filter buckets that have at least min(cycle_lengths) wallets
        min_k = min(self.cycle_lengths)
        valid_bucket_keys = [k for k, v in buckets.items() if len(v) >= min_k]
        if not valid_bucket_keys:
            valid_bucket_keys = list(buckets.keys())

        injected_candidates: List[CandidateExample] = []
        new_edges: List[Tuple[str, str, Dict[str, Any]]] = list(base_edgelist)
        manifest_records: List[Dict[str, Any]] = []

        for i in range(self.num_injections):
            # Select cycle length deterministically
            k = self.cycle_lengths[i % len(self.cycle_lengths)]

            # Select a bucket
            bucket_key = valid_bucket_keys[i % len(valid_bucket_keys)]
            bucket_wallets = buckets[bucket_key]

            if len(bucket_wallets) < k:
                # If bucket too small, cycle through available
                sampled_nodes = [bucket_wallets[j % len(bucket_wallets)] for j in range(k)]
            else:
                sampled_nodes = rng.sample(bucket_wallets, k)

            # Construct directed cycle edges: (v0, v1), (v1, v2), ..., (vk-1, v0)
            cycle_edges: List[Tuple[str, str, Dict[str, Any]]] = []
            cand_edges: List[Tuple[str, str, Dict[str, Any]]] = []
            timestep_val = float(bucket_key[1])

            for step in range(k):
                u = sampled_nodes[step]
                v = sampled_nodes[(step + 1) % k]
                edge_attr = {
                    "injected": True,
                    "injected_ring_id": f"ring_{i+1:04d}",
                    "timestep": timestep_val,
                    "amount": 10.0 + (i * 1.5),
                }
                edge_tuple = (u, v, edge_attr)
                cycle_edges.append(edge_tuple)
                cand_edges.append(edge_tuple)
                new_edges.append(edge_tuple)

            ring_id = f"inj_ring_{i+1:04d}"
            candidate = CandidateExample(
                candidate_id=ring_id,
                dataset_track="elliptic_actors",
                temporal_bounds=(timestep_val, timestep_val),
                participant_ids=sampled_nodes,
                edges=cand_edges,
                node_features={node: [0.0] * 56 for node in sampled_nodes},
                target_y=1,
                typology_label="CYCLE",
                group_id=f"group_{ring_id}",
                metadata={
                    "cycle_length": k,
                    "injected": True,
                    "partition": bucket_key[0],
                    "timestep": bucket_key[1],
                },
            )
            injected_candidates.append(candidate)

            manifest_records.append(
                {
                    "injected_ring_id": ring_id,
                    "cycle_length": k,
                    "participants": sampled_nodes,
                    "partition": bucket_key[0],
                    "timestep": bucket_key[1],
                    "edges": [(e[0], e[1]) for e in cycle_edges],
                }
            )

        injection_manifest = {
            "seed": self.seed,
            "num_injections": self.num_injections,
            "cycle_lengths": self.cycle_lengths,
            "total_edges_injected": sum(len(r["edges"]) for r in manifest_records),
            "injected_rings": manifest_records,
            "split_boundary_violations": 0,
        }

        return injected_candidates, new_edges, injection_manifest
