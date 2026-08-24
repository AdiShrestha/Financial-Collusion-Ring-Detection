"""Pattern group and cycle typology feasibility auditor for IBM AMLworld datasets.

Contract C10-03 (T-DESC): Parses laundering pattern blocks, computes cycle length
distributions (k in {3,4,5,6}), evaluates account-overlap clustering, and measures
transaction join rates against the main transaction log.
"""

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from source.data.audit_data import parse_amlworld_pattern_blocks
from source.data.streaming_loader import StreamingTransactionLoader, parse_timestamp_to_epoch


class PatternGroupAuditor:
    """Audits laundering pattern blocks, cycle topologies, and group independence."""

    def __init__(self):
        self.loader = StreamingTransactionLoader()

    def audit_patterns(
        self,
        patterns_source: Union[str, io.TextIOBase, io.StringIO],
        trans_source: Optional[Union[str, io.TextIOBase, io.StringIO]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive pattern group and cycle typology audit."""
        blocks = parse_amlworld_pattern_blocks(patterns_source)

        typology_counts: Dict[str, int] = {}
        cycle_lengths: Dict[int, int] = {3: 0, 4: 0, 5: 0, 6: 0}
        cycle_durations: List[float] = []
        cycle_accounts: Set[str] = set()
        total_cycle_patterns = 0

        # Pattern overlap graph
        G = nx.Graph()
        for b in blocks:
            p_id = b["pattern_id"]
            typology = b.get("typology", "UNKNOWN").upper()
            typology_counts[typology] = typology_counts.get(typology, 0) + 1
            G.add_node(p_id, typology=typology, participants=b["participants"])

            if typology == "CYCLE":
                total_cycle_patterns += 1
                participants = b.get("participants", [])
                k = len(participants)
                if k in cycle_lengths:
                    cycle_lengths[k] += 1
                else:
                    cycle_lengths[k] = 1

                for acc in participants:
                    cycle_accounts.add(str(acc))

                # Compute duration
                txs = b.get("transactions", [])
                epochs = []
                for tx in txs:
                    ts_str = tx.get("timestamp", "")
                    ep = parse_timestamp_to_epoch(ts_str)
                    if ep > 0:
                        epochs.append(ep)
                if epochs:
                    dur = max(epochs) - min(epochs)
                    cycle_durations.append(dur)
                else:
                    cycle_durations.append(0.0)

        # Build edges based on shared participant accounts
        num_blocks = len(blocks)
        for i in range(num_blocks):
            p_i = blocks[i]["pattern_id"]
            set_i = set(str(x) for x in blocks[i].get("participants", []))
            for j in range(i + 1, num_blocks):
                p_j = blocks[j]["pattern_id"]
                set_j = set(str(x) for x in blocks[j].get("participants", []))
                if set_i & set_j:
                    G.add_edge(p_i, p_j)

        components = list(nx.connected_components(G))
        connected_components_count = len(components)
        largest_comp_size = max((len(c) for c in components), default=0)

        # Duration stats
        if cycle_durations:
            mean_dur = sum(cycle_durations) / len(cycle_durations)
            min_dur = min(cycle_durations)
            max_dur = max(cycle_durations)
        else:
            mean_dur = 0.0
            min_dur = 0.0
            max_dur = 0.0

        # Join rate verification
        total_pattern_txs = 0
        matched_pattern_txs = 0
        join_rate = 1.0

        if trans_source is not None:
            # Build set of transaction signature keys from trans_source
            if hasattr(trans_source, "seek"):
                trans_source.seek(0)

            main_tx_keys: Set[Tuple[str, str, float]] = set()
            for chunk in self.loader.stream_transactions(trans_source):
                for tx in chunk:
                    f_acc = str(tx.get("from_account", ""))
                    t_acc = str(tx.get("to_account", ""))
                    amt = round(float(tx.get("amount_paid", 0.0)), 2)
                    main_tx_keys.add((f_acc, t_acc, amt))

            for b in blocks:
                for tx in b.get("transactions", []):
                    total_pattern_txs += 1
                    f_acc = str(tx.get("from_account", ""))
                    t_acc = str(tx.get("to_account", ""))
                    try:
                        amt = round(float(tx.get("amount_paid", 0.0)), 2)
                    except (ValueError, TypeError):
                        amt = 0.0

                    if (f_acc, t_acc, amt) in main_tx_keys:
                        matched_pattern_txs += 1

            if total_pattern_txs > 0:
                join_rate = matched_pattern_txs / total_pattern_txs

        return {
            "total_patterns_by_typology": typology_counts,
            "total_pattern_blocks": len(blocks),
            "cycle_pattern_stats": {
                "total_cycle_patterns": total_cycle_patterns,
                "length_distribution": cycle_lengths,
                "duration_stats": {
                    "mean_duration_seconds": mean_dur,
                    "min_duration_seconds": min_dur,
                    "max_duration_seconds": max_dur,
                },
                "total_cycle_accounts": len(cycle_accounts),
            },
            "cluster_independence_stats": {
                "connected_components_count": connected_components_count,
                "largest_component_size": largest_comp_size,
                "independent_groups_count": connected_components_count,
            },
            "transaction_join_rate": join_rate,
            "total_pattern_transactions": total_pattern_txs,
            "matched_pattern_transactions": matched_pattern_txs,
        }
