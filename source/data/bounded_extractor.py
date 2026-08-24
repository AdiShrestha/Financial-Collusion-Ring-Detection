"""Bounded directed cycle candidate extractor for genuine benign transaction networks.

Contract C16-01 (T-COMP): Enumerates simple directed cycles (k in [3..12]) label-blind
over the master Parquet transaction graph, validates strict laundering account exclusion,
and exports the benign candidate pool to artifacts/candidates/benign_pool.parquet.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.audit_data import parse_amlworld_pattern_blocks


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
    """Extracts bounded directed benign cycle candidates (k in [3..12]) from transaction graphs."""

    def __init__(
        self,
        min_cycle_len: int = 3,
        max_cycle_len: int = 12,
        max_candidates: int = 2000,
    ):
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.max_candidates = max_candidates

    def extract_benign_pool(
        self,
        transactions_parquet_path: str = "artifacts/raw/transactions.parquet",
        patterns_txt_path: str = "data/raw/HI-Small_Patterns.txt",
        output_parquet_path: str = "artifacts/candidates/benign_pool.parquet",
    ) -> Dict[str, Any]:
        """Extract benign directed cycles with verified zero laundering and pattern overlap."""
        if not os.path.exists(transactions_parquet_path):
            raise FileNotFoundError(f"Parquet table missing: {transactions_parquet_path}")
        if not os.path.exists(patterns_txt_path):
            raise FileNotFoundError(f"Patterns file missing: {patterns_txt_path}")

        # 1. Build blacklist from all 370 pattern blocks
        pattern_blocks = parse_amlworld_pattern_blocks(patterns_txt_path)
        pattern_blacklist: Set[str] = set()
        for b in pattern_blocks:
            for acc in b.get("participants", []):
                pattern_blacklist.add(str(acc))

        # 2. Read master transactions
        table = pq.read_table(
            transactions_parquet_path,
            columns=[
                "transaction_id",
                "source_line_number",
                "from_account",
                "to_account",
                "timestamp_raw",
                "timestamp_epoch",
                "amount_paid",
                "payment_format",
                "is_laundering",
                "raw_row_sha256",
            ],
        )

        p_from = table.column("from_account").to_pylist()
        p_to = table.column("to_account").to_pylist()
        p_is_l = table.column("is_laundering").to_pylist()
        p_txid = table.column("transaction_id").to_pylist()
        p_line = table.column("source_line_number").to_pylist()
        p_ts = table.column("timestamp_raw").to_pylist()
        p_epoch = table.column("timestamp_epoch").to_pylist()
        p_amt = table.column("amount_paid").to_pylist()
        p_fmt = table.column("payment_format").to_pylist()
        p_sha = table.column("raw_row_sha256").to_pylist()

        # Build directed graph and edge transaction lookup
        G = nx.DiGraph()
        edge_txs: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        n = len(p_from)
        for i in range(n):
            u = str(p_from[i])
            v = str(p_to[i])
            # Filter strictly benign and non-blacklisted
            if u in pattern_blacklist or v in pattern_blacklist or p_is_l[i] == 1 or u == v:
                continue

            G.add_edge(u, v)
            edge_txs[(u, v)].append({
                "transaction_id": p_txid[i],
                "source_line_number": p_line[i],
                "from_account": u,
                "to_account": v,
                "timestamp_raw": p_ts[i],
                "timestamp_epoch": p_epoch[i],
                "amount_paid": p_amt[i],
                "payment_format": p_fmt[i],
                "raw_row_sha256": p_sha[i],
            })

        # 3. Find strongly connected components with >= min_cycle_len nodes
        sccs = [c for c in nx.strongly_connected_components(G) if len(c) >= self.min_cycle_len]

        seen_cycles: Set[Tuple[str, ...]] = set()
        candidates: List[Dict[str, Any]] = []

        for scc in sccs:
            sub = G.subgraph(scc)
            adj = {u: list(sub.successors(u)) for u in sub}

            for start_node in adj:
                stack = [(start_node, [start_node])]
                while stack:
                    curr, path = stack.pop()
                    if len(path) > self.max_cycle_len:
                        continue

                    for nxt in adj.get(curr, []):
                        if nxt == start_node and len(path) >= self.min_cycle_len:
                            can = canonical_cycle(path)
                            if can not in seen_cycles:
                                seen_cycles.add(can)
                                k = len(path)
                                # Build ordered transaction records
                                ordered_tx_ids = []
                                ordered_tx_meta = []
                                for step in range(k):
                                    u_step = path[step]
                                    v_step = path[(step + 1) % k]
                                    tx_sample = edge_txs[(u_step, v_step)][0]
                                    ordered_tx_ids.append(tx_sample["transaction_id"])
                                    ordered_tx_meta.append({
                                        **tx_sample,
                                        "role": "backbone",
                                        "cycle_position": step,
                                    })

                                cand_idx = len(candidates) + 1
                                cand_id = f"ben_pool_{cand_idx:05d}"
                                content_hash = hashlib.sha256(
                                    json.dumps({"id": cand_id, "accounts": path, "txs": ordered_tx_ids}).encode("utf-8")
                                ).hexdigest()

                                candidates.append({
                                    "candidate_id": cand_id,
                                    "source_kind": "extracted_benign_cycle",
                                    "label": 0,
                                    "cycle_length": k,
                                    "ordered_cycle_accounts": list(path),
                                    "cycle_transaction_ids": ordered_tx_ids,
                                    "transactions_json": json.dumps(ordered_tx_meta),
                                    "candidate_content_sha256": content_hash,
                                })

                        elif nxt not in path and len(path) < self.max_cycle_len:
                            stack.append((nxt, path + [nxt]))

                    if len(candidates) >= self.max_candidates:
                        break

            if len(candidates) >= self.max_candidates:
                break

        # Save to Parquet
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
        arrow_schema = pa.schema([
            ("candidate_id", pa.string()),
            ("source_kind", pa.string()),
            ("label", pa.int64()),
            ("cycle_length", pa.int64()),
            ("ordered_cycle_accounts", pa.list_(pa.string())),
            ("cycle_transaction_ids", pa.list_(pa.string())),
            ("transactions_json", pa.string()),
            ("candidate_content_sha256", pa.string()),
        ])

        batch_dict = {
            "candidate_id": [c["candidate_id"] for c in candidates],
            "source_kind": [c["source_kind"] for c in candidates],
            "label": [c["label"] for c in candidates],
            "cycle_length": [c["cycle_length"] for c in candidates],
            "ordered_cycle_accounts": [c["ordered_cycle_accounts"] for c in candidates],
            "cycle_transaction_ids": [c["cycle_transaction_ids"] for c in candidates],
            "transactions_json": [c["transactions_json"] for c in candidates],
            "candidate_content_sha256": [c["candidate_content_sha256"] for c in candidates],
        }

        out_table = pa.Table.from_pydict(batch_dict, schema=arrow_schema)
        pq.write_table(out_table, output_parquet_path, compression="snappy")

        return {
            "status": "EXTRACTED",
            "total_benign_candidates": len(candidates),
            "output_path": output_parquet_path,
        }


if __name__ == "__main__":
    extractor = BoundedCycleExtractor()
    res = extractor.extract_benign_pool()
    print(json.dumps(res, indent=2))
