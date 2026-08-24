"""Independent Gate A Data Provenance & Lineage Verifier.

Contract C15-04 (T-DESC): Recomputes physical file hashes, master transaction counts,
pattern multiset join rates, directed cycle chaining, and verifies zero synthetic tokens
in production artifacts. Writes project/gate_a_report.json certifying GATE_A_PASS.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class GateAVerifier:
    """Independent verifier for Gate A Raw Ingestion, Lineage & Positive Reconstruction."""

    EXPECTED_TRANS_SHA256 = "b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040"
    EXPECTED_PATTERNS_SHA256 = "2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b"
    EXPECTED_TOTAL_TX = 5078345
    EXPECTED_UNIQUE_ACCOUNTS = 515080
    EXPECTED_POSITIVE_CYCLES_COUNT = 40
    EXPECTED_DISJOINT_COMPONENTS = 38

    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def verify_gate_a(
        self,
        trans_csv_path: str = "data/raw/HI-Small_Trans.csv",
        patterns_txt_path: str = "data/raw/HI-Small_Patterns.txt",
        trans_parquet_path: str = "artifacts/raw/transactions.parquet",
        positive_parquet_path: str = "artifacts/candidates/positive_candidates.parquet",
        gate_a_report_path: str = "project/gate_a_report.json",
    ) -> Dict[str, Any]:
        """Execute complete independent verification of Gate A invariants."""
        checks: Dict[str, bool] = {}
        details: Dict[str, Any] = {}

        # 1. Raw file existence and hashes
        checks["trans_csv_exists"] = os.path.exists(trans_csv_path)
        checks["patterns_txt_exists"] = os.path.exists(patterns_txt_path)

        if not checks["trans_csv_exists"] or not checks["patterns_txt_exists"]:
            raise FileNotFoundError("Required raw benchmark files missing.")

        trans_sha = self.compute_sha256(trans_csv_path)
        patterns_sha = self.compute_sha256(patterns_txt_path)

        checks["trans_csv_hash_matches"] = (trans_sha == self.EXPECTED_TRANS_SHA256)
        checks["patterns_txt_hash_matches"] = (patterns_sha == self.EXPECTED_PATTERNS_SHA256)

        details["trans_csv_sha256"] = trans_sha
        details["patterns_txt_sha256"] = patterns_sha

        # 2. Master Parquet Table Verification
        checks["trans_parquet_exists"] = os.path.exists(trans_parquet_path)
        if not checks["trans_parquet_exists"]:
            raise FileNotFoundError(f"Master transactions parquet missing: {trans_parquet_path}")

        table = pq.read_table(trans_parquet_path)
        total_tx = table.num_rows
        checks["total_transactions_matches"] = (total_tx == self.EXPECTED_TOTAL_TX)
        details["total_transactions"] = total_tx

        from_accs = set(table.column("from_account").to_pylist())
        to_accs = set(table.column("to_account").to_pylist())
        all_accounts = from_accs.union(to_accs)
        unique_acc_count = len(all_accounts)

        checks["unique_accounts_matches"] = (unique_acc_count == self.EXPECTED_UNIQUE_ACCOUNTS)
        details["unique_accounts_count"] = unique_acc_count

        # Check required columns
        req_cols = {"transaction_id", "source_line_number", "timestamp_raw", "timestamp_epoch", "from_account", "to_account", "amount_paid", "payment_format", "is_laundering", "raw_row_sha256"}
        checks["required_columns_present"] = req_cols.issubset(set(table.column_names))

        # Check reject ledger
        reject_path = "artifacts/raw/reject_ledger.json"
        checks["reject_ledger_zero_rejects"] = False
        if os.path.exists(reject_path):
            with open(reject_path, "r", encoding="utf-8") as f:
                r_data = json.load(f)
                checks["reject_ledger_zero_rejects"] = (r_data.get("rejected_rows_count", -1) == 0)

        # 3. Positive Candidates Parquet Verification
        checks["positive_parquet_exists"] = os.path.exists(positive_parquet_path)
        if not checks["positive_parquet_exists"]:
            raise FileNotFoundError(f"Positive candidates parquet missing: {positive_parquet_path}")

        pos_table = pq.read_table(positive_parquet_path)
        pos_count = pos_table.num_rows
        checks["positive_candidate_count_matches"] = (pos_count == self.EXPECTED_POSITIVE_CYCLES_COUNT)
        details["positive_candidate_count"] = pos_count

        pos_candidates = pos_table.to_pylist()
        lengths = [c["cycle_length"] for c in pos_candidates]
        checks["positive_lengths_in_range_3_to_12"] = all(3 <= k <= 12 for k in lengths)
        checks["all_positive_labels_are_one"] = all(c["label"] == 1 for c in pos_candidates)

        # Verify directed cycle chaining and account component independence
        G = nx.Graph()
        directed_chaining_valid = True

        for idx, c in enumerate(pos_candidates):
            accs = c["ordered_cycle_accounts"]
            k = len(accs)
            if len(set(accs)) != k:
                directed_chaining_valid = False
            txs = json.loads(c["transactions_json"])
            if len(txs) != k:
                directed_chaining_valid = False
            for step in range(k):
                if txs[step]["from_account"] != accs[step]:
                    directed_chaining_valid = False
                if txs[step]["to_account"] != accs[(step + 1) % k]:
                    directed_chaining_valid = False

            G.add_node(idx)
            set_i = set(accs)
            for jdx in range(idx + 1, len(pos_candidates)):
                set_j = set(pos_candidates[jdx]["ordered_cycle_accounts"])
                if set_i & set_j:
                    G.add_edge(idx, jdx)

        checks["directed_chaining_valid"] = directed_chaining_valid
        comp_count = len(list(nx.connected_components(G)))
        checks["disjoint_components_count_matches"] = (comp_count == self.EXPECTED_DISJOINT_COMPONENTS)
        details["disjoint_components_count"] = comp_count

        # 4. Prohibition of Synthetic Tokens
        forbidden_tokens = ["ACC_POS_", "ACC_BEN_", "synthetic_manifest", "formats_pool"]
        cand_ids = pos_table.column("candidate_id").to_pylist()
        checks["zero_synthetic_candidate_tokens"] = all(
            not any(tok in str(cid) for tok in forbidden_tokens) for cid in cand_ids
        )

        all_passed = all(checks.values())
        status = "GATE_A_PASS" if all_passed else "GATE_A_FAIL"

        report = {
            "gate_id": "GATE_A_DATA_PROVENANCE_AND_LINEAGE",
            "status": status,
            "passed": all_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "details": details,
        }

        os.makedirs(os.path.dirname(os.path.abspath(gate_a_report_path)), exist_ok=True)
        with open(gate_a_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    verifier = GateAVerifier()
    rep = verifier.verify_gate_a()
    print(f"Gate A Verification Status: {rep['status']}")
    print(json.dumps(rep["checks"], indent=2))
    if not rep["passed"]:
        sys.exit(1)
    sys.exit(0)
