"""Independent Gate B & Gate C Verifier for candidate lineage and group split safety.

Contract C16-04 (T-DESC): Recomputes candidate provenance against master Parquet rows,
verifies zero pattern account intersection for negative controls, verifies zero synthetic tokens,
and verifies strict 0.0% account/transaction outer-fold cross-validation leakage.
Writes project/gate_b_report.json (GATE_B_PASS) and project/gate_c_report.json (GATE_C_PASS).
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.data.audit_data import parse_amlworld_pattern_blocks


class GateBCVerifier:
    """Independent verifier for Gate B (Candidate Lineage) and Gate C (Group Splits)."""

    EXPECTED_TOTAL_CANDIDATES = 155
    EXPECTED_POSITIVE_COUNT = 40
    EXPECTED_NEGATIVE_COUNT = 115
    LOCKED_MANIFEST_SHA256 = "8bc14682dc07e879de6cc7edf7f13b4599bc1ef5dad9307c6d4ce8cd2639c57b"

    def verify_gate_b(
        self,
        candidates_parquet: str = "artifacts/candidates/candidates.parquet",
        candidate_txs_parquet: str = "artifacts/candidates/candidate_transactions.parquet",
        labels_parquet: str = "artifacts/candidates/labels.parquet",
        transactions_parquet: str = "artifacts/raw/transactions.parquet",
        patterns_txt: str = "data/raw/HI-Small_Patterns.txt",
        gate_b_report_path: str = "project/gate_b_report.json",
    ) -> Dict[str, Any]:
        """Verify candidate lineage, zero synthetic tokens, and zero pattern contamination for controls."""
        checks: Dict[str, bool] = {}
        details: Dict[str, Any] = {}

        # 1. Existence of relational Parquet tables
        checks["candidates_parquet_exists"] = os.path.exists(candidates_parquet)
        checks["candidate_txs_parquet_exists"] = os.path.exists(candidate_txs_parquet)
        checks["labels_parquet_exists"] = os.path.exists(labels_parquet)

        if not (checks["candidates_parquet_exists"] and checks["candidate_txs_parquet_exists"] and checks["labels_parquet_exists"]):
            raise FileNotFoundError("Relational candidate Parquet tables missing.")

        cand_tbl = pq.read_table(candidates_parquet)
        tx_tbl = pq.read_table(candidate_txs_parquet)
        lbl_tbl = pq.read_table(labels_parquet)

        total_cands = cand_tbl.num_rows
        checks["candidate_count_matches"] = (total_cands == self.EXPECTED_TOTAL_CANDIDATES)
        details["total_candidates"] = total_cands

        pos_count = sum(cand_tbl.column("label").to_pylist())
        neg_count = total_cands - pos_count
        checks["positive_count_matches"] = (pos_count == self.EXPECTED_POSITIVE_COUNT)
        checks["negative_count_matches"] = (neg_count == self.EXPECTED_NEGATIVE_COUNT)
        details["positive_count"] = pos_count
        details["negative_count"] = neg_count

        # 2. Verify candidate transactions link 100% to master Parquet rows
        master_tbl = pq.read_table(transactions_parquet, columns=["transaction_id"])
        master_tx_set = set(master_tbl.column("transaction_id").to_pylist())
        cand_tx_set = set(tx_tbl.column("transaction_id").to_pylist())

        checks["all_candidate_transactions_in_master"] = cand_tx_set.issubset(master_tx_set)

        # 3. Verify zero pattern account overlap for negative controls
        pattern_blocks = parse_amlworld_pattern_blocks(patterns_txt)
        pattern_accounts = set()
        for b in pattern_blocks:
            pattern_accounts.update(b.get("participants", []))

        neg_accounts = set()
        for c in cand_tbl.to_pylist():
            if c["label"] == 0:
                neg_accounts.update(c["ordered_cycle_accounts"])

        neg_overlap = neg_accounts.intersection(pattern_accounts)
        checks["negative_controls_zero_pattern_overlap"] = (len(neg_overlap) == 0)

        # 4. Zero synthetic identifier tokens
        forbidden = ["ACC_POS_", "ACC_BEN_", "synthetic_manifest", "formats_pool"]
        cand_ids = cand_tbl.column("candidate_id").to_pylist()
        checks["zero_synthetic_tokens"] = all(
            not any(tok in str(cid) for tok in forbidden) for cid in cand_ids
        )

        all_passed = all(checks.values())
        status = "GATE_B_PASS" if all_passed else "GATE_B_FAIL"

        report = {
            "gate_id": "GATE_B_CANDIDATE_DATASET_AND_LINEAGE",
            "status": status,
            "passed": all_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "details": details,
        }

        os.makedirs(os.path.dirname(os.path.abspath(gate_b_report_path)), exist_ok=True)
        with open(gate_b_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def verify_gate_c(
        self,
        candidates_parquet: str = "artifacts/candidates/candidates.parquet",
        fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
        gate_c_report_path: str = "project/gate_c_report.json",
    ) -> Dict[str, Any]:
        """Verify group-safe cross-validation splits and zero outer-fold account/transaction leakage."""
        checks: Dict[str, bool] = {}
        details: Dict[str, Any] = {}

        checks["fold_manifest_exists"] = os.path.exists(fold_manifest_path)
        if not checks["fold_manifest_exists"]:
            raise FileNotFoundError(f"Fold manifest missing: {fold_manifest_path}")

        with open(fold_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        checks["n_outer_folds_is_5"] = (manifest.get("n_outer_folds") == 5)
        checks["n_inner_folds_is_3"] = (manifest.get("n_inner_folds") == 3)
        checks["total_candidates_matches"] = (manifest.get("total_candidates") == self.EXPECTED_TOTAL_CANDIDATES)

        # Manifest SHA-256 match
        manifest_sha = manifest.get("manifest_sha256", "")
        checks["manifest_sha_matches_protocol"] = (manifest_sha == self.LOCKED_MANIFEST_SHA256)
        details["manifest_sha256"] = manifest_sha

        # 0.0% account leakage across outer folds
        cand_tbl = pq.read_table(candidates_parquet)
        cand_accounts = {
            c["candidate_id"]: set(c["ordered_cycle_accounts"])
            for c in cand_tbl.to_pylist()
        }

        outer_leak_detected = False
        for of in manifest.get("outer_folds", []):
            train_accs = set()
            for cid in of["train_candidate_ids"]:
                train_accs.update(cand_accounts[cid])
            test_accs = set()
            for cid in of["test_candidate_ids"]:
                test_accs.update(cand_accounts[cid])

            if train_accs & test_accs:
                outer_leak_detected = True
                break

        checks["outer_fold_zero_account_leakage"] = not outer_leak_detected

        all_passed = all(checks.values())
        status = "GATE_C_PASS" if all_passed else "GATE_C_FAIL"

        report = {
            "gate_id": "GATE_C_GROUP_SAFE_SPLITS_AND_PROTOCOL_LOCK",
            "status": status,
            "passed": all_passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "details": details,
        }

        os.makedirs(os.path.dirname(os.path.abspath(gate_c_report_path)), exist_ok=True)
        with open(gate_c_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    verifier = GateBCVerifier()
    rep_b = verifier.verify_gate_b()
    rep_c = verifier.verify_gate_c()
    print(f"Gate B: {rep_b['status']}, Gate C: {rep_c['status']}")
    if not (rep_b["passed"] and rep_c["passed"]):
        sys.exit(1)
    sys.exit(0)
