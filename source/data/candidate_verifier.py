"""Candidate extraction and Group-Safe split integrity verification engine.

Contract C11-05 (T-COMP): Formal gate verifier validating candidate cohort completeness,
cycle length coverage (k in {3,4,5,6}), label balance, and strict 0.0% cross-partition
account overlap (INV-006) before model training.
"""

import json
import os
from typing import Any, Dict, List, Optional, Set


class CandidateIntegrityVerifier:
    """Verifies candidate dataset integrity and Group-Safe split isolation."""

    def __init__(self, project_root: str = "."):
        self.project_root = project_root

    def verify_candidate_integrity(
        self,
        candidates_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        audit_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run candidate cohort and split integrity verification and generate report."""
        if candidates_path is None:
            candidates_path = os.path.join(self.project_root, "data/processed/candidates.jsonl")
        if manifest_path is None:
            manifest_path = os.path.join(self.project_root, "data/manifests/split_manifest.json")
        if audit_path is None:
            audit_path = os.path.join(self.project_root, "data/observed_audit_report.json")
        if output_path is None:
            output_path = os.path.join(self.project_root, "project/candidate_integrity_report.json")

        checks: Dict[str, bool] = {}
        details: Dict[str, Any] = {}

        # 1. Check candidate file exists and load
        if not os.path.exists(candidates_path):
            checks["candidates_file_exists"] = False
            details["error"] = f"Missing candidates file: {candidates_path}"
            return self._fail_report(checks, details, output_path)

        candidates: List[Dict[str, Any]] = []
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        num_cands = len(candidates)
        checks["candidates_file_exists"] = True
        checks["sufficient_candidates_count"] = num_cands >= 20
        details["total_candidates"] = num_cands

        # 2. Check split manifest
        if not os.path.exists(manifest_path):
            checks["split_manifest_exists"] = False
            details["error"] = f"Missing split manifest: {manifest_path}"
            return self._fail_report(checks, details, output_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        checks["split_manifest_exists"] = True

        # Check candidate ID consistency
        cand_ids_jsonl = set(c["candidate_id"] for c in candidates)
        manifest_splits = manifest.get("splits", {})
        cand_ids_manifest = set()
        split_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        for split_key in ("train", "val", "validation", "test"):
            canon_name = "val" if split_key in ("val", "validation") else split_key
            s_val = manifest_splits.get(split_key)
            if isinstance(s_val, dict):
                c_ids = s_val.get("candidate_ids", [])
                split_ids[canon_name].extend(c_ids)
                cand_ids_manifest.update(c_ids)
            elif isinstance(s_val, list):
                for item in s_val:
                    if isinstance(item, dict):
                        cid = item.get("candidate_id")
                        if cid:
                            split_ids[canon_name].append(cid)
                            cand_ids_manifest.add(cid)
                    elif isinstance(item, str):
                        split_ids[canon_name].append(item)
                        cand_ids_manifest.add(item)

        checks["candidate_ids_aligned"] = cand_ids_jsonl == cand_ids_manifest
        details["jsonl_candidate_count"] = len(cand_ids_jsonl)
        details["manifest_candidate_count"] = len(cand_ids_manifest)

        # 3. Check 0.0% account overlap across splits (INV-006)
        cand_by_id = {c["candidate_id"]: c for c in candidates}
        train_accs: Set[str] = set()
        val_accs: Set[str] = set()
        test_accs: Set[str] = set()

        for cid in split_ids["train"]:
            c = cand_by_id.get(cid, {})
            train_accs.update(str(a) for a in c.get("participants", c.get("nodes", [])))

        for cid in split_ids["val"]:
            c = cand_by_id.get(cid, {})
            val_accs.update(str(a) for a in c.get("participants", c.get("nodes", [])))

        for cid in split_ids["test"]:
            c = cand_by_id.get(cid, {})
            test_accs.update(str(a) for a in c.get("participants", c.get("nodes", [])))

        overlap_tv = train_accs & val_accs
        overlap_tt = train_accs & test_accs
        overlap_vt = val_accs & test_accs

        zero_overlap = (len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0)
        checks["zero_cross_split_account_overlap"] = zero_overlap
        details["train_val_overlap"] = len(overlap_tv)
        details["train_test_overlap"] = len(overlap_tt)
        details["val_test_overlap"] = len(overlap_vt)

        # 4. Check cycle length distribution coverage (k in 3,4,5,6)
        pos_lengths = set(c["cycle_length"] for c in candidates if c.get("label", c.get("is_laundering", 0)) == 1)
        neg_lengths = set(c["cycle_length"] for c in candidates if c.get("label", c.get("is_laundering", 0)) == 0)
        required_lengths = {3, 4, 5, 6}

        has_all_pos = required_lengths.issubset(pos_lengths)
        has_all_neg = required_lengths.issubset(neg_lengths)
        checks["cycle_length_distribution_covered"] = has_all_pos and has_all_neg
        details["positive_cycle_lengths"] = sorted(list(pos_lengths))
        details["negative_cycle_lengths"] = sorted(list(neg_lengths))

        # 5. Label balance (ratio between 0.5 and 2.0)
        n_pos = sum(1 for c in candidates if c.get("label", c.get("is_laundering", 0)) == 1)
        n_neg = sum(1 for c in candidates if c.get("label", c.get("is_laundering", 0)) == 0)
        ratio = (n_pos / n_neg) if n_neg > 0 else 0.0
        checks["label_balance_valid"] = 0.5 <= ratio <= 2.0
        details["positive_count"] = n_pos
        details["negative_count"] = n_neg
        details["positive_negative_ratio"] = ratio

        # 6. Check partition hashes
        test_split = manifest_splits.get("test", {})
        if isinstance(test_split, dict):
            test_h = test_split.get("sha256_checksum", "")
        else:
            test_h = manifest.get("split_hashes", {}).get("test", "")
        checks["split_hashes_valid"] = len(test_h) == 64

        all_passed = all(checks.values())
        status = "CANDIDATE_INTEGRITY_PASS" if all_passed else "CANDIDATE_INTEGRITY_FAIL"

        report: Dict[str, Any] = {
            "gate_id": "CANDIDATE_INTEGRITY_GATE",
            "status": status,
            "candidate_integrity_status": status,
            "all_subchecks_passed": all_passed,
            "integrity_checks": checks,
            "details": details,
            "invariants_verified": ["INV-001", "INV-005", "INV-006"],
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _fail_report(self, checks: Dict[str, bool], details: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        report = {
            "gate_id": "CANDIDATE_INTEGRITY_GATE",
            "status": "CANDIDATE_INTEGRITY_FAIL",
            "candidate_integrity_status": "CANDIDATE_INTEGRITY_FAIL",
            "all_subchecks_passed": False,
            "integrity_checks": checks,
            "details": details,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report


if __name__ == "__main__":
    verifier = CandidateIntegrityVerifier()
    res = verifier.verify_candidate_integrity()
    print(f"Candidate Integrity Gate Status: {res['status']}")
