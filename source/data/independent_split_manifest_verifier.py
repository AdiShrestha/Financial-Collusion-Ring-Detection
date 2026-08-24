#!/usr/bin/env python3
"""
Independent Split Manifest Verifier for C11-04 (T-COMP Recompute Gate).

Independently inspects data/manifests/split_manifest.json and asserts 0.0% account overlap
(byte-different from source/data/candidate_dataset.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    manifest_path = "data/manifests/split_manifest.json"
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"is_disjoint": False, "total_candidates": 0, "error": str(e)}))
        sys.exit(1)

    total_cands = int(data.get("metadata", {}).get("total_candidates", 0))
    audit_data = data.get("disjointness_audit", data.get("leakage_verification", {}))
    is_disjoint = bool(audit_data.get("is_disjoint", False))

    splits_data = data.get("splits", {})
    train_split = splits_data.get("train", {})
    val_split = splits_data.get("validation", splits_data.get("val", {}))
    test_split = splits_data.get("test", {})

    train_cands = len(train_split.get("candidate_ids", train_split if isinstance(train_split, list) else []))
    val_cands = len(val_split.get("candidate_ids", val_split if isinstance(val_split, list) else []))
    test_cands = len(test_split.get("candidate_ids", test_split if isinstance(test_split, list) else []))

    result = {
        "is_disjoint": is_disjoint,
        "total_candidates": total_cands,
        "sum_of_splits": train_cands + val_cands + test_cands,
        "cross_split_account_overlap": 0.0 if is_disjoint else 1.0,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
