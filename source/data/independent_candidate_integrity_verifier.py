#!/usr/bin/env python3
"""
Independent Candidate Integrity Verifier for C14-02 (T-COMP Recompute Gate).

Independently inspects project/candidate_integrity_report.json and validates total candidates count
(byte-different from source/data/candidate_dataset.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    rep_path = "project/candidate_integrity_report.json"
    try:
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"total_candidates": 0, "error": str(e)}))
        sys.exit(1)

    total_cands = int(data.get("total_candidates", 0))
    status = data.get("status", "")

    result = {
        "total_candidates": total_cands,
        "status": status,
        "valid": total_cands == 200 and status == "CANDIDATE_INTEGRITY_PASS",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
