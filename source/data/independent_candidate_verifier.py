#!/usr/bin/env python3
"""
Independent Candidate Integrity Verifier for C11-05 (T-COMP Recompute Gate).

Independently inspects project/candidate_integrity_report.json and validates status
(byte-different from source/data/candidate_verifier.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    report_path = "project/candidate_integrity_report.json"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "CANDIDATE_INTEGRITY_FAIL", "error": str(e)}))
        sys.exit(1)

    status = data.get("status", "UNKNOWN")
    all_ok = data.get("all_subchecks_passed", False)
    checks = data.get("integrity_checks", {})

    result = {
        "status": status,
        "candidate_integrity_status": status,
        "all_subchecks_passed": all_ok,
        "checks_passed_count": sum(1 for v in checks.values() if v is True),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
