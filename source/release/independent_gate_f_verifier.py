#!/usr/bin/env python3
"""
Independent Gate F Verifier for C09-05 (T-COMP Recompute Gate).

Re-implements gate chain verification from scratch (byte-different from
source/release/gate_f_verifier.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    report_path = "project/gate_f_report.json"
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"gate_f_status": "GATE_F_FAIL", "error": str(e)}))
        sys.exit(1)

    status = report.get("gate_f_status", "UNKNOWN")
    all_ok = report.get("all_sections_passed", False)
    num_sections = len(report.get("sections", {}))

    result = {
        "gate_f_status": status,
        "all_sections_passed": all_ok,
        "num_sections_verified": num_sections,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
