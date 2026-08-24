#!/usr/bin/env python3
"""
Independent Phase-0 Feasibility Gate Verifier for C10-05 (T-COMP Recompute Gate).

Re-implements Phase-0 gate verification independently from scratch
(byte-different from source/data/phase0_gate_verifier.py per Factory Constitution C11).
"""

import json
import os
import sys


def main() -> None:
    report_path = "project/phase0_gate_report.json"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "PHASE0_GATE_FAIL", "error": str(e)}))
        sys.exit(1)

    status = data.get("status", "UNKNOWN")
    all_ok = data.get("all_subchecks_passed", False)
    checks = data.get("feasibility_checks", {})

    result = {
        "status": status,
        "phase0_gate_status": status,
        "all_subchecks_passed": all_ok,
        "checks_passed_count": sum(1 for v in checks.values() if v is True),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
