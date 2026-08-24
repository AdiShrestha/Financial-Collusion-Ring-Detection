#!/usr/bin/env python3
"""
Independent Production Gate Verifier for C12-05 (T-COMP Recompute Gate).

Independently inspects project/production_training_report.json and validates status
(byte-different from source/training/production_gate_verifier.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    report_path = "project/production_training_report.json"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "PRODUCTION_TRAINING_FAIL", "error": str(e)}))
        sys.exit(1)

    status = data.get("status", "UNKNOWN")
    all_ok = data.get("all_subchecks_passed", False)
    total_audited = int(data.get("total_checkpoints_audited", 0))

    result = {
        "status": status,
        "production_training_status": status,
        "all_subchecks_passed": all_ok,
        "total_checkpoints_audited": total_audited,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
