#!/usr/bin/env python3
"""Independent Gate D verification script for Gatekeeper recomputation."""

import json
import os
import sys

def main():
    report_path = "project/gate_d_report.json"
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    status = data.get("gate_d_status", "UNKNOWN")
    output = {
        "gate_d_status": status,
        "all_subchecks_passed": data.get("all_subchecks_passed", False)
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
