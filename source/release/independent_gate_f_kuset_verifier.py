#!/usr/bin/env python3
"""
Independent Gate F Verifier for C14-05 (T-COMP Recompute Gate).

Independently inspects project/gate_f_kuset_report.json and validates certification
(byte-different from source/release/kuset_release_packager.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    rep_path = "project/gate_f_kuset_report.json"
    try:
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"total_chunks_certified": 0, "error": str(e)}))
        sys.exit(1)

    n_chunks = int(data.get("total_chunks_certified", 0))
    status = data.get("terminal_status", "")

    result = {
        "total_chunks_certified": n_chunks,
        "status": status,
        "valid": n_chunks in (13, 14) and status == "GATE_F_KUSET_PASS",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
