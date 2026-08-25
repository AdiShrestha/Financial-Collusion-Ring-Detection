#!/usr/bin/env python3
"""Independent structural check of the KUSET Gate F report."""

import json
import sys


def main() -> None:
    try:
        data = json.load(open("project/gate_f_kuset_report.json", encoding="utf-8"))
    except Exception as exc:
        print(json.dumps({"valid": False, "error": str(exc)})); raise SystemExit(1)
    valid = (
        data.get("total_chunks_certified") == 19
        and data.get("total_checkpoints_trained") == 200
        and data.get("terminal_status") == "GATE_F_KUSET_PASS"
        and data.get("passed") is True
    )
    print(json.dumps({"total_chunks_certified": data.get("total_chunks_certified"), "total_checkpoints": data.get("total_checkpoints_trained"), "status": data.get("terminal_status"), "valid": valid}))
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
