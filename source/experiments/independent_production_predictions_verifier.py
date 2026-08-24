#!/usr/bin/env python3
"""
Independent Predictions Verifier for C13-01 (T-COMP Recompute Gate).

Independently inspects runs/production_confirmatory/predictions.json and validates count
(byte-different from source/experiments/production_confirmatory_runner.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    pred_path = "runs/production_confirmatory/predictions.json"
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"total_test_candidates": 0, "error": str(e)}))
        sys.exit(1)

    preds = data.get("predictions", [])
    count = len(preds)

    result = {
        "total_test_candidates": count,
        "valid": count == 28,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
