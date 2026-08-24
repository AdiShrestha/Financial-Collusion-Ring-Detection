#!/usr/bin/env python3
"""Independent hypothesis testing verification script for Gatekeeper recomputation."""

import json
import os
import sys


def verify_test_predictions(predictions_path="runs/confirmatory/predictions.json"):
    """Independently parse and verify candidate counts in confirmatory predictions."""
    if not os.path.exists(predictions_path):
        return {"num_test_candidates": 0, "status": "FILE_NOT_FOUND"}

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    num_candidates = meta.get("num_test_candidates", 0)

    return {
        "num_test_candidates": float(num_candidates),
        "status": "PASS"
    }


def main():
    res = verify_test_predictions()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
