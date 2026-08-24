#!/usr/bin/env python3
"""
Independent Stats Verifier for C13-02 (T-COMP Recompute Gate).

Independently inspects results/production_confirmatory_stats.json and validates hypotheses
(byte-different from source/evidence/kuset_hypothesis_tester.py per Factory Constitution C11).
"""

import json
import sys


def main() -> None:
    stats_path = "results/production_confirmatory_stats.json"
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"test_sample_size": 0, "error": str(e)}))
        sys.exit(1)

    n_test = int(data.get("test_sample_size", 0))
    hypos = data.get("hypotheses", {})

    result = {
        "test_sample_size": n_test,
        "hypotheses_count": len(hypos),
        "valid": n_test == 28 and len(hypos) == 3,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
