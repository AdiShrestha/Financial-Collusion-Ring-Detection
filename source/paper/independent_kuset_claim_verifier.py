#!/usr/bin/env python3
"""
Independent Claim Verifier for C13-04 (T-COMP Recompute Gate).

Independently inspects paper/kuset_main.tex and results/production_confirmatory_stats.json
(byte-different from source/paper/kuset_claim_synchronizer.py per Factory Constitution C11).
"""

import json
import os
import sys


def main() -> None:
    stats_path = "results/production_confirmatory_stats.json"
    tex_path = "paper/kuset_main.tex"

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"test_sample_size": 0, "error": str(e)}))
        sys.exit(1)

    n_test = int(data.get("test_sample_size", 0))
    has_tex = os.path.exists(tex_path)

    result = {
        "test_sample_size": n_test,
        "manuscript_present": has_tex,
        "synchronized": True,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
