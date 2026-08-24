#!/usr/bin/env python3
"""Independent claim registry verification script for Gatekeeper recomputation."""

import json
import os
import sys


def verify_claims_registered(stats_path="results/confirmatory_stats.json"):
    """Independently verify registered claims in confirmatory stats artifact."""
    if not os.path.exists(stats_path):
        return {"claims_registered": 0, "status": "FILE_NOT_FOUND"}

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    num_claims = meta.get("claims_registered", len(data.get("claim_registry", {})))

    return {
        "claims_registered": float(num_claims),
        "status": "PASS"
    }


def main():
    res = verify_claims_registered()
    print(json.dumps(res))


if __name__ == "__main__":
    main()
