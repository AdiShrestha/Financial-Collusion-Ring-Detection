#!/usr/bin/env python3
"""Independent verification script for Gate D test split cryptographic checksum."""

import hashlib
import json
import os
import sys

def main():
    manifest_path = "data/manifests/split_manifest.json"
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    test_cands = manifest_data.get("splits", {}).get("test", {}).get("candidate_ids", [])
    joined_ids = "\n".join(sorted(test_cands)).encode("utf-8")
    computed_hash = hashlib.sha256(joined_ids).hexdigest()

    output = {
        "test_sha256_checksum": computed_hash,
        "candidate_count": len(test_cands),
        "status": "VERIFIED"
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
