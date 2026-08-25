#!/usr/bin/env python3
"""Read-only verification of existing Chunk 19 release artifacts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.release.packager import verify_existing_artifacts


def verify_existing_release():
    return verify_existing_artifacts(ROOT)


def run_clean_room_replication():
    """Backward-compatible name; performs existing-artifact verification only."""
    return verify_existing_release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(verify_existing_release(), indent=2))
