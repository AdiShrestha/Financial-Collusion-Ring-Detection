#!/usr/bin/env python3
"""One-command reproduction script for TopoRingNet results."""
import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd} (exit {result.returncode})")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("TopoRingNet Replication Script")
    print("=" * 60)

    # Step 1: Run full test suite
    run("python3 -m pytest tests/ -v")

    print("\n" + "=" * 60)
    print("REPLICATION COMPLETE — all tests passed.")
    print("=" * 60)
