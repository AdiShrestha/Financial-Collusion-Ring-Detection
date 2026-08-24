#!/usr/bin/env python3
"""
Independent Claim Synchronization Verifier for C09-03 (T-COMP Recompute Gate).

This script independently audits the numerical synchronization between
paper/main.tex and results/confirmatory_stats.json by re-implementing
the parsing and comparison logic from scratch (byte-different from
source/paper/claim_synchronizer.py per Factory Constitution C11).
"""

import json
import re
import sys


def main() -> None:
    tex_path = "paper/main.tex"
    stats_path = "results/confirmatory_stats.json"
    tolerance = 1e-4

    with open(tex_path, "r") as f:
        tex = f.read()
    with open(stats_path, "r") as f:
        stats = json.load(f)

    benchmarks = stats["model_benchmarks"]
    total_checks = 0
    mismatches = 0

    # Parse table rows independently
    for line in tex.split("\n"):
        line = line.strip()
        m = re.match(
            r"^(\w+)\s+&\s+\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$"
            r"\s+&\s+\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$"
            r"\s+&\s+\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$",
            line,
        )
        if not m:
            continue
        model = m.group(1)
        if model not in benchmarks:
            mismatches += 1
            continue
        ref = benchmarks[model]
        pairs = [
            (float(m.group(2)), ref["mean_pr_auc"]),
            (float(m.group(3)), ref["std_pr_auc"]),
            (float(m.group(4)), ref["mean_roc_auc"]),
            (float(m.group(5)), ref["std_roc_auc"]),
            (float(m.group(6)), ref["mean_f1_macro"]),
            (float(m.group(7)), ref["std_f1_macro"]),
        ]
        for paper_v, data_v in pairs:
            total_checks += 1
            if abs(paper_v - data_v) >= tolerance:
                mismatches += 1

    # Parse hypothesis verdicts independently
    verdicts_ref = stats.get("hypothesis_testing", {}).get("overall_verdicts", {})
    for hid in ["H1", "H2", "H3", "H4"]:
        total_checks += 1
        if hid in verdicts_ref:
            if verdicts_ref[hid] not in tex:
                mismatches += 1

    all_synced = mismatches == 0 and total_checks > 0

    result = {
        "independent_sync_check": all_synced,
        "num_checked": total_checks,
        "num_mismatches": mismatches,
        "claims_registered": 4 if all_synced else 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
