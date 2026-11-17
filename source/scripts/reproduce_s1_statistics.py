#!/usr/bin/env python3
"""Reproduce conditional replication statistics from published world AP means."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np

ARMS = (
    "gbdt_fixed20",
    "fixed20_tabular_neural_control",
    "directed_local_edge_gnn",
    "simplicial_mpsn_style_local",
    "cellular_hasse_mechanism_control",
    "cellular_cwn_style_local",
)
WORLDS = tuple(f"W{i}" for i in range(5, 35))
BOOTSTRAP_SEED = 2026101103
BOOTSTRAP_RESAMPLES = 20000


def load_world_means(path):
    """Require the whole separate replication panel, including missingness reasons."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"phase", "arm", "world", "mean_ap", "initialization_count", "undefined_reason"}
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("missing or duplicate CSV columns")
        if not required.issubset(reader.fieldnames):
            raise ValueError("incomplete world-AP columns")
        values = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("malformed CSV record")
            key = row["arm"], row["world"]
            if key in values:
                raise ValueError("duplicate arm/world row")
            if row["phase"] != "replication" or row["initialization_count"] != "5":
                raise ValueError("replication phase and all five settings required")
            if row["mean_ap"] == "":
                if not row["undefined_reason"]:
                    raise ValueError("undefined AP lacks reason")
                value = None
            else:
                value = float(row["mean_ap"])
                if not math.isfinite(value) or not 0 <= value <= 1 or row["undefined_reason"]:
                    raise ValueError("invalid finite AP or missingness")
            values[key] = value
    if set(values) != {(a, w) for a in ARMS for w in WORLDS}:
        raise ValueError("complete six-arm/thirty-world replication coverage required")
    return values


def exact_signed_rank(differences):
    """Inclusive two-sided conditional sign tails with zero omission and midranks.

    Only rank arithmetic uses twelve-decimal differences. Integer doubled ranks
    preserve half-ranks. This conditional sign-symmetry test is not specifically
    a test of the arithmetic mean difference.
    """
    raw = [float(v) for v in differences]
    if not raw or len(raw) > 30 or any(not math.isfinite(v) for v in raw):
        raise ValueError("one to thirty finite paired differences required")
    rounded = [round(v, 12) for v in raw]
    groups = {}
    for i, v in enumerate(rounded):
        if v != 0:
            groups.setdefault(abs(v), []).append(i)
    rank_twice = [0] * len(raw)
    ties = []
    next_rank = 1
    for magnitude, indexes in sorted(groups.items()):
        last_rank = next_rank + len(indexes) - 1
        doubled = next_rank + last_rank
        for i in indexes:
            rank_twice[i] = doubled
        if len(indexes) > 1:
            ties.append(
                {"absolute_difference": magnitude, "count": len(indexes), "average_rank": doubled / 2}
            )
        next_rank = last_rank + 1
    distribution = {0: 1}
    nonzero_ranks = [r for r in rank_twice if r]
    for r in nonzero_ranks:
        updated = dict(distribution)
        for subtotal, count in distribution.items():
            updated[subtotal + r] = updated.get(subtotal + r, 0) + count
        distribution = updated
    observed = sum(r for r, v in zip(rank_twice, rounded) if v > 0)
    total = sum(nonzero_ranks)
    le = sum(c for subtotal, c in distribution.items() if subtotal <= observed)
    ge = sum(c for subtotal, c in distribution.items() if subtotal >= observed)
    denominator = 2 ** len(nonzero_ranks)
    p = min(Fraction(1), Fraction(2 * min(le, ge), denominator))
    return dict(
        status="EXACT_CONDITIONAL_SIGN_DISTRIBUTION" if nonzero_ranks else "DEGENERATE_ALL_ZERO",
        statistic=min(observed, total - observed) / 2,
        w_plus=observed / 2,
        w_minus=(total - observed) / 2,
        n_nonzero=len(nonzero_ranks),
        zero_count=len(raw) - len(nonzero_ranks),
        tie_groups=ties,
        tied_observation_count=sum(t["count"] for t in ties),
        tail_le_count=le,
        tail_ge_count=ge,
        denominator=denominator,
        p_value=float(p),
        rounding_decimal_places=12,
        rank_twice_by_original_index=rank_twice,
        rounded_differences=rounded,
        mean_signed_difference=math.fsum(raw) / len(raw),
    )


def holm(p_values):
    if not p_values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p_values.values()):
        raise ValueError("finite p-values in [0,1] required")
    order = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * p_values[key]))
        adjusted[key] = running
    return adjusted


def reproduce(values):
    if set(values) != {(a, w) for a in ARMS for w in WORLDS}:
        raise ValueError("complete replication panel required")
    if any(v is None for v in values.values()):
        return {
            "status": "UNAVAILABLE_REQUIRED_AP",
            "pilot_pooled": False,
            "reason": "A required world AP is unavailable; no row was dropped.",
        }
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("finite world AP in [0,1] required")
    matrix = np.array([[values[a, w] for a in ARMS] for w in WORLDS], dtype=np.float64)
    contrasts = {a: [values[a, w] - values[ARMS[0], w] for w in WORLDS] for a in ARMS[1:]}
    tests = {a: exact_signed_rank(contrasts[a]) for a in ARMS[1:]}
    adjusted = holm({a: t["p_value"] for a, t in tests.items()})
    for a in tests:
        tests[a]["holm_adjusted_p_value"] = adjusted[a]
    targets = np.column_stack([matrix, *(np.asarray(contrasts[a], dtype=np.float64) for a in ARMS[1:])])
    names = [*ARMS, *(a + "_minus_gbdt_fixed20" for a in ARMS[1:])]
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(WORLDS), size=(BOOTSTRAP_RESAMPLES, len(WORLDS)), dtype=np.int64
    )
    samples = targets[indices].mean(axis=1)
    intervals = {}
    for i, name in enumerate(names):
        lower, upper = np.quantile(samples[:, i], [0.025, 0.975], method="linear")
        intervals[name] = dict(
            estimate=float(targets[:, i].mean()),
            lower_95=float(lower),
            upper_95=float(upper),
            width=float(upper - lower),
            scope="marginal conditional percentile interval",
        )
    return dict(
        status="COMPLETE_PUBLISHED_TABLE_ARITHMETIC",
        panel="prospective_replication",
        pilot_pooled=False,
        worlds=list(WORLDS),
        equal_world_mean_ap={a: math.fsum(values[a, w] for w in WORLDS) / len(WORLDS) for a in ARMS},
        equal_world_mean_differences={a: math.fsum(d) / len(d) for a, d in contrasts.items()},
        signed_rank_family=tests,
        holm_family_size=len(tests),
        paired_world_bootstrap=dict(
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
            rng="PCG64",
            interval_method="linear percentile",
            intervals=intervals,
        ),
        scope="Fixed states and generator; conditional sign assumptions; marginal intervals, not simultaneous.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-ap-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_json.exists():
        raise FileExistsError("existing arithmetic output will not be overwritten")
    result = reproduce(load_world_means(args.world_ap_csv))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
