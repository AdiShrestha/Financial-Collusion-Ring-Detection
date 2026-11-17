#!/usr/bin/env python3
"""Plot complete saved simulator metrics; no predictors or empirical data required."""

from __future__ import annotations
import argparse
import csv
import itertools
import math
from pathlib import Path

ARMS = (
    "gbdt_fixed20",
    "fixed20_tabular_neural_control",
    "directed_local_edge_gnn",
    "simplicial_mpsn_style_local",
    "cellular_hasse_mechanism_control",
    "cellular_cwn_style_local",
)
LABELS = dict(
    zip(
        ARMS,
        (
            "GBDT (summary)",
            "Summary MLP",
            "Directed edge GNN",
            "Simplicial (MPSN-style)",
            "Cellular, Hasse variant",
            "Cellular (CWN-style)",
        ),
    )
)
WORLDS = tuple(f"W{i}" for i in range(1, 5))
SETTINGS = tuple(f"I{i}" for i in range(1, 6))
REPLICATION_WORLDS = tuple(f"W{i}" for i in range(5, 35))


def load_metrics(path, worlds=WORLDS):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("missing or duplicate CSV columns")
        required = {
            "arm",
            "world",
            "initialization",
            "ap",
            "ap_undefined_reason",
            "auroc",
            "auroc_undefined_reason",
            "bce",
            "candidate_count",
            "positives",
            "negatives",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        }
        if not required.issubset(reader.fieldnames):
            raise ValueError("incomplete metric columns")
        rows = []
        keys = set()
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError("malformed CSV record")
            key = (raw["arm"], raw["world"], raw["initialization"])
            if key in keys:
                raise ValueError("duplicate arm/world/initialization")
            keys.add(key)
            row = dict(raw)
            for field in ("ap", "auroc", "bce"):
                value = raw[field]
                if value == "" and field != "bce":
                    if not raw[field + "_undefined_reason"]:
                        raise ValueError("undefined metric lacks reason")
                    row[field] = None
                else:
                    value = float(value)
                    if not math.isfinite(value) or value < 0 or (field != "bce" and value > 1):
                        raise ValueError("invalid finite metric range")
                    if field != "bce" and raw[field + "_undefined_reason"]:
                        raise ValueError("defined metric has undefined reason")
                    row[field] = value
            for field in (
                "candidate_count",
                "positives",
                "negatives",
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
            ):
                if not raw[field].isdigit():
                    raise ValueError("counts must be nonnegative integers")
                row[field] = int(raw[field])
            if row["candidate_count"] <= 0 or row["positives"] + row["negatives"] != row["candidate_count"]:
                raise ValueError("class counts do not reconcile")
            if (
                row["true_positive"] + row["false_negative"] != row["positives"]
                or row["true_negative"] + row["false_positive"] != row["negatives"]
            ):
                raise ValueError("confusion counts do not reconcile")
            if (row["ap"] is None) != (row["positives"] == 0):
                raise ValueError("AP support and missingness disagree")
            if (row["auroc"] is None) != (row["positives"] == 0 or row["negatives"] == 0):
                raise ValueError("AUROC support and missingness disagree")
            rows.append(row)
    expected = set(itertools.product(ARMS, worlds, SETTINGS))
    if keys != expected:
        raise ValueError("complete six-arm/registered-world/five-setting coverage required")
    for world in worlds:
        counts = {(r["candidate_count"], r["positives"], r["negatives"]) for r in rows if r["world"] == world}
        if len(counts) != 1:
            raise ValueError("world class counts differ across model settings")
    expected_phase = "heldout" if tuple(worlds) == WORLDS else "replication"
    if any(r.get("phase") != expected_phase for r in rows):
        raise ValueError(f"every row must carry phase={expected_phase!r}; panels are never mixed")
    return rows


def aggregate(rows, worlds=WORLDS):
    """Average metrics across settings, then give each registered world equal weight."""
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for world in worlds:
            values = [r["ap"] for r in rows if r["arm"] == arm and r["world"] == world]
            if len(values) != len(SETTINGS):
                raise ValueError("missing settings in aggregation")
            means[arm][world] = None if any(x is None for x in values) else math.fsum(values) / len(values)
    overall = {
        arm: None if any(v is None for v in d.values()) else math.fsum(d.values()) / len(worlds)
        for arm, d in means.items()
    }
    contrasts = {
        arm: {
            w: (
                None
                if means[arm][w] is None or means[ARMS[0]][w] is None
                else means[arm][w] - means[ARMS[0]][w]
            )
            for w in worlds
        }
        for arm in ARMS[1:]
    }
    return means, overall, contrasts


COLORS = ("#222222", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#7a5d00")
MARKERS = ("o", "s", "^", "D", "v", "P")


def _pyplot():
    """Configure matplotlib so figures match the manuscript body (Computer Modern, 9 pt)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "s1-result-plots",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
        }
    )
    return plt


def _save(plt, fig, output_dir, name, preview_dir=None):
    """Write the exact vector files used by the manuscript (PDF) and a web copy (SVG)."""
    fig.savefig(
        output_dir / (name + ".pdf"),
        metadata={
            "Title": name.replace("_", " "),
            "Author": "",
            "Creator": "plot_s1_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_dir / (name + ".svg"), metadata={"Date": None, "Creator": "plot_s1_results.py"})
    if preview_dir is not None:
        fig.savefig(preview_dir / (name + ".png"), dpi=160, metadata={"Software": "plot_s1_results.py"})
    plt.close(fig)


def _guard(output_dir, names, preview_dir=None):
    for name in names:
        for suffix in (".pdf", ".svg"):
            if (output_dir / (name + suffix)).exists():
                raise FileExistsError("existing figures will not be overwritten")
        if preview_dir is not None and (preview_dir / (name + ".png")).exists():
            raise FileExistsError("existing previews will not be overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)


def make_plots(rows, output_dir, preview_dir=None):
    """Pilot (W1-W4) figures: AP by world and contrasts against GBDT."""
    plt = _pyplot()
    output_dir = Path(output_dir)
    preview_dir = Path(preview_dir) if preview_dir is not None else None
    names = ("average_precision_by_world", "descriptive_ap_contrasts")
    _guard(output_dir, names, preview_dir)
    means, overall, contrasts = aggregate(rows)
    for plot_index, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(6.5, 2.9))
        arms = ARMS if plot_index == 0 else ARMS[1:]
        for arm in arms:
            i = ARMS.index(arm)
            series = means[arm] if plot_index == 0 else contrasts[arm]
            ax.plot(
                range(len(WORLDS)),
                [float("nan") if series[w] is None else series[w] for w in WORLDS],
                color=COLORS[i],
                marker=MARKERS[i],
                linewidth=1,
                markersize=4,
                label=LABELS[arm],
            )
        ax.set_xticks(range(len(WORLDS)), WORLDS)
        ax.set_xlim(-0.3, len(WORLDS) - 0.7)
        ax.set_xlabel("World")
        ax.set_ylabel("Mean AP" if plot_index == 0 else "AP difference from GBDT")
        ax.grid(axis="y", alpha=0.25)
        if plot_index == 0:
            ax.set_ylim(-0.02, 1.04)
        else:
            ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_ylim(-1.03, 1.03)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.subplots_adjust(left=0.1, right=0.72, bottom=0.16, top=0.97)
        _save(plt, fig, output_dir, name, preview_dir)
    return means, overall, contrasts


def load_intervals(path, expected_arms, estimate_field):
    """Read published marginal intervals; plotting does not perform inference."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "phase",
            "arm",
            estimate_field,
            "lower_95",
            "upper_95",
            "interval_width",
            "interval_scope",
        }
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("missing or duplicate interval columns")
        if not required.issubset(reader.fieldnames):
            raise ValueError("incomplete interval columns")
        result = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("malformed interval row")
            arm = row["arm"]
            if arm in result or arm not in expected_arms:
                raise ValueError("duplicate or unknown interval arm")
            if (
                row["phase"] != "replication"
                or row["interval_scope"] != "marginal conditional percentile interval"
            ):
                raise ValueError("separate replication and marginal interval scope required")
            values = [float(row[k]) for k in (estimate_field, "lower_95", "upper_95", "interval_width")]
            estimate, lower, upper, width = values
            if (
                any(not math.isfinite(v) for v in values)
                or lower > upper
                or abs(width - (upper - lower)) > 1e-8
            ):
                raise ValueError("invalid finite interval or width")
            result[arm] = estimate, lower, upper
        if set(result) != set(expected_arms):
            raise ValueError("complete interval arm coverage required")
    return result


def make_replication_plots(rows, mean_intervals_csv, contrast_intervals_csv, output_dir, preview_dir=None):
    """Held-out (W5-W34) figures: AP by world, and means and contrasts with intervals."""
    means, overall, contrasts = aggregate(rows, REPLICATION_WORLDS)
    if any(v is None for v in overall.values()):
        raise ValueError("required replication AP is unavailable; no world can be dropped")
    arm_intervals = load_intervals(mean_intervals_csv, ARMS, "mean_ap")
    contrast_intervals = load_intervals(contrast_intervals_csv, ARMS[1:], "mean_ap_difference")
    for arm in ARMS:
        if abs(arm_intervals[arm][0] - overall[arm]) > 1e-8:
            raise ValueError("arm interval estimate differs from complete world means")
    for arm in ARMS[1:]:
        estimate = math.fsum(contrasts[arm].values()) / len(REPLICATION_WORLDS)
        if abs(contrast_intervals[arm][0] - estimate) > 1e-8:
            raise ValueError("contrast interval estimate differs from complete paired worlds")
    plt = _pyplot()
    names = ("replication_ap_by_world", "replication_means_and_contrasts")
    output_dir = Path(output_dir)
    preview_dir = Path(preview_dir) if preview_dir is not None else None
    _guard(output_dir, names, preview_dir)

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4.4))
    for j, ax in enumerate(axs):
        worlds = REPLICATION_WORLDS[15 * j : 15 * (j + 1)]
        for i, arm in enumerate(ARMS):
            ax.plot(
                range(len(worlds)),
                [means[arm][w] for w in worlds],
                color=COLORS[i],
                marker=MARKERS[i],
                linewidth=1,
                markersize=3.5,
                label=LABELS[arm],
            )
        ax.set_xticks(range(len(worlds)), worlds)
        ax.set_xlim(-0.4, len(worlds) - 0.6)
        ax.set_ylim(-0.02, 1.04)
        ax.set_ylabel("Mean AP")
        ax.grid(axis="y", alpha=0.25)
    axs[1].set_xlabel("World")
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.53, 0),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.98, bottom=0.2, hspace=0.35)
    _save(plt, fig, output_dir, names[0], preview_dir)

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [6, 5]})
    for ax, arms, intervals, xlabel, limits in (
        (axs[0], ARMS, arm_intervals, "Mean AP (pointwise 95% interval)", (0, 1.02)),
        (axs[1], ARMS[1:], contrast_intervals, "AP difference from GBDT", (-1, 0.05)),
    ):
        for arm in arms:
            estimate, lower, upper = intervals[arm]
            i = y = ARMS.index(arm)  # both panels share one row per model
            ax.errorbar(
                estimate,
                y,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=MARKERS[i],
                color=COLORS[i],
                capsize=2.5,
                markersize=4,
                linewidth=1,
            )
        ax.set_yticks(range(len(ARMS)), [LABELS[a] for a in ARMS])
        ax.set_ylim(len(ARMS) - 0.5, -0.5)
        ax.set_xlim(*limits)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
    axs[1].axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    axs[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.215, right=0.98, top=0.97, bottom=0.19, wspace=0.08)
    _save(plt, fig, output_dir, names[1], preview_dir)
    return means, overall, contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, help="Optional PNG previews for local inspection")
    parser.add_argument("--panel", choices=("pilot", "replication"), default="pilot")
    parser.add_argument("--mean-intervals-csv", type=Path)
    parser.add_argument("--contrast-intervals-csv", type=Path)
    args = parser.parse_args(argv)
    if args.panel == "replication":
        if args.mean_intervals_csv is None or args.contrast_intervals_csv is None:
            parser.error("replication plots require both interval CSV arguments")
        make_replication_plots(
            load_metrics(args.metrics_csv, REPLICATION_WORLDS),
            args.mean_intervals_csv,
            args.contrast_intervals_csv,
            args.output_dir,
            args.preview_dir,
        )
    else:
        make_plots(load_metrics(args.metrics_csv), args.output_dir, args.preview_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
