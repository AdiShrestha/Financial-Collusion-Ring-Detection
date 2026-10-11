#!/usr/bin/env python3
"""Plot complete saved simulator metrics; no predictors or empirical data required."""
from __future__ import annotations
import argparse
import csv
import itertools
import math
from pathlib import Path

ARMS = (
    'gbdt_fixed20', 'fixed20_tabular_neural_control', 'directed_local_edge_gnn',
    'simplicial_mpsn_style_local', 'cellular_hasse_mechanism_control', 'cellular_cwn_style_local',
)
LABELS = dict(zip(ARMS, ('Summary GBDT', 'Summary neural', 'Directed edge GNN',
    'Local simplicial', 'Hasse control', 'Local cellular')))
WORLDS = tuple(f'W{i}' for i in range(1, 5))
SETTINGS = tuple(f'I{i}' for i in range(1, 6))
REPLICATION_WORLDS = tuple(f'W{i}' for i in range(5, 35))


def load_metrics(path, worlds=WORLDS):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('missing or duplicate CSV columns')
        required = {'arm', 'world', 'initialization', 'ap', 'ap_undefined_reason', 'auroc',
            'auroc_undefined_reason', 'bce', 'candidate_count', 'positives', 'negatives',
            'true_positive', 'false_positive', 'true_negative', 'false_negative'}
        if not required.issubset(reader.fieldnames):
            raise ValueError('incomplete metric columns')
        rows = []
        keys = set()
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError('malformed CSV record')
            key = (raw['arm'], raw['world'], raw['initialization'])
            if key in keys:
                raise ValueError('duplicate arm/world/initialization')
            keys.add(key)
            row = dict(raw)
            for field in ('ap', 'auroc', 'bce'):
                value = raw[field]
                if value == '' and field != 'bce':
                    if not raw[field + '_undefined_reason']:
                        raise ValueError('undefined metric lacks reason')
                    row[field] = None
                else:
                    value = float(value)
                    if not math.isfinite(value) or value < 0 or (field != 'bce' and value > 1):
                        raise ValueError('invalid finite metric range')
                    if field != 'bce' and raw[field + '_undefined_reason']:
                        raise ValueError('defined metric has undefined reason')
                    row[field] = value
            for field in ('candidate_count', 'positives', 'negatives', 'true_positive',
                          'false_positive', 'true_negative', 'false_negative'):
                if not raw[field].isdigit():
                    raise ValueError('counts must be nonnegative integers')
                row[field] = int(raw[field])
            if row['candidate_count'] <= 0 or row['positives'] + row['negatives'] != row['candidate_count']:
                raise ValueError('class counts do not reconcile')
            if row['true_positive'] + row['false_negative'] != row['positives'] or row['true_negative'] + row['false_positive'] != row['negatives']:
                raise ValueError('confusion counts do not reconcile')
            if (row['ap'] is None) != (row['positives'] == 0):
                raise ValueError('AP support and missingness disagree')
            if (row['auroc'] is None) != (row['positives'] == 0 or row['negatives'] == 0):
                raise ValueError('AUROC support and missingness disagree')
            rows.append(row)
    expected = set(itertools.product(ARMS, worlds, SETTINGS))
    if keys != expected:
        raise ValueError('complete six-arm/registered-world/five-setting coverage required')
    for world in worlds:
        counts = {(r['candidate_count'], r['positives'], r['negatives']) for r in rows if r['world'] == world}
        if len(counts) != 1:
            raise ValueError('world class counts differ across model settings')
    if tuple(worlds) != WORLDS and any(r.get('phase') != 'replication' for r in rows):
        raise ValueError('replication rows require a separate replication phase')
    return rows


def aggregate(rows, worlds=WORLDS):
    """Average metrics across settings, then give each registered world equal weight."""
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for world in worlds:
            values = [r['ap'] for r in rows if r['arm'] == arm and r['world'] == world]
            if len(values) != len(SETTINGS):
                raise ValueError('missing settings in aggregation')
            means[arm][world] = None if any(x is None for x in values) else math.fsum(values) / len(values)
    overall = {arm: None if any(v is None for v in d.values()) else math.fsum(d.values()) / len(worlds)
               for arm, d in means.items()}
    contrasts = {arm: {w: None if means[arm][w] is None or means[ARMS[0]][w] is None
        else means[arm][w] - means[ARMS[0]][w] for w in worlds} for arm in ARMS[1:]}
    return means, overall, contrasts


def make_plots(rows, output_dir, preview_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 's1-result-plots',
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    output_dir = Path(output_dir)
    names = ('average_precision_by_world', 'descriptive_ap_contrasts')
    if any((output_dir / (name + '.svg')).exists() for name in names):
        raise FileExistsError('existing figures will not be overwritten')
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        if any((preview_dir / (name + '.png')).exists() for name in names):
            raise FileExistsError('existing previews will not be overwritten')
        preview_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    means, overall, contrasts = aggregate(rows)
    colors = ('#222222', '#0072b2', '#d55e00', '#009e73', '#cc79a7', '#7a5d00')
    markers = ('o', 's', '^', 'D', 'v', 'P')
    for plot_index, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        arms = ARMS if plot_index == 0 else ARMS[1:]
        for arm in arms:
            i = ARMS.index(arm)
            series = means[arm] if plot_index == 0 else contrasts[arm]
            ax.plot(range(len(WORLDS)), [float('nan') if series[w] is None else series[w] for w in WORLDS],
                    color=colors[i], marker=markers[i], linewidth=1.8, markersize=7, label=LABELS[arm])
        ax.set_xticks(range(len(WORLDS)), WORLDS)
        ax.set_xlabel('Registered simulator world (neutral alias)')
        ax.set_ylabel('Mean average precision' if plot_index == 0 else 'Mean AP difference: neural minus GBDT')
        ax.set_title('Frozen panel on four simulator worlds — descriptive results')
        ax.grid(axis='y', alpha=.25)
        if plot_index == 0:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
        else:
            ax.axhline(0, color='#555555', linestyle='--', linewidth=1)
            ax.set_ylim(-1, 1)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, .5), frameon=False)
        fig.subplots_adjust(left=.085, right=.70, bottom=.23, top=.88)
        fig.text(.085, .06, 'Each point averages all five initialization settings within its world.\n'
                 'All registered worlds retained; no population intervals or superiority inference.', fontsize=10)
        fig.savefig(output_dir / (name + '.svg'), metadata={'Date': None, 'Creator': 'Saved-metric plotting utility'})
        if preview_dir is not None:
            fig.savefig(preview_dir / (name + '.png'), dpi=160, metadata={'Software': 'Saved-metric plotting utility'})
        plt.close(fig)
    return means, overall, contrasts


def load_intervals(path, expected_arms, estimate_field):
    """Read published marginal intervals; plotting does not perform inference."""
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'phase', 'arm', estimate_field, 'lower_95', 'upper_95', 'interval_width', 'interval_scope'}
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('missing or duplicate interval columns')
        if not required.issubset(reader.fieldnames):
            raise ValueError('incomplete interval columns')
        result = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('malformed interval row')
            arm = row['arm']
            if arm in result or arm not in expected_arms:
                raise ValueError('duplicate or unknown interval arm')
            if row['phase'] != 'replication' or row['interval_scope'] != 'marginal conditional percentile interval':
                raise ValueError('separate replication and marginal interval scope required')
            values = [float(row[k]) for k in (estimate_field, 'lower_95', 'upper_95', 'interval_width')]
            estimate, lower, upper, width = values
            if any(not math.isfinite(v) for v in values) or lower > upper or abs(width - (upper - lower)) > 1e-8:
                raise ValueError('invalid finite interval or width')
            result[arm] = estimate, lower, upper
        if set(result) != set(expected_arms):
            raise ValueError('complete interval arm coverage required')
    return result


def make_replication_plots(rows, mean_intervals_csv, contrast_intervals_csv, output_dir, preview_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'replication-result-plots',
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    means, overall, contrasts = aggregate(rows, REPLICATION_WORLDS)
    if any(v is None for v in overall.values()):
        raise ValueError('required replication AP is unavailable; no world can be dropped')
    arm_intervals = load_intervals(mean_intervals_csv, ARMS, 'mean_ap')
    contrast_intervals = load_intervals(contrast_intervals_csv, ARMS[1:], 'mean_ap_difference')
    for arm in ARMS:
        if abs(arm_intervals[arm][0] - overall[arm]) > 1e-8:
            raise ValueError('arm interval estimate differs from complete world means')
    for arm in ARMS[1:]:
        estimate = math.fsum(contrasts[arm].values()) / len(REPLICATION_WORLDS)
        if abs(contrast_intervals[arm][0] - estimate) > 1e-8:
            raise ValueError('contrast interval estimate differs from complete paired worlds')
    names = ('replication_ap_by_world', 'replication_mean_ap_intervals', 'replication_ap_contrast_intervals')
    output_dir = Path(output_dir)
    preview_dir = Path(preview_dir) if preview_dir is not None else None
    if any((output_dir / (name + '.svg')).exists() for name in names):
        raise FileExistsError('existing replication figures will not be overwritten')
    if preview_dir is not None and any((preview_dir / (name + '.png')).exists() for name in names):
        raise FileExistsError('existing replication previews will not be overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    colors = ('#222222', '#0072b2', '#d55e00', '#009e73', '#cc79a7', '#7a5d00')
    markers = ('o', 's', '^', 'D', 'v', 'P')

    def save(fig, name):
        fig.savefig(output_dir / (name + '.svg'), metadata={'Date': None, 'Creator': 'Published-table plotting utility'})
        if preview_dir is not None:
            fig.savefig(preview_dir / (name + '.png'), dpi=150, metadata={'Software': 'Published-table plotting utility'})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, arm in enumerate(ARMS):
        ax.plot(range(len(REPLICATION_WORLDS)), [means[arm][w] for w in REPLICATION_WORLDS],
                color=colors[i], marker=markers[i], linewidth=1.25, markersize=5, label=LABELS[arm])
    ax.set_xticks(range(len(REPLICATION_WORLDS)), REPLICATION_WORLDS, rotation=55)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('Average precision: mean across five settings within world')
    ax.set_xlabel('Prospectively registered simulator realization (neutral alias)')
    ax.set_title('Separate thirty-world prospective replication: every arm and world')
    ax.grid(axis='y', alpha=.25)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.05), ncol=3, frameon=False)
    fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.77)
    fig.text(.075, .055, 'All worlds retained; connecting lines show aliases, not time. Pilot worlds are excluded.\n'
             'Each point is a five-setting mean; no per-world uncertainty interval is shown.', fontsize=10)
    save(fig, names[0])

    for name, arms, intervals, xlabel, limits in (
        (names[1], ARMS, arm_intervals, 'Equal-world mean average precision', (0, 1.02)),
        (names[2], ARMS[1:], contrast_intervals, 'Equal-world mean AP difference: neural minus GBDT', (-1, .12)),
    ):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for y, arm in enumerate(arms):
            estimate, lower, upper = intervals[arm]
            color = colors[ARMS.index(arm)]
            ax.hlines(y, lower, upper, color=color, linewidth=2.3)
            ax.vlines([lower, upper], y - .10, y + .10, color=color, linewidth=1.5)
            ax.scatter([estimate], [y], color=color, marker=markers[ARMS.index(arm)], s=65, zorder=3)
        ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_title('Thirty-world prospective replication: marginal 95% paired-world intervals')
        ax.grid(axis='x', alpha=.25)
        fig.subplots_adjust(left=.23, right=.96, bottom=.25, top=.88)
        fig.text(.23, .07, 'Points are observed means; bars are percentile intervals from 20,000 common world resamples.\n'
                 'Intervals are marginal, not simultaneous; fixed states and generator define the scope.', fontsize=10)
        save(fig, name)
    return means, overall, contrasts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview-dir', type=Path, help='Optional PNG previews for local inspection')
    parser.add_argument('--panel', choices=('pilot', 'replication'), default='pilot')
    parser.add_argument('--mean-intervals-csv', type=Path)
    parser.add_argument('--contrast-intervals-csv', type=Path)
    args = parser.parse_args(argv)
    if args.panel == 'replication':
        if args.mean_intervals_csv is None or args.contrast_intervals_csv is None:
            parser.error('replication plots require both interval CSV arguments')
        make_replication_plots(load_metrics(args.metrics_csv, REPLICATION_WORLDS),
            args.mean_intervals_csv, args.contrast_intervals_csv, args.output_dir, args.preview_dir)
    else:
        make_plots(load_metrics(args.metrics_csv), args.output_dir, args.preview_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
