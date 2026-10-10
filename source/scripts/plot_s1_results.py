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


def load_metrics(path):
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
    expected = set(itertools.product(ARMS, WORLDS, SETTINGS))
    if keys != expected:
        raise ValueError('complete six-arm/four-world/five-setting coverage required')
    for world in WORLDS:
        counts = {(r['candidate_count'], r['positives'], r['negatives']) for r in rows if r['world'] == world}
        if len(counts) != 1:
            raise ValueError('world class counts differ across model settings')
    return rows


def aggregate(rows):
    """Average metrics across settings, then give each registered world equal weight."""
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for world in WORLDS:
            values = [r['ap'] for r in rows if r['arm'] == arm and r['world'] == world]
            if len(values) != len(SETTINGS):
                raise ValueError('missing settings in aggregation')
            means[arm][world] = None if any(x is None for x in values) else math.fsum(values) / len(values)
    overall = {arm: None if any(v is None for v in d.values()) else math.fsum(d.values()) / len(WORLDS)
               for arm, d in means.items()}
    contrasts = {arm: {w: None if means[arm][w] is None or means[ARMS[0]][w] is None
        else means[arm][w] - means[ARMS[0]][w] for w in WORLDS} for arm in ARMS[1:]}
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview-dir', type=Path, help='Optional PNG previews for local inspection')
    args = parser.parse_args(argv)
    make_plots(load_metrics(args.metrics_csv), args.output_dir, args.preview_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
