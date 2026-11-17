#!/usr/bin/env python3
"""Post hoc span-rule and cycle-length references (exploratory; not part of the held-out protocol).

For every world, score each candidate by whether the span of its selected transfers
(maximum minus minimum simulator step) equals 29, and separately by cycle length alone,
then compute grouped-tie AP and tie-aware AUROC with exact rational arithmetic.
Also checks the frame composition and that every member spans exactly 29 steps.

  python posthoc_span_rule.py --world-index data/world_index.json --output-csv span_rule.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from fractions import Fraction
from pathlib import Path

STEPS = 120
EXPECTED_FRAME = {2: 36, 3: 14, 4: 14, 5: 7, 6: 7}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def grouped_ap(labels, scores):
    """Grouped-threshold AP: tied scores enter together."""
    members = sum(labels)
    groups = collections.defaultdict(list)
    for y, s in zip(labels, scores):
        groups[s].append(y)
    cumulative_pos = cumulative = 0
    total = Fraction(0)
    for score in sorted(groups, reverse=True):
        block = groups[score]
        added = sum(block)
        cumulative_pos += added
        cumulative += len(block)
        if added:
            total += Fraction(added, members) * Fraction(cumulative_pos, cumulative)
    return total


def tie_auroc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    wins = sum(Fraction(1) if p > n else Fraction(1, 2) if p == n else 0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def analyse(world_dir):
    obs = read_jsonl(world_dir / 'candidate_observations.jsonl')
    labels_by_id = {r['candidate_id']: r['label'] for r in read_jsonl(world_dir / 'candidate_labels.jsonl')}
    ids = [r['candidate_id'] for r in read_jsonl(world_dir / 'candidate_identity.jsonl')]
    if len(ids) != len(obs) or set(ids) != set(labels_by_id):
        raise ValueError(f'{world_dir}: observation, identity and label rows do not align')
    labels = [labels_by_id[i] for i in ids]
    lengths = [r['cycle_length'] for r in obs]
    spans = []
    for r in obs:
        steps = [round(t * STEPS) for t in r['time_steps']]
        spans.append(max(steps) - min(steps))
    frame = collections.Counter(lengths)
    if dict(frame) != EXPECTED_FRAME:
        raise ValueError(f'{world_dir}: unexpected candidate frame {dict(frame)}')
    member_spans = sorted({s for s, y in zip(spans, labels) if y})
    rule = [int(s == 29) for s in spans]
    length_ref = [int(k >= 3) for k in lengths]
    return {
        'candidates': len(labels), 'members': sum(labels),
        'member_spans': ' '.join(map(str, member_spans)),
        'coincidental_matches': sum(1 for r, y in zip(rule, labels) if r and not y),
        'span_rule_ap': grouped_ap(labels, rule), 'span_rule_auroc': tie_auroc(labels, rule),
        'cycle_length_ap': grouped_ap(labels, length_ref), 'cycle_length_auroc': tie_auroc(labels, length_ref),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--world-index', type=Path, required=True,
                        help='JSON {"worlds": [{"world_id", "split", "alias", "path"}]}; paths relative to the index')
    parser.add_argument('--output-csv', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_csv.exists():
        raise FileExistsError(args.output_csv)
    base = args.world_index.resolve().parent.parent if args.world_index.parent.name == 'data' else args.world_index.resolve().parent
    worlds = json.loads(args.world_index.read_text())['worlds']
    rows = []
    for w in worlds:
        result = analyse(base / w['path'])
        rows.append({'split': w['split'], 'alias': w['alias'], 'world_id': w['world_id'],
                     **{k: (f'{float(v):.10f}' if isinstance(v, Fraction) else v) for k, v in result.items()},
                     'span_rule_ap_exact': str(result['span_rule_ap'])})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in dict.fromkeys(r['split'] for r in rows):
        part = [r for r in rows if r['split'] == split]
        ap = sum(float(r['span_rule_ap']) for r in part) / len(part)
        auc = sum(float(r['span_rule_auroc']) for r in part) / len(part)
        print(f'{split}: {len(part)} worlds, span-rule mean AP {ap:.4f}, mean AUROC {auc:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
