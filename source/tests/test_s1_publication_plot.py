"""Synthetic coverage and aggregation checks for saved-metric plotting."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'plot_s1_results.py'
spec = importlib.util.spec_from_file_location('publication_plot', MODULE_PATH)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def fixture():
    rows = []
    for ai, arm in enumerate(plot.ARMS):
        for wi, world in enumerate(plot.WORLDS):
            for ii, setting in enumerate(plot.SETTINGS):
                rows.append(dict(arm=arm, world=world, initialization=setting,
                    ap=(ai + wi + ii) / 14, ap_undefined_reason='', auroc=.5, auroc_undefined_reason='',
                    bce=.7, candidate_count=9, positives=2, negatives=7,
                    true_positive=1, false_negative=1, true_negative=6, false_positive=1))
    return rows


class SavedMetricPlots(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            return plot.load_metrics(path)

    def test_complete_panel_and_equal_world_mean(self):
        means, overall, contrasts = plot.aggregate(self.load(fixture()))
        self.assertAlmostEqual(means[plot.ARMS[0]]['W4'], 5/14)
        self.assertAlmostEqual(overall[plot.ARMS[0]], 3.5/14)
        self.assertAlmostEqual(contrasts[plot.ARMS[1]]['W1'], 1/14)

    def test_duplicate_block_rejected(self):
        rows = fixture(); rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.load(rows)

    def test_missing_block_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'): self.load(fixture()[:-1])

    def test_unknown_alias_rejected(self):
        rows = fixture(); rows[0]['world'] = 'W5'
        with self.assertRaisesRegex(ValueError, 'coverage'): self.load(rows)

    def test_nonfinite_rejected(self):
        rows = fixture(); rows[0]['ap'] = 'nan'
        with self.assertRaisesRegex(ValueError, 'finite'): self.load(rows)

    def test_confusions_reconcile(self):
        rows = fixture(); rows[0]['true_positive'] = 2
        with self.assertRaisesRegex(ValueError, 'confusion'): self.load(rows)

    def test_undefined_not_dropped(self):
        rows = fixture()
        for r in rows:
            if r['world'] == 'W1':
                r.update(ap='', ap_undefined_reason='no positives', auroc='',
                    auroc_undefined_reason='single class', positives=0, negatives=9,
                    true_positive=0, false_negative=0, true_negative=8, false_positive=1)
        means, overall, contrasts = plot.aggregate(self.load(rows))
        self.assertTrue(all(v is None for v in overall.values()))
        self.assertIsNone(contrasts[plot.ARMS[1]]['W1'])

    def single_class(self, all_positive=False):
        rows=fixture()
        for r in rows:
            if r['world']=='W1':
                r.update(positives=9 if all_positive else 0,negatives=0 if all_positive else 9,
                    true_positive=9 if all_positive else 0,false_negative=0,
                    true_negative=0 if all_positive else 9,false_positive=0,
                    ap=1 if all_positive else '',ap_undefined_reason='' if all_positive else 'no positives',
                    auroc='',auroc_undefined_reason='single class')
        return rows

    def test_zero_positive_defined_ap_rejected(self):
        rows=self.single_class();rows[0].update(ap=0,ap_undefined_reason='')
        with self.assertRaisesRegex(ValueError,'AP support'):self.load(rows)

    def test_single_class_defined_auroc_rejected(self):
        for all_positive in (False,True):
            with self.subTest(all_positive=all_positive):
                rows=self.single_class(all_positive);rows[0].update(auroc=.5,auroc_undefined_reason='')
                with self.assertRaisesRegex(ValueError,'AUROC support'):self.load(rows)

    def test_all_positive_ap_defined_and_auroc_missing(self):
        rows=self.load(self.single_class(True))
        means,overall,contrasts=plot.aggregate(rows)
        self.assertEqual(means[plot.ARMS[0]]['W1'],1)
        self.assertTrue(all(r['auroc'] is None for r in rows if r['world']=='W1'))

    def test_positive_undefined_ap_rejected(self):
        rows=fixture();rows[0].update(ap='',ap_undefined_reason='incorrect')
        with self.assertRaisesRegex(ValueError,'AP support'):self.load(rows)

    def test_two_class_undefined_auroc_rejected(self):
        rows=fixture();rows[0].update(auroc='',auroc_undefined_reason='incorrect')
        with self.assertRaisesRegex(ValueError,'AUROC support'):self.load(rows)

    def test_svg_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot.make_plots(self.load(fixture()), root)
            expected={'average_precision_by_world.svg','descriptive_ap_contrasts.svg'}
            self.assertEqual({p.name for p in root.glob('*.svg')},expected)
            for path in sorted(root.glob('*.svg')):
                text = path.read_text()
                self.assertIn('W1', text)
                self.assertNotIn('<dc:date>', text)
                self.assertNotIn(str(root), text)
            with self.assertRaises(FileExistsError): plot.make_plots(self.load(fixture()), root)


if __name__ == '__main__':
    unittest.main()
