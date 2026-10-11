"""Synthetic checks for separate-panel table arithmetic and vector plots."""
import csv
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


stats = module('reproduce_s1_statistics')
plot = module('plot_s1_results')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def world_rows():
    return [dict(phase='replication', arm=a, world=w, mean_ap=(ai + wi) / 40,
                 initialization_count=5, undefined_reason='')
            for ai, a in enumerate(stats.ARMS) for wi, w in enumerate(stats.WORLDS)]


def metric_rows():
    return [dict(phase='replication', arm=a, world=w, initialization=i,
                 ap=(ai + wi + ii) / 40, ap_undefined_reason='', auroc=.6,
                 auroc_undefined_reason='', bce=.7, candidate_count=10,
                 positives=2, negatives=8, true_positive=1, false_negative=1,
                 true_negative=6, false_positive=2)
            for ai, a in enumerate(stats.ARMS) for wi, w in enumerate(stats.WORLDS)
            for ii, i in enumerate(plot.SETTINGS)]


class ReplicationTables(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'worlds.csv'
            write_csv(path, rows)
            return stats.load_world_means(path)

    def test_complete_separate_coverage(self):
        values = self.load(world_rows())
        self.assertEqual(len(values), 180)
        self.assertEqual(values[stats.ARMS[0], 'W34'], 29 / 40)

    def test_pilot_pooling_rejected(self):
        rows = world_rows()
        rows[0]['world'] = 'W1'
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.load(rows)

    def test_duplicate_and_missing_rejected(self):
        for rows, message in ((world_rows() + [world_rows()[0]], 'duplicate'), (world_rows()[:-1], 'coverage')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load(rows)

    def test_nonfinite_and_phase_rejected(self):
        for column, value in (('mean_ap', 'nan'), ('phase', 'heldout')):
            rows = world_rows()
            rows[0][column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.load(rows)

    def test_undefined_blocks_primary_instead_of_dropping(self):
        rows = world_rows()
        rows[0].update(mean_ap='', undefined_reason='no positives')
        result = stats.reproduce(self.load(rows))
        self.assertEqual(result['status'], 'UNAVAILABLE_REQUIRED_AP')
        self.assertFalse(result['pilot_pooled'])

    def test_exact_sign_distribution_matches_enumeration_with_ties_and_zero(self):
        result = stats.exact_signed_rank([1, -1, 2, 0])
        ranks = [3, 3, 6]
        sums = [sum(r for r, use in zip(ranks, signs) if use)
                for signs in itertools.product((0, 1), repeat=3)]
        observed = 9
        le = sum(s <= observed for s in sums)
        ge = sum(s >= observed for s in sums)
        self.assertEqual(result['tail_le_count'], le)
        self.assertEqual(result['tail_ge_count'], ge)
        self.assertEqual(result['p_value'], min(1, 2 * min(le, ge) / 8))
        self.assertEqual(result['rank_twice_by_original_index'], [3, 3, 6, 0])
        self.assertEqual(result['tied_observation_count'], 2)
        self.assertEqual(result['zero_count'], 1)

    def test_rounding_only_for_ranks_and_all_zero(self):
        result = stats.exact_signed_rank([1e-14, -1e-14])
        self.assertEqual(result['denominator'], 1)
        self.assertEqual(result['p_value'], 1)
        self.assertEqual(result['zero_count'], 2)
        result = stats.exact_signed_rank([1e-14, 1])
        self.assertEqual(result['mean_signed_difference'], (1 + 1e-14) / 2)

    def test_holm_full_family_and_nonzero_p(self):
        self.assertEqual(stats.holm({'a': .01, 'b': .04, 'c': .03}), {'a': .03, 'c': .06, 'b': .06})
        result = stats.exact_signed_rank([-1] * 30)
        self.assertEqual(result['p_value'], 2 / (2 ** 30))
        self.assertGreater(result['p_value'], 0)

    def test_metric_coverage_and_five_setting_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            write_csv(path, metric_rows())
            rows = plot.load_metrics(path, plot.REPLICATION_WORLDS)
            means, overall, contrasts = plot.aggregate(rows, plot.REPLICATION_WORLDS)
            self.assertAlmostEqual(means[stats.ARMS[0]]['W5'], 2 / 40)
            self.assertAlmostEqual(overall[stats.ARMS[0]], 16.5 / 40)
            self.assertAlmostEqual(contrasts[stats.ARMS[1]]['W34'], 1 / 40)

    def test_paired_world_bootstrap_keeps_constant_arm_shift(self):
        values = {(a, w): .2 + wi / 100 + ai / 20
                  for ai, a in enumerate(stats.ARMS) for wi, w in enumerate(stats.WORLDS)}
        result = stats.reproduce(values)
        bootstrap = result['paired_world_bootstrap']
        self.assertEqual(bootstrap['seed'], stats.BOOTSTRAP_SEED)
        self.assertEqual(bootstrap['resamples'], 20000)
        self.assertEqual(len(bootstrap['intervals']), 11)
        for ai, arm in enumerate(stats.ARMS[1:], 1):
            interval = bootstrap['intervals'][arm + '_minus_gbdt_fixed20']
            self.assertAlmostEqual(interval['lower_95'], ai / 20)
            self.assertAlmostEqual(interval['upper_95'], ai / 20)

    def test_vector_plot_coverage_intervals_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / 'metrics.csv'
            write_csv(metrics, metric_rows())
            rows = plot.load_metrics(metrics, plot.REPLICATION_WORLDS)
            means, overall, contrasts = plot.aggregate(rows, plot.REPLICATION_WORLDS)
            arm_rows = [dict(phase='replication', arm=a, mean_ap=v, lower_95=v-.05, upper_95=v+.05,
                            interval_width=.1, interval_scope='marginal conditional percentile interval')
                        for a, v in overall.items()]
            contrast_rows = [dict(phase='replication', arm=a, mean_ap_difference=1/40*ai,
                                 lower_95=1/40*ai-.05, upper_95=1/40*ai+.05,
                                 interval_width=.1, interval_scope='marginal conditional percentile interval')
                             for ai, a in enumerate(stats.ARMS[1:], 1)]
            mean_path, contrast_path = root/'mean.csv', root/'contrast.csv'
            write_csv(mean_path, arm_rows)
            write_csv(contrast_path, contrast_rows)
            out = root / 'figures'
            plot.make_replication_plots(rows, mean_path, contrast_path, out)
            svgs = list(out.glob('*.svg'))
            self.assertEqual(len(svgs), 3)
            all_worlds = (out / 'replication_ap_by_world.svg').read_text()
            for world in stats.WORLDS:
                self.assertIn('>' + world + '<', all_worlds)
            for path in svgs:
                text = path.read_text()
                self.assertNotIn(str(root), text)
                self.assertNotIn('<dc:date>', text)
            self.assertIn('marginal 95%', (out / 'replication_mean_ap_intervals.svg').read_text())
            with self.assertRaises(FileExistsError):
                plot.make_replication_plots(rows, mean_path, contrast_path, out)


if __name__ == '__main__':
    unittest.main()
