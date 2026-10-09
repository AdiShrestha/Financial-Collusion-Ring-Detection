"""Data-independent mathematical and capacity checks for the S1 benchmark."""

import importlib.util
import json
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "scripts" / "check_s1_benchmark.py"
SPEC = importlib.util.spec_from_file_location("s1_benchmark_contract", SCRIPT)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


class S1BenchmarkContractTests(unittest.TestCase):
    def test_integer_boundary_accepts_cycle_and_rejects_open_chain(self):
        import numpy as np

        b1 = np.asarray([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=np.int64)
        self.assertTrue(BENCH.closed(b1, np.ones((3, 1), dtype=np.int64)))
        self.assertFalse(BENCH.closed(b1, np.asarray([[1], [1], [0]], dtype=np.int64)))

    def test_tied_scores_receive_grouped_ap_and_half_credit_auc(self):
        y = [1, 0, 1, 0]
        scores = [1.0, 1.0, 0.0, 0.0]
        self.assertEqual(BENCH.grouped_ap(y, scores), 0.5)
        self.assertEqual(BENCH.tie_auc(y, scores), 0.5)

    def test_registered_hidden_widths_have_the_observed_matched_counts(self):
        config_path = SOURCE / "configs" / "s1_matched_smoke.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        widths = config["hidden_widths"]
        models = {
            "cellular_cwn_style_local": BENCH.Cell(widths["cellular_cwn_style_local"]),
            "simplicial_mpsn_style_local": BENCH.Simp(widths["simplicial_mpsn_style_local"]),
            "directed_local_edge_gnn": BENCH.Directed(widths["directed_local_edge_gnn"]),
            "cellular_hasse_mechanism_control": BENCH.Cell(
                widths["cellular_hasse_mechanism_control"], hasse=True
            ),
        }
        observed = {
            name: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            for name, model in models.items()
        }
        self.assertEqual(observed, {
            "cellular_cwn_style_local": 8467,
            "simplicial_mpsn_style_local": 8533,
            "directed_local_edge_gnn": 8390,
            "cellular_hasse_mechanism_control": 8431,
        })
        self.assertEqual(BENCH.DEFAULT_HIDDEN, 24)


if __name__ == "__main__":
    unittest.main()
