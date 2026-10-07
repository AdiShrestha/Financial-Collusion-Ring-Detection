"""Small, data-independent regression tests for result-changing invariants."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx

from source.src.data.candidate_extractor import CandidateExtractor, CandidateSubgraph
from source.src.data.splits import GroupSafeSplitter
from source.src.data.ring_injector import SemiSyntheticRingInjector
from source.src.data.elliptic_loader import EllipticActorsLoader
from source.src.data.amlworld_patterns import parse_amlworld_patterns, parse_transaction_line
from source.src.models.train_harness import compute_classification_metrics, compute_optimal_threshold
from source.src.models.graph_baselines import GCNBaseline
from source.src.models.feature_encoders import StructuralMotifEncoder
from source.src.ph.gudhi_backend import PersistenceDiagram
from source.src.ph.cache_features import PHFeatureCache
from source.src.ph.vectorize import PersistenceLandscapeVectorizer
from source.src.ph.graph_filtration import build_temporal_filtration
from source.src.topology.incidence import build_b1, build_b2_cellular, build_b2_simplicial
from source.src.topology.cell_view import lift_to_cell_complex
from source.scripts.run_elliptic_benchmark import match_elliptic_controls_by_structure


class ResearchInvariantTests(unittest.TestCase):
    def test_multigraph_preserves_transactions_and_extraction_is_label_blind(self):
        graph = nx.MultiDiGraph()
        for u, v, tid, label in [
            ("a", "b", "ab1", 1), ("a", "b", "ab2", 0),
            ("b", "c", "bc", 1), ("c", "a", "ca", 1),
        ]:
            graph.add_edge(u, v, tx_id=tid, amount=2.0, timestamp=1.0,
                           is_laundering=label)
        extractor = CandidateExtractor(min_k=3, max_k=3, max_cycles=10)
        first = extractor.find_simple_cycles(graph)
        for _, _, data in graph.edges(data=True):
            data["is_laundering"] ^= 1
        self.assertEqual(first, extractor.find_simple_cycles(graph))
        candidate = extractor.extract_subgraph(graph, first[0])
        self.assertEqual(len(candidate.edges), 4)
        self.assertEqual({e["tx_id"] for e in candidate.edges}, {"ab1", "ab2", "bc", "ca"})

    def test_elliptic_loader_preserves_parallel_transaction_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text("input_address,output_address\na,b\na,b\nb,c\n", encoding="utf-8")
            graph = EllipticActorsLoader(
                edgelist_path=str(path), features_path=None, classes_path=None
            ).load_actor_graph(max_edges=3)
        self.assertTrue(graph.is_multigraph())
        self.assertEqual(graph.number_of_edges("a", "b"), 2)

    def test_pattern_parser_rejects_invalid_rows_and_mismatched_block_types(self):
        valid = "2020/01/01 00:00,B1,A1,B2,A2,10,USD,10,USD,Wire,1"
        self.assertIsNotNone(parse_transaction_line(valid))
        self.assertIsNone(parse_transaction_line(valid + ",extra"))
        self.assertIsNone(parse_transaction_line(valid.rsplit(",", 1)[0] + ",1.5"))
        self.assertIsNone(parse_transaction_line(valid.replace(",10,USD,10,", ",nan,USD,10,")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.txt"
            path.write_text(
                "BEGIN LAUNDERING ATTEMPT - Cycle\n"
                + valid + "\n"
                + "END LAUNDERING ATTEMPT - Smurfing\n",
                encoding="utf-8",
            )
            _, audit = parse_amlworld_patterns(path)
        self.assertEqual(audit["malformed_lines_count"], 1)

    def test_undirected_cycles_are_canonical_and_cycle_cap_fails_closed(self):
        graph = nx.cycle_graph(["a", "b", "c", "d"])
        extractor = CandidateExtractor(min_k=4, max_k=4, max_cycles=1)
        self.assertEqual(extractor.find_simple_cycles(graph), [["a", "b", "c", "d"]])
        with self.assertRaisesRegex(ValueError, "candidate population is incomplete"):
            CandidateExtractor(min_k=3, max_k=4, max_cycles=1).find_simple_cycles(
                nx.complete_graph(["a", "b", "c", "d"])
            )

    def test_group_split_rejects_unlabelled_and_duplicate_ids(self):
        c = CandidateSubgraph("one", ["a", "b", "c"], [], 3, ["a", "b", "c"])
        splitter = GroupSafeSplitter(n_splits=3)
        with self.assertRaises(ValueError):
            splitter.split([c])
        c.label = 1
        with self.assertRaises(ValueError):
            splitter.split([c, c])

    def test_group_split_keeps_shared_accounts_together(self):
        candidates = [
            CandidateSubgraph("one", ["a", "b"], [], 3, [], 1),
            CandidateSubgraph("two", ["b", "c"], [], 3, [], 0),
            CandidateSubgraph("three", ["x", "y"], [], 3, [], 0),
        ]
        assignment = GroupSafeSplitter(n_splits=3).split(candidates)
        self.assertEqual(assignment["one"], assignment["two"])

    def test_group_split_keeps_matched_controls_together(self):
        candidates = [
            CandidateSubgraph("positive", ["p1", "p2"], [], 3, [], 1,
                              matched_group_id="pair-1"),
            CandidateSubgraph("control", ["n1", "n2"], [], 3, [], 0,
                              matched_group_id="pair-1"),
        ]
        splitter = GroupSafeSplitter(n_splits=3)
        assignment = splitter.split(candidates)
        self.assertEqual(assignment["positive"], assignment["control"])

    def test_elliptic_matching_requires_exact_node_and_edge_support(self):
        positive = CandidateSubgraph("p", ["a", "b"], [
            {"source": "a", "target": "b"}, {"source": "b", "target": "a"},
        ], 2, [], 1)
        unmatched = CandidateSubgraph("n", ["x", "y"], [
            {"source": "x", "target": "y"},
        ], 2, [], 0)
        with self.assertRaisesRegex(ValueError, "exact node-and-edge-count matched cohort"):
            match_elliptic_controls_by_structure([positive, unmatched], 1, control_seed=17)
        control = CandidateSubgraph("n2", ["u", "v"], [
            {"source": "u", "target": "v"}, {"source": "v", "target": "u"},
        ], 2, [], 0)
        matched, provenance = match_elliptic_controls_by_structure([positive, control], 1, control_seed=17)
        self.assertEqual(len(matched), 2)
        self.assertEqual(matched[0].matched_group_id, matched[1].matched_group_id)
        self.assertEqual(provenance["control_selection_seed"], 17)

    def test_undirected_structural_cycle_feature_counts_once(self):
        nodes = ["a", "b", "c", "d"]
        edges = [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "d"},
            {"source": "d", "target": "a"},
        ]
        features = StructuralMotifEncoder().encode(nodes, edges, {n: i for i, n in enumerate(nodes)})
        np.testing.assert_array_equal(features[:, 2], np.ones(4, dtype=np.float32))

    def test_reciprocal_cell_boundary_and_simplicial_missing_edge(self):
        edges = [("a", "b"), ("b", "a"), ("b", "c"), ("c", "a")]
        b1 = build_b1(["a", "b", "c"], edges)
        b2 = build_b2_cellular(edges, [["a", "b", "c"]])
        self.assertEqual(b2.toarray()[:, 0].tolist(), [1.0, 0.0, 1.0, 1.0])
        self.assertEqual((b1 @ b2).nnz, 0)
        with self.assertRaises(ValueError):
            build_b2_simplicial([("a", "b"), ("b", "c")], [("a", "b", "c")])

    def test_cell_lift_does_not_invent_boundary_edges(self):
        candidate = CandidateSubgraph("x", ["a", "b", "c"], [
            {"source": "a", "target": "b", "amount": 1.0, "timestamp": 0.0},
            {"source": "b", "target": "c", "amount": 1.0, "timestamp": 1.0},
        ], 3, ["a", "b", "c"], 0)
        with self.assertRaises(ValueError):
            lift_to_cell_complex(candidate)

    def test_cell_features_align_with_collapsed_parallel_edges(self):
        candidate = CandidateSubgraph("x", ["a", "b", "c"], [
            {"source": "a", "target": "b", "amount": 100.0, "timestamp": 2.0},
            {"source": "a", "target": "b", "amount": 200.0, "timestamp": 1.0},
            {"source": "b", "target": "c", "amount": 1.0, "timestamp": 3.0},
            {"source": "c", "target": "a", "amount": 1.0, "timestamp": 4.0},
        ], 3, ["a", "b", "c"], 1)
        cell = lift_to_cell_complex(candidate)
        self.assertEqual(cell.B1.shape[1], cell.x_1.shape[0])
        self.assertAlmostEqual(cell.x_1[cell.edges.index(("a", "b")), 0], np.log1p(2))

    def test_metrics_use_fixed_or_validation_threshold_and_do_not_fake_auc(self):
        y = np.array([0, 1])
        p = np.array([0.2, 0.4])
        fixed = compute_classification_metrics(y, p)
        self.assertEqual(fixed["best_threshold"], 0.5)
        self.assertEqual(fixed["f1"], 0.0)
        threshold = compute_optimal_threshold(y, p)
        self.assertEqual(threshold, 0.4)
        self.assertEqual(compute_classification_metrics(y, p, threshold)["f1"], 1.0)
        self.assertIsNone(compute_classification_metrics([1, 1], [0.8, 0.9])["auroc"])

    def test_ph_landscape_uses_common_normalized_coordinate(self):
        vectorizer = PersistenceLandscapeVectorizer()
        a = PersistenceDiagram(h0_intervals=[(0.0, 2.0)],
                               h1_intervals=[(2.0, float("inf"))], max_filtration_time=4.0)
        b = PersistenceDiagram(h0_intervals=[(0.0, 4.0)],
                               h1_intervals=[(4.0, float("inf"))], max_filtration_time=8.0)
        np.testing.assert_allclose(vectorizer.vectorize(a), vectorizer.vectorize(b))
        with self.assertRaises(ValueError):
            PersistenceLandscapeVectorizer(resolution=0)

    def test_temporal_filtration_rejects_missing_times_and_isolated_nodes(self):
        missing_time = CandidateSubgraph("missing-time", ["a", "b"], [
            {"source": "a", "target": "b"},
        ], 0, [], 0)
        with self.assertRaisesRegex(ValueError, "lacks source, target, or timestamp"):
            build_temporal_filtration(missing_time)
        isolated = CandidateSubgraph("isolated", ["a", "b", "c"], [
            {"source": "a", "target": "b", "timestamp": 1.0},
        ], 0, [], 0)
        with self.assertRaisesRegex(ValueError, "has no observed transaction time"):
            build_temporal_filtration(isolated)

    def test_ph_cache_binds_fold_hash_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = PHFeatureCache(tmp)
            cache.extract_vector = lambda candidate: np.zeros(cache.vectorizer.feature_dim, dtype=np.float32)
            candidate = CandidateSubgraph("cache-candidate", ["a"], [], 0, [], 0)
            metadata = cache.cache_fold([candidate], fold_id=0, source_fold_sha256="source-hash")
            manifest = cache.write_manifest(source_dataset_manifest_sha256="dataset-hash")
            self.assertTrue(manifest.exists())
            self.assertEqual(cache.load_fold(0)["candidate_ids"], ["cache-candidate"])
            (Path(tmp) / metadata["filename"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "failed manifest integrity"):
                cache.load_fold(0)

    def test_ring_flow_verifier_checks_hop_continuity(self):
        injector = SemiSyntheticRingInjector(seed=42)
        _, rings = injector.inject_rings(nx.DiGraph(), n_rings=1)
        ring = rings[0]
        self.assertTrue(injector.verify_flow_conservation(ring))
        ring.edges[1]["amount"] += 10.0
        ring.edges[1]["amount_next"] = round(ring.edges[1]["amount"] * (1 - ring.fee_rate), 6)
        self.assertFalse(injector.verify_flow_conservation(ring))

    def test_graph_baseline_requires_and_consumes_edge_features(self):
        model = GCNBaseline(in_dim=3, hidden_dim=8, ph_dim=None, dropout=0.0)
        x = np.zeros((3, 3), dtype=np.float32)
        import torch
        x = torch.tensor(x)
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        attr = torch.tensor([[1.0, 0.0], [2.0, 0.5], [3.0, 1.0]])
        self.assertEqual(tuple(model(x, edges, edge_attr=attr).shape), (1,))
        with self.assertRaises(ValueError):
            model(x, edges)

    def test_graph_baseline_edge_features_are_pooled_per_graph(self):
        import torch
        model = GCNBaseline(in_dim=3, hidden_dim=8, ph_dim=None, dropout=0.0)
        x = torch.zeros((4, 3))
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        graph_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        edge_attr = torch.tensor([[1.0, 0.0], [3.0, 2.0], [10.0, 4.0], [14.0, 8.0]])
        model.eval()
        logits = model(x, edge_index, batch=graph_batch, edge_attr=edge_attr)
        self.assertEqual(tuple(logits.shape), (2,))
        first = model(x[:2], torch.tensor([[0, 1], [1, 0]]), edge_attr=edge_attr[:2])[0]
        second = model(x[2:], torch.tensor([[0, 1], [1, 0]]), edge_attr=edge_attr[2:])[0]
        torch.testing.assert_close(logits, torch.stack([first, second]))
        with self.assertRaisesRegex(ValueError, "align one-to-one"):
            model(x, edge_index, batch=graph_batch, edge_attr=edge_attr[:3])


if __name__ == "__main__":
    unittest.main()
