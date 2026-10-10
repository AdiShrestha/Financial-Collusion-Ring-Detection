"""Data-independent checks for the neutral S1 heldout inference adapter."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import evaluate_s1_heldout as adapter  # noqa: E402


FEATURES = [f"feature_{index}" for index in range(20)]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_fixture(root: Path, candidate_count: int = 2, world_id: str = "S1_test_fixture"):
    attempt = root / "worlds" / world_id / "attempts" / "run_01"
    attempt.mkdir(parents=True)
    write_jsonl(attempt / "candidate_observations.jsonl", [
        {
            "cycle_length": 2,
            "world_event_time_steps": [0.1, 0.3],
            "world_event_log1p_amounts": [1.0, 2.0],
            "central_node_mask": [1, 1],
            "selected_event_mask": [1, 1],
        }
        for _ in range(candidate_count)
    ])
    identities = [
        {
            "candidate_id": f"cand{index:07d}",
            "world_id": world_id,
            "event_ids": [1, 2],
            "cycle_accounts": ["A0", "A1"],
        }
        for index in range(candidate_count)
    ]
    write_jsonl(attempt / "candidate_identity.jsonl", identities)
    write_jsonl(attempt / "cellular_polygonal_boundaries.jsonl", [
        {"candidate_id": row["candidate_id"], "dimension": 2, "boundary": [{"event_id": 1}, {"event_id": 2}]}
        for row in identities
    ])
    write_jsonl(attempt / "simplicial_subdivision.jsonl", [
        {"candidate_id": row["candidate_id"], "triangles": []} for row in identities
    ])
    write_jsonl(attempt / "hasse_incidence.jsonl", [
        {"candidate_id": row["candidate_id"], "incidence": []} for row in identities
    ])
    write_jsonl(attempt / "world_directed_multigraph_identity.jsonl", [
        {"physical_event_id": 1, "source_account": "A0", "target_account": "A1"},
        {"physical_event_id": 2, "source_account": "A1", "target_account": "A0"},
    ])
    write_json(attempt / "world_node_identity.json", {
        "world_id": world_id,
        "node_order": ["A0", "A1"],
        "event_order": [1, 2],
    })
    # A trap target exists so an accidental read is observable.
    write_jsonl(attempt / "candidate_labels.jsonl", [
        {"candidate_id": row["candidate_id"], "label": index % 2}
        for index, row in enumerate(identities)
    ])
    index_path = root / "world_index.json"
    if not index_path.exists():
        write_json(index_path, {"schema": "fixture"})
    entry = {
        "world_id": world_id,
        "seed": 101,
        "role": "heldout_test",
        "status": "SOURCE_RECONCILED",
        "attempt_path": str(attempt),
    }
    return index_path, entry, attempt


def load_test_benchmark():
    path = SCRIPTS / "check_s1_benchmark.py"
    binding = adapter.file_binding(path)
    lock = {
        "source_reference": [{
            "path": str(path),
            "bytes": binding["bytes"],
            "sha256": binding["sha256"],
            "copy": dict(binding),
        }]
    }
    return adapter.load_benchmark(lock)[0]


class SimpleInference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_grad_modes = []

    def forward(self, x):
        self.seen_grad_modes.append(torch.is_grad_enabled())
        return x.sum().reshape(1)


class StagedScores:
    def __init__(self):
        self.yielded = 0

    def staged_decision_function(self, features):
        for step in range(1, 801):
            self.yielded += 1
            yield np.full((len(features),), step, dtype=np.float64)


class HeldoutAdapterTests(unittest.TestCase):
    def test_pinned_helper_is_passed_explicitly_and_fixture_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, entry, _ = make_fixture(root)
            benchmark = load_test_benchmark()
            world = adapter._join_and_build_world(entry, index_path, benchmark)
            self.assertEqual(world["candidate_count"], 2)
            self.assertEqual(len(world["views"]), 2)
            self.assertTrue(np.array_equal(
                world["views"][0]["cell"][0].astype(np.int64)
                @ world["views"][0]["cell"][1].astype(np.int64),
                np.zeros((2, 1), dtype=np.int64),
            ))
            self.assertEqual(len(world["data"]["event_order"]), 2)
            self.assertNotIn("labels_path", world["data"])
            self.assertTrue(all(path.name in adapter.INPUT_ALLOWLIST for path in world["data"]["inputs"]))

    def test_label_open_and_fit_traps_and_variable_candidate_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, entry, _ = make_fixture(root, candidate_count=3)
            benchmark = load_test_benchmark()
            original_open = Path.open
            original_read_bytes = Path.read_bytes

            def guarded_open(path, *args, **kwargs):
                if path.name == "candidate_labels.jsonl":
                    raise AssertionError("prediction adapter attempted a label open")
                return original_open(path, *args, **kwargs)

            def guarded_read_bytes(path, *args, **kwargs):
                if path.name == "candidate_labels.jsonl":
                    raise AssertionError("prediction adapter attempted a label read")
                return original_read_bytes(path, *args, **kwargs)

            with (
                patch.object(Path, "open", guarded_open),
                patch.object(Path, "read_bytes", guarded_read_bytes),
                patch("torch.optim.Adam", side_effect=AssertionError("optimizer construction forbidden")),
                patch("sklearn.ensemble.HistGradientBoostingClassifier.fit", side_effect=AssertionError("fit forbidden")),
            ):
                world = adapter._join_and_build_world(entry, index_path, benchmark)
                transform = {"mean_float64": [0.0] * 20, "scale_float64": [1.0] * 20}
                raw, normalized = adapter.summary_feature_matrices(world, benchmark, transform)
                model = SimpleInference()
                logits = adapter.predict_neural_logits(model, adapter.SUMMARY_ARM, world, benchmark, normalized)
            self.assertEqual(world["candidate_count"], 3)
            self.assertEqual(raw.shape, (3, 20))
            self.assertEqual(normalized.shape, (3, 20))
            self.assertEqual(normalized.dtype, np.float32)
            self.assertEqual(logits.shape, (3,))
            self.assertEqual(model.seen_grad_modes, [False, False, False])

    def test_locked_selected_state_and_frozen_float64_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = load_test_benchmark()
            original = adapter._new_neural_model({"arm": adapter.SUMMARY_ARM}, benchmark)
            with torch.no_grad():
                for index, parameter in enumerate(original.parameters()):
                    parameter.fill_(index + 0.25)
            checkpoint = root / "selected.pt"
            torch.save({
                "epoch": 375,
                "arm": adapter.SUMMARY_ARM,
                "seed": 11,
                "learning_rate": 0.001,
                "model": original.state_dict(),
            }, checkpoint)
            binding = adapter.file_binding(checkpoint)
            record = {
                "model_id": f"{adapter.SUMMARY_ARM}/seed_11",
                "arm": adapter.SUMMARY_ARM,
                "seed": 11,
                "selected_epoch": 375,
                "frozen": {"path": str(checkpoint), "bytes": binding["bytes"], "sha256": binding["sha256"]},
            }
            loaded, _ = adapter.load_neural_model_record(record, benchmark)
            for name, expected in original.state_dict().items():
                self.assertTrue(torch.equal(loaded["model"].state_dict()[name], expected))
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded["model"].parameters()))
            self.assertFalse(loaded["model"].training)

            transform_path = root / "fitted_transform.json"
            transform_value = {
                "columns": FEATURES,
                "fit_candidate_count": 312,
                "fit_worlds": ["training_a", "training_b", "training_c", "training_d"],
                "normalization_dtype": "float64",
                "model_tensor_dtype": "float32",
                "constant_columns": [FEATURES[3]],
                "mean_float64": [float(index) for index in range(20)],
                "scale_float64": [1.0] * 20,
            }
            write_json(transform_path, transform_value)
            transform_binding = adapter.file_binding(transform_path)
            lock = {"summary_transform": {"refit_authorized": False, "fit_worlds": transform_value["fit_worlds"], "frozen": {
                "path": str(transform_path),
                "bytes": transform_binding["bytes"],
                "sha256": transform_binding["sha256"],
            }}}
            study = {"features": {"feature_order": FEATURES}, "training_worlds_unchanged": transform_value["fit_worlds"]}
            loaded_transform, _ = adapter.load_summary_transform(lock, study)
            self.assertEqual(loaded_transform["scale_float64"][3], 1.0)
            self.assertEqual(loaded_transform["mean_float64"], transform_value["mean_float64"])
            lock["summary_transform"]["refit_authorized"] = True
            with self.assertRaisesRegex(ValueError, "prohibit transform refitting"):
                adapter.load_summary_transform(lock, study)

    def test_gbdt_is_one_based_stage_70_and_stops_there(self):
        model = StagedScores()
        scores = adapter.selected_gbdt_scores(model, np.zeros((4, 20), dtype=np.float64), 70)
        self.assertTrue(np.array_equal(scores, np.full(4, 70.0)))
        self.assertEqual(model.yielded, 70)

    def test_mask_coordinate_join_duplicate_and_symlink_errors_are_rejected(self):
        benchmark = load_test_benchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, entry, attempt = make_fixture(root)
            observations_path = attempt / "candidate_observations.jsonl"
            rows = adapter.strict_jsonl_bytes(observations_path.read_bytes(), str(observations_path))
            rows[0]["selected_event_mask"] = [1, 0]
            write_jsonl(observations_path, rows)
            with self.assertRaisesRegex(ValueError, "mask|identity"):
                adapter._join_and_build_world(entry, index_path, benchmark)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, entry, attempt = make_fixture(root)
            node_path = attempt / "world_node_identity.json"
            node = json.loads(node_path.read_text())
            node["event_order"] = [1, 3]
            write_json(node_path, node)
            with self.assertRaisesRegex(ValueError, "event coordinate|coordinate"):
                adapter._join_and_build_world(entry, index_path, benchmark)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, entry, attempt = make_fixture(root)
            sidecar_path = attempt / "hasse_incidence.jsonl"
            rows = adapter.strict_jsonl_bytes(sidecar_path.read_bytes(), str(sidecar_path))
            rows[1]["candidate_id"] = rows[0]["candidate_id"]
            write_jsonl(sidecar_path, rows)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                adapter._join_and_build_world(entry, index_path, benchmark)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, entry, attempt = make_fixture(root)
            outside = root / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            sidecar_path = attempt / "hasse_incidence.jsonl"
            sidecar_path.unlink()
            sidecar_path.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink|escapes"):
                adapter._join_and_build_world(entry, index_path, benchmark)

    def test_source_has_no_training_or_optimizer_imports(self):
        source = (SCRIPTS / "evaluate_s1_heldout.py").read_text(encoding="utf-8")
        self.assertNotIn("train_s1_development", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("HistGradientBoostingClassifier.fit", source)
        self.assertNotIn('EXPERIMENT_ID =', source)


if __name__ == "__main__":
    unittest.main()
