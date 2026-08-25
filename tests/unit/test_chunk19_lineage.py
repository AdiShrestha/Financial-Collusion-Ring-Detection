"""Regression tests for Chunk 19 canonical evidence lineage."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from source.experiments.oof_evaluator import export_oof_prediction_ledger
from source.training.cross_val_runner import _validate_manifest_candidate_lineage


def test_manifest_candidate_known_answer_rejects_missing_id():
    manifest = {
        "total_groups": 1,
        "candidate_assignments": {
            "a": {"group_id": "group_01"},
            "b": {"group_id": "group_01"},
        },
        "outer_folds": [
            {
                "outer_fold_id": 0,
                "train_candidate_ids": ["a"],
                "test_candidate_ids": ["b"],
            },
            {
                "outer_fold_id": 1,
                "train_candidate_ids": ["b"],
                "test_candidate_ids": ["a"],
            },
        ],
    }
    with pytest.raises(ValueError, match="lineage mismatch"):
        _validate_manifest_candidate_lineage(manifest, {"a": {}})


def test_canonical_oof_ledger_exact_lineage():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "artifacts/splits/fold_manifest.json").read_text())
    candidates = pq.read_table(root / "artifacts/candidates/candidates.parquet").to_pylist()
    ledger = pq.read_table(root / "artifacts/predictions/oof_predictions.parquet").to_pandas()

    candidate_by_id = {c["candidate_id"]: c for c in candidates}
    assignments = manifest["candidate_assignments"]
    fold_by_id = {
        cid: fold["outer_fold_id"]
        for fold in manifest["outer_folds"]
        for cid in fold["test_candidate_ids"]
    }

    assert set(candidate_by_id) == set(assignments) == set(ledger.candidate_id)
    assert set(ledger.seed) == {42, 43, 44, 45, 46}
    assert ledger.model_name.nunique() == 8
    assert ledger.group_id.nunique() == manifest["total_groups"] == 18
    assert len(ledger) == 155 * 8 * 5
    assert not ledger.duplicated(["candidate_id", "model_name", "seed"]).any()
    assert all(
        row.group_id == assignments[row.candidate_id]["group_id"]
        and int(row.fold_id) == int(fold_by_id[row.candidate_id])
        and int(row.y_true) == int(candidate_by_id[row.candidate_id]["label"])
        for row in ledger.itertuples()
    )


def test_export_rejects_stale_history(tmp_path):
    root = Path(__file__).resolve().parents[2]
    history = json.loads((root / "artifacts/training/training_history.json").read_text())
    model = history["models"][0]
    seed = f"seed_{history['seeds'][0]}"
    fold = next(iter(history["results"][model][seed]))
    history["results"][model][seed][fold]["test_candidate_ids"][0] = "stale_candidate"
    stale = tmp_path / "history.json"
    stale.write_text(json.dumps(history))
    with pytest.raises(ValueError, match="unknown candidate"):
        export_oof_prediction_ledger(
            training_history_path=str(stale),
            candidates_parquet_path=str(root / "artifacts/candidates/candidates.parquet"),
            fold_manifest_path=str(root / "artifacts/splits/fold_manifest.json"),
            output_parquet_path=str(tmp_path / "oof.parquet"),
        )

