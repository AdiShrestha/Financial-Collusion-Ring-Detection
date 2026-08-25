"""Long-form Out-of-Fold (OOF) Prediction Ledger Exporter.

Contract C18-01 (T-COMP): Compiles out-of-fold predictions across all candidates,
models, and seeds into artifacts/predictions/oof_predictions.parquet.
"""

import json
import hashlib
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def export_oof_prediction_ledger(
    training_history_path: str = "artifacts/training/training_history.json",
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    output_parquet_path: str = "artifacts/predictions/oof_predictions.parquet",
) -> Dict[str, Any]:
    """Export long-form OOF predictions table."""
    if not os.path.exists(training_history_path):
        raise FileNotFoundError(f"Missing training history: {training_history_path}")
    if not os.path.exists(candidates_parquet_path):
        raise FileNotFoundError(f"Missing candidates: {candidates_parquet_path}")
    if not os.path.exists(fold_manifest_path):
        raise FileNotFoundError(f"Missing fold manifest: {fold_manifest_path}")

    with open(training_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    c_table = pq.read_table(candidates_parquet_path)
    candidates = c_table.to_pylist()
    assignments = manifest.get("candidate_assignments", {})
    candidate_ids = {c["candidate_id"] for c in candidates}
    if candidate_ids != set(assignments):
        raise ValueError("Candidate parquet and manifest assignments are not identical")
    cand_info = {
        c["candidate_id"]: {
            "label": int(c["label"]),
            "cycle_length": int(c["cycle_length"]),
            "group_id": str(assignments[c["candidate_id"]]["group_id"]),
        }
        for c in candidates
    }

    # Map candidate to test fold id and group id
    cand_to_fold = {}
    for of in manifest["outer_folds"]:
        f_id = of["outer_fold_id"]
        for cid in of["test_candidate_ids"]:
            cand_to_fold[cid] = f_id

    records = []
    results = history.get("results", {})
    expected_models = list(history.get("models", []))
    expected_seeds = [int(seed) for seed in history.get("seeds", [])]
    if set(results) != set(expected_models):
        raise ValueError("Training history model keys are incomplete")
    if len(expected_seeds) != 5 or len(set(expected_seeds)) != 5:
        raise ValueError("Canonical OOF export requires five distinct seeds")

    for model_name in expected_models:
        seed_dict = results[model_name]
        actual_seeds = {int(key.replace("seed_", "")) for key in seed_dict}
        if actual_seeds != set(expected_seeds):
            raise ValueError(f"Model {model_name} has incomplete seed coverage")
        for seed_str, fold_dict in seed_dict.items():
            seed = int(seed_str.replace("seed_", ""))
            for fold_str, fold_res in fold_dict.items():
                f_id = int(fold_str.replace("fold_", ""))
                test_cids = fold_res["test_candidate_ids"]
                y_trues = fold_res["y_true"]
                y_probs = fold_res["y_prob"]
                y_preds = fold_res.get("y_pred", [1 if p >= 0.5 else 0 for p in y_probs])
                if not (len(test_cids) == len(y_trues) == len(y_probs) == len(y_preds)):
                    raise ValueError(f"Length mismatch for {model_name}/{seed_str}/{fold_str}")

                for cid, yt, yp, ypred in zip(test_cids, y_trues, y_probs, y_preds):
                    if cid not in cand_info:
                        raise ValueError(f"History contains unknown candidate {cid}")
                    c_meta = cand_info[cid]
                    expected_fold = cand_to_fold.get(cid)
                    if expected_fold != f_id:
                        raise ValueError(
                            f"Candidate {cid} recorded in fold {f_id}, expected {expected_fold}"
                        )
                    if int(yt) != c_meta["label"]:
                        raise ValueError(f"Label mismatch for candidate {cid}")
                    records.append({
                        "candidate_id": cid,
                        "model_name": model_name,
                        "seed": seed,
                        "fold_id": f_id,
                        "group_id": c_meta["group_id"],
                        "cycle_length": int(c_meta["cycle_length"]),
                        "y_true": int(yt),
                        "y_pred_prob": float(yp),
                        "y_pred_binary": int(ypred),
                    })

    schema = pa.schema([
        ("candidate_id", pa.string()),
        ("model_name", pa.string()),
        ("seed", pa.int64()),
        ("fold_id", pa.int64()),
        ("group_id", pa.string()),
        ("cycle_length", pa.int64()),
        ("y_true", pa.int64()),
        ("y_pred_prob", pa.float64()),
        ("y_pred_binary", pa.int64()),
    ])

    expected_keys = len(candidate_ids) * len(expected_models) * len(expected_seeds)
    keys = {(r["candidate_id"], r["model_name"], r["seed"]) for r in records}
    if len(records) != expected_keys or len(keys) != expected_keys:
        raise ValueError(
            f"OOF ledger coverage failure: rows={len(records)}, unique_keys={len(keys)}, "
            f"expected={expected_keys}"
        )
    for model_name in expected_models:
        for seed in expected_seeds:
            covered = {
                r["candidate_id"]
                for r in records
                if r["model_name"] == model_name and r["seed"] == seed
            }
            if covered != candidate_ids:
                raise ValueError(f"Incomplete OOF coverage for {model_name}/seed_{seed}")

    def digest(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    metadata = {
        b"manifest_sha256": digest(fold_manifest_path).encode(),
        b"candidates_sha256": digest(candidates_parquet_path).encode(),
        b"training_history_sha256": digest(training_history_path).encode(),
        b"candidate_count": str(len(candidate_ids)).encode(),
        b"group_count": str(manifest["total_groups"]).encode(),
        b"seed_count": str(len(expected_seeds)).encode(),
    }
    table = pa.Table.from_pylist(records, schema=schema).replace_schema_metadata(metadata)
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    pq.write_table(table, output_parquet_path, compression="snappy")

    return {
        "status": "OOF_LEDGER_EXPORTED",
        "output_path": output_parquet_path,
        "total_records": len(records),
        "models": list(results.keys()),
        "seeds": expected_seeds,
        "candidate_count": len(candidate_ids),
        "group_count": int(manifest["total_groups"]),
    }


if __name__ == "__main__":
    res = export_oof_prediction_ledger()
    print(json.dumps(res, indent=2))
