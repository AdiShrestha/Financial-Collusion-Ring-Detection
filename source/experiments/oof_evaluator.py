"""Long-form Out-of-Fold (OOF) Prediction Ledger Exporter.

Contract C18-01 (T-COMP): Compiles out-of-fold predictions across all candidates,
models, and seeds into artifacts/predictions/oof_predictions.parquet.
"""

import json
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
    cand_info = {
        c["candidate_id"]: {
            "label": int(c["label"]),
            "cycle_length": int(c["cycle_length"]),
        }
        for c in candidates
    }

    # Map candidate to test fold id
    cand_to_fold = {}
    for of in manifest["outer_folds"]:
        f_id = of["outer_fold_id"]
        for cid in of["test_candidate_ids"]:
            cand_to_fold[cid] = f_id

    records = []
    oof = history["oof_predictions"]

    for model_name, seed_dict in oof.items():
        for seed_str, pred_dict in seed_dict.items():
            seed = int(seed_str)
            for cid, prob in pred_dict.items():
                c_meta = cand_info.get(cid, {"label": 0, "cycle_length": 3})
                records.append({
                    "candidate_id": cid,
                    "model_name": model_name,
                    "seed": seed,
                    "fold_id": cand_to_fold.get(cid, 0),
                    "cycle_length": c_meta["cycle_length"],
                    "y_true": c_meta["label"],
                    "y_pred_prob": float(prob),
                })

    schema = pa.schema([
        ("candidate_id", pa.string()),
        ("model_name", pa.string()),
        ("seed", pa.int64()),
        ("fold_id", pa.int64()),
        ("cycle_length", pa.int64()),
        ("y_true", pa.int64()),
        ("y_pred_prob", pa.float64()),
    ])

    table = pa.Table.from_pylist(records, schema=schema)
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    pq.write_table(table, output_parquet_path, compression="snappy")

    return {
        "status": "OOF_LEDGER_EXPORTED",
        "output_path": output_parquet_path,
        "total_records": len(records),
        "models": list(oof.keys()),
    }


if __name__ == "__main__":
    res = export_oof_prediction_ledger()
    print(json.dumps(res, indent=2))
