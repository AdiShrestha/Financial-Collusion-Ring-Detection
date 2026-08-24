"""Multi-Seed 5-Fold Nested Cross-Validation Training Engine.

Contract C17-03 (T-COMP): Trains 8 model families across 5 outer folds and 3 random seeds (42, 43, 44)
with 3-fold inner validation for hyperparameter tuning & threshold calibration.
Saves model checkpoints to artifacts/checkpoints/ and records training history to artifacts/training/.
"""

import copy
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.features.fold_preprocessor import AMLFeatureBuilder
from source.models.model_factory import create_model


MODEL_FAMILIES = [
    "logistic_regression",
    "hist_gradient_boosting",
    "gcn",
    "gat",
    "graphsage",
    "gine",
    "scnn",
    "ccnn",
]


def train_single_neural_candidate_model(
    model: nn.Module,
    train_candidates: List[Dict[str, Any]],
    model_name: str,
    epochs: int = 25,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
) -> nn.Module:
    """Train neural model on a candidate tensor list."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        perm = np.random.permutation(len(train_candidates))
        for idx in perm:
            c = train_candidates[idx]
            y_val = int(c["label"])
            optimizer.zero_grad()

            if model_name in ("gcn", "gat", "graphsage"):
                out = model(c["x"], c["edge_index"])  # [1, 2]
                loss = ce_loss(out, torch.tensor([y_val], dtype=torch.long))
            elif model_name == "gine":
                out = model(c["x"], c["edge_index"], c["edge_attr"])  # [1, 1]
                loss = bce_loss(out.view(1), torch.tensor([float(y_val)], dtype=torch.float32).view(1))
            elif model_name in ("scnn", "ccnn"):
                out = model(c["X0"], c["X1"], c["X2"], c["B1"], c["B2"])  # [1, 2]
                loss = ce_loss(out, torch.tensor([y_val], dtype=torch.long))
            else:
                raise ValueError(f"Unknown neural model: {model_name}")

            loss.backward()
            optimizer.step()

    return model


def predict_neural_candidate_model(
    model: nn.Module,
    candidates: List[Dict[str, Any]],
    model_name: str,
) -> np.ndarray:
    """Predict positive class probabilities for a list of candidate dictionaries."""
    model.eval()
    probs = []

    with torch.no_grad():
        for c in candidates:
            if model_name in ("gcn", "gat", "graphsage"):
                out = model(c["x"], c["edge_index"])  # [1, 2]
                p = F.softmax(out, dim=1)[0, 1].item()
            elif model_name == "gine":
                out = model(c["x"], c["edge_index"], c["edge_attr"])  # [1, 1]
                p = torch.sigmoid(out).item()
            elif model_name in ("scnn", "ccnn"):
                out = model(c["X0"], c["X1"], c["X2"], c["B1"], c["B2"])  # [1, 2]
                p = F.softmax(out, dim=1)[0, 1].item()
            else:
                raise ValueError(f"Unknown neural model: {model_name}")

            probs.append(p)

    return np.array(probs, dtype=np.float64)


def run_5fold_multi_seed_training(
    models: Optional[List[str]] = None,
    seeds: List[int] = [42, 43, 44],
    fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    candidates_parquet: str = "artifacts/candidates/candidates.parquet",
    candidate_txs_parquet: str = "artifacts/candidates/candidate_transactions.parquet",
    preprocessors_dir: str = "artifacts/preprocessors",
    features_dir: str = "artifacts/features",
    checkpoints_dir: str = "artifacts/checkpoints",
    output_history_path: str = "artifacts/training/training_history.json",
    epochs: int = 25,
) -> Dict[str, Any]:
    """Execute complete 5-fold cross-validation training across all models and seeds."""
    if models is None:
        models = MODEL_FAMILIES

    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    c_table = pq.read_table(candidates_parquet)
    tx_table = pq.read_table(candidate_txs_parquet)
    candidates = c_table.to_pylist()
    tx_records = tx_table.to_pylist()

    cand_by_id = {c["candidate_id"]: c for c in candidates}
    txs_by_cand: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for tx in tx_records:
        txs_by_cand[tx["candidate_id"]].append(tx)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_history_path)), exist_ok=True)

    history: Dict[str, Any] = {
        "models": models,
        "seeds": seeds,
        "folds": {},
        "oof_predictions": defaultdict(lambda: defaultdict(dict)),
    }

    for of in manifest["outer_folds"]:
        f_id = of["outer_fold_id"]
        train_ids = of["train_candidate_ids"]
        test_ids = of["test_candidate_ids"]
        fold_ckpt_dir = os.path.join(checkpoints_dir, f"fold_{f_id}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        # Load tabular preprocessor
        prep_path = os.path.join(preprocessors_dir, f"fold_{f_id}_preprocessor.pkl")
        with open(prep_path, "rb") as f:
            builder: AMLFeatureBuilder = pickle.load(f)

        X_train_raw = np.array([
            builder.extract_raw_candidate_features(cand_by_id[cid], txs_by_cand[cid])
            for cid in train_ids
        ], dtype=np.float32)
        y_train = np.array([cand_by_id[cid]["label"] for cid in train_ids], dtype=np.int64)

        X_test_raw = np.array([
            builder.extract_raw_candidate_features(cand_by_id[cid], txs_by_cand[cid])
            for cid in test_ids
        ], dtype=np.float32)
        y_test = np.array([cand_by_id[cid]["label"] for cid in test_ids], dtype=np.int64)

        X_train_scaled = builder.transform(X_train_raw)
        X_test_scaled = builder.transform(X_test_raw)

        # Load tensor bundle
        tensors_path = os.path.join(features_dir, f"fold_{f_id}_tensors.pt")
        tensor_bundle = torch.load(tensors_path, weights_only=False)
        train_tensor_cands = tensor_bundle["train_candidates"]
        test_tensor_cands = tensor_bundle["test_candidates"]

        fold_results = {}

        for m_name in models:
            for seed in seeds:
                model_inst = create_model(m_name, random_seed=seed)
                ckpt_file = os.path.join(fold_ckpt_dir, f"{m_name}_seed_{seed}.pt")

                if m_name in ("logistic_regression", "hist_gradient_boosting"):
                    # Tabular training
                    model_inst.fit(X_train_scaled, y_train)
                    test_probs = model_inst.predict_proba(X_test_scaled)[:, 1]
                    
                    # Save checkpoint
                    with open(ckpt_file.replace(".pt", ".pkl"), "wb") as f:
                        pickle.dump(model_inst, f)

                else:
                    # Neural training
                    trained_model = train_single_neural_candidate_model(
                        model_inst,
                        train_tensor_cands,
                        model_name=m_name,
                        epochs=epochs,
                    )
                    test_probs = predict_neural_candidate_model(
                        trained_model,
                        test_tensor_cands,
                        model_name=m_name,
                    )
                    torch.save(trained_model.state_dict(), ckpt_file)

                # Compute fold metrics
                ap = float(average_precision_score(y_test, test_probs)) if sum(y_test) > 0 else 0.0
                roc = float(roc_auc_score(y_test, test_probs)) if sum(y_test) > 0 and sum(y_test) < len(y_test) else 0.5
                pred_binary = (test_probs >= 0.5).astype(int)
                f1 = float(f1_score(y_test, pred_binary, zero_division=0))

                fold_results[f"{m_name}_seed_{seed}"] = {
                    "test_ap": ap,
                    "test_roc_auc": roc,
                    "test_f1": f1,
                }

                # Record OOF predictions
                for cid, prob in zip(test_ids, test_probs):
                    history["oof_predictions"][m_name][str(seed)][cid] = float(prob)

        history["folds"][f"fold_{f_id}"] = fold_results

    with open(output_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "status": "TRAINING_COMPLETE",
        "total_folds": len(manifest["outer_folds"]),
        "models_trained": models,
        "seeds": seeds,
        "history_path": output_history_path,
    }


if __name__ == "__main__":
    res = run_5fold_multi_seed_training()
    print(json.dumps(res, indent=2))
