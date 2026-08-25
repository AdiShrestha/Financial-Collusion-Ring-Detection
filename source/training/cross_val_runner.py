"""Multi-seed stratified group cross-validation training and evaluation engine.

Contract C17-03 & Scientific Remediation:
- Executes 5 outer folds x 3 inner threshold-selection folds x 5 seeds (42--46).
- Selects optimal classification threshold tau_i* per fold via inner validation.
- Differentiates SCNN (simplicial complex with triangulated 2-simplices) and CCNN (native polygonal CW complex).
- Evaluates 8 model families across 200 production checkpoints.
"""

import json
import hashlib
import math
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

CONFIRMATORY_SEEDS = (42, 43, 44, 45, 46)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest_candidate_lineage(
    manifest: Dict[str, Any], cand_by_id: Dict[str, Any]
) -> Dict[str, int]:
    """Fail closed unless candidate and outer-fold membership are exactly canonical."""
    manifest_ids = set(manifest.get("candidate_assignments", {}))
    candidate_ids = set(cand_by_id)
    if candidate_ids != manifest_ids:
        raise ValueError(
            "Candidate/manifest lineage mismatch: "
            f"missing_from_manifest={sorted(candidate_ids - manifest_ids)[:10]}, "
            f"missing_from_candidates={sorted(manifest_ids - candidate_ids)[:10]}"
        )

    seen_test: List[str] = []
    for fold in manifest.get("outer_folds", []):
        train_ids = set(fold["train_candidate_ids"])
        test_ids = set(fold["test_candidate_ids"])
        if train_ids & test_ids:
            raise ValueError(f"Outer fold {fold['outer_fold_id']} has train/test overlap")
        if train_ids | test_ids != manifest_ids:
            raise ValueError(f"Outer fold {fold['outer_fold_id']} does not cover the cohort")
        seen_test.extend(fold["test_candidate_ids"])

    if len(seen_test) != len(set(seen_test)) or set(seen_test) != manifest_ids:
        raise ValueError("Outer test folds must partition every candidate exactly once")

    group_count = len(
        {str(v["group_id"]) for v in manifest["candidate_assignments"].values()}
    )
    if group_count != int(manifest["total_groups"]):
        raise ValueError("Manifest group count disagrees with candidate assignments")
    return {"candidate_count": len(manifest_ids), "group_count": group_count}


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "AMLFeatureBuilder":
            return AMLFeatureBuilder
        return super().find_class(module, name)


def safe_load_pickle(file_path: str) -> Any:
    """Safely unpickle preprocessor across different __main__ execution contexts."""
    with open(file_path, "rb") as f:
        return CustomUnpickler(f).load()


def forward_candidate_model(model: nn.Module, model_name: str, c: Dict[str, Any]) -> torch.Tensor:
    """Execute forward pass for candidate subgraph according to architecture requirements."""
    if model_name in ("gcn", "gat", "graphsage"):
        return model(c["x"], c["edge_index"])
    elif model_name == "gine":
        return model(c["x"], c["edge_index"], c["edge_attr"])
    elif model_name == "scnn":
        return model(c["X0_simp"], c["X1_simp"], c["X2_simp"], c["B1_simp"], c["B2_simp"])
    elif model_name == "ccnn":
        return model(c["X0"], c["X1"], c["X2"], c["B1"], c["B2"])
    else:
        raise ValueError(f"Unknown neural model: {model_name}")


def train_neural_candidate_model(
    model: nn.Module,
    model_name: str,
    train_candidates: List[Dict[str, Any]],
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    random_seed: int = 42,
) -> nn.Module:
    """Train neural candidate classifier with gradient clipping."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
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

            out = forward_candidate_model(model, model_name, c)

            if model_name == "gine":
                loss = bce_loss(out.view(1), torch.tensor([float(y_val)], dtype=torch.float32).view(1))
            else:
                loss = ce_loss(out, torch.tensor([y_val], dtype=torch.long))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model


def predict_neural_candidate_model(
    model: nn.Module,
    model_name: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute predicted probabilities and true labels."""
    model.eval()
    y_true = []
    y_prob = []

    with torch.no_grad():
        for c in candidates:
            y_true.append(int(c["label"]))
            out = forward_candidate_model(model, model_name, c)

            if model_name == "gine":
                prob = torch.sigmoid(out.view(1)).item()
            else:
                prob = torch.softmax(out, dim=-1)[0, 1].item()
            y_prob.append(prob)

    return np.array(y_prob), np.array(y_true)


def tune_threshold_on_inner_folds(
    model_name: str,
    train_candidates: List[Dict[str, Any]],
    inner_folds_spec: List[Dict[str, Any]],
    preprocessor: Any,
    cand_by_id: Dict[str, Any],
    txs_by_cand: Dict[str, List[Dict[str, Any]]],
    random_seed: int = 42,
) -> float:
    """Tune decision threshold using 3-fold inner cross-validation."""
    c_map = {c["candidate_id"]: c for c in train_candidates}
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thresh_f1_scores = defaultdict(list)

    for inner_spec in inner_folds_spec:
        in_train_cids = inner_spec["train_candidate_ids"]
        in_val_cids = inner_spec["val_candidate_ids"]

        in_train = [c_map[cid] for cid in in_train_cids if cid in c_map]
        in_val = [c_map[cid] for cid in in_val_cids if cid in c_map]

        if not in_train or not in_val:
            continue

        if model_name in ("logistic_regression", "hist_gradient_boosting"):
            X_tr, y_tr = preprocessor.extract_and_transform_candidates(in_train_cids, cand_by_id, txs_by_cand)
            X_va, y_va = preprocessor.extract_and_transform_candidates(in_val_cids, cand_by_id, txs_by_cand)
            m = create_model(model_name, random_seed=random_seed)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_va)[:, 1]
            y_val_arr = y_va
        else:
            m = create_model(model_name, random_seed=random_seed)
            m = train_neural_candidate_model(m, model_name, in_train, epochs=20, random_seed=random_seed)
            probs, y_val_arr = predict_neural_candidate_model(m, model_name, in_val)

        for th in thresholds:
            pred_b = (probs >= th).astype(int)
            f1 = f1_score(y_val_arr, pred_b, zero_division=0)
            thresh_f1_scores[th].append(f1)

    if not thresh_f1_scores:
        return 0.5

    mean_f1 = {th: np.mean(scores) for th, scores in thresh_f1_scores.items()}
    best_th = max(mean_f1.keys(), key=lambda th: mean_f1[th])
    return float(best_th)


def run_5fold_multi_seed_training(
    features_dir: str = "artifacts/features",
    preprocessors_dir: str = "artifacts/preprocessors",
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    candidate_txs_parquet_path: str = "artifacts/candidates/candidate_transactions.parquet",
    fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    checkpoints_dir: str = "artifacts/checkpoints",
    history_output_path: str = "artifacts/training/training_history.json",
    seeds: Sequence[int] = CONFIRMATORY_SEEDS,
    models: Sequence[str] = (
        "logistic_regression",
        "hist_gradient_boosting",
        "gcn",
        "gat",
        "graphsage",
        "gine",
        "scnn",
        "ccnn",
    ),
    epochs: int = 50,
    output_history_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Train all models with inner-fold threshold selection and strict lineage checks."""
    if output_history_path is not None:
        history_output_path = output_history_path

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(history_output_path)), exist_ok=True)

    c_tbl = pq.read_table(candidates_parquet_path)
    cand_by_id = {c["candidate_id"]: c for c in c_tbl.to_pylist()}

    tx_tbl = pq.read_table(candidate_txs_parquet_path)
    txs_by_cand = defaultdict(list)
    for t in tx_tbl.to_pylist():
        txs_by_cand[t["candidate_id"]].append(t)

    with open(fold_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    lineage = _validate_manifest_candidate_lineage(manifest, cand_by_id)
    if tuple(seeds) != CONFIRMATORY_SEEDS:
        raise ValueError(
            f"Confirmatory execution requires seeds {CONFIRMATORY_SEEDS}; got {tuple(seeds)}"
        )
    if tuple(models) != tuple(MODEL_FAMILIES):
        raise ValueError("Confirmatory execution requires the complete ordered model family")

    n_outer_folds = manifest["n_outer_folds"]
    history: Dict[str, Any] = {
        "dataset_name": manifest["dataset_name"],
        "n_outer_folds": n_outer_folds,
        "seeds": list(seeds),
        "models": list(models),
        "lineage": {
            **lineage,
            "manifest_sha256": _sha256_file(fold_manifest_path),
            "candidates_sha256": _sha256_file(candidates_parquet_path),
            "candidate_transactions_sha256": _sha256_file(candidate_txs_parquet_path),
            "selection_protocol": "three inner group folds select F1 threshold; model hyperparameters fixed symmetrically before outer evaluation",
            "fixed_training_config": {
                "epochs": int(epochs),
                "neural_learning_rate": 0.005,
                "weight_decay": 0.0001,
                "threshold_grid": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            },
        },
        "results": defaultdict(lambda: defaultdict(dict)),
    }

    total_checkpoints = 0

    for fold_idx in range(n_outer_folds):
        tensor_path = os.path.join(features_dir, f"fold_{fold_idx}_tensors.pt")
        prep_path = os.path.join(preprocessors_dir, f"fold_{fold_idx}_preprocessor.pkl")

        fold_tensors = torch.load(tensor_path, weights_only=False)
        train_cands = fold_tensors["train_candidates"]
        test_cands = fold_tensors["test_candidates"]
        inner_folds_spec = fold_tensors.get("inner_folds", [])

        loaded_prep = safe_load_pickle(prep_path)
        preprocessor = loaded_prep.get("preprocessor", loaded_prep) if isinstance(loaded_prep, dict) else loaded_prep

        train_cids = [c["candidate_id"] for c in train_cands]
        test_cids = [c["candidate_id"] for c in test_cands]
        manifest_fold = manifest["outer_folds"][fold_idx]
        if set(train_cids) != set(manifest_fold["train_candidate_ids"]):
            raise ValueError(f"Fold {fold_idx} tensor training candidates are stale")
        if set(test_cids) != set(manifest_fold["test_candidate_ids"]):
            raise ValueError(f"Fold {fold_idx} tensor test candidates are stale")

        for seed in seeds:
            for m_name in models:
                ckpt_filename = f"{m_name}_fold{fold_idx}_seed{seed}.pt"
                ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)

                # Tune threshold via inner validation
                optimal_tau = tune_threshold_on_inner_folds(
                    m_name, train_cands, inner_folds_spec, preprocessor, cand_by_id, txs_by_cand, random_seed=seed
                )

                if m_name in ("logistic_regression", "hist_gradient_boosting"):
                    X_tr, y_tr = preprocessor.extract_and_transform_candidates(train_cids, cand_by_id, txs_by_cand)
                    X_te, y_te = preprocessor.extract_and_transform_candidates(test_cids, cand_by_id, txs_by_cand)

                    model = create_model(m_name, random_seed=seed)
                    model.fit(X_tr, y_tr)

                    probs = model.predict_proba(X_te)[:, 1]
                    y_true = y_te

                    with open(ckpt_path.replace(".pt", ".pkl"), "wb") as mf:
                        pickle.dump(model, mf)

                else:
                    model = create_model(m_name, random_seed=seed)
                    model = train_neural_candidate_model(
                        model, m_name, train_cands, epochs=epochs, random_seed=seed
                    )
                    probs, y_true = predict_neural_candidate_model(model, m_name, test_cands)
                    torch.save(model.state_dict(), ckpt_path)

                total_checkpoints += 1

                # Evaluate metrics using inner-selected threshold
                ap = float(average_precision_score(y_true, probs))
                roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
                pred_binary = (probs >= optimal_tau).astype(int)
                f1 = float(f1_score(y_true, pred_binary, zero_division=0))

                history["results"][m_name][f"seed_{seed}"][f"fold_{fold_idx}"] = {
                    "ap": ap,
                    "roc_auc": roc,
                    "f1": f1,
                    "optimal_threshold": optimal_tau,
                    "test_candidate_ids": test_cids,
                    "y_true": [int(y) for y in y_true],
                    "y_prob": [float(p) for p in probs],
                    "y_pred": [int(b) for b in pred_binary],
                    "selection_ancestry": {
                        "inner_fold_count": len(inner_folds_spec),
                        "threshold_grid": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        "selected_threshold": optimal_tau,
                        "outer_test_used_for_selection": False,
                    },
                }

    # Construct OOF predictions dictionary
    oof_predictions = defaultdict(lambda: defaultdict(dict))
    for m_name in models:
        for seed in seeds:
            for fold_idx in range(n_outer_folds):
                fold_res = history["results"][m_name][f"seed_{seed}"][f"fold_{fold_idx}"]
                for cid, prob, yt in zip(fold_res["test_candidate_ids"], fold_res["y_prob"], fold_res["y_true"]):
                    pred_entry = {
                        "probability": prob,
                        "true_label": yt,
                        "fold": fold_idx,
                    }
                    oof_predictions[m_name][f"seed_{seed}"][cid] = pred_entry
                    oof_predictions[m_name][str(seed)][cid] = pred_entry

    history["oof_predictions"] = oof_predictions

    # Convert defaultdict to regular dict for serialization
    serializable_history = json.loads(json.dumps(history))
    with open(history_output_path, "w", encoding="utf-8") as hf:
        json.dump(serializable_history, hf, indent=2)

    return {
        "status": "TRAINING_COMPLETE",
        "total_checkpoints": total_checkpoints,
        "n_outer_folds": n_outer_folds,
        "seeds": list(seeds),
        "models": list(models),
        "history_file": history_output_path,
    }


if __name__ == "__main__":
    res = run_5fold_multi_seed_training()
    print(json.dumps(res, indent=2))
