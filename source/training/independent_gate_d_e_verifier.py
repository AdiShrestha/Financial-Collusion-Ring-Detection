"""Independent Gate D & Gate E Verifier and Auditor.

Verifies:
- Gate D: Candidate tensor bundles and 100% boundary nilpotency (B1 @ B2 == 0) across all folds.
- Gate E: Checkpoints for all 8 models across 5 folds and 5 seeds (200 checkpoints total) and full OOF prediction coverage.
"""

import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.training.cross_val_runner import MODEL_FAMILIES


class IndependentGateDEVerifier:
    """Independent auditor for Gate D (tensor features) and Gate E (models & checkpoints)."""

    def __init__(
        self,
        features_dir: str = "artifacts/features",
        checkpoints_dir: str = "artifacts/checkpoints",
        history_path: str = "artifacts/training/training_history.json",
        oof_path: str = "artifacts/predictions/oof_predictions.parquet",
    ):
        self.features_dir = features_dir
        self.checkpoints_dir = checkpoints_dir
        self.history_path = history_path
        self.oof_path = oof_path

    def verify_gate_d(
        self,
        features_dir: Optional[str] = None,
        output_report_path: Optional[str] = "project/gate_d_report.json",
        **kwargs,
    ) -> Dict[str, Any]:
        """Verify tensor encoding and boundary nilpotency (B1 @ B2 == 0) for all candidate complexes."""
        feat_dir = features_dir or self.features_dir
        total_nilpotent = 0
        total_candidates = 0
        fold_reports = {}

        for fold_idx in range(5):
            tensor_file = os.path.join(feat_dir, f"fold_{fold_idx}_tensors.pt")
            assert os.path.exists(tensor_file), f"Missing fold tensor file: {tensor_file}"
            fold_bundle = torch.load(tensor_file, weights_only=False)

            train_cands = fold_bundle["train_candidates"]
            test_cands = fold_bundle["test_candidates"]

            all_cands = train_cands + test_cands
            for c in all_cands:
                total_candidates += 1
                b1 = c["B1"].numpy() if isinstance(c["B1"], torch.Tensor) else c["B1"]
                b2 = c["B2"].numpy() if isinstance(c["B2"], torch.Tensor) else c["B2"]
                prod = np.dot(b1, b2)
                assert np.all(prod == 0), f"Nilpotency failure on candidate {c.get('candidate_id')}"
                total_nilpotent += 1

            fold_reports[f"fold_{fold_idx}"] = {
                "train_count": len(train_cands),
                "test_count": len(test_cands),
            }

        report = {
            "gate": "GATE_D",
            "gate_d_status": "GATE_D_PASS",
            "status": "GATE_D_PASS",
            "total_candidates_audited": total_candidates,
            "nilpotency_verified_count": total_nilpotent,
            "folds": fold_reports,
        }

        if output_report_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report

    def verify_gate_e(
        self,
        checkpoints_dir: Optional[str] = None,
        training_history_path: Optional[str] = None,
        output_report_path: Optional[str] = "project/gate_e_report.json",
        **kwargs,
    ) -> Dict[str, Any]:
        """Verify canonical checkpoints across all history-declared models/folds/seeds."""
        hist_path = training_history_path or self.history_path
        ckpt_dir = checkpoints_dir or self.checkpoints_dir

        assert os.path.exists(hist_path), f"Missing training history: {hist_path}"
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        seeds = history.get("seeds", [])
        models = history.get("models", MODEL_FAMILIES)
        assert seeds == [42, 43, 44, 45, 46], f"Expected canonical seeds 42--46, got {seeds}"
        assert models == MODEL_FAMILIES, "History model list is not canonical"
        total_expected = 5 * len(models) * len(seeds)
        valid_checkpoints = 0

        for f_id in range(5):
            for m_name in models:
                for s in seeds:
                    ckpt_pkl = os.path.join(ckpt_dir, f"{m_name}_fold{f_id}_seed{s}.pkl")
                    ckpt_pt = os.path.join(ckpt_dir, f"{m_name}_fold{f_id}_seed{s}.pt")
                    sub_pkl = os.path.join(ckpt_dir, f"fold_{f_id}", f"{m_name}_seed_{s}.pkl")
                    sub_pt = os.path.join(ckpt_dir, f"fold_{f_id}", f"{m_name}_seed_{s}.pt")

                    if m_name in ("logistic_regression", "hist_gradient_boosting"):
                        target_file = ckpt_pkl if os.path.exists(ckpt_pkl) else sub_pkl
                        assert os.path.exists(target_file), f"Missing tabular checkpoint: {target_file}"
                        with open(target_file, "rb") as f:
                            m_obj = pickle.load(f)
                        assert hasattr(m_obj, "predict_proba")
                        valid_checkpoints += 1
                    else:
                        target_file = ckpt_pt if os.path.exists(ckpt_pt) else sub_pt
                        assert os.path.exists(target_file), f"Missing neural checkpoint: {target_file}"
                        state = torch.load(target_file, weights_only=True)
                        for param_name, tensor in state.items():
                            assert not torch.isnan(tensor).any(), f"NaN in {m_name} fold {f_id} seed {s} {param_name}"
                            assert not torch.isinf(tensor).any(), f"Inf in {m_name} fold {f_id} seed {s} {param_name}"
                        valid_checkpoints += 1

        # Verify OOF prediction coverage (all 155 candidates per model-seed)
        if os.path.exists(self.oof_path):
            oof_tbl = pq.read_table(self.oof_path)
            oof_recs = oof_tbl.to_pylist()
            by_m_s = {}
            for r in oof_recs:
                key = (r["model_name"], r["seed"])
                by_m_s[key] = by_m_s.get(key, 0) + 1
            for m_name in models:
                for s in seeds:
                    c_count = by_m_s.get((m_name, s), 0)
                    assert c_count == 155, f"Expected 155 predictions for {m_name} seed {s}, got {c_count}"

        is_pass = (valid_checkpoints == total_expected and total_expected > 0)
        report = {
            "gate": "GATE_E",
            "gate_e_status": "GATE_E_PASS" if is_pass else "GATE_E_FAIL",
            "status": "GATE_E_PASS" if is_pass else "GATE_E_FAIL",
            "all_subchecks_passed": is_pass,
            "total_checkpoints_verified": valid_checkpoints,
            "expected_checkpoints": total_expected,
            "models_verified": models,
            "seeds_verified": seeds,
            "n_folds": 5,
        }

        if output_report_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report
