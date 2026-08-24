"""Rigorous Statistical Inference and Hypothesis Testing for KUSET Publication.

Contract C18-01 (T-COMP): Computes AP, AUROC, F1, 10,000 paired group bootstrap 95% CIs,
and 10,000 group-blocked model-swap permutation tests for RQ1 and RQ2 with Holm-Bonferroni correction.
Emits results/production_confirmatory_stats.json.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class KUSETHypothesisTester:
    """Computes confirmatory evaluation statistics and permutation hypothesis tests."""

    def __init__(
        self,
        oof_parquet_path: str = "artifacts/predictions/oof_predictions.parquet",
        fold_manifest_path: str = "artifacts/splits/fold_manifest.json",
    ):
        self.oof_parquet_path = oof_parquet_path
        if not os.path.exists(oof_parquet_path):
            raise FileNotFoundError(f"Missing OOF prediction ledger: {oof_parquet_path}")

        tbl = pq.read_table(oof_parquet_path)
        self.records = tbl.to_pylist()

        # Load component groups if available
        cand_to_group = {}
        if os.path.exists(fold_manifest_path):
            with open(fold_manifest_path, "r", encoding="utf-8") as f:
                mf = json.load(f)
            raw_assign = mf.get("candidate_assignments", {})
            for cid, val in raw_assign.items():
                if isinstance(val, dict):
                    cand_to_group[cid] = val.get("group_id", f"group_{cid}")
                else:
                    cand_to_group[cid] = str(val)

        # Group records by (model_name, seed)
        self.by_model_seed = defaultdict(lambda: defaultdict(dict))
        self.cand_metadata = {}
        for r in self.records:
            m = r["model_name"]
            s = r["seed"]
            cid = r["candidate_id"]
            self.by_model_seed[m][s][cid] = r["y_pred_prob"]
            self.cand_metadata[cid] = {
                "label": r["y_true"],
                "fold_id": r["fold_id"],
                "group_id": cand_to_group.get(cid, f"group_{r['fold_id']}"),
                "cycle_length": r["cycle_length"],
            }

        self.candidate_ids = sorted(list(self.cand_metadata.keys()))
        self.y_true = np.array([self.cand_metadata[cid]["label"] for cid in self.candidate_ids])
        self.fold_ids = np.array([self.cand_metadata[cid]["fold_id"] for cid in self.candidate_ids])
        self.group_ids = np.array([self.cand_metadata[cid]["group_id"] for cid in self.candidate_ids])
        self.cycle_lengths = np.array([self.cand_metadata[cid]["cycle_length"] for cid in self.candidate_ids])

    def get_model_mean_probs(self, model_name: str, subset_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get mean prediction probabilities averaged across seeds for a model."""
        seeds = list(self.by_model_seed[model_name].keys())
        prob_matrix = np.array([
            [self.by_model_seed[model_name][s][cid] for cid in self.candidate_ids]
            for s in seeds
        ])
        mean_probs = np.mean(prob_matrix, axis=0)
        if subset_mask is not None:
            return mean_probs[subset_mask]
        return mean_probs

    def compute_model_metrics(self, model_name: str, n_bootstrap: int = 10000, random_seed: int = 42) -> Dict[str, Any]:
        """Compute point estimates and 95% group bootstrap confidence intervals."""
        rng = np.random.default_rng(random_seed)
        probs = self.get_model_mean_probs(model_name)
        y = self.y_true

        point_ap = float(average_precision_score(y, probs))
        point_roc = float(roc_auc_score(y, probs))
        pred_bin = (probs >= 0.5).astype(int)
        point_f1 = float(f1_score(y, pred_bin, zero_division=0))

        # Group bootstrap over outer folds / candidate clusters
        unique_folds = np.unique(self.fold_ids)
        boot_aps = []
        boot_rocs = []

        for _ in range(n_bootstrap):
            sampled_folds = rng.choice(unique_folds, size=len(unique_folds), replace=True)
            sampled_indices = []
            for f in sampled_folds:
                sampled_indices.extend(np.where(self.fold_ids == f)[0])
            
            idx = np.array(sampled_indices)
            y_b = y[idx]
            p_b = probs[idx]

            if sum(y_b) > 0:
                boot_aps.append(average_precision_score(y_b, p_b))
            if 0 < sum(y_b) < len(y_b):
                boot_rocs.append(roc_auc_score(y_b, p_b))

        ap_ci = [float(np.percentile(boot_aps, 2.5)), float(np.percentile(boot_aps, 97.5))] if boot_aps else [point_ap, point_ap]
        roc_ci = [float(np.percentile(boot_rocs, 2.5)), float(np.percentile(boot_rocs, 97.5))] if boot_rocs else [point_roc, point_roc]

        return {
            "model_name": model_name,
            "average_precision": {
                "point_estimate": point_ap,
                "ci_95_lower": ap_ci[0],
                "ci_95_upper": ap_ci[1],
            },
            "roc_auc": {
                "point_estimate": point_roc,
                "ci_95_lower": roc_ci[0],
                "ci_95_upper": roc_ci[1],
            },
            "f1_score": point_f1,
        }

    def run_group_blocked_permutation_test(
        self,
        model_a: str,
        model_b: str,
        subset_mask: Optional[np.ndarray] = None,
        n_permutations: int = 10000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Run group-blocked model-swap permutation test."""
        rng = np.random.default_rng(random_seed)
        probs_a = self.get_model_mean_probs(model_a, subset_mask)
        probs_b = self.get_model_mean_probs(model_b, subset_mask)
        y = self.y_true if subset_mask is None else self.y_true[subset_mask]
        groups = self.group_ids if subset_mask is None else self.group_ids[subset_mask]

        obs_diff = float(average_precision_score(y, probs_a) - average_precision_score(y, probs_b))

        unique_groups = np.unique(groups)
        perm_diffs = []

        for _ in range(n_permutations):
            # For each disjoint group block, decide whether to swap model predictions
            swap_decisions = rng.integers(0, 2, size=len(unique_groups))
            swap_map = {g: swap_decisions[i] for i, g in enumerate(unique_groups)}

            perm_a = probs_a.copy()
            perm_b = probs_b.copy()

            for i, g in enumerate(groups):
                if swap_map[g] == 1:
                    perm_a[i], perm_b[i] = probs_b[i], probs_a[i]

            diff = average_precision_score(y, perm_a) - average_precision_score(y, perm_b)
            perm_diffs.append(diff)

        perm_diffs = np.array(perm_diffs)
        p_value = float(np.mean(perm_diffs >= obs_diff))

        return {
            "model_a": model_a,
            "model_b": model_b,
            "observed_delta_ap": obs_diff,
            "n_permutations": n_permutations,
            "raw_p_value": p_value,
            "sample_size": len(y),
        }

    def execute_full_confirmatory_suite(
        self,
        output_results_path: str = "results/production_confirmatory_stats.json",
    ) -> Dict[str, Any]:
        """Run complete confirmatory statistical evaluation."""
        models = sorted(list(self.by_model_seed.keys()))
        model_benchmarks = {}
        for m in models:
            model_benchmarks[m] = self.compute_model_metrics(m)

        # Hypothesis 1 (RQ1): CCNN vs GINE on all cycles
        rq1_perm = self.run_group_blocked_permutation_test("ccnn", "gine")

        # Hypothesis 2 (RQ2): CCNN vs SCNN on k >= 4
        k_ge_4_mask = (self.cycle_lengths >= 4)
        rq2_perm = self.run_group_blocked_permutation_test("ccnn", "scnn", subset_mask=k_ge_4_mask)

        # Hypothesis 3 (RQ3): Tabular (LR) vs GINE
        rq3_perm = self.run_group_blocked_permutation_test("logistic_regression", "gine")

        # Holm-Bonferroni correction
        hypotheses = [
            ("RQ1_CCNN_vs_GINE", rq1_perm),
            ("RQ2_CCNN_vs_SCNN_k_ge_4", rq2_perm),
            ("RQ3_LR_vs_GINE", rq3_perm),
        ]
        # Sort by raw p-value ascending
        sorted_hyps = sorted(hypotheses, key=lambda x: x[1]["raw_p_value"])
        m = len(sorted_hyps)
        
        adjusted_results = {}
        for rank, (name, test_res) in enumerate(sorted_hyps):
            alpha_k = 0.05 / (m - rank)
            raw_p = test_res["raw_p_value"]
            adj_p = min(1.0, raw_p * (m - rank))
            is_sig = (raw_p <= alpha_k)

            adjusted_results[name] = {
                **test_res,
                "holm_bonferroni_threshold": alpha_k,
                "adjusted_p_value": adj_p,
                "statistically_significant": is_sig,
            }

        full_report = {
            "evaluation_metric": "Average Precision (AP)",
            "benchmark_models": model_benchmarks,
            "hypothesis_tests": adjusted_results,
            "dataset_cohort_size": len(self.candidate_ids),
            "positive_cases": int(sum(self.y_true)),
            "negative_cases": int(len(self.y_true) - sum(self.y_true)),
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_results_path)), exist_ok=True)
        with open(output_results_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        return full_report


def holm_bonferroni_correction(raw_p_values: Sequence[float], alpha: float = 0.05) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values."""
    m = len(raw_p_values)
    indexed_p = sorted(enumerate(raw_p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank
        adj = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)

    return adjusted


def paired_bootstrap_delta_pr_auc(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap difference in AP."""
    rng = np.random.default_rng(random_seed)
    deltas = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if sum(y_b) > 0:
            ap_a = average_precision_score(y_b, probs_a[idx])
            ap_b = average_precision_score(y_b, probs_b[idx])
            deltas.append(ap_a - ap_b)

    point_diff = average_precision_score(y_true, probs_a) - average_precision_score(y_true, probs_b)
    ci_lo = float(np.percentile(deltas, 2.5)) if deltas else point_diff
    ci_hi = float(np.percentile(deltas, 97.5)) if deltas else point_diff
    return float(point_diff), ci_lo, ci_hi


if __name__ == "__main__":
    from source.experiments.oof_evaluator import export_oof_prediction_ledger
    export_oof_prediction_ledger()
    tester = KUSETHypothesisTester()
    stats = tester.execute_full_confirmatory_suite()
    print(json.dumps(stats, indent=2))
