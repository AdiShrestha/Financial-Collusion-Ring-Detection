"""Group-valid statistical inference for the KUSET corrective analysis.

Contract C18-01 (T-COMP) & Scientific Remediation:
- Computes AP, AUROC, F1, 10,000 paired group bootstrap 95% CIs across the 18 disjoint component groups.
- Executes 10,000 group-blocked model-swap permutation tests across the 18 disjoint component groups with Holm-Bonferroni correction.
- Fails closed on any ledger/manifest lineage mismatch.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def holm_bonferroni_correction(p_values: Sequence[float]) -> List[float]:
    """Perform Holm-Bonferroni step-down multi-hypothesis correction."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj_p = min(1.0, p_val * (m - rank))
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = running_max

    return adjusted


def paired_bootstrap_delta_pr_auc(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute point estimate and 95% bootstrap CI for delta AP."""
    rng = np.random.default_rng(random_seed)
    obs_delta = float(average_precision_score(y_true, probs_a) - average_precision_score(y_true, probs_b))

    deltas = []
    n_samples = len(y_true)

    if group_ids is not None:
        unique_grps = np.unique(group_ids)
        n_grps = len(unique_grps)
        for _ in range(n_bootstrap):
            sampled_grps = rng.choice(unique_grps, size=n_grps, replace=True)
            sampled_idx = []
            for g in sampled_grps:
                sampled_idx.extend(np.where(group_ids == g)[0])
            idx = np.array(sampled_idx)
            y_b = y_true[idx]
            if 0 < sum(y_b) < len(y_b):
                d_b = average_precision_score(y_b, probs_a[idx]) - average_precision_score(y_b, probs_b[idx])
                deltas.append(d_b)
    else:
        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            y_b = y_true[idx]
            if 0 < sum(y_b) < len(y_b):
                d_b = average_precision_score(y_b, probs_a[idx]) - average_precision_score(y_b, probs_b[idx])
                deltas.append(d_b)

    ci_lo = float(np.percentile(deltas, 2.5)) if deltas else obs_delta
    ci_hi = float(np.percentile(deltas, 97.5)) if deltas else obs_delta
    return obs_delta, ci_lo, ci_hi


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
        if not os.path.exists(fold_manifest_path):
            raise FileNotFoundError(f"Missing fold manifest: {fold_manifest_path}")

        tbl = pq.read_table(oof_parquet_path)
        self.records = tbl.to_pylist()

        with open(fold_manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        raw_assign = self.manifest.get("candidate_assignments", {})
        cand_to_group = {cid: str(val["group_id"]) for cid, val in raw_assign.items()}
        cand_to_fold = {
            cid: int(fold["outer_fold_id"])
            for fold in self.manifest["outer_folds"]
            for cid in fold["test_candidate_ids"]
        }
        if not self.records:
            raise ValueError("OOF ledger is empty")
        ledger_ids = {str(r["candidate_id"]) for r in self.records}
        if ledger_ids != set(raw_assign):
            raise ValueError("OOF candidate IDs do not equal manifest assignments")

        # Group records by (model_name, seed)
        self.by_model_seed = defaultdict(lambda: defaultdict(dict))
        self.by_model_seed_bin = defaultdict(lambda: defaultdict(dict))
        self.cand_metadata = {}
        observed_keys = set()
        for r in self.records:
            m = r["model_name"]
            s = r["seed"]
            cid = r["candidate_id"]
            key = (cid, m, int(s))
            if key in observed_keys:
                raise ValueError(f"Duplicate OOF key: {key}")
            observed_keys.add(key)
            if str(r["group_id"]) != cand_to_group[cid]:
                raise ValueError(f"Ledger/manifest group mismatch for {cid}")
            if int(r["fold_id"]) != cand_to_fold[cid]:
                raise ValueError(f"Ledger/manifest fold mismatch for {cid}")
            if cid in self.cand_metadata:
                prior = self.cand_metadata[cid]
                if (
                    int(prior["label"]) != int(r["y_true"])
                    or int(prior["fold_id"]) != int(r["fold_id"])
                    or int(prior["cycle_length"]) != int(r["cycle_length"])
                ):
                    raise ValueError(f"Inconsistent metadata across ledger rows for {cid}")
            self.by_model_seed[m][s][cid] = r["y_pred_prob"]
            self.by_model_seed_bin[m][s][cid] = r.get("y_pred_binary", 1 if r["y_pred_prob"] >= 0.5 else 0)
            self.cand_metadata[cid] = {
                "label": r["y_true"],
                "fold_id": r["fold_id"],
                "group_id": cand_to_group[cid],
                "cycle_length": r["cycle_length"],
            }

        self.candidate_ids = sorted(list(self.cand_metadata.keys()))
        self.y_true = np.array([self.cand_metadata[cid]["label"] for cid in self.candidate_ids])
        self.fold_ids = np.array([self.cand_metadata[cid]["fold_id"] for cid in self.candidate_ids])
        self.group_ids = np.array([self.cand_metadata[cid]["group_id"] for cid in self.candidate_ids])
        self.cycle_lengths = np.array([self.cand_metadata[cid]["cycle_length"] for cid in self.candidate_ids])
        self.unique_groups = np.unique(self.group_ids)
        self.models = sorted(self.by_model_seed)
        self.seeds = sorted({int(s) for by_seed in self.by_model_seed.values() for s in by_seed})
        if len(self.candidate_ids) != int(self.manifest["total_candidates"]):
            raise ValueError("Ledger candidate count disagrees with manifest")
        if len(self.unique_groups) != int(self.manifest["total_groups"]):
            raise ValueError("Ledger group count disagrees with manifest")
        if self.seeds != [42, 43, 44, 45, 46]:
            raise ValueError(f"Canonical inference requires seeds 42--46; got {self.seeds}")
        expected_models = {
            "logistic_regression", "hist_gradient_boosting", "gcn", "gat",
            "graphsage", "gine", "scnn", "ccnn",
        }
        if set(self.models) != expected_models:
            raise ValueError("Ledger model family is incomplete")
        expected_ids = set(self.candidate_ids)
        for model in self.models:
            if sorted(int(s) for s in self.by_model_seed[model]) != self.seeds:
                raise ValueError(f"Incomplete seed coverage for {model}")
            for seed in self.seeds:
                if set(self.by_model_seed[model][seed]) != expected_ids:
                    raise ValueError(f"Incomplete candidate coverage for {model}/seed_{seed}")

        self.provenance = {
            "oof_sha256": self._sha256_file(oof_parquet_path),
            "manifest_sha256": self._sha256_file(fold_manifest_path),
        }

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def get_model_mean_probs(self, model_name: str, subset_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get mean prediction probabilities averaged across seeds for a model."""
        seeds = sorted(self.by_model_seed[model_name].keys())
        prob_matrix = np.array([
            [self.by_model_seed[model_name][s][cid] for cid in self.candidate_ids]
            for s in seeds
        ])
        mean_probs = np.mean(prob_matrix, axis=0)
        if subset_mask is not None:
            return mean_probs[subset_mask]
        return mean_probs

    def get_model_mean_binary(self, model_name: str, subset_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Get ensemble majority binary predictions for a model."""
        seeds = sorted(self.by_model_seed_bin[model_name].keys())
        bin_matrix = np.array([
            [self.by_model_seed_bin[model_name][s][cid] for cid in self.candidate_ids]
            for s in seeds
        ])
        maj_bin = (np.mean(bin_matrix, axis=0) >= 0.5).astype(int)
        if subset_mask is not None:
            return maj_bin[subset_mask]
        return maj_bin

    def compute_model_metrics(self, model_name: str, n_bootstrap: int = 10000, random_seed: int = 42) -> Dict[str, Any]:
        """Compute point estimates and 95% group bootstrap confidence intervals over disjoint component groups."""
        rng = np.random.default_rng(random_seed)
        probs = self.get_model_mean_probs(model_name)
        pred_bin = self.get_model_mean_binary(model_name)
        y = self.y_true

        point_ap = float(average_precision_score(y, probs))
        point_roc = float(roc_auc_score(y, probs))
        point_f1 = float(f1_score(y, pred_bin, zero_division=0))

        n_grps = len(self.unique_groups)
        boot_aps = []
        boot_rocs = []
        invalid_draws = 0

        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(self.unique_groups, size=n_grps, replace=True)
            sampled_indices = []
            for g in sampled_groups:
                sampled_indices.extend(np.where(self.group_ids == g)[0])

            idx = np.array(sampled_indices)
            y_b = y[idx]
            p_b = probs[idx]

            if 0 < sum(y_b) < len(y_b):
                boot_aps.append(average_precision_score(y_b, p_b))
                boot_rocs.append(roc_auc_score(y_b, p_b))
            else:
                invalid_draws += 1

        ap_ci = [float(np.percentile(boot_aps, 2.5)), float(np.percentile(boot_aps, 97.5))] if boot_aps else [point_ap, point_ap]
        roc_ci = [float(np.percentile(boot_rocs, 2.5)), float(np.percentile(boot_rocs, 97.5))] if boot_rocs else [point_roc, point_roc]

        return {
            "model_name": model_name,
            "average_precision": point_ap,
            "average_precision_ci_95": ap_ci,
            "roc_auc": point_roc,
            "roc_auc_ci_95": roc_ci,
            "f1_score": point_f1,
            "n_bootstrap_requested": n_bootstrap,
            "n_bootstrap_valid": len(boot_aps),
            "n_bootstrap_invalid_one_class": invalid_draws,
        }

    def paired_group_bootstrap_delta(
        self,
        model_a: str,
        model_b: str,
        subset_mask: Optional[np.ndarray] = None,
        n_bootstrap: int = 10000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Bootstrap paired AP delta by resampling whole manifest groups."""
        rng = np.random.default_rng(random_seed)
        probs_a = self.get_model_mean_probs(model_a, subset_mask)
        probs_b = self.get_model_mean_probs(model_b, subset_mask)
        y = self.y_true[subset_mask] if subset_mask is not None else self.y_true
        groups = self.group_ids[subset_mask] if subset_mask is not None else self.group_ids
        active_groups = np.unique(groups)
        observed = float(average_precision_score(y, probs_a) - average_precision_score(y, probs_b))
        deltas: List[float] = []
        invalid = 0
        for _ in range(n_bootstrap):
            sampled_groups = rng.choice(active_groups, size=len(active_groups), replace=True)
            sampled_idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled_groups])
            y_draw = y[sampled_idx]
            if not (0 < int(y_draw.sum()) < len(y_draw)):
                invalid += 1
                continue
            deltas.append(
                float(
                    average_precision_score(y_draw, probs_a[sampled_idx])
                    - average_precision_score(y_draw, probs_b[sampled_idx])
                )
            )
        if not deltas:
            raise ValueError("All paired group bootstrap draws were one-class")
        return {
            "delta_ap": observed,
            "delta_ap_ci_95": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
            "n_bootstrap_requested": n_bootstrap,
            "n_bootstrap_valid": len(deltas),
            "n_bootstrap_invalid_one_class": invalid,
        }

    def run_group_permutation_test(
        self,
        model_a: str,
        model_b: str,
        subset_mask: Optional[np.ndarray] = None,
        n_permutations: int = 10000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Perform paired group-blocked model-swap permutation test across disjoint component groups."""
        rng = np.random.default_rng(random_seed)
        probs_a = self.get_model_mean_probs(model_a, subset_mask)
        probs_b = self.get_model_mean_probs(model_b, subset_mask)
        y = self.y_true[subset_mask] if subset_mask is not None else self.y_true
        grp_arr = self.group_ids[subset_mask] if subset_mask is not None else self.group_ids
        active_groups = np.unique(grp_arr)

        obs_delta_ap = float(average_precision_score(y, probs_a) - average_precision_score(y, probs_b))
        obs_delta_roc = float(roc_auc_score(y, probs_a) - roc_auc_score(y, probs_b))

        perm_deltas_ap = []
        n_grps = len(active_groups)

        for _ in range(n_permutations):
            swap_flags = rng.integers(0, 2, size=n_grps)
            swapped_a = probs_a.copy()
            swapped_b = probs_b.copy()

            for g_idx, g in enumerate(active_groups):
                if swap_flags[g_idx] == 1:
                    idx = np.where(grp_arr == g)[0]
                    swapped_a[idx] = probs_b[idx]
                    swapped_b[idx] = probs_a[idx]

            delta_ap_perm = average_precision_score(y, swapped_a) - average_precision_score(y, swapped_b)
            perm_deltas_ap.append(delta_ap_perm)

        perm_deltas_ap = np.array(perm_deltas_ap)
        extreme_count = int(np.count_nonzero(perm_deltas_ap >= obs_delta_ap))
        p_val_ap = float((extreme_count + 1) / (n_permutations + 1))

        return {
            "model_a": model_a,
            "model_b": model_b,
            "delta_ap": obs_delta_ap,
            "delta_roc_auc": obs_delta_roc,
            "p_value_raw": p_val_ap,
            "alternative": "greater",
            "extreme_permutations": extreme_count,
            "monte_carlo_plus_one": True,
            "n_permutations": n_permutations,
            "n_groups": len(active_groups),
            "n_candidates": len(y),
        }

    def evaluate_all(self, output_json_path: str = "results/production_confirmatory_stats.json", output_results_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Compute full statistical report with Holm-Bonferroni corrected hypothesis tests."""
        target_path = output_results_path or output_json_path
        models = [
            "logistic_regression",
            "hist_gradient_boosting",
            "gcn",
            "gat",
            "graphsage",
            "gine",
            "scnn",
            "ccnn",
        ]

        model_stats = {}
        for m in models:
            model_stats[m] = self.compute_model_metrics(m)

        # Locked corrective primary comparison family:
        # RQ1: CCNN vs GINE on full cohort
        rq1 = self.run_group_permutation_test("ccnn", "gine")
        rq1.update(self.paired_group_bootstrap_delta("ccnn", "gine"))

        # RQ2: CCNN vs SCNN on k >= 4
        mask_k4 = self.cycle_lengths >= 4
        rq2 = self.run_group_permutation_test("ccnn", "scnn", subset_mask=mask_k4)
        rq2.update(self.paired_group_bootstrap_delta("ccnn", "scnn", subset_mask=mask_k4))

        # RQ3: Logistic Regression vs GINE
        rq3 = self.run_group_permutation_test("logistic_regression", "gine")
        rq3.update(self.paired_group_bootstrap_delta("logistic_regression", "gine"))

        raw_ps = [rq1["p_value_raw"], rq2["p_value_raw"]]
        adj_ps = holm_bonferroni_correction(raw_ps)

        rq1["p_value_adjusted"] = adj_ps[0]
        rq1["statistically_significant_alpha_0_05"] = bool(adj_ps[0] < 0.05)

        rq2["p_value_adjusted"] = adj_ps[1]
        rq2["statistically_significant_alpha_0_05"] = bool(adj_ps[1] < 0.05)

        rq3["p_value_adjusted"] = None
        rq3["statistically_significant_alpha_0_05"] = None
        rq3["analysis_role"] = "exploratory"
        rq1["analysis_role"] = "primary_confirmatory_family"
        rq2["analysis_role"] = "primary_confirmatory_family"

        rq1["raw_p_value"] = rq1["p_value_raw"]
        rq1["observed_delta_ap"] = rq1["delta_ap"]
        rq1["statistically_significant"] = rq1["statistically_significant_alpha_0_05"]

        rq2["raw_p_value"] = rq2["p_value_raw"]
        rq2["observed_delta_ap"] = rq2["delta_ap"]
        rq2["statistically_significant"] = rq2["statistically_significant_alpha_0_05"]

        rq3["raw_p_value"] = rq3["p_value_raw"]
        rq3["observed_delta_ap"] = rq3["delta_ap"]
        rq3["statistically_significant"] = None

        hyp_dict = {
            "RQ1_ccnn_vs_gine": rq1,
            "RQ1_CCNN_vs_GINE": rq1,
            "RQ2_ccnn_vs_scnn_kge4": rq2,
            "RQ2_CCNN_vs_SCNN_kge4": rq2,
            "RQ3_lr_vs_gine": rq3,
            "RQ3_LR_vs_GINE": rq3,
        }

        report = {
            "schema_version": "chunk19-1.0",
            "benchmark_dataset": "IBM AMLworld HI-Small",
            "total_candidates": len(self.candidate_ids),
            "total_groups": len(self.unique_groups),
            "seeds": self.seeds,
            "models": self.models,
            "provenance": self.provenance,
            "inference_protocol": {
                "seed_aggregation": "mean prediction per candidate before inference",
                "protected_unit": "manifest account-connected group",
                "bootstrap_resamples": 10000,
                "permutation_resamples": 10000,
                "permutation_alternative": "greater",
                "monte_carlo_plus_one": True,
                "holm_primary_family": ["RQ1_ccnn_vs_gine", "RQ2_ccnn_vs_scnn_kge4"],
                "exploratory_comparisons": ["RQ3_lr_vs_gine"],
            },
            "model_benchmark": model_stats,
            "benchmark_models": model_stats,
            "model_metrics": model_stats,
            "confirmatory_hypothesis_tests": hyp_dict,
            "hypothesis_tests": hyp_dict,
            "metrics": [
                {
                    "name": f"{model}_average_precision",
                    "value": values["average_precision"],
                    "sample_count": len(self.candidate_ids),
                }
                for model, values in model_stats.items()
            ],
        }

        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def execute_full_confirmatory_suite(self, output_json_path: str = "results/production_confirmatory_stats.json", output_results_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Convenience alias for full confirmatory statistical evaluation."""
        return self.evaluate_all(output_json_path=output_json_path, output_results_path=output_results_path, **kwargs)


if __name__ == "__main__":
    tester = KUSETHypothesisTester()
    res = tester.evaluate_all()
    print(json.dumps(res, indent=2))
