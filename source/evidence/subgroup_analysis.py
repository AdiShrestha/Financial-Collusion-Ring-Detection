"""Subgroup stratification and track robustness analyzer."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from source.evidence.statistics import compute_classification_metrics


class SubgroupAnalyzer:
    """Evaluates detection metrics stratified across AML typologies and dataset tracks."""

    TYPOLOGIES = ["CYCLE", "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "BIPARTITE", "NEGATIVE_CANDIDATE"]
    TRACKS = ["amlworld", "elliptic_actors"]

    def __init__(self):
        pass

    def evaluate_model_subgroup(
        self,
        model_predictions: Dict[str, List[Dict[str, Any]]],
        filter_key: str,
        filter_val: str,
    ) -> Dict[str, Any]:
        """Compute mean and std metrics across seeds for candidates matching filter_key == filter_val."""
        seed_metrics = []
        total_matching_candidates = 0

        for seed_str, cand_list in model_predictions.items():
            matching = [x for x in cand_list if x.get(filter_key) == filter_val]
            if not matching:
                continue

            total_matching_candidates = len(matching)
            y_trues = [x["y_true"] for x in matching]
            y_probs = [x["y_prob"] for x in matching]

            m = compute_classification_metrics(y_trues, y_probs)
            seed_metrics.append(m)

        if not seed_metrics:
            return {
                "count": 0,
                "status": "EMPTY_SUBGROUP",
                "mean_pr_auc": 0.0,
                "std_pr_auc": 0.0,
                "mean_roc_auc": 0.0,
                "std_roc_auc": 0.0,
                "mean_f1_macro": 0.0,
                "std_f1_macro": 0.0,
            }

        pr_scores = [m["pr_auc"] for m in seed_metrics]
        roc_scores = [m["roc_auc"] for m in seed_metrics]
        f1_scores = [m["f1_macro"] for m in seed_metrics]

        return {
            "count": total_matching_candidates,
            "status": "EVALUATED",
            "num_seeds": len(seed_metrics),
            "mean_pr_auc": float(np.mean(pr_scores)),
            "std_pr_auc": float(np.std(pr_scores)),
            "mean_roc_auc": float(np.mean(roc_scores)),
            "std_roc_auc": float(np.std(roc_scores)),
            "mean_f1_macro": float(np.mean(f1_scores)),
            "std_f1_macro": float(np.std(f1_scores)),
        }

    def analyze_subgroups(self, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete subgroup analysis across all typologies, tracks, and models."""
        preds = predictions_data.get("predictions", predictions_data)
        model_names = list(preds.keys())

        by_typology: Dict[str, Dict[str, Any]] = {}
        for typ in self.TYPOLOGIES:
            by_typology[typ] = {}
            for model_name in model_names:
                by_typology[typ][model_name] = self.evaluate_model_subgroup(
                    model_predictions=preds[model_name],
                    filter_key="typology",
                    filter_val=typ,
                )

        by_track: Dict[str, Dict[str, Any]] = {}
        for track in self.TRACKS:
            by_track[track] = {}
            for model_name in model_names:
                by_track[track][model_name] = self.evaluate_model_subgroup(
                    model_predictions=preds[model_name],
                    filter_key="dataset_track",
                    filter_val=track,
                )

        return {
            "by_typology": by_typology,
            "by_track": by_track,
            "models_evaluated": model_names,
            "typologies_evaluated": self.TYPOLOGIES,
            "tracks_evaluated": self.TRACKS,
        }
