"""Confirmatory paired hypothesis testing engine with Holm-Bonferroni correction."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from source.evidence.protocol_lock import ProtocolLock
from source.evidence.statistics import (
    compute_classification_metrics,
    compute_cliffs_delta,
    evaluate_hypothesis_verdict,
    holm_bonferroni_correction,
    paired_wilcoxon_test,
)


class ConfirmatoryHypothesisTester:
    """Evaluates pre-registered hypotheses H1–H4 on confirmatory test split predictions."""

    HYPOTHESIS_IDS = ["H1", "H2", "H3", "H4"]

    def __init__(self, alpha: float = 0.05, min_effect: float = 0.147):
        self.alpha = alpha
        self.min_effect = min_effect

    def extract_group_metrics(
        self,
        predictions_by_model: Dict[str, Dict[str, List[Dict[str, Any]]]],
        model_name: str,
        filter_typologies: Optional[Sequence[str]] = None,
        filter_track: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each candidate group across seeds for specified model."""
        model_seeds = predictions_by_model.get(model_name, {})
        group_metrics: Dict[str, List[Dict[str, float]]] = {}

        for seed_str, cand_list in model_seeds.items():
            # Partition by group_id
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for item in cand_list:
                if filter_typologies and item.get("typology") not in filter_typologies:
                    continue
                if filter_track and item.get("dataset_track") != filter_track:
                    continue

                grp = item["group_id"]
                if grp not in groups:
                    groups[grp] = []
                groups[grp].append(item)

            for grp, items in groups.items():
                y_trues = [x["y_true"] for x in items]
                y_probs = [x["y_prob"] for x in items]
                m = compute_classification_metrics(y_trues, y_probs)
                if grp not in group_metrics:
                    group_metrics[grp] = []
                group_metrics[grp].append(m)

        # Average across seeds per group
        avg_group_metrics: Dict[str, Dict[str, float]] = {}
        for grp, m_list in group_metrics.items():
            if not m_list:
                continue
            avg_group_metrics[grp] = {
                "pr_auc": float(np.mean([x["pr_auc"] for x in m_list])),
                "f1_macro": float(np.mean([x["f1_macro"] for x in m_list])),
                "roc_auc": float(np.mean([x["roc_auc"] for x in m_list])),
            }

        return avg_group_metrics

    def evaluate_all_hypotheses(self, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute confirmatory hypothesis evaluation on test prediction archive."""
        preds = predictions_data.get("predictions", predictions_data)

        # Determine strongest baseline GNN on PR-AUC
        baseline_names = ["GCNBaseline", "GATBaseline", "GraphSAGEBaseline"]
        baseline_pr_scores = {}
        for b in baseline_names:
            grp_m = self.extract_group_metrics(preds, b)
            if grp_m:
                baseline_pr_scores[b] = float(np.mean([v["pr_auc"] for v in grp_m.values()]))
            else:
                baseline_pr_scores[b] = 0.0

        strongest_gnn = max(baseline_pr_scores, key=baseline_pr_scores.get) if baseline_pr_scores else "GraphSAGEBaseline"

        # --- H1: TopoRingNet vs strongest GNN on PR-AUC ---
        topo_h1 = self.extract_group_metrics(preds, "TopoRingNet")
        base_h1 = self.extract_group_metrics(preds, strongest_gnn)
        common_h1 = sorted(list(set(topo_h1.keys()).intersection(set(base_h1.keys()))))
        scores_topo_h1 = [topo_h1[g]["pr_auc"] for g in common_h1]
        scores_base_h1 = [base_h1[g]["pr_auc"] for g in common_h1]
        w_stat_h1, p_val_h1 = paired_wilcoxon_test(scores_topo_h1, scores_base_h1)
        cliff_h1 = compute_cliffs_delta(scores_topo_h1, scores_base_h1, alpha=self.alpha)

        # --- H2: CellularComplexNet vs SimplicialComplexNet on F1-Macro for k >= 4 polygonal cycles ---
        poly_typologies = ["CYCLE", "NEGATIVE_CANDIDATE"]
        cell_h2 = self.extract_group_metrics(preds, "CellularComplexNet", filter_typologies=poly_typologies)
        simp_h2 = self.extract_group_metrics(preds, "SimplicialComplexNet", filter_typologies=poly_typologies)
        common_h2 = sorted(list(set(cell_h2.keys()).intersection(set(simp_h2.keys()))))
        scores_cell_h2 = [cell_h2[g]["f1_macro"] for g in common_h2]
        scores_simp_h2 = [simp_h2[g]["f1_macro"] for g in common_h2]
        w_stat_h2, p_val_h2 = paired_wilcoxon_test(scores_cell_h2, scores_simp_h2)
        cliff_h2 = compute_cliffs_delta(scores_cell_h2, scores_simp_h2, alpha=self.alpha)

        # --- H3: TopoRingNet vs CellularComplexNet (structural ablation) on PR-AUC ---
        cell_h3 = self.extract_group_metrics(preds, "CellularComplexNet")
        common_h3 = sorted(list(set(topo_h1.keys()).intersection(set(cell_h3.keys()))))
        scores_topo_h3 = [topo_h1[g]["pr_auc"] for g in common_h3]
        scores_cell_h3 = [cell_h3[g]["pr_auc"] for g in common_h3]
        w_stat_h3, p_val_h3 = paired_wilcoxon_test(scores_topo_h3, scores_cell_h3)
        cliff_h3 = compute_cliffs_delta(scores_topo_h3, scores_cell_h3, alpha=self.alpha)

        # --- H4: TopoRingNet cross-track superiority (AMLworld and Elliptic++) ---
        topo_h4_aml = self.extract_group_metrics(preds, "TopoRingNet", filter_track="amlworld")
        base_h4_aml = self.extract_group_metrics(preds, strongest_gnn, filter_track="amlworld")
        topo_h4_ell = self.extract_group_metrics(preds, "TopoRingNet", filter_track="elliptic_actors")
        base_h4_ell = self.extract_group_metrics(preds, strongest_gnn, filter_track="elliptic_actors")

        common_h4_aml = sorted(list(set(topo_h4_aml.keys()).intersection(set(base_h4_aml.keys()))))
        common_h4_ell = sorted(list(set(topo_h4_ell.keys()).intersection(set(base_h4_ell.keys()))))

        scores_topo_h4 = [topo_h4_aml[g]["pr_auc"] for g in common_h4_aml] + [topo_h4_ell[g]["pr_auc"] for g in common_h4_ell]
        scores_base_h4 = [base_h4_aml[g]["pr_auc"] for g in common_h4_aml] + [base_h4_ell[g]["pr_auc"] for g in common_h4_ell]
        w_stat_h4, p_val_h4 = paired_wilcoxon_test(scores_topo_h4, scores_base_h4)
        cliff_h4 = compute_cliffs_delta(scores_topo_h4, scores_base_h4, alpha=self.alpha)

        # Apply Holm-Bonferroni correction across all 4 hypotheses
        raw_p_values = [p_val_h1, p_val_h2, p_val_h3, p_val_h4]
        corrected = holm_bonferroni_correction(raw_p_values, alpha=self.alpha)

        # Assign verdicts
        hypotheses_results = {}
        raw_tests = [
            ("H1", "TopoRingNet vs strongest GNN (PR-AUC)", w_stat_h1, p_val_h1, cliff_h1, "TopoRingNet", strongest_gnn),
            ("H2", "CellularComplexNet vs SimplicialComplexNet on k>=4 cycles (F1-Macro)", w_stat_h2, p_val_h2, cliff_h2, "CellularComplexNet", "SimplicialComplexNet"),
            ("H3", "TopoRingNet vs CellularComplexNet structural ablation (PR-AUC)", w_stat_h3, p_val_h3, cliff_h3, "TopoRingNet", "CellularComplexNet"),
            ("H4", "TopoRingNet cross-track superiority (PR-AUC)", w_stat_h4, p_val_h4, cliff_h4, "TopoRingNet", strongest_gnn),
        ]

        for i, (hid, desc, w_stat, p_raw, c_res, m_a, m_b) in enumerate(raw_tests):
            p_adj = corrected[i]["adjusted_p_value"]
            is_sig = corrected[i]["is_significant"]
            d_val = c_res["delta"]
            verdict = evaluate_hypothesis_verdict(
                p_adj=p_adj,
                delta=d_val,
                alpha=self.alpha,
                delta_threshold=self.min_effect,
            )

            hypotheses_results[hid] = {
                "hypothesis_id": hid,
                "description": desc,
                "model_a": m_a,
                "model_b": m_b,
                "primary_metric": "pr_auc" if hid != "H2" else "f1_macro",
                "raw_p_value": p_raw,
                "adjusted_p_value": p_adj,
                "wilcoxon_w_statistic": w_stat,
                "cliffs_delta": d_val,
                "cliffs_delta_ci_lower": c_res.get("ci_low", 0.0),
                "cliffs_delta_ci_upper": c_res.get("ci_high", 0.0),
                "cliffs_delta_interpretation": c_res.get("magnitude", "negligible"),
                "is_significant_after_correction": is_sig,
                "verdict": verdict,
            }

        return {
            "metadata": {
                "alpha": self.alpha,
                "min_effect_size": self.min_effect,
                "correction_method": "Holm-Bonferroni Step-Down",
                "hypotheses_count": len(self.HYPOTHESIS_IDS),
                "strongest_baseline_gnn": strongest_gnn,
            },
            "hypotheses": hypotheses_results,
            "overall_verdicts": {hid: hypotheses_results[hid]["verdict"] for hid in self.HYPOTHESIS_IDS},
        }
