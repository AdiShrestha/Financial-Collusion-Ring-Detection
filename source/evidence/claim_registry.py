"""Structured evidence claim registry and confirmatory statistical report compiler."""

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from source.evidence.hypothesis_tester import ConfirmatoryHypothesisTester
from source.evidence.subgroup_analysis import SubgroupAnalyzer


@dataclass
class ClaimRecord:
    """Represents a formal confirmatory scientific claim with statistical provenance."""

    claim_id: str
    hypothesis_id: str
    statement: str
    primary_metric: str
    p_value_raw: float
    p_value_adj: float
    effect_size_delta: float
    effect_ci_low: float
    effect_ci_high: float
    verdict: str
    sample_size: int
    reproduction_command: str


class ClaimRegistry:
    """Manages formal empirical claims mapped to statistical test results."""

    def __init__(self):
        self._claims: Dict[str, ClaimRecord] = {}

    def register_claim(self, record: ClaimRecord) -> None:
        """Register a formal claim record, validating required fields and reproduction command."""
        if not record.reproduction_command or not record.reproduction_command.strip():
            raise ValueError(f"BLOCKED — CLAIM REGISTRY INTEGRITY FAILURE: Claim {record.claim_id} missing mandatory reproduction command")
        if record.verdict not in {"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"}:
            raise ValueError(f"BLOCKED — CLAIM REGISTRY INTEGRITY FAILURE: Invalid verdict {record.verdict} in claim {record.claim_id}")

        self._claims[record.claim_id] = record

    def get_claim(self, claim_id: str) -> ClaimRecord:
        """Retrieve registered claim record by ID."""
        if claim_id not in self._claims:
            raise KeyError(f"Claim ID {claim_id} not registered")
        return self._claims[claim_id]

    def all_claims(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered claims as serializable dictionaries."""
        return {cid: asdict(rec) for cid, rec in self._claims.items()}

    def build_full_confirmatory_package(
        self,
        predictions_path: str = "runs/confirmatory/predictions.json",
        output_path: str = "results/confirmatory_stats.json",
    ) -> Dict[str, Any]:
        """Compile complete confirmatory statistics artifact including hypothesis tests, subgroups, and claims."""
        with open(predictions_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        # 1. Hypothesis Testing
        hyp_tester = ConfirmatoryHypothesisTester(alpha=0.05, min_effect=0.147)
        hyp_results = hyp_tester.evaluate_all_hypotheses(pred_data)

        # 2. Subgroup Analysis
        sub_analyzer = SubgroupAnalyzer()
        sub_results = sub_analyzer.analyze_subgroups(pred_data)

        # 3. Model Benchmark Summary across test split
        preds = pred_data.get("predictions", {})
        model_summary = {}
        for model_name, seed_dict in preds.items():
            seed_metrics = []
            for s_str, cand_list in seed_dict.items():
                y_trues = [x["y_true"] for x in cand_list]
                y_probs = [x["y_prob"] for x in cand_list]
                from source.evidence.statistics import compute_classification_metrics
                m = compute_classification_metrics(y_trues, y_probs)
                seed_metrics.append(m)

            pr_scores = [m["pr_auc"] for m in seed_metrics]
            roc_scores = [m["roc_auc"] for m in seed_metrics]
            f1_scores = [m["f1_macro"] for m in seed_metrics]

            import numpy as np
            model_summary[model_name] = {
                "mean_pr_auc": float(np.mean(pr_scores)),
                "std_pr_auc": float(np.std(pr_scores)),
                "mean_roc_auc": float(np.mean(roc_scores)),
                "std_roc_auc": float(np.std(roc_scores)),
                "mean_f1_macro": float(np.mean(f1_scores)),
                "std_f1_macro": float(np.std(f1_scores)),
                "num_seeds": len(seed_metrics),
            }

        # 4. Register formal claims CLM-001 through CLM-004
        h1_info = hyp_results["hypotheses"]["H1"]
        h2_info = hyp_results["hypotheses"]["H2"]
        h3_info = hyp_results["hypotheses"]["H3"]
        h4_info = hyp_results["hypotheses"]["H4"]

        num_cands = pred_data.get("metadata", {}).get("num_test_candidates", 28)

        self.register_claim(ClaimRecord(
            claim_id="CLM-001",
            hypothesis_id="H1",
            statement="TopoRingNet significantly outperforms the strongest GNN baseline on test split PR-AUC.",
            primary_metric="pr_auc",
            p_value_raw=h1_info["raw_p_value"],
            p_value_adj=h1_info["adjusted_p_value"],
            effect_size_delta=h1_info["cliffs_delta"],
            effect_ci_low=h1_info["cliffs_delta_ci_lower"],
            effect_ci_high=h1_info["cliffs_delta_ci_upper"],
            verdict=h1_info["verdict"],
            sample_size=num_cands,
            reproduction_command="python3 -m pytest tests/unit/test_hypothesis_tester.py",
        ))

        self.register_claim(ClaimRecord(
            claim_id="CLM-002",
            hypothesis_id="H2",
            statement="CellularComplexNet significantly outperforms SimplicialComplexNet on k>=4 polygonal cycles on F1-Macro.",
            primary_metric="f1_macro",
            p_value_raw=h2_info["raw_p_value"],
            p_value_adj=h2_info["adjusted_p_value"],
            effect_size_delta=h2_info["cliffs_delta"],
            effect_ci_low=h2_info["cliffs_delta_ci_lower"],
            effect_ci_high=h2_info["cliffs_delta_ci_upper"],
            verdict=h2_info["verdict"],
            sample_size=num_cands,
            reproduction_command="python3 -m pytest tests/unit/test_hypothesis_tester.py",
        ))

        self.register_claim(ClaimRecord(
            claim_id="CLM-003",
            hypothesis_id="H3",
            statement="TopoRingNet significantly outperforms CellularComplexNet structural ablation on PR-AUC.",
            primary_metric="pr_auc",
            p_value_raw=h3_info["raw_p_value"],
            p_value_adj=h3_info["adjusted_p_value"],
            effect_size_delta=h3_info["cliffs_delta"],
            effect_ci_low=h3_info["cliffs_delta_ci_lower"],
            effect_ci_high=h3_info["cliffs_delta_ci_upper"],
            verdict=h3_info["verdict"],
            sample_size=num_cands,
            reproduction_command="python3 -m pytest tests/unit/test_hypothesis_tester.py",
        ))

        self.register_claim(ClaimRecord(
            claim_id="CLM-004",
            hypothesis_id="H4",
            statement="TopoRingNet achieves superior detection across both IBM AMLworld and Elliptic++ tracks on PR-AUC.",
            primary_metric="pr_auc",
            p_value_raw=h4_info["raw_p_value"],
            p_value_adj=h4_info["adjusted_p_value"],
            effect_size_delta=h4_info["cliffs_delta"],
            effect_ci_low=h4_info["cliffs_delta_ci_lower"],
            effect_ci_high=h4_info["cliffs_delta_ci_upper"],
            verdict=h4_info["verdict"],
            sample_size=num_cands,
            reproduction_command="python3 -m pytest tests/unit/test_hypothesis_tester.py",
        ))

        package = {
            "metadata": {
                "test_split_sha256": pred_data.get("metadata", {}).get("test_split_sha256", ""),
                "seeds": pred_data.get("metadata", {}).get("seeds", [42, 43, 44, 45, 46]),
                "num_test_candidates": num_cands,
                "claims_registered": len(self._claims),
            },
            "model_benchmarks": model_summary,
            "hypothesis_testing": hyp_results,
            "subgroup_analysis": sub_results,
            "claim_registry": self.all_claims(),
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(package, f, indent=2)

        return package
