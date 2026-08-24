"""Formal Gate D (Pre-Registration & Protocol Lock) Verification Engine."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.evidence.protocol_lock import ProtocolLock
from source.evidence.statistics import (
    compute_classification_metrics,
    compute_cliffs_delta,
    evaluate_hypothesis_verdict,
    holm_bonferroni_correction,
    paired_wilcoxon_test,
)


class GateDVerifier:
    """Formal verification engine for Gate D Pre-Registration & Protocol Lock."""

    EXPECTED_MODELS = [
        "GCNBaseline",
        "GATBaseline",
        "GraphSAGEBaseline",
        "SimplicialComplexNet",
        "CellularComplexNet",
        "TopoRingNet",
    ]

    def __init__(
        self,
        output_report_path: str = "project/gate_d_report.json",
        manifest_path: str = "data/manifests/split_manifest.json",
        pre_reg_path: str = "project/PRE_REGISTRATION.md",
        hyperparams_path: str = "data/cache/tuned_hyperparameters.json",
    ):
        self.output_report_path = output_report_path
        self.manifest_path = manifest_path
        self.pre_reg_path = pre_reg_path
        self.hyperparams_path = hyperparams_path
        self.lock = ProtocolLock(manifest_path=manifest_path, pre_reg_path=pre_reg_path)

    def verify_pre_registration(self) -> Tuple[bool, List[str]]:
        """Verify presence and completeness of PRE_REGISTRATION.md."""
        return self.lock.verify_pre_registration_document()

    def verify_test_partition_hash(self) -> Tuple[bool, str]:
        """Verify cryptographic SHA-256 test partition hash."""
        return self.lock.verify_test_partition_hash()

    def verify_hyperparameter_freeze(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify that tuned hyperparameters are frozen for all 6 candidate models."""
        if not os.path.exists(self.hyperparams_path):
            return False, {"error": f"Hyperparameters file not found at '{self.hyperparams_path}'"}

        try:
            with open(self.hyperparams_path, "r", encoding="utf-8") as f:
                hp_data = json.load(f)

            missing_models = [m for m in self.EXPECTED_MODELS if m not in hp_data]
            if missing_models:
                return False, {"missing_models": missing_models, "present": list(hp_data.keys())}

            return True, {"frozen_models": self.EXPECTED_MODELS}
        except Exception as e:
            return False, {"error": str(e)}

    def verify_statistics_engine(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify functionality of statistical inference and step-down correction engine."""
        try:
            scores_a = [0.90, 0.88, 0.92, 0.94, 0.91, 0.89]
            scores_b = [0.75, 0.72, 0.78, 0.80, 0.74, 0.71]

            stat, p_val = paired_wilcoxon_test(scores_a, scores_b)
            d_res = compute_cliffs_delta(scores_a, scores_b, num_bootstraps=200, seed=42)
            hb_res = holm_bonferroni_correction([p_val, 0.04, 0.01], alpha=0.05)

            is_valid = (
                p_val < 0.05
                and d_res["delta"] > 0.0
                and len(hb_res) == 3
                and hb_res[0]["adjusted_p_value"] <= hb_res[1]["adjusted_p_value"]
            )

            return is_valid, {
                "wilcoxon_p_value": p_val,
                "cliffs_delta": d_res["delta"],
                "cliffs_magnitude": d_res["magnitude"],
                "holm_bonferroni_verified": True,
            }
        except Exception as e:
            return False, {"error": str(e)}

    def verify_gate_d(self) -> Dict[str, Any]:
        """Execute complete Gate D verification protocol and write gate_d_report.json."""
        pr_ok, pr_msgs = self.verify_pre_registration()
        hash_ok, hash_str = self.verify_test_partition_hash()
        hp_ok, hp_details = self.verify_hyperparameter_freeze()
        stat_ok, stat_details = self.verify_statistics_engine()

        all_passed = pr_ok and hash_ok and hp_ok and stat_ok
        gate_status = "GATE_D_PASS" if all_passed else "GATE_D_FAIL"

        report = {
            "gate_d_status": gate_status,
            "all_subchecks_passed": all_passed,
            "subchecks": {
                "pre_registration_document": {
                    "passed": pr_ok,
                    "errors": pr_msgs,
                },
                "test_partition_immutability": {
                    "passed": hash_ok,
                    "test_hash": hash_str,
                    "expected_hash": ProtocolLock.LOCKED_TEST_HASH,
                },
                "hyperparameter_freeze": {
                    "passed": hp_ok,
                    "details": hp_details,
                },
                "statistics_engine": {
                    "passed": stat_ok,
                    "details": stat_details,
                },
            },
            "protocol_specifications": {
                "evaluation_seeds": ProtocolLock.LOCKED_SEEDS,
                "hypotheses": ProtocolLock.HYPOTHESES,
                "alpha": ProtocolLock.ALPHA,
                "cliffs_delta_threshold": ProtocolLock.CLIFFS_DELTA_THRESHOLD,
                "verdict_classes": ["SUPPORTED", "FALSIFIED", "INCONCLUSIVE"],
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(self.output_report_path)), exist_ok=True)
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    verifier = GateDVerifier()
    res = verifier.verify_gate_d()
    print(f"Gate D Status: {res['gate_d_status']}")

