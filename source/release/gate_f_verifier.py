"""Evidence-derived Gate F verifier for the Chunk 19 release package."""

import json
from pathlib import Path
from typing import Any

from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer


class GateFVerifier:
    GATE_REPORTS = {
        name: (f"project/{name}_report.json", f"{name}_status", f"{name.upper()}_PASS")
        for name in ("gate_b", "gate_c", "gate_d", "gate_e")
    }
    RELEASE_FILES = (
        "artifacts/candidates/candidates.parquet", "artifacts/splits/fold_manifest.json",
        "artifacts/training/training_history.json", "artifacts/predictions/oof_predictions.parquet",
        "results/production_confirmatory_stats.json", "paper/kuset_main.tex",
        "paper/generated/tab_cohort_stats.tex", "paper/generated/tab_model_benchmark.tex",
        "paper/generated/tab_hypothesis_tests.tex", "paper/generated/tab_ablation.tex",
        "paper/figures/fig_pr_curves.png", "paper/figures/fig_k_ablation.png", "release/replicate.py",
    )

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

    def _check_gate_chain(self):
        checks = []
        for name, (relative, key, expected) in self.GATE_REPORTS.items():
            path = self.project_root / relative
            actual = None
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    report = json.load(handle)
                actual = report.get(key, report.get("status"))
            checks.append({"check": name, "passed": actual == expected, "detail": f"expected={expected}, actual={actual}"})
        return all(item["passed"] for item in checks), checks

    def _check_release_files(self):
        checks = []
        for relative in self.RELEASE_FILES:
            path = self.project_root / relative
            passed = path.exists() and path.stat().st_size > 0
            checks.append({"check": relative, "passed": passed, "detail": "present and non-empty" if passed else "missing or empty"})
        return all(item["passed"] for item in checks), checks

    def _check_claims(self):
        tex = self.project_root / "paper/kuset_main.tex"
        stats = self.project_root / "results/production_confirmatory_stats.json"
        if not tex.exists() or not stats.exists():
            return False, [{"check": "claim_sync", "passed": False, "detail": "inputs missing"}]
        audit = KUSETClaimSynchronizer(str(tex), str(stats)).audit_claim_synchronization()
        return bool(audit["all_synchronized"]), [{"check": "claim_sync", "passed": bool(audit["all_synchronized"]), "detail": audit}]

    def verify_gate_f(self, output_path: str = "project/gate_f_report.json") -> dict[str, Any]:
        chain_ok, chain = self._check_gate_chain()
        files_ok, files = self._check_release_files()
        claims_ok, claims = self._check_claims()
        all_passed = chain_ok and files_ok and claims_ok
        result = {
            "gate_f_status": "GATE_F_PASS" if all_passed else "GATE_F_FAIL",
            "all_sections_passed": all_passed,
            "sections": {
                "gate_chain": {"passed": chain_ok, "subchecks": chain},
                "release_artifacts": {"passed": files_ok, "subchecks": files},
                "claim_synchronization": {"passed": claims_ok, "subchecks": claims},
            },
            "proof_boundary": "artifact presence, upstream gate status, and manuscript/result synchronization; not publication acceptance or raw-data reacquisition",
        }
        target = self.project_root / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


if __name__ == "__main__":
    result = GateFVerifier().verify_gate_f()
    print(json.dumps(result, indent=2))
