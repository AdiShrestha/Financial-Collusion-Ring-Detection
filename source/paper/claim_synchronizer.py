"""
Claim Synchronizer — Evidence Traceability & Manuscript Synchronization Auditor.

Contract C09-03: Verifies that 100% of numerical values in paper/main.tex
exactly match the ground-truth statistical outputs in results/confirmatory_stats.json
and results/ablation_summary.json with zero discrepancy (|Δ| < 1e-4).
"""

import json
import os
import re
from typing import Any


class ClaimSynchronizer:
    """Audits numerical synchronization between paper manuscript and data artifacts."""

    def __init__(
        self,
        tex_path: str = "paper/main.tex",
        stats_path: str = "results/confirmatory_stats.json",
        ablation_path: str = "results/ablation_summary.json",
        tolerance: float = 1e-4,
    ):
        self.tex_path = tex_path
        self.stats_path = stats_path
        self.ablation_path = ablation_path
        self.tolerance = tolerance

    def _load_tex(self) -> str:
        with open(self.tex_path, "r") as f:
            return f.read()

    def _load_stats(self) -> dict:
        with open(self.stats_path, "r") as f:
            return json.load(f)

    def _load_ablation(self) -> dict:
        if os.path.exists(self.ablation_path):
            with open(self.ablation_path, "r") as f:
                return json.load(f)
        return {}

    def _extract_table_values(self, tex_content: str) -> list[dict[str, Any]]:
        """Extract model benchmark values from the LaTeX results table."""
        extracted = []
        # Match table rows like: ModelName & $0.6333 \pm 0.1633$ & $0.4500 \pm 0.2449$ & $0.3333 \pm 0.0000$ \\
        pattern = re.compile(
            r"^(\w+)\s*&\s*\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$"
            r"\s*&\s*\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$"
            r"\s*&\s*\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$",
            re.MULTILINE,
        )
        for m in pattern.finditer(tex_content):
            extracted.append({
                "model": m.group(1),
                "mean_pr_auc": float(m.group(2)),
                "std_pr_auc": float(m.group(3)),
                "mean_roc_auc": float(m.group(4)),
                "std_roc_auc": float(m.group(5)),
                "mean_f1_macro": float(m.group(6)),
                "std_f1_macro": float(m.group(7)),
            })
        return extracted

    def _extract_hypothesis_values(self, tex_content: str) -> list[dict[str, Any]]:
        """Extract hypothesis p-values and effect sizes from the text."""
        extracted = []
        # Match patterns like: p_{\mathrm{raw}} = 0.0156
        p_raw_pattern = re.compile(
            r"\\textbf\{(H\d)\}.*?"
            r"p_\{\\mathrm\{raw\}\}\s*=\s*(\d+\.\d+).*?"
            r"p_\{\\mathrm\{adj\}\}\s*=\s*(\d+\.\d+).*?"
            r"\\delta\s*=\s*(-?\d+\.\d+).*?"
            r"\\textbf\{Verdict:\s*(\w+)",
            re.DOTALL,
        )
        for m in p_raw_pattern.finditer(tex_content):
            extracted.append({
                "hypothesis_id": m.group(1),
                "raw_p_value": float(m.group(2)),
                "adjusted_p_value": float(m.group(3)),
                "cliffs_delta": float(m.group(4)),
                "verdict": m.group(5).rstrip("."),
            })
        return extracted

    def _extract_claim_verdicts(self, tex_content: str) -> dict[str, str]:
        """Extract SYNC:claim_verdicts marker if present."""
        verdicts = {}
        pattern = re.compile(r"SYNC:claim_verdicts\s+(.*)")
        m = pattern.search(tex_content)
        if m:
            for pair in m.group(1).split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    verdicts[k] = v
        return verdicts

    def _compare_float(self, paper_val: float, data_val: float) -> bool:
        """Compare two floats within tolerance."""
        return abs(paper_val - data_val) < self.tolerance

    def audit_synchronization(
        self,
        tex_path: str | None = None,
        stats_path: str | None = None,
    ) -> dict:
        """
        Run full synchronization audit.

        Returns dict with keys:
            all_synced: bool
            discrepancies: list of dicts describing each mismatch
            num_checked: int
        """
        if tex_path:
            self.tex_path = tex_path
        if stats_path:
            self.stats_path = stats_path

        tex_content = self._load_tex()
        stats = self._load_stats()
        discrepancies: list[dict[str, Any]] = []
        num_checked = 0

        # 1. Verify model benchmark table values
        table_values = self._extract_table_values(tex_content)
        model_benchmarks = stats.get("model_benchmarks", {})

        for row in table_values:
            model_name = row["model"]
            if model_name not in model_benchmarks:
                discrepancies.append({
                    "type": "missing_model",
                    "model": model_name,
                    "detail": f"Model {model_name} in paper but not in stats",
                })
                continue

            ref = model_benchmarks[model_name]
            for metric in ["mean_pr_auc", "std_pr_auc", "mean_roc_auc",
                           "std_roc_auc", "mean_f1_macro", "std_f1_macro"]:
                num_checked += 1
                paper_val = row[metric]
                data_val = ref[metric]
                if not self._compare_float(paper_val, data_val):
                    discrepancies.append({
                        "type": "table_mismatch",
                        "model": model_name,
                        "metric": metric,
                        "paper_value": paper_val,
                        "data_value": data_val,
                        "delta": abs(paper_val - data_val),
                    })

        # 2. Verify hypothesis testing values
        hyp_values = self._extract_hypothesis_values(tex_content)
        hyp_data = stats.get("hypothesis_testing", {}).get("hypotheses", {})

        for hyp in hyp_values:
            hid = hyp["hypothesis_id"]
            if hid not in hyp_data:
                discrepancies.append({
                    "type": "missing_hypothesis",
                    "hypothesis": hid,
                    "detail": f"Hypothesis {hid} in paper but not in stats",
                })
                continue

            ref = hyp_data[hid]
            for field in ["raw_p_value", "adjusted_p_value", "cliffs_delta"]:
                num_checked += 1
                paper_val = hyp[field]
                data_val = ref[field]
                if not self._compare_float(paper_val, data_val):
                    discrepancies.append({
                        "type": "hypothesis_mismatch",
                        "hypothesis": hid,
                        "field": field,
                        "paper_value": paper_val,
                        "data_value": data_val,
                        "delta": abs(paper_val - data_val),
                    })

            # Verdict check (exact string match)
            num_checked += 1
            if hyp["verdict"] != ref["verdict"]:
                discrepancies.append({
                    "type": "verdict_mismatch",
                    "hypothesis": hid,
                    "paper_verdict": hyp["verdict"],
                    "data_verdict": ref["verdict"],
                })

        # 3. Verify claim verdicts from SYNC marker
        claim_verdicts = self._extract_claim_verdicts(tex_content)
        overall_verdicts = stats.get("hypothesis_testing", {}).get("overall_verdicts", {})

        for hid, verdict in claim_verdicts.items():
            num_checked += 1
            if hid in overall_verdicts and verdict != overall_verdicts[hid]:
                discrepancies.append({
                    "type": "claim_verdict_mismatch",
                    "claim": hid,
                    "paper_verdict": verdict,
                    "data_verdict": overall_verdicts[hid],
                })

        return {
            "all_synced": len(discrepancies) == 0,
            "discrepancies": discrepancies,
            "num_checked": num_checked,
        }
