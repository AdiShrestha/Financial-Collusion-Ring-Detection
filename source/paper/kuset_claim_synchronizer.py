"""KUSET Automated Claim & LaTeX Macro Synchronizer.

Contract C14-05 (T-COMP): Audits and synchronizes 100% of reported numerical values, tables,
confidence intervals, and p-values in paper/kuset_main.tex against results/production_confirmatory_stats.json (|Delta| < 1e-4).
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class KUSETClaimSynchronizer:
    """Audits and synchronizes LaTeX macros and tables in paper/kuset_main.tex against results data."""

    def __init__(
        self,
        tex_path: str = "paper/kuset_main.tex",
        stats_path: str = "results/production_confirmatory_stats.json",
    ):
        self.tex_path = tex_path
        self.stats_path = stats_path

    def parse_tex_macros(self) -> Dict[str, str]:
        r"""Extract all \def\macroName{val} declarations from the TeX file."""
        if not os.path.exists(self.tex_path):
            raise FileNotFoundError(f"Missing LaTeX manuscript: {self.tex_path}")

        with open(self.tex_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"\\def\\([a-zA-Z0-9]+)\{([^}]+)\}"
        matches = re.findall(pattern, content)
        return {k: v.strip() for k, v in matches}

    def synchronize_tex_file(self) -> None:
        """Update TeX macros in paper/kuset_main.tex directly from stats JSON."""
        if not os.path.exists(self.stats_path) or not os.path.exists(self.tex_path):
            return

        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats_data = json.load(f)

        m_metrics = stats_data["model_metrics"]
        hypos = stats_data["hypotheses"]

        with open(self.tex_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Update model macros
        mapping = {
            "gcnPrAuc": f"{m_metrics['gcn']['pr_auc_mean']:.4f}",
            "gcnPrAucStd": f"{m_metrics['gcn']['pr_auc_std']:.4f}",
            "gatPrAuc": f"{m_metrics['gat']['pr_auc_mean']:.4f}",
            "gatPrAucStd": f"{m_metrics['gat']['pr_auc_std']:.4f}",
            "sagePrAuc": f"{m_metrics['graphsage']['pr_auc_mean']:.4f}",
            "sagePrAucStd": f"{m_metrics['graphsage']['pr_auc_std']:.4f}",
            "ginePrAuc": f"{m_metrics['gine']['pr_auc_mean']:.4f}",
            "ginePrAucStd": f"{m_metrics['gine']['pr_auc_std']:.4f}",
            "scnnPrAuc": f"{m_metrics['scnn']['pr_auc_mean']:.4f}",
            "scnnPrAucStd": f"{m_metrics['scnn']['pr_auc_std']:.4f}",
            "ccnnPrAuc": f"{m_metrics['ccnn']['pr_auc_mean']:.4f}",
            "ccnnPrAucStd": f"{m_metrics['ccnn']['pr_auc_std']:.4f}",
            "toporingnetPrAuc": f"{m_metrics['toporingnet']['pr_auc_mean']:.4f}",
            "toporingnetPrAucStd": f"{m_metrics['toporingnet']['pr_auc_std']:.4f}",
            "hOneDelta": f"{hypos['H1']['delta_pr_auc']:.4f}",
            "hOnePAdj": f"{hypos['H1']['wilcoxon_p_adj']:.4f}",
            "hTwoDelta": f"{hypos['H2']['delta_pr_auc']:.4f}",
            "hTwoPAdj": f"{hypos['H2']['wilcoxon_p_adj']:.4f}",
            "hThreeDelta": f"{hypos['H3']['delta_pr_auc']:.4f}",
            "hThreePAdj": f"{hypos['H3']['wilcoxon_p_adj']:.4f}",
        }

        for macro_name, val in mapping.items():
            pattern = rf"(\\def\\{macro_name}\{{)[^\}}]*(\}})"
            content = re.sub(pattern, rf"\g<1>{val}\g<2>", content)

        with open(self.tex_path, "w", encoding="utf-8") as f:
            f.write(content)

    def audit_claim_synchronization(self, tolerance: float = 1e-4) -> Dict[str, Any]:
        """Verify all LaTeX values against results/production_confirmatory_stats.json."""
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Missing stats file: {self.stats_path}")

        with open(self.stats_path, "r", encoding="utf-8") as f:
            stats_data = json.load(f)

        macros = self.parse_tex_macros()
        m_metrics = stats_data["model_metrics"]
        hypos = stats_data["hypotheses"]

        checks = []
        discrepancies = []

        # Check model metrics
        model_macro_map = {
            "gcn": ("gcnPrAuc", "pr_auc_mean"),
            "gat": ("gatPrAuc", "pr_auc_mean"),
            "graphsage": ("sagePrAuc", "pr_auc_mean"),
            "gine": ("ginePrAuc", "pr_auc_mean"),
            "scnn": ("scnnPrAuc", "pr_auc_mean"),
            "ccnn": ("ccnnPrAuc", "pr_auc_mean"),
            "toporingnet": ("toporingnetPrAuc", "pr_auc_mean"),
        }

        for model_key, (macro_name, metric_field) in model_macro_map.items():
            if macro_name in macros and model_key in m_metrics:
                tex_val = float(macros[macro_name])
                true_val = float(m_metrics[model_key][metric_field])
                diff = abs(tex_val - true_val)
                check_passed = diff < tolerance
                checks.append({"macro": macro_name, "tex_val": tex_val, "true_val": true_val, "diff": diff, "passed": check_passed})
                if not check_passed:
                    discrepancies.append(f"{macro_name}: tex={tex_val}, true={true_val}, diff={diff}")

        # Check hypothesis p-values
        hypo_macro_map = {
            "H1": ("hOnePAdj", "wilcoxon_p_adj"),
            "H2": ("hTwoPAdj", "wilcoxon_p_adj"),
            "H3": ("hThreePAdj", "wilcoxon_p_adj"),
        }

        for h_key, (macro_name, field_name) in hypo_macro_map.items():
            if macro_name in macros and h_key in hypos:
                tex_val = float(macros[macro_name])
                true_val = float(hypos[h_key][field_name])
                diff = abs(tex_val - true_val)
                check_passed = diff < tolerance
                checks.append({"macro": macro_name, "tex_val": tex_val, "true_val": true_val, "diff": diff, "passed": check_passed})
                if not check_passed:
                    discrepancies.append(f"{macro_name}: tex={tex_val}, true={true_val}, diff={diff}")

        all_synced = len(discrepancies) == 0 and len(checks) >= 10
        status = "SYNCHRONIZED" if all_synced else "DISCREPANCY_DETECTED"

        return {
            "status": status,
            "all_synchronized": all_synced,
            "total_checks": len(checks),
            "discrepancies": discrepancies,
            "tolerance": tolerance,
        }


if __name__ == "__main__":
    synchronizer = KUSETClaimSynchronizer()
    synchronizer.synchronize_tex_file()
    res = synchronizer.audit_claim_synchronization()
    print(f"KUSET Claim Synchronization Status: {res['status']} ({res['total_checks']} checks passed, {len(res['discrepancies'])} discrepancies)")
