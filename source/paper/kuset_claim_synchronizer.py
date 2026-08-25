"""Fail-closed synchronization of KUSET manuscript claims to canonical results."""

import json
import os
import re
from typing import Any, Dict


class KUSETClaimSynchronizer:
    """Synchronize result macros and reject known stale release claims."""

    def __init__(self, tex_path="paper/kuset_main.tex", stats_path="results/production_confirmatory_stats.json"):
        self.tex_path = tex_path
        self.stats_path = stats_path

    def _load(self):
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Missing statistics: {self.stats_path}")
        if not os.path.exists(self.tex_path):
            raise FileNotFoundError(f"Missing manuscript: {self.tex_path}")
        with open(self.stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)
        with open(self.tex_path, encoding="utf-8") as handle:
            tex = handle.read()
        return stats, tex

    def parse_tex_macros(self) -> Dict[str, str]:
        _, tex = self._load()
        return dict(re.findall(r"\\def\\([A-Za-z0-9]+)\{([^}]*)\}", tex))

    @staticmethod
    def expected_macros(stats: Dict[str, Any]) -> Dict[str, str]:
        models = stats["model_benchmark"]
        rq = stats["hypothesis_tests"]
        mapping = {
            "candidateCount": str(stats["total_candidates"]), "groupCount": str(stats["total_groups"]),
            "seedCount": str(len(stats["seeds"])), "checkpointCount": str(len(stats["seeds"]) * len(stats["models"]) * 5),
        }
        model_keys = {
            "lr": "logistic_regression", "hgb": "hist_gradient_boosting", "gcn": "gcn", "gat": "gat",
            "sage": "graphsage", "gine": "gine", "scnn": "scnn", "ccnn": "ccnn",
        }
        for prefix, key in model_keys.items():
            model = models[key]
            mapping[f"{prefix}AP"] = f"{model['average_precision']:.4f}"
            mapping[f"{prefix}APLo"] = f"{model['average_precision_ci_95'][0]:.3f}"
            mapping[f"{prefix}APHi"] = f"{model['average_precision_ci_95'][1]:.3f}"
            mapping[f"{prefix}ROC"] = f"{model['roc_auc']:.4f}"
            mapping[f"{prefix}FOne"] = f"{model['f1_score']:.4f}"
        for prefix, key in (("rqOne", "RQ1_ccnn_vs_gine"), ("rqTwo", "RQ2_ccnn_vs_scnn_kge4"), ("rqThree", "RQ3_lr_vs_gine")):
            result = rq[key]
            mapping[f"{prefix}Delta"] = f"{result['delta_ap']:.4f}"
            mapping[f"{prefix}RawP"] = f"{result['p_value_raw']:.4f}"
            if result["p_value_adjusted"] is not None:
                mapping[f"{prefix}AdjP"] = f"{result['p_value_adjusted']:.4f}"
        return mapping

    def synchronize_tex_file(self) -> None:
        stats, tex = self._load()
        for name, value in self.expected_macros(stats).items():
            pattern = rf"(\\def\\{re.escape(name)}\{{)[^}}]*(\}})"
            if not re.search(pattern, tex):
                raise ValueError(f"Required manuscript macro missing: {name}")
            tex = re.sub(pattern, rf"\g<1>{value}\g<2>", tex)
        with open(self.tex_path, "w", encoding="utf-8") as handle:
            handle.write(tex)

    def audit_claim_synchronization(self, tolerance: float = 1e-4) -> Dict[str, Any]:
        stats, tex = self._load()
        expected = self.expected_macros(stats)
        actual = dict(re.findall(r"\\def\\([A-Za-z0-9]+)\{([^}]*)\}", tex))
        discrepancies = [f"{name}: expected={value}, actual={actual.get(name)}" for name, value in expected.items() if actual.get(name) != value]
        banned_patterns = {
            "three_seed_claim": r"3 (?:evaluation |random )?seeds|seeds? \\?\{42, 43, 44\\?\}",
            "stale_p_0012": r"0\.0012", "stale_p_1108": r"0\.1108", "stale_p_0116": r"0\.0116",
            "stale_toporingnet_result": r"\\def\\toporingnet|TopoRingNet AP",
            "false_preregistration": r"pre-registered (?:hypotheses|confirmatory)",
            "false_clean_room": r"turnkey clean-room|clean-room replication",
            "fabricated_author": r"Antigravity Research Group|research@ku\.edu\.np",
        }
        banned_hits = [name for name, pattern in banned_patterns.items() if re.search(pattern, tex, re.I)]
        required = [
            "corrective analysis", "pattern-grounded", "synthetic", "18 protected", "five seeds",
            "cohort construction", "not a prospectively preregistered", "same-model self-adversarial", "Aditya Shrestha",
        ]
        missing = [phrase for phrase in required if phrase.lower() not in tex.lower()]
        for fragment in ("generated/tab_cohort_stats.tex", "generated/tab_model_benchmark.tex", "generated/tab_hypothesis_tests.tex", "generated/tab_ablation.tex"):
            if rf"\input{{{fragment}}}" not in tex:
                discrepancies.append(f"missing fragment input: {fragment}")
        passed = not discrepancies and not banned_hits and not missing
        return {
            "status": "SYNCHRONIZED" if passed else "DISCREPANCY_DETECTED", "all_synced": passed,
            "all_synchronized": passed, "total_checks": len(expected), "passed_checks": len(expected) - len(discrepancies),
            "discrepancies_count": len(discrepancies) + len(banned_hits) + len(missing), "discrepancies": discrepancies,
            "banned_hits": banned_hits, "missing_required_phrases": missing,
        }


if __name__ == "__main__":
    synchronizer = KUSETClaimSynchronizer()
    synchronizer.synchronize_tex_file()
    print(json.dumps(synchronizer.audit_claim_synchronization(), indent=2))
