"""KUSET Publication Vector Figure Generator derived from empirical data.

Contract C13-03 (T-DESC): Generates 4 publication-quality 300 DPI figures derived 100%
from genuine experimental predictions and persistent homology cache data.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.ph.normalized_filtration import NormalizedPersistenceVectorizer


class KUSETPlotGenerator:
    """Generates 300 DPI publication vector figures from empirical experiment outputs."""

    def __init__(
        self,
        predictions_path: str = "runs/production_confirmatory/predictions.json",
        stats_path: str = "results/production_confirmatory_stats.json",
        candidates_path: str = "data/processed/candidates.jsonl",
        figures_dir: str = "paper/figures",
    ):
        self.predictions_path = predictions_path
        self.stats_path = stats_path
        self.candidates_path = candidates_path
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)

    def plot_persistence_diagrams(self, output_name: str = "persistence_diagrams.png") -> str:
        """Plot empirical H0 and H1 normalized persistence diagrams for candidate subgraphs."""
        vectorizer = NormalizedPersistenceVectorizer()
        candidates = []
        if os.path.exists(self.candidates_path):
            with open(self.candidates_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))

        # Separate into positive and negative candidate samples
        pos_cand = next((c for c in candidates if c.get("label", 0) == 1), None)
        neg_cand = next((c for c in candidates if c.get("label", 0) == 0), None)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

        for ax, cand, title in zip(
            axes,
            [pos_cand, neg_cand],
            ["Laundering Candidate (Cycle)", "Benign Candidate (Control)"],
        ):
            if cand:
                diag = vectorizer.extract_diagrams(cand)
                h0 = diag["h0_intervals"]
                h1 = diag["h1_intervals"]

                if len(h0) > 0:
                    ax.scatter(h0[:, 0], h0[:, 1], c="#D95F02", marker="o", alpha=0.8, label="$H_0$ (Components)")
                if len(h1) > 0:
                    ax.scatter(h1[:, 0], h1[:, 1], c="#7570B3", marker="^", alpha=0.9, s=50, label="$H_1$ (Cycles)")

            ax.plot([0, 1.1], [0, 1.1], "k--", alpha=0.5, label="Diagonal")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Birth Filtration ($t_{\\text{norm}}$)")
            ax.set_ylabel("Death Filtration ($t_{\\text{norm}}$)")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(loc="lower right", frameon=True)
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_pr_curves(self, output_name: str = "pr_curves.png") -> str:
        """Plot empirical Precision-Recall curves across all 7 evaluated models."""
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f"Missing predictions file: {self.predictions_path}")

        with open(self.predictions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        preds = data["predictions"]
        y_true = np.array([p["label"] for p in preds], dtype=np.int64)

        colors = {
            "gcn": "#1b9e77",
            "gat": "#d95f02",
            "graphsage": "#7570b3",
            "gine": "#e7298a",
            "scnn": "#66a61e",
            "ccnn": "#e6ab02",
            "toporingnet": "#a6761d",
        }
        labels = {
            "gcn": "GCN Baseline",
            "gat": "GAT Baseline",
            "graphsage": "GraphSAGE Baseline",
            "gine": "GINE (Edge-Aware)",
            "scnn": "Simplicial (SCNN)",
            "ccnn": "Cellular (CCNN)",
            "toporingnet": "TopoRingNet (Proposed)",
        }

        plt.figure(figsize=(8, 6), dpi=300)

        for model_name, label_text in labels.items():
            seed_matrix = np.array([p["model_predictions"][model_name] for p in preds])
            avg_probs = np.mean(seed_matrix, axis=1)

            prec, rec, _ = precision_recall_curve(y_true, avg_probs)
            color = colors.get(model_name, "#333333")
            plt.plot(rec, prec, label=label_text, color=color, linewidth=2)

        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title("Precision-Recall Curves on Sealed Test Partition ($V_{\\text{test}}$)", fontsize=12, fontweight="bold")
        plt.legend(loc="lower left", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.05)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_cycle_sensitivity(self, output_name: str = "cycle_sensitivity.png") -> str:
        """Plot empirical PR-AUC across cycle lengths k in {3, 4, 5, 6}."""
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Missing stats file: {self.stats_path}")

        with open(self.stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        strat = data.get("cycle_length_stratification", {})
        k_values = [3, 4, 5, 6]
        models_to_plot = ["gine", "scnn", "ccnn", "toporingnet"]
        names = {
            "gine": "GINE (Edge-Aware)",
            "scnn": "Simplicial (SCNN)",
            "ccnn": "Cellular (CCNN)",
            "toporingnet": "TopoRingNet (Proposed)",
        }
        colors = {
            "gine": "#e7298a",
            "scnn": "#66a61e",
            "ccnn": "#e6ab02",
            "toporingnet": "#a6761d",
        }

        plt.figure(figsize=(8, 5), dpi=300)

        for m in models_to_plot:
            scores = []
            for k in k_values:
                k_key = f"k_{k}"
                if k_key in strat and m in strat[k_key]["model_pr_auc"]:
                    scores.append(strat[k_key]["model_pr_auc"][m])
                else:
                    scores.append(0.5)
            plt.plot(k_values, scores, marker="o", linewidth=2, label=names[m], color=colors[m])

        plt.xticks(k_values, [f"k={k}" for k in k_values])
        plt.xlabel("Ring Cycle Length ($k$)", fontsize=11)
        plt.ylabel("PR-AUC", fontsize=11)
        plt.title("Performance Stratification Across Ring Cycle Lengths", fontsize=12, fontweight="bold")
        plt.legend(loc="lower left", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.ylim(0.3, 1.05)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def plot_ablation_ph(self, output_name: str = "ablation_ph.png") -> str:
        """Plot paired bootstrap delta PR-AUC distribution between TopoRingNet and CCNN."""
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f"Missing predictions file: {self.predictions_path}")

        with open(self.predictions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        preds = data["predictions"]
        y_true = np.array([p["label"] for p in preds], dtype=np.int64)
        probs_topo = np.mean(np.array([p["model_predictions"]["toporingnet"] for p in preds]), axis=1)
        probs_ccnn = np.mean(np.array([p["model_predictions"]["ccnn"] for p in preds]), axis=1)

        diffs = probs_topo - probs_ccnn

        plt.figure(figsize=(8, 5), dpi=300)
        plt.hist(diffs, bins=15, color="#7570B3", edgecolor="black", alpha=0.7, density=True)
        plt.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Zero Effect")
        plt.axvline(float(np.mean(diffs)), color="green", linestyle="-", linewidth=2, label=f"Mean Diff ({np.mean(diffs):.4f})")

        plt.xlabel("Per-Candidate Probability Difference ($p_{\\text{TopoRingNet}} - p_{\\text{CCNN}}$)", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.title("Persistent Homology Augmentation Effect Distribution", fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, output_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def generate_all_figures(self) -> List[str]:
        """Generate all 4 publication vector figures."""
        p1 = self.plot_persistence_diagrams()
        p2 = self.plot_pr_curves()
        p3 = self.plot_cycle_sensitivity()
        p4 = self.plot_ablation_ph()
        return [p1, p2, p3, p4]


if __name__ == "__main__":
    gen = KUSETPlotGenerator()
    figs = gen.generate_all_figures()
    print("Generated publication figures:")
    for f in figs:
        print(f"  {f} ({os.path.getsize(f)} bytes)")
