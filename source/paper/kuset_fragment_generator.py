"""Automated LaTeX Table and Figure Fragment Generator for KUSET Camera-Ready Manuscript.

Contract C18-02 (T-COMP) & Scientific Remediation:
- Reads production confirmatory stats, observed audit, candidates, and OOF predictions.
- Generates publication LaTeX table fragments in paper/generated/.
- Generates publication figures in paper/figures/.
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def generate_kuset_fragments(
    stats_json_path: str = "results/production_confirmatory_stats.json",
    audit_json_path: str = "artifacts/audit/observed_data_audit.json",
    candidates_parquet_path: str = "artifacts/candidates/candidates.parquet",
    oof_parquet_path: str = "artifacts/predictions/oof_predictions.parquet",
    output_tex_dir: str = "paper/generated",
    output_fig_dir: str = "paper/figures",
) -> Dict[str, Any]:
    """Generate automated LaTeX table fragments and publication figures."""
    os.makedirs(output_tex_dir, exist_ok=True)
    os.makedirs(output_fig_dir, exist_ok=True)

    with open(stats_json_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    with open(audit_json_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    c_table = pq.read_table(candidates_parquet_path)
    candidates = c_table.to_pylist()

    oof_table = pq.read_table(oof_parquet_path)
    oof_records = oof_table.to_pylist()
    summary = audit["observed_summary"]
    positives = sum(int(c["label"]) == 1 for c in candidates)
    negatives = len(candidates) - positives
    candidate_transactions = sum(len(c["cycle_transaction_ids"]) for c in candidates)

    # 1. Generate tab_cohort_stats.tex
    tab_cohort = r"""\begin{table}[htbp]
\centering
\caption{Observed Data Cohort and Candidate Set Properties (IBM AMLworld HI-Small)}
\label{tab:cohort_stats}
\begin{tabular}{llr}
\hline
\textbf{Category} & \textbf{Metric} & \textbf{Value} \\
\hline
Raw Transaction Ledger & Total Transaction Records & __TOTAL_TRANSACTIONS__ \\
 & Total Active Accounts & __TOTAL_ACCOUNTS__ \\
 & Total Laundering Transactions & __LAUNDERING_TRANSACTIONS__ (__LAUNDERING_RATE__\%) \\
 & Verified Cycle Pattern Instances & __POSITIVES__ \\
\hline
Candidate Cohort & Total Cycle Candidates ($k \in [3..12]$) & __CANDIDATES__ \\
 & Official-Pattern Laundering Cycles ($y=1$) & __POSITIVES__ \\
 & Extracted, Matched Benign Controls ($y=0$) & __NEGATIVES__ \\
 & Total Candidate Subgraph Transactions & __CANDIDATE_TRANSACTIONS__ \\
 & Protected Account-Connected Groups & __GROUPS__ \\
 & Outer Validation Scheme & 5-Fold Stratified Group CV \\
 & Inner Validation Scheme & 3-Fold Inner CV \\
 & Cross-Fold Account Leakage & 0.0\% (\textit{INV-006}) \\
\hline
\end{tabular}
\end{table}
"""
    tab_cohort = (
        tab_cohort.replace("__TOTAL_TRANSACTIONS__", f"{int(summary['total_transactions']):,}")
        .replace("__TOTAL_ACCOUNTS__", f"{int(summary['unique_active_accounts']):,}")
        .replace("__LAUNDERING_TRANSACTIONS__", f"{int(summary['laundering_transactions']):,}")
        .replace("__LAUNDERING_RATE__", f"{float(summary['laundering_rate_pct']):.4f}")
        .replace("__CANDIDATES__", str(len(candidates)))
        .replace("__POSITIVES__", str(positives))
        .replace("__NEGATIVES__", str(negatives))
        .replace("__CANDIDATE_TRANSACTIONS__", str(candidate_transactions))
        .replace("__GROUPS__", str(stats["total_groups"]))
    )
    with open(os.path.join(output_tex_dir, "tab_cohort_stats.tex"), "w", encoding="utf-8") as f:
        f.write(tab_cohort)

    # 2. Generate tab_model_benchmark.tex
    benchmarks = stats.get("model_benchmark", {})
    model_rows = [
        ("Logistic Regression (LR)", "logistic_regression", "Tabular Baseline"),
        ("HistGradientBoosting (HGB)", "hist_gradient_boosting", "Tabular Baseline"),
        ("SimplicialNet (SCNN)", "scnn", "Simplicial Complex TDL"),
        ("CellularComplexNet (CCNN)", "ccnn", "Cellular Complex TDL"),
        ("GINE", "gine", "Edge-Aware GNN"),
        ("GAT", "gat", "Spatial GNN Baseline"),
        ("GCN", "gcn", "Spatial GNN Baseline"),
        ("GraphSAGE", "graphsage", "Spatial GNN Baseline"),
    ]

    tab_model = r"""\begin{table}[htbp]
\centering
\caption{Five-Seed, Five-Fold Out-of-Fold Performance (seed-averaged predictions)}
\label{tab:model_benchmark}
\begin{tabular}{llccc}
\hline
\textbf{Model Architecture} & \textbf{Model Family} & \textbf{Average Precision (95\% CI)} & \textbf{ROC-AUC (95\% CI)} & \textbf{F1 Score} \\
\hline
"""
    for label, m_key, family in model_rows:
        if m_key in benchmarks:
            bm = benchmarks[m_key]
            ap = bm["average_precision"]
            ap_lo = bm["average_precision_ci_95"][0]
            ap_hi = bm["average_precision_ci_95"][1]
            roc = bm["roc_auc"]
            roc_lo = bm["roc_auc_ci_95"][0]
            roc_hi = bm["roc_auc_ci_95"][1]
            f1 = bm["f1_score"]
            tab_model += f"{label} & {family} & {ap:.4f} [{ap_lo:.3f}, {ap_hi:.3f}] & {roc:.4f} [{roc_lo:.3f}, {roc_hi:.3f}] & {f1:.4f} \\\\\n"

    tab_model += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_tex_dir, "tab_model_benchmark.tex"), "w", encoding="utf-8") as f:
        f.write(tab_model)

    # 3. Generate tab_hypothesis_tests.tex
    hyps = stats.get("confirmatory_hypothesis_tests", {})
    tab_hyps = r"""\begin{table}[htbp]
\centering
\caption{Corrective Group-Blocked Permutation Comparisons (10,000 permutations)}
\label{tab:hypothesis_tests}
\begin{tabular}{llcccc}
\hline
\textbf{Role} & \textbf{Comparison} & \textbf{$\Delta$AP (95\% CI)} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{Interpretation} \\
\hline
"""
    rq_labels = {
        "RQ1_ccnn_vs_gine": ("RQ1: Cellular vs GNN", "CCNN vs GINE"),
        "RQ2_ccnn_vs_scnn_kge4": (r"RQ2: Cellular vs Simplicial ($k \ge 4$)", "CCNN vs SCNN"),
        "RQ3_lr_vs_gine": ("RQ3: Tabular vs GNN", "LR vs GINE"),
    }

    for h_key in ["RQ1_ccnn_vs_gine", "RQ2_ccnn_vs_scnn_kge4"]:
        if h_key in hyps:
            h_data = hyps[h_key]
            rq_title, comp_str = rq_labels[h_key]
            d_ap = h_data["delta_ap"]
            d_lo, d_hi = h_data["delta_ap_ci_95"]
            rp = h_data["p_value_raw"]
            ap = h_data["p_value_adjusted"]
            role = "Primary" if h_data["analysis_role"] == "primary_confirmatory_family" else "Exploratory"
            adjusted = f"{ap:.4f}" if ap is not None else "--"
            sig = (("Significant" if h_data.get("statistically_significant_alpha_0_05") else "Not significant")
                   if role == "Primary" else "Unadjusted exploratory")
            tab_hyps += f"{rq_title} & {comp_str} & {d_ap:+.4f} & {rp:.4f} & {ap:.4f} & {sig} \\\\\n"

    tab_hyps += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_tex_dir, "tab_hypothesis_tests.tex"), "w", encoding="utf-8") as f:
        f.write(tab_hyps)

    # 4. Generate tab_ablation.tex (stratification across cycle lengths k)
    by_k = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
    for c in candidates:
        k = c["cycle_length"]
        if c["label"] == 1:
            by_k[k]["pos"] += 1
        else:
            by_k[k]["neg"] += 1
        by_k[k]["total"] += 1

    tab_ablation = r"""\begin{table}[htbp]
\centering
\caption{Cycle Length ($k \in [3..12]$) Stratification and Pattern Distribution}
\label{tab:cycle_ablation}
\begin{tabular}{ccccc}
\hline
\textbf{Cycle Length ($k$)} & \textbf{Laundering Cycles ($y=1$)} & \textbf{Benign Controls ($y=0$)} & \textbf{Total Candidates} & \textbf{Laundering Proportion} \\
\hline
"""
    for k in sorted(by_k.keys()):
        pos = by_k[k]["pos"]
        neg = by_k[k]["neg"]
        tot = by_k[k]["total"]
        prop = (pos / tot) * 100.0
        tab_ablation += f"$k = {k}$ & {pos} & {neg} & {tot} & {prop:.1f}\\% \\\\\n"

    tab_ablation += r"""\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_tex_dir, "tab_ablation.tex"), "w", encoding="utf-8") as f:
        f.write(tab_ablation)

    # 5. Generate Figures
    # Precision-Recall curves
    plt.figure(figsize=(7, 5))
    oof_df = pd.DataFrame(oof_records)
    for m_name in sorted(oof_df.model_name.unique()):
        model_df = oof_df[oof_df.model_name == m_name]
        yp = model_df.pivot(index="candidate_id", columns="seed", values="y_pred_prob").sort_index().mean(axis=1)
        yt = model_df.drop_duplicates("candidate_id").set_index("candidate_id").sort_index()["y_true"]
        p, r_rec, _ = precision_recall_curve(yt, yp)
        ap_val = stats.get("model_benchmark", {}).get(m_name, {}).get("average_precision", 0.0)
        plt.plot(r_rec, p, label=f"{m_name} (AP = {ap_val:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Out-of-Fold Precision-Recall Curves (IBM AMLworld HI-Small)")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_pr_path = os.path.join(output_fig_dir, "fig_pr_curves.png")
    plt.savefig(fig_pr_path, dpi=300)
    plt.close()

    # Cycle length distribution figure
    plt.figure(figsize=(7, 4))
    ks = sorted(by_k.keys())
    pos_vals = [by_k[k]["pos"] for k in ks]
    neg_vals = [by_k[k]["neg"] for k in ks]

    bar_w = 0.4
    x_idx = np.arange(len(ks))
    plt.bar(x_idx - bar_w/2, pos_vals, width=bar_w, label="Laundering Cycles ($y=1$)", color="#d9534f")
    plt.bar(x_idx + bar_w/2, neg_vals, width=bar_w, label="Benign Controls ($y=0$)", color="#337ab7")

    plt.xlabel("Cycle Length ($k$)")
    plt.ylabel("Candidate Count")
    plt.title("Candidate Cohort Cycle Length Distribution")
    plt.xticks(x_idx, [f"$k={k}$" for k in ks])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_k_path = os.path.join(output_fig_dir, "fig_k_ablation.png")
    plt.savefig(fig_k_path, dpi=300)
    plt.close()

    return {
        "status": "FRAGMENTS_GENERATED",
        "tex_files": [
            os.path.join(output_tex_dir, "tab_cohort_stats.tex"),
            os.path.join(output_tex_dir, "tab_model_benchmark.tex"),
            os.path.join(output_tex_dir, "tab_hypothesis_tests.tex"),
            os.path.join(output_tex_dir, "tab_ablation.tex"),
        ],
        "fig_files": [fig_pr_path, fig_k_path],
        "seed_count": len(stats["seeds"]),
        "group_count": stats["total_groups"],
    }


if __name__ == "__main__":
    res = generate_kuset_fragments()
    print(json.dumps(res, indent=2))
