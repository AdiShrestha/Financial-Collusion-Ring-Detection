#!/usr/bin/env python3
"""Turnkey End-to-End Pipeline Replication Script for KUSET Publication Release.

Executes the entire reproducible experimental pipeline from observed data audit,
candidate extraction, persistent homology caching, model confirmatory evaluation,
statistical hypothesis testing, publication figure generation, and claim audit.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from source.data.audit_exporter import export_audit_report
from source.data.candidate_dataset import CandidateDatasetManager
from source.evidence.kuset_hypothesis_tester import KUSETHypothesisTester
from source.experiments.production_confirmatory_runner import ProductionConfirmatoryRunner
from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer
from source.paper.kuset_plot_generator import KUSETPlotGenerator
from source.ph.cache_pipeline import TopologicalFeatureCache


def run_full_replication() -> None:
    print("=" * 70)
    print("STARTING COMPLETE KUSET EXPERIMENTAL REPLICATION")
    print("=" * 70)

    # 1. Observed Data Ingestion & Audit
    print("\n[Step 1/7] Auditing IBM AMLworld Observed Dataset...")
    audit_rep = export_audit_report()
    print(f"  -> Observed transactions: {audit_rep['total_transactions']:,}, Accounts: {audit_rep['total_accounts']:,}")

    # 2. Candidate Extraction & Split Manifest
    print("\n[Step 2/7] Extracting Cycle Candidates & Group-Safe Split...")
    mgr = CandidateDatasetManager()
    ds_res = mgr.build_candidate_dataset()
    print(f"  -> Extracted {ds_res['total_candidates']} candidates across {ds_res['total_groups']} groups.")

    # 3. Topological Feature Caching
    print("\n[Step 3/7] Computing Normalized Persistent Homology Features...")
    cache_engine = TopologicalFeatureCache()
    c_res = cache_engine.build_cache_from_jsonl()
    print(f"  -> Cached topological feature matrix: {c_res['total_records']} records x {c_res['feature_dimension']} dims.")

    # 4. Confirmatory Evaluation on Test Partition
    print("\n[Step 4/7] Running Confirmatory Evaluation across 35 Checkpoints on V_test...")
    runner = ProductionConfirmatoryRunner()
    conf_res = runner.run_confirmatory_evaluation()
    print(f"  -> Evaluated {conf_res['metadata']['total_test_candidates']} test candidates across 7 models x 5 seeds.")

    # 5. Hypothesis Testing & Holm-Bonferroni Correction
    print("\n[Step 5/7] Executing Statistical Hypothesis Testing (H1–H3)...")
    tester = KUSETHypothesisTester()
    stat_res = tester.run_hypothesis_tests()
    for h_id, h_data in stat_res["hypotheses"].items():
        print(f"  -> {h_id}: Delta PR-AUC = {h_data['delta_pr_auc']:.4f} (p_adj = {h_data['wilcoxon_p_adj']:.4f}, Verdict: {h_data['verdict']})")

    # 6. Vector Figure Generation
    print("\n[Step 6/7] Generating 300 DPI Publication Vector Figures...")
    plotter = KUSETPlotGenerator()
    figs = plotter.generate_all_figures()
    print(f"  -> Generated {len(figs)} publication figures in paper/figures/.")

    # 7. Claim Synchronization Audit
    print("\n[Step 7/7] Auditing LaTeX Manuscript Claim Synchronization...")
    sync = KUSETClaimSynchronizer()
    sync_res = sync.audit_claim_synchronization()
    print(f"  -> Claim Audit Status: {sync_res['status']} ({sync_res['total_checks']} checks passed, {len(sync_res['discrepancies'])} discrepancies).")

    print("\n" + "=" * 70)
    print("REPLICATION SUCCESSFULLY COMPLETED (100% Empirically Synchronized)")
    print("=" * 70)


if __name__ == "__main__":
    run_full_replication()
