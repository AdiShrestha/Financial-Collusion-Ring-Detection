"""Unified End-to-End Execution Pipeline CLI.

Contract C18-04 (T-DESC): Orchestrates full pipeline from raw CSV ingestion to
candidate extraction, fold splitting, cell complex encoding, model training,
confirmatory evaluation, and publication fragment generation.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from source.data.streaming_loader import convert_raw_csv_to_parquet
from source.data.audit_exporter import generate_observed_data_audit
from source.data.positive_reconstruction import reconstruct_exact_positive_cycles
from source.data.bounded_extractor import extract_benign_candidate_pool
from source.data.negative_cycle_sampler import assemble_caliper_matched_cohort
from source.data.splits import build_grouped_cross_validation_splits
from source.features.fold_preprocessor import build_fold_preprocessors
from source.topology.cell_complex_encoder import encode_and_serialize_fold_tensors
from source.training.cross_val_runner import run_5fold_multi_seed_training
from source.experiments.oof_evaluator import export_oof_prediction_ledger
from source.evidence.kuset_hypothesis_tester import KUSETHypothesisTester
from source.paper.kuset_fragment_generator import generate_kuset_fragments


def run_pipeline(stage: str = "all") -> None:
    """Execute stages of the pipeline."""
    print(f"=== Running Autonomous Pipeline Stage: {stage} ===")

    if stage in ("all", "ingest"):
        print("1. Ingesting raw CSV to Parquet...")
        convert_raw_csv_to_parquet()

    if stage in ("all", "audit"):
        print("2. Generating observed data audit...")
        generate_observed_data_audit()

    if stage in ("all", "reconstruct"):
        print("3. Reconstructing genuine positive cycles...")
        reconstruct_exact_positive_cycles()

    if stage in ("all", "extract"):
        print("4. Extracting benign candidate pool...")
        extract_benign_candidate_pool()

    if stage in ("all", "match"):
        print("5. Assembling caliper-matched cohort...")
        assemble_caliper_matched_cohort()

    if stage in ("all", "split"):
        print("6. Constructing leak-free 5-fold cross-validation manifest...")
        build_grouped_cross_validation_splits()

    if stage in ("all", "preprocess"):
        print("7. Fitting fold-isolated feature preprocessors...")
        build_fold_preprocessors()

    if stage in ("all", "encode"):
        print("8. Encoding candidate cell complexes (B1 B2 = 0)...")
        encode_and_serialize_fold_tensors()

    if stage in ("all", "train"):
        print("9. Training 8 model families across 5 folds and 3 seeds...")
        run_5fold_multi_seed_training()

    if stage in ("all", "evaluate"):
        print("10. Exporting OOF prediction ledger & running permutation tests...")
        export_oof_prediction_ledger()
        tester = KUSETHypothesisTester()
        tester.execute_full_confirmatory_suite()

    if stage in ("all", "report"):
        print("11. Generating publication LaTeX tables and figures...")
        generate_kuset_fragments()

    print(f"=== Pipeline Stage '{stage}' Finished Successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AML TDL Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "ingest", "audit", "reconstruct", "extract", "match", "split", "preprocess", "encode", "train", "evaluate", "report"],
        help="Stage to execute",
    )
    args = parser.parse_args()
    run_pipeline(args.stage)
