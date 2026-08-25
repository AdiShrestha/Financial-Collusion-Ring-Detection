# Chunk 19 Analysis-Protocol Reconciliation

**Date:** 2026-08-24  
**Status:** Prospective lock for the Chunk 19 rerun; not represented as an original preregistration.

## Why this document exists

Two historical documents disagree. `project/PRE_REGISTRATION.md` describes a single 28-candidate test partition, TopoRingNet, four hypotheses, Wilcoxon tests, and five seeds. `protocol/ANALYSIS_PROTOCOL.md` describes a later 155-candidate five-fold OOF benchmark, eight evaluated models, group bootstrap, and model-swap permutation tests, but contains stale cohort hashes/counts. Chunk 18 then executed only three seeds and mixed an older training history with the current cohort.

Those conflicts invalidate the phrase “pre-registered confirmatory result” for the Chunk 18 analysis. This document does not rewrite that history. It declares the reproducible analysis used for the corrective Chunk 19 rerun and requires the manuscript to describe it as a locked corrective analysis, with the historical deviation disclosed.

## Canonical Chunk 19 estimand and execution

- Cohort: the exact 155 candidates in `artifacts/candidates/candidates.parquet`.
- Protected unit: the exact 18 account-connected groups in `artifacts/splits/fold_manifest.json`.
- Outer evaluation: five group-disjoint folds; each candidate appears in an outer test fold exactly once.
- Seeds: `42,43,44,45,46`; predictions are averaged per candidate before statistical inference.
- Models: Logistic Regression, HistGradientBoosting, GCN, GAT, GraphSAGE, GINE, SCNN, CCNN.
- Model hyperparameters: fixed symmetrically before outer evaluation. Three inner group folds select only the positive-class F1 threshold from `{0.2,...,0.8}`. No claim of hyperparameter search is made.
- Primary metric: Average Precision.
- Primary comparison family: RQ1 CCNN versus GINE on all candidates; RQ2 CCNN versus SCNN on candidates with `k >= 4`.
- Primary inference: 10,000 whole-group bootstrap resamples and 10,000 group-blocked model-swap permutations; plus-one Monte Carlo p-values; Holm correction across RQ1 and RQ2.
- Exploratory comparison: Logistic Regression versus GINE, reported with an unadjusted p-value and explicitly labeled exploratory.
- Non-significance is not equivalence. Negative deltas are preserved as falsifying evidence for directional superiority.

## Declared deviations and supersession

1. The legacy four-hypothesis TopoRingNet/Elliptic++ plan was not completed and is not tested by this cohort.
2. Chunk 18's three-seed analysis is superseded; Chunk 19 uses the originally required five seeds.
3. The measured protected-group count is 18, not the earlier expected 38 positive components; the grouping includes both classes through shared-account connectivity.
4. Positive candidates are reconstructed from official cycle-pattern blocks while benign controls are extracted and matched. This is pattern-grounded candidate classification, not label-blind candidate discovery or whole-network detection.
5. Because the corrective plan was reconciled after defects were observed, statistical significance is presented with this limitation and not called prospectively preregistered.

