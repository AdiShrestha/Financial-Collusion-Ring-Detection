# Pre-Registered Analysis & Validation Protocol: IBM AMLworld Cycle Benchmark

**Version:** 2.2.0  
**Status:** FROZEN & LOCKED  
**Authority:** Factory Constitution §8 / Remediation Specification §14  
**Date:** 2026-08-24  

---

## 1. Study Specification & Benchmark Scope

- **Study Title:** Benchmarking Graph and Cellular-Complex Models for Pattern-Grounded Money-Laundering Cycle Classification on IBM AMLworld HI-Small
- **Dataset Identification:**
  - Raw Transactions: `data/raw/HI-Small_Trans.csv` (SHA-256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`, 5,078,345 transactions, 515,080 unique transaction accounts)
  - Raw Patterns: `data/raw/HI-Small_Patterns.txt` (SHA-256: `2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b`, 370 pattern blocks)
- **Target Task:** Pattern-grounded directed simple cycle classification for lengths $k \in [3..12]$.
- **Cohort Composition:**
  - Positive Laundering Cycles ($y=1$): Exactly 40 cycles ($k \in [3..12]$) from official `CYCLE` blocks, forming 38 account-disjoint components.
  - Matched Benign Cycles ($y=0$): 115 label-blind extracted cycles matched via exact length $k$ and duration/amount calipers.
  - Total Candidate Cohort: **155 candidates** (963 transactions).

---

## 2. Frozen Split Manifest & Leakage Boundaries

- **Manifest File:** `artifacts/splits/fold_manifest.json`
- **Locked Manifest SHA-256:** `8bc14682dc07e879de6cc7edf7f13b4599bc1ef5dad9307c6d4ce8cd2639c57b`
- **Validation Design:**
  - 5-Fold Outer Stratified Group Cross-Validation.
  - 3-Fold Inner Group Cross-Validation per outer training fold (for hyperparameter selection and F1 threshold tuning).
  - Invariant `INV-006`: 0.0% account and transaction leakage across outer folds.

---

## 3. Model Zoo & Evaluation Plan

### Model Families
1. **Tabular Baselines:** Logistic Regression (L2-penalized), HistGradientBoosting.
2. **Standard GNN Baselines:** GCN, GAT, GraphSAGE.
3. **Primary Edge-Aware GNN:** GINE (Graph Isomorphism Network with Edge Features).
4. **Topological Cellular Complexes:** Simplicial CNN (SCNN), Cell Complex Neural Network (CCNN) with boundary operators $B_1 \in \{-1,0,1\}^{|V| \times |E|}$, $B_2 \in \{-1,0,1\}^{|E| \times |C|}$, verified $B_1 B_2 = 0$.

### Evaluation Metrics
- **Primary Confirmatory Metric:** Average Precision (AP / PR-AUC) evaluated on out-of-fold predictions.
- **Secondary Metrics:** Macro F1 (threshold selected via inner CV), AUROC, Precision@K.

### Statistical Inference Standards
- **Bootstrap 95% Confidence Intervals:** Group-resampled paired bootstrap over 10,000 resamples.
- **Hypothesis Testing:** Group-blocked model-swap permutation test (swapping prediction vectors across independent account components and recomputing AP differences over 10,000 permutations) + Holm-Bonferroni multi-comparison correction.
