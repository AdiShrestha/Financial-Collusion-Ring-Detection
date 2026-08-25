# TopoRing: Topological Deep Learning for Financial Collusion-Ring Detection

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Evaluation](https://img.shields.io/badge/Protocol-5--Fold%20Group%20CV-success.svg)](#3-benchmark-results)
[![Reproducibility](https://img.shields.io/badge/Auditing-Immutable%20Ledger-orange.svg)](#6-execution--pipeline-cli)

> **Benchmarking Graph, Simplicial, and Cellular-Complex Models for Pattern-Grounded Money-Laundering Cycle Classification.**

---

## 1. Overview

**TopoRing** is a rigorous, reproducible research framework that investigates whether **higher-order topological deep learning (TDL)** representations—specifically **simplicial complexes** and **cellular (CW) complexes**—provide a predictive advantage over standard **Graph Neural Networks (GNNs)** and tabular baselines when classifying candidate financial collusion rings and money-laundering cycles.

Operating on candidate subgraphs from the synthetic [IBM AMLworld](https://arxiv.org/abs/2306.16424) (HI-Small) benchmark, each directed transaction cycle is mapped to mathematically isolated representations to evaluate the inductive bias of higher-order message passing.

```
                    ┌─────────────────────────┐
                    │ Raw Transaction Ledger  │ (5.08M txs, 515k accounts)
                    └───────────┬─────────────┘
                                │
                 Candidate Subgraph Construction
            (40 Pattern Cycles + 115 Caliper-Matched Negatives)
                                │
    ┌───────────────────────────┼───────────────────────────┐
    ▼                           ▼                           ▼
1-Skeleton Graph        Simplicial Complex           Cellular Complex
  (GNN Baselines)      (Fan Triangulation)       (Native Polygonal 2-Cells)
  [GCN, GAT, SAGE, GINE]  [SCNN / MPSN]             [CCNN / CWN, B1 B2 = 0]
```

---

## 2. Key Methodological Features

- **Strict Algebraic Invariants:** Native polygonal cell complexes enforce the fundamental boundary-of-boundary identity:
  $$\mathbf{B}_1 \mathbf{B}_2 = \mathbf{0}$$
- **Leak-Free Group Partitioning:** 155 candidates are partitioned into **18 account-connected groups** across a 5-fold outer cross-validation scheme to ensure zero account or transaction leakage across splits.
- **Fair Model Comparison:** Tabular, spatial GNN, edge-aware GNN, simplicial, and cellular architectures receive identical underlying features and standardized seed-averaged evaluation budgets (5 seeds $\times$ 5 folds = 200 model checkpoints).
- **Corrective Statistical Testing:** Group-blocked model-swap permutation tests (10,000 permutations) paired with whole-group bootstrap 95% confidence intervals and Holm multiplicity corrections.

---

## 3. Benchmark Results

Summary of 5-fold outer cross-validation performance across 5 seeds (seed-averaged predictions) on the 155-candidate cohort:

| Model Architecture | Paradigm | Average Precision (95% CI) | ROC-AUC (95% CI) | F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression (LR)** | Tabular Baseline | **0.8534** [0.700, 0.970] | **0.9433** [0.863, 0.989] | **0.7532** |
| **HistGradientBoosting (HGB)** | Tabular Baseline | 0.8225 [0.623, 0.954] | 0.9211 [0.813, 0.982] | 0.6667 |
| **SimplicialNet (SCNN)** | Simplicial TDL (MPSN) | 0.7384 [0.555, 0.941] | 0.9097 [0.831, 0.972] | 0.7160 |
| **CellularComplexNet (CCNN)** | Cellular TDL (CWN) | 0.7229 [0.491, 0.969] | 0.9061 [0.781, 0.986] | 0.7059 |
| **GAT** | Spatial GNN Baseline | 0.3628 [0.331, 0.616] | 0.6183 [0.559, 0.754] | 0.3738 |
| **GINE** | Edge-Aware GNN | 0.3392 [0.292, 0.604] | 0.6591 [0.570, 0.823] | 0.3529 |
| **GCN** | Spatial GNN Baseline | 0.2868 [0.265, 0.376] | 0.5088 [0.475, 0.564] | 0.3704 |
| **GraphSAGE** | Spatial GNN Baseline | 0.2816 [0.278, 0.384] | 0.5215 [0.469, 0.626] | 0.3952 |

### Confirmatory Hypothesis Testing
- **RQ1 (CCNN vs. GINE):** $\Delta\text{AP} = +0.3838$ ($p_{\text{raw}} = 0.0881, p_{\text{adj}} = 0.1762$). Non-significant after Holm adjustment.
- **RQ2 (CCNN vs. SCNN on $k \ge 4$):** $\Delta\text{AP} = -0.0025$ ($p_{\text{adj}} = 0.5324$). No evidence of cellular superiority over triangulated simplicial views.
- **RQ3 (Exploratory: LR vs. GINE):** $\Delta\text{AP} = +0.5142$ ($p_{\text{raw}} = 0.0047$). Global flow attributes strongly separate cycles relative to local one-hop message passing.

---

## 4. Repository Structure

```text
├── source/
│   ├── pipeline.py                 # Unified CLI execution orchestrator
│   ├── data/
│   │   ├── streaming_loader.py     # Streamed CSV-to-Parquet conversion
│   │   ├── audit_exporter.py       # Ledger accounting and multiset verification
│   │   ├── positive_reconstruction.py # Exact CYCLE pattern backbone reconstruction
│   │   ├── bounded_extractor.py    # Temporal windowed cycle extraction
│   │   ├── negative_cycle_sampler.py # Caliper-matched negative control selection
│   │   └── splits.py               # Union-find 5-fold group-stratified splitter
│   ├── topology/
│   │   ├── cell_complex_encoder.py # Multi-view topological lifting & B1 @ B2 check
│   │   ├── incidence.py            # Signed boundary operators and Laplacians
│   │   └── oracles.py              # Mathematical and algebraic verification oracles
│   ├── features/
│   │   └── fold_preprocessor.py    # Fold-isolated AML feature scalers
│   ├── models/
│   │   ├── factory.py              # Centralized ModelFactory (8 architectures)
│   │   ├── simplicial_net.py       # Simplicial Complex Neural Network
│   │   ├── cell_net.py             # Cellular Complex Neural Network
│   │   └── edge_aware_gnn.py       # GINE model
│   ├── training/
│   │   └── cross_val_runner.py     # 5-fold x 5-seed training engine
│   ├── experiments/
│   │   └── oof_evaluator.py        # Out-of-fold inference & metrics aggregator
│   ├── evidence/
│   │   └── kuset_hypothesis_tester.py # Group-bootstrap & permutation testing suite
│   └── paper/
│       ├── kuset_fragment_generator.py # Automated LaTeX table/macro synchronizer
│       └── kuset_plot_generator.py     # Publication figure generator
├── paper/                          # LaTeX manuscript and synchronized figures
├── tests/                          # Automated unit, integration, and oracle test suite
├── project/                        # Architectural contracts, manifests, and specifications
└── requirements.txt                # Pinned dependency environment
```

---

## 5. Getting Started

### Prerequisites
- Python 3.10+
- PyTorch >= 2.0.0

```bash
# Clone the repository
git clone https://github.com/your-username/toporing.git
cd toporing

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Execution & Pipeline CLI

The pipeline is fully automated via `source/pipeline.py`:

```bash
# Run complete end-to-end pipeline (ingestion to manuscript generation)
python source/pipeline.py --stage all
```

### Individual Pipeline Stages
You can also execute discrete stages independently:
```bash
python source/pipeline.py --stage ingest       # 1. Convert raw CSV ledger to Parquet
python source/pipeline.py --stage audit        # 2. Generate observed data audit
python source/pipeline.py --stage reconstruct  # 3. Reconstruct positive cycle backbones
python source/pipeline.py --stage extract      # 4. Extract benign candidate pool
python source/pipeline.py --stage match        # 5. Assemble caliper-matched cohort
python source/pipeline.py --stage split        # 6. Generate 5-fold group splits
python source/pipeline.py --stage preprocess   # 7. Fit fold-isolated feature preprocessors
python source/pipeline.py --stage encode       # 8. Lift subgraphs into cell/simplicial complexes
python source/pipeline.py --stage train        # 9. Train 8 model families (5 folds x 5 seeds)
python source/pipeline.py --stage evaluate     # 10. Run OOF evaluation & permutation tests
python source/pipeline.py --stage report       # 11. Generate publication LaTeX tables & plots
```

---

## 7. Verification & Testing

To run the algebraic integrity oracles, leak-free split tests, and gate verifications:

```bash
# Execute full test suite
pytest tests/ -v
```

## 8. Author & Affiliation

- **Author:** Aditya Shrestha
- **Affiliation:** Department of Computer Science and Engineering, Kathmandu University, Dhulikhel, Nepal
- **Role:** Final Year Computer Engineering Student
- **Email:** [adityashrestha39@gmail.com](mailto:adityashrestha39@gmail.com)
- **GitHub:** [@AdiShrestha](https://github.com/AdiShrestha)

---

## 9. Citation

If you build upon this benchmark, representations, or methodology, please cite:

```bibtex
@article{shrestha2026toporing,
  title={Benchmarking Graph and Cellular-Complex Models for Pattern-Grounded Money-Laundering Cycle Classification},
  author={Shrestha, Aditya},
  journal={Preprint, Kathmandu University},
  year={2026},
  url={https://github.com/AdiShrestha/Financial-Collusion-Ring-Detection}
}
```
