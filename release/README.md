# TopoRingNet Replication Package

## Overview
This package contains all code, data, and artifacts needed to reproduce the
results reported in the TopoRingNet paper on topological deep learning for
anti-money laundering detection.

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. One-Command Reproduction
```bash
python3 release/replicate.py
```

This script will:
1. Verify the test split checksum (INV-006).
2. Run confirmatory predictions across all 6 models and 5 locked seeds.
3. Execute hypothesis testing with Holm-Bonferroni correction.
4. Generate the claim registry.
5. Verify all claims against the paper manuscript.

### 3. Verify Results
```bash
python3 -m pytest tests/ -v
```

## Directory Structure
```
├── source/          # All source modules
│   ├── topology/    # Incidence matrices, oracles, persistent homology
│   ├── models/      # GCN, GAT, GraphSAGE, Simplicial, Cellular, TopoRingNet
│   ├── experiments/ # Confirmatory runner
│   ├── evidence/    # Hypothesis tester, claim registry, gate verifiers
│   ├── paper/       # Compiler, plot generator, claim synchronizer
│   └── release/     # Packager, gate F verifier
├── tests/           # Pytest test suites
├── results/         # Confirmatory statistics and ablation summaries
├── paper/           # LaTeX manuscript and figures
├── project/         # Gate reports and project documentation
└── release/         # This replication package
```

## Pre-Registered Hypotheses
- H1: TopoRingNet vs strongest GNN (PR-AUC)
- H2: CellularComplexNet vs SimplicialComplexNet on k>=4 cycles (F1-Macro)
- H3: TopoRingNet vs CellularComplexNet structural ablation (PR-AUC)
- H4: TopoRingNet cross-track superiority (PR-AUC)

All hypotheses used paired Wilcoxon signed-rank tests with Cliff's delta
effect sizes, corrected via Holm-Bonferroni step-down at alpha=0.05.

## Factory Invariants
- INV-001: True empirical data — no fabricated constants
- INV-002: Simplicial vs cellular realism (k>=4 cycles are rank-2 polygonal cells)
- INV-004: Boundary operator nilpotence (B1 B2 = 0)
- INV-006: Test split isolation (SHA-256 verified)
- INV-008: Factory frozen file immutability

## License
See project documentation for licensing terms.
