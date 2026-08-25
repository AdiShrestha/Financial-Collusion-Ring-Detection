# Pattern-Grounded AML Cycle Classification Artifact Package

## Overview
This package verifies the existing artifacts used by the KUSET manuscript.
It does not reacquire the raw dataset and is not a clean-room reproduction.

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. One-Command Reproduction
```bash
python3 release/replicate.py --verify-only
```

This read-only command checks cohort/manifest identity, five-seed and eight-model
ledger cardinality, 200 checkpoints, the Gate A--F chain, and manuscript claims.

### 3. Verify Results
```bash
python3 -m pytest tests/ -v
```

## Directory Structure
```
├── source/          # All source modules
│   ├── topology/    # Incidence matrices, oracles, persistent homology
│   ├── models/      # LR, HGB, spatial GNN, simplicial, and cellular models
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

## Analysis status
Chunk 19 is a disclosed corrective analysis, not a prospective preregistration.
RQ1 and RQ2 form the Holm-adjusted primary family; LR-versus-GINE is exploratory.

## Factory Invariants
- INV-001: True empirical data — no fabricated constants
- INV-002: Simplicial vs cellular realism (k>=4 cycles are rank-2 polygonal cells)
- INV-004: Boundary operator nilpotence (B1 B2 = 0)
- INV-006: Test split isolation (SHA-256 verified)
- INV-008: Factory frozen file immutability

## License
See project documentation for licensing terms.
