"""
Release Packager — Standalone Replication Bundle Assembly.

Contract C09-04: Assembles a self-contained replication package in release/
with environment documentation, reproduction scripts, and verification manifests.
"""

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict


def verify_existing_artifacts(project_root: str | Path = ".") -> Dict[str, Any]:
    """Derive release facts from existing artifacts and fail closed on any mismatch."""
    root = Path(project_root)
    required = [
        "artifacts/audit/observed_data_audit.json",
        "artifacts/candidates/candidates.parquet",
        "artifacts/splits/fold_manifest.json",
        "artifacts/predictions/oof_predictions.parquet",
        "artifacts/training/training_history.json",
        "results/production_confirmatory_stats.json",
        "paper/kuset_main.tex",
        "project/gate_a_report.json",
        "project/gate_b_report.json",
        "project/gate_c_report.json",
        "project/gate_d_report.json",
        "project/gate_e_report.json",
        "project/gate_f_report.json",
        "project/gate_f_kuset_report.json",
    ]
    missing = [rel for rel in required if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Required artifact missing: {missing[0]}")

    def load_json(rel: str) -> Dict[str, Any]:
        with (root / rel).open(encoding="utf-8") as handle:
            return json.load(handle)

    import pyarrow.parquet as pq
    from source.paper.kuset_claim_synchronizer import KUSETClaimSynchronizer

    audit = load_json("artifacts/audit/observed_data_audit.json")
    manifest = load_json("artifacts/splits/fold_manifest.json")
    history = load_json("artifacts/training/training_history.json")
    stats = load_json("results/production_confirmatory_stats.json")
    candidates = pq.read_table(root / "artifacts/candidates/candidates.parquet")
    ledger = pq.read_table(root / "artifacts/predictions/oof_predictions.parquet")
    candidate_ids = set(candidates.column("candidate_id").to_pylist())
    ledger_ids = set(ledger.column("candidate_id").to_pylist())
    manifest_ids = set(manifest["candidate_assignments"])
    seeds = sorted(set(ledger.column("seed").to_pylist()))
    models = sorted(set(ledger.column("model_name").to_pylist()))
    groups = {str(v["group_id"]) for v in manifest["candidate_assignments"].values()}
    checkpoints = len(seeds) * len(models) * int(manifest["n_outer_folds"])

    gate_paths = {
        "gate_a": "project/gate_a_report.json", "gate_b": "project/gate_b_report.json",
        "gate_c": "project/gate_c_report.json", "gate_d": "project/gate_d_report.json",
        "gate_e": "project/gate_e_report.json",
    }
    gate_statuses = {}
    for gate, rel in gate_paths.items():
        report = load_json(rel)
        status = report.get("status") or report.get(f"{gate}_status")
        gate_statuses[gate] = status
    gate_f = load_json("project/gate_f_report.json")
    gate_f_kuset = load_json("project/gate_f_kuset_report.json")
    gate_statuses["gate_f"] = gate_f.get("gate_f_status")
    gate_statuses["gate_f_kuset"] = gate_f_kuset.get("terminal_status")

    claim_audit = KUSETClaimSynchronizer(
        str(root / "paper/kuset_main.tex"),
        str(root / "results/production_confirmatory_stats.json"),
    ).audit_claim_synchronization()
    total_candidates = candidates.num_rows
    expected_rows = total_candidates * len(seeds) * len(models)
    expected_gates = {
        "gate_a": "GATE_A_PASS", "gate_b": "GATE_B_PASS", "gate_c": "GATE_C_PASS",
        "gate_d": "GATE_D_PASS", "gate_e": "GATE_E_PASS", "gate_f": "GATE_F_PASS",
        "gate_f_kuset": "GATE_F_KUSET_PASS",
    }
    checks = {
        "candidate_manifest_identity": candidate_ids == manifest_ids,
        "ledger_candidate_identity": ledger_ids == manifest_ids,
        "candidate_count": total_candidates == manifest["total_candidates"] == stats["total_candidates"],
        "group_count": len(groups) == manifest["total_groups"] == stats["total_groups"],
        "five_seed_requirement": seeds == [42, 43, 44, 45, 46] == history["seeds"] == stats["seeds"],
        "model_family_requirement": models == sorted(history["models"]) == sorted(stats["models"]),
        "ledger_key_cardinality": ledger.num_rows == expected_rows,
        "checkpoint_cardinality": checkpoints == 200,
        "gate_chain": gate_statuses == expected_gates,
        "manuscript_claims": claim_audit["all_synchronized"],
        "manifest_hash_lineage": history["lineage"]["manifest_sha256"] == stats["provenance"]["manifest_sha256"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Artifact verification failed: {failed}")
    observed = audit["observed_summary"]
    labels = candidates.column("label").to_pylist()
    return {
        "status": "ARTIFACT_VERIFICATION_PASS",
        "verification_scope": "existing artifacts; raw-data reacquisition is outside this command",
        "total_transactions": observed["total_transactions"],
        "total_candidates": total_candidates,
        "positive_cycles": sum(int(label) == 1 for label in labels),
        "matched_controls": sum(int(label) == 0 for label in labels),
        "total_groups": len(groups),
        "n_outer_folds": manifest["n_outer_folds"],
        "seeds": seeds,
        "models": models,
        "total_checkpoints": checkpoints,
        "ledger_rows": ledger.num_rows,
        "gate_statuses": gate_statuses,
        "checks": checks,
    }


class ReleasePackager:
    """Assembles a complete, self-contained replication bundle."""

    REQUIRED_SOURCE_MODULES = [
        "source/topology/incidence.py",
        "source/topology/oracles.py",
        "source/models/gnn_baselines.py",
        "source/models/ph_augmented_net.py",
        "source/experiments/confirmatory_runner.py",
        "source/evidence/hypothesis_tester.py",
        "source/evidence/claim_registry.py",
        "source/paper/compiler.py",
        "source/paper/claim_synchronizer.py",
    ]

    REQUIRED_MANIFESTS = [
        "project/project_description.md",
        "project/architecture.md",
        "project/invariants.md",
        "project/PRE_REGISTRATION.md",
    ]

    REQUIRED_RESULTS = [
        "results/confirmatory_stats.json",
        "results/ablation_summary.json",
    ]

    REQUIRED_GATE_REPORTS = [
        "project/gate_b_report.json",
        "project/gate_c_report.json",
        "project/gate_d_report.json",
        "project/gate_e_report.json",
    ]

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)

    def _file_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _check_file_exists(self, rel_path: str) -> bool:
        """Check if a file exists relative to project root."""
        return (self.project_root / rel_path).exists()

    def _generate_environment_lock(self) -> dict[str, Any]:
        """Generate cryptographic environment state."""
        env_lock: dict[str, Any] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "file_hashes": {},
        }

        all_files = (
            self.REQUIRED_SOURCE_MODULES
            + self.REQUIRED_MANIFESTS
            + self.REQUIRED_RESULTS
            + self.REQUIRED_GATE_REPORTS
        )

        for rel_path in sorted(all_files):
            full_path = self.project_root / rel_path
            if full_path.exists():
                env_lock["file_hashes"][rel_path] = self._file_sha256(full_path)

        return env_lock

    def _write_readme(self, output_dir: Path) -> None:
        """Write release/README.md with reproduction instructions."""
        readme_content = """# Pattern-Grounded AML Cycle Classification Artifact Package

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
"""
        (output_dir / "README.md").write_text(readme_content)

    def _write_replicate_script(self, output_dir: Path) -> None:
        """Write release/replicate.py one-command reproduction script."""
        script = '''#!/usr/bin/env python3
"""Read-only verification of existing Chunk 19 release artifacts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.release.packager import verify_existing_artifacts


def verify_existing_release():
    return verify_existing_artifacts(ROOT)


def run_clean_room_replication():
    """Backward-compatible name; performs existing-artifact verification only."""
    return verify_existing_release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(verify_existing_release(), indent=2))
'''
        (output_dir / "replicate.py").write_text(script)

    def package_release(self, output_dir: str = "release") -> dict:
        """
        Assemble complete replication bundle.

        Returns packaging manifest dict with keys:
            success: bool
            output_dir: str
            files_created: list[str]
            missing_prerequisites: list[str]
            file_count: int
        """
        out = self.project_root / output_dir
        os.makedirs(out, exist_ok=True)

        missing = []
        for f in (
            self.REQUIRED_SOURCE_MODULES
            + self.REQUIRED_MANIFESTS
            + self.REQUIRED_RESULTS
        ):
            if not self._check_file_exists(f):
                missing.append(f)

        # Generate environment lock
        env_lock = self._generate_environment_lock()
        env_lock_path = out / "environment_lock.json"
        with open(env_lock_path, "w") as f:
            json.dump(env_lock, f, indent=2)

        # Write README
        self._write_readme(out)

        # Write replicate script
        self._write_replicate_script(out)

        files_created = [
            str(out / "README.md"),
            str(out / "environment_lock.json"),
            str(out / "replicate.py"),
        ]

        return {
            "success": len(missing) == 0,
            "output_dir": str(out),
            "files_created": files_created,
            "missing_prerequisites": missing,
            "file_count": len(files_created),
        }
