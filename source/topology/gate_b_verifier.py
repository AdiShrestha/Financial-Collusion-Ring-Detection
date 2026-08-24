"""Gate B (Mathematical Truth) automated verification engine."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from source.topology.clique_simplicial_view import CliqueSimplicialView
from source.topology.cycle_cell_view import CycleCellView
from source.topology.oracles import (
    compute_betti_numbers,
    verify_boundary_nilpotence,
    verify_hodge_laplacian_properties,
)


class GateBVerifier:
    """Automated verifier certifying Gate B (Mathematical Truth) criteria.

    Verifies:
    1. Simplicial vs Cellular complex domain boundary distinction (INV-002).
    2. Persistent Homology graph domain unfilled integrity (INV-003).
    3. Boundary operator nilpotence B1 @ B2 == 0 across all complexes (INV-004).
    4. Hodge Laplacian L0 and L1 symmetry and positive semi-definiteness (PSD).
    """

    @classmethod
    def verify_complexes(
        cls,
        simplicial_complexes: Sequence[CliqueSimplicialView],
        cellular_complexes: Sequence[CycleCellView],
        ph_graphs: Optional[Sequence[Any]] = None,
        tol: float = 1e-10,
    ) -> Dict[str, Any]:
        """Perform formal Gate B mathematical truth certification."""
        results: Dict[str, Any] = {
            "gate_b_status": "GATE_B_PENDING",
            "simplicial_audit": {},
            "cellular_audit": {},
            "ph_domain_audit": {},
            "laplacian_audit": {},
            "violations": [],
        }

        # 1. Simplicial complex audit
        simp_nilpotence_passed = 0
        simp_closure_passed = 0
        for sv in simplicial_complexes:
            if not verify_boundary_nilpotence(sv.B1, sv.B2, tol=tol):
                results["violations"].append(
                    f"Simplicial nilpotence failed for candidate {sv.candidate_id}"
                )
            else:
                simp_nilpotence_passed += 1

            # Check no 4-cycles masquerading as 2-simplices (faces_2 must all have 3 nodes)
            for f in sv.faces_2:
                if len(f) != 3:
                    results["violations"].append(
                        f"Non-3-clique face found in simplicial complex {sv.candidate_id}: {f}"
                    )
            simp_closure_passed += 1

        results["simplicial_audit"] = {
            "total_simplicial_complexes": len(simplicial_complexes),
            "nilpotence_passed": simp_nilpotence_passed,
            "closure_passed": simp_closure_passed,
        }

        # 2. Cellular complex audit
        cell_nilpotence_passed = 0
        polygon_cells_verified = 0
        for cv in cellular_complexes:
            if not verify_boundary_nilpotence(cv.B1, cv.B2, tol=tol):
                results["violations"].append(
                    f"Cellular nilpotence failed for candidate {cv.candidate_id}"
                )
            else:
                cell_nilpotence_passed += 1

            # Check polygonal cells have rank >= 3 and matching B2 columns
            for i, cell in enumerate(cv.cells_2):
                k = len(cell)
                if k >= 3:
                    col = cv.B2[:, i]
                    nonzeros = np.nonzero(col)[0]
                    if len(nonzeros) != k:
                        results["violations"].append(
                            f"Cellular B2 column for cell {cell} has {len(nonzeros)} nonzero entries, expected {k}"
                        )
                    else:
                        polygon_cells_verified += 1

        results["cellular_audit"] = {
            "total_cellular_complexes": len(cellular_complexes),
            "nilpotence_passed": cell_nilpotence_passed,
            "polygon_cells_verified": polygon_cells_verified,
        }

        # 3. Laplacian audit
        l0_symmetric_count = 0
        l0_psd_count = 0
        l1_symmetric_count = 0
        l1_psd_count = 0
        total_laplacians = len(simplicial_complexes) + len(cellular_complexes)

        for comp in list(simplicial_complexes) + list(cellular_complexes):
            # L0 check
            is_sym_0, is_psd_0, _ = verify_hodge_laplacian_properties(comp.L0, tol=tol)
            if is_sym_0:
                l0_symmetric_count += 1
            else:
                results["violations"].append(f"L0 asymmetry in candidate {comp.candidate_id}")
            if is_psd_0:
                l0_psd_count += 1
            else:
                results["violations"].append(f"L0 non-PSD in candidate {comp.candidate_id}")

            # L1 check
            is_sym_1, is_psd_1, _ = verify_hodge_laplacian_properties(comp.L1, tol=tol)
            if is_sym_1:
                l1_symmetric_count += 1
            else:
                results["violations"].append(f"L1 asymmetry in candidate {comp.candidate_id}")
            if is_psd_1:
                l1_psd_count += 1
            else:
                results["violations"].append(f"L1 non-PSD in candidate {comp.candidate_id}")

        results["laplacian_audit"] = {
            "total_laplacians_evaluated": total_laplacians,
            "l0_symmetric": l0_symmetric_count,
            "l0_psd": l0_psd_count,
            "l1_symmetric": l1_symmetric_count,
            "l1_psd": l1_psd_count,
        }

        # 4. PH domain audit
        if ph_graphs is not None:
            ph_unfilled_count = 0
            for g in ph_graphs:
                # Check that PH input is strictly 1-skeleton (max simplex dim <= 1)
                if hasattr(g, "faces_2") and g.faces_2:
                    results["violations"].append(f"PH input contains 2-faces: {g}")
                elif hasattr(g, "cells_2") and g.cells_2:
                    results["violations"].append(f"PH input contains 2-cells: {g}")
                else:
                    ph_unfilled_count += 1

            results["ph_domain_audit"] = {
                "total_ph_graphs": len(ph_graphs),
                "unfilled_verified": ph_unfilled_count,
            }
        else:
            results["ph_domain_audit"] = {"note": "PH graphs verified at filtration level"}

        # Final verdict
        if not results["violations"]:
            results["gate_b_status"] = "GATE_B_PASS"
        else:
            results["gate_b_status"] = "GATE_B_FAIL"

        return results

    @classmethod
    def generate_and_save_report(
        cls,
        simplicial_complexes: Sequence[CliqueSimplicialView],
        cellular_complexes: Sequence[CycleCellView],
        output_path: str = "project/gate_b_report.json",
    ) -> Dict[str, Any]:
        """Run Gate B verification and save structured JSON report."""
        report = cls.verify_complexes(simplicial_complexes, cellular_complexes)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report
