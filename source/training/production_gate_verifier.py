"""Production Training & Checkpoint Integrity Certification Engine.

Contract C12-05 (T-COMP): Formal gate verifier auditing the physical existence, weight health,
and non-zero parameter norms of all 35 model checkpoints (7 models x 5 seeds), certifying
PRODUCTION_TRAINING_PASS before confirmatory hypothesis testing.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.training.production_trainer import LOCKED_SEEDS, MODEL_NAMES


class ProductionTrainingGateVerifier:
    """Audits production model checkpoints and certifies training integrity."""

    def __init__(self, project_root: str = "."):
        self.project_root = project_root

    def verify_production_checkpoints(
        self,
        checkpoint_dir: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Audit all 35 checkpoints and export gate certification report."""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        if output_path is None:
            output_path = os.path.join(self.project_root, "project/production_training_report.json")

        audited_checkpoints: List[Dict[str, Any]] = []
        missing_checkpoints: List[str] = []
        corrupt_checkpoints: List[str] = []

        for model_name in MODEL_NAMES:
            for seed in LOCKED_SEEDS:
                fname = f"{model_name}_seed{seed}.pt"
                fpath = os.path.join(checkpoint_dir, fname)

                if not os.path.exists(fpath) or os.path.getsize(fpath) < 1000:
                    missing_checkpoints.append(fname)
                    continue

                try:
                    data = torch.load(fpath, map_location="cpu", weights_only=False)
                    state_dict = data.get("model_state_dict", {})
                    if not state_dict:
                        corrupt_checkpoints.append(f"{fname}: empty state_dict")
                        continue

                    has_nan = False
                    total_norm = 0.0
                    for p_tensor in state_dict.values():
                        if torch.isnan(p_tensor).any() or torch.isinf(p_tensor).any():
                            has_nan = True
                            break
                        total_norm += float(torch.norm(p_tensor.float()).item())

                    if has_nan or total_norm <= 0.0:
                        corrupt_checkpoints.append(f"{fname}: non-finite or zero weights")
                        continue

                    audited_checkpoints.append({
                        "model_name": model_name,
                        "seed": seed,
                        "best_val_auprc": float(data.get("best_val_auprc", 0.5)),
                        "best_epoch": int(data.get("best_epoch", 0)),
                        "weight_norm": total_norm,
                    })
                except Exception as e:
                    corrupt_checkpoints.append(f"{fname}: load error ({str(e)})")

        total_audited = len(audited_checkpoints)
        all_passed = (total_audited == 35 and len(missing_checkpoints) == 0 and len(corrupt_checkpoints) == 0)
        status = "PRODUCTION_TRAINING_PASS" if all_passed else "PRODUCTION_TRAINING_FAIL"

        report = {
            "gate_id": "PRODUCTION_TRAINING_GATE",
            "status": status,
            "production_training_status": status,
            "all_subchecks_passed": all_passed,
            "total_checkpoints_audited": total_audited,
            "required_checkpoints_count": 35,
            "models_verified": MODEL_NAMES,
            "seeds_verified": LOCKED_SEEDS,
            "missing_checkpoints": missing_checkpoints,
            "corrupt_checkpoints": corrupt_checkpoints,
            "invariants_verified": ["INV-001", "INV-005", "INV-006"],
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    verifier = ProductionTrainingGateVerifier()
    res = verifier.verify_production_checkpoints()
    print(f"Production Training Gate Status: {res['status']}")
