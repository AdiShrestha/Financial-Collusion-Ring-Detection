"""Experimental Protocol Lock and Pre-Registration Verification Engine (Gate D)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ProtocolLock:
    """Enforces pre-registration protocol lock and verifies cryptographic test set immutability."""

    LOCKED_TEST_HASH = "d99fbc094a1b072da17090d1ad56dfce3478427abefc79e943594ebd5680452c"
    LOCKED_SEEDS = [42, 43, 44, 45, 46]
    ALPHA = 0.05
    CLIFFS_DELTA_THRESHOLD = 0.147
    HYPOTHESES = ["H1", "H2", "H3", "H4"]

    def __init__(
        self,
        manifest_path: str = "data/manifests/split_manifest.json",
        pre_reg_path: str = "project/PRE_REGISTRATION.md",
    ):
        self.manifest_path = manifest_path
        self.pre_reg_path = pre_reg_path

    def verify_test_partition_hash(self) -> Tuple[bool, str]:
        """Verify that split_manifest.json test candidate partition matches the locked SHA-256 hash."""
        if not os.path.exists(self.manifest_path):
            return False, f"Manifest file not found: {self.manifest_path}"

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            test_split = manifest_data.get("splits", {}).get("test", {})
            recorded_hash = test_split.get("sha256_checksum", "")
            candidate_ids = test_split.get("candidate_ids", [])

            # Recompute hash over sorted candidate IDs
            computed_hash = hashlib.sha256("\n".join(sorted(candidate_ids)).encode("utf-8")).hexdigest()

            if recorded_hash != self.LOCKED_TEST_HASH:
                return False, f"Recorded hash {recorded_hash} does not match locked hash {self.LOCKED_TEST_HASH}"

            if computed_hash != self.LOCKED_TEST_HASH:
                return False, f"Computed hash {computed_hash} does not match locked hash {self.LOCKED_TEST_HASH}"

            return True, computed_hash
        except Exception as e:
            return False, f"Error verifying manifest: {str(e)}"

    def verify_pre_registration_document(self) -> Tuple[bool, List[str]]:
        """Verify that PRE_REGISTRATION.md is present and contains all required sections."""
        if not os.path.exists(self.pre_reg_path):
            return False, [f"Missing pre-registration document at '{self.pre_reg_path}'"]

        with open(self.pre_reg_path, "r", encoding="utf-8") as f:
            content = f.read()

        missing_sections = []
        required_elements = [
            "H1",
            "H2",
            "H3",
            "H4",
            "Wilcoxon",
            "Cliff's delta",
            "Holm-Bonferroni",
            "PR-AUC",
            "SUPPORTED",
            "FALSIFIED",
            "INCONCLUSIVE",
            self.LOCKED_TEST_HASH,
        ]

        for req in required_elements:
            if req not in content:
                missing_sections.append(f"Missing required element: '{req}'")

        return len(missing_sections) == 0, missing_sections

    @staticmethod
    def compute_verdict(
        p_adj: float,
        effect_size: float,
        alpha: float = 0.05,
        effect_threshold: float = 0.147,
    ) -> str:
        """Evaluate three-valued hypothesis verdict based on adjusted p-value and Cliff's delta."""
        if p_adj < alpha and effect_size >= effect_threshold:
            return "SUPPORTED"
        elif p_adj < alpha and effect_size <= -effect_threshold:
            return "FALSIFIED"
        else:
            return "INCONCLUSIVE"

    def get_protocol_summary(self) -> Dict[str, Any]:
        """Return standardized protocol metadata dictionary."""
        is_hash_valid, hash_msg = self.verify_test_partition_hash()
        is_doc_valid, doc_msgs = self.verify_pre_registration_document()

        return {
            "locked_test_hash": self.LOCKED_TEST_HASH,
            "is_test_hash_valid": is_hash_valid,
            "hash_message": hash_msg,
            "is_pre_registration_valid": is_doc_valid,
            "document_validation_errors": doc_msgs,
            "evaluation_seeds": self.LOCKED_SEEDS,
            "alpha": self.ALPHA,
            "cliffs_delta_threshold": self.CLIFFS_DELTA_THRESHOLD,
            "hypotheses": self.HYPOTHESES,
            "verdict_classes": ["SUPPORTED", "FALSIFIED", "INCONCLUSIVE"],
        }
