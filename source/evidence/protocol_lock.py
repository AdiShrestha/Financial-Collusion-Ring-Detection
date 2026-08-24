"""Experimental Protocol Lock and Pre-Registration Verification Engine (Gate D)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ProtocolLock:
    """Enforces pre-registration protocol lock and verifies cryptographic test set immutability."""

    LOCKED_TEST_HASH = "43ece91757021473cc7fb5060ef034083637fa35a68a85cc7095fc18715eae37"
    LOCKED_TEST_HASHES = {
        "43ece91757021473cc7fb5060ef034083637fa35a68a85cc7095fc18715eae37",
        "d99fbc094a1b072da17090d1ad56dfce3478427abefc79e943594ebd5680452c",
    }
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

            if recorded_hash not in self.LOCKED_TEST_HASHES:
                return False, f"Recorded hash {recorded_hash} not in locked hashes"

            if computed_hash != recorded_hash:
                return False, f"Computed hash {computed_hash} does not match recorded hash {recorded_hash}"

            return True, computed_hash
        except Exception as e:
            return False, f"Error verifying manifest: {str(e)}"

    def verify_pre_registration_document(self) -> Tuple[bool, List[str]]:
        """Verify that PRE_REGISTRATION.md is present and contains all required sections."""
        if not os.path.exists(self.pre_reg_path):
            return False, [f"Missing pre-registration document at '{self.pre_reg_path}'"]

        with open(self.pre_reg_path, "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = [
            "Scientific Objective",
            "Confirmatory Hypotheses",
            "Evaluation Metrics",
            "Statistical Testing Protocol",
        ]

        missing = []
        for sec in required_sections:
            if sec.lower() not in content.lower():
                missing.append(sec)

        if missing:
            return False, [f"Missing required section in PRE_REGISTRATION.md: '{s}'" for s in missing]

        return True, []

    @staticmethod
    def compute_verdict(p_adj: float, effect_size: float, alpha: float = 0.05, threshold: float = 0.147) -> str:
        """Evaluate hypothesis verdict using three-valued logic."""
        if p_adj < alpha:
            if effect_size >= threshold:
                return "SUPPORTED"
            elif effect_size <= -threshold:
                return "FALSIFIED"
        return "INCONCLUSIVE"

    def get_protocol_summary(self) -> Dict[str, Any]:
        """Return protocol summary dictionary for Gate D verification."""
        is_test_hash_valid, _ = self.verify_test_partition_hash()
        is_pre_reg_valid, _ = self.verify_pre_registration_document()
        return {
            "is_test_hash_valid": is_test_hash_valid,
            "is_pre_registration_valid": is_pre_reg_valid,
            "evaluation_seeds": self.LOCKED_SEEDS,
            "alpha": self.ALPHA,
            "cliffs_delta_threshold": self.CLIFFS_DELTA_THRESHOLD,
            "hypotheses": self.HYPOTHESES,
        }

    def verify_protocol_integrity(self) -> Dict[str, Any]:
        """Comprehensive verification of pre-registration compliance."""
        test_hash_valid, hash_msg = self.verify_test_partition_hash()
        doc_valid, missing_sections = self.verify_pre_registration_document()

        passed = test_hash_valid and doc_valid

        return {
            "passed": passed,
            "test_hash_valid": test_hash_valid,
            "test_hash_details": hash_msg,
            "doc_valid": doc_valid,
            "missing_sections": missing_sections,
            "locked_seeds": self.LOCKED_SEEDS,
            "alpha": self.ALPHA,
            "hypotheses": self.HYPOTHESES,
        }
