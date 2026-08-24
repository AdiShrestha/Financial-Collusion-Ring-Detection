"""Oracle tests for Pre-Registration Document & Protocol Lock (Gate D)."""

import pytest
from source.evidence.protocol_lock import ProtocolLock


def test_pre_registration_document_exists_and_complete():
    """Verify PRE_REGISTRATION.md is present and contains required hypothesis specifications and statistical rules."""
    lock = ProtocolLock()
    is_valid, errors = lock.verify_pre_registration_document()
    assert is_valid, f"Pre-registration document validation failed with errors: {errors}"


def test_test_set_hash_immutability():
    """Assert that the test split hash in split_manifest.json strictly matches the locked Gate D hash."""
    lock = ProtocolLock()
    is_valid, msg = lock.verify_test_partition_hash()
    assert is_valid, f"Test partition hash verification failed: {msg}"
    assert msg == ProtocolLock.LOCKED_TEST_HASH


def test_verdict_three_valued_logic():
    """Verify three-valued decision logic maps accurately to SUPPORTED, FALSIFIED, and INCONCLUSIVE."""
    # SUPPORTED: p_adj < 0.05 and d >= 0.147
    assert ProtocolLock.compute_verdict(p_adj=0.01, effect_size=0.25) == "SUPPORTED"
    assert ProtocolLock.compute_verdict(p_adj=0.049, effect_size=0.147) == "SUPPORTED"

    # FALSIFIED: p_adj < 0.05 and d <= -0.147
    assert ProtocolLock.compute_verdict(p_adj=0.01, effect_size=-0.30) == "FALSIFIED"
    assert ProtocolLock.compute_verdict(p_adj=0.049, effect_size=-0.147) == "FALSIFIED"

    # INCONCLUSIVE: p_adj >= 0.05 or |d| < 0.147
    assert ProtocolLock.compute_verdict(p_adj=0.06, effect_size=0.50) == "INCONCLUSIVE"  # High effect, non-significant p
    assert ProtocolLock.compute_verdict(p_adj=0.01, effect_size=0.10) == "INCONCLUSIVE"  # Significant p, negligible effect
    assert ProtocolLock.compute_verdict(p_adj=0.20, effect_size=0.02) == "INCONCLUSIVE"


def test_protocol_summary_structure():
    """Verify that protocol summary contains all mandatory evaluation metadata."""
    lock = ProtocolLock()
    summary = lock.get_protocol_summary()

    assert summary["is_test_hash_valid"] is True
    assert summary["is_pre_registration_valid"] is True
    assert summary["evaluation_seeds"] == [42, 43, 44, 45, 46]
    assert summary["alpha"] == 0.05
    assert summary["cliffs_delta_threshold"] == 0.147
    assert len(summary["hypotheses"]) == 4
