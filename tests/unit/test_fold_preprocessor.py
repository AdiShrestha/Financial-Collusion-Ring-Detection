"""Unit tests for fold-safe AML feature builder and preprocessors (Contract C17-01)."""

import os
import pickle
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from source.features.fold_preprocessor import AMLFeatureBuilder, build_fold_preprocessors


def test_aml_feature_builder_fit_transform_isolation():
    """Verify AMLFeatureBuilder fit/transform separation and exact scaling."""
    builder = AMLFeatureBuilder()
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    X_test = np.array([[2.0, 3.0]], dtype=np.float32)

    with pytest.raises(RuntimeError):
        builder.transform(X_test)

    X_train_scaled = builder.fit_transform(X_train)
    assert np.allclose(np.mean(X_train_scaled, axis=0), [0.0, 0.0], atol=1e-5)

    X_test_scaled = builder.transform(X_test)
    assert X_test_scaled.shape == (1, 2)


def test_build_fold_preprocessors_across_5_folds():
    """Verify preprocessors are built and serialized cleanly for all 5 outer folds."""
    res = build_fold_preprocessors(
        candidates_parquet_path="artifacts/candidates/candidates.parquet",
        candidate_txs_parquet_path="artifacts/candidates/candidate_transactions.parquet",
        fold_manifest_path="artifacts/splits/fold_manifest.json",
        output_dir="artifacts/preprocessors",
    )

    assert res["status"] == "PREPROCESSORS_BUILT"
    assert res["n_outer_folds"] == 5

    for f_id in range(5):
        pkl_path = f"artifacts/preprocessors/fold_{f_id}_preprocessor.pkl"
        assert os.path.exists(pkl_path)

        with open(pkl_path, "rb") as f:
            builder = pickle.load(f)
        assert builder.fitted is True
