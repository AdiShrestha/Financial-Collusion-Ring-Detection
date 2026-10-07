#!/usr/bin/env python3
"""
Group-Level Clustered Bootstrap & TOST Practical Equivalence Statistical Engine.

Implements:
1. Clustered paired bootstrap resampling preserving group/pattern-level independence (MAR-3, C65).
2. Empirical percentile confidence intervals and standardized effect size estimation (C66, D-063).
3. Two One-Sided Tests (TOST) for practical equivalence against pre-registered delta_equiv (MAR-7).
4. Holm-Bonferroni step-down multiple-comparisons correction for confirmatory hypothesis family (D-064).

Upholds Invariants:
- INV-001 (No Mock Data in Production): Operates strictly on real score distributions.
- INV-008 (Self-Contained Verification Scripts): Standalone numerical verification.
- MAR-3 (Statistical Power & Analysis Plan).
- MAR-7 (Pre-Registered Seeds & Evaluation Freeze Compliance).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_standardized_difference(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Computes Cohen's d standardized effect size for two score samples.
    d = (mean_a - mean_b) / s_pooled
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    n_a = len(a)
    n_b = len(b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    s_pooled = math.sqrt(max(1e-12, pooled_var))
    return round(float(mean_diff / s_pooled), 4)


def cluster_bootstrap_paired_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: List[Any],
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs cluster bootstrap resampling by pattern group cluster.

    Args:
        scores_a: Metric scores from model A.
        scores_b: Metric scores from model B.
        cluster_ids: Cluster / group identifier for each observation.
        n_bootstraps: Number of bootstrap resamples (default 2000).
        alpha: Significance level (default 0.05).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dictionary with point estimate, percentile CI, empirical p-value, and effect size.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    clusters = np.asarray(cluster_ids)

    if len(a) != len(b) or len(a) != len(clusters):
        raise ValueError(f"Length mismatch: len(a)={len(a)}, len(b)={len(b)}, len(clusters)={len(clusters)}")

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        raise ValueError("Need at least 2 unique clusters for clustered bootstrap.")

    point_estimate = float(np.mean(a) - np.mean(b))
    std_diff = compute_standardized_difference(a, b)

    # Group index lookup
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    rng = np.random.RandomState(seed)
    bootstrap_diffs = np.empty(n_bootstraps, dtype=np.float64)

    for b_idx in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_idx_list = [cluster_indices[c] for c in sampled_clusters]
        sampled_indices = np.concatenate(sampled_idx_list)
        bootstrap_diffs[b_idx] = np.mean(a[sampled_indices]) - np.mean(b[sampled_indices])

    # Percentile confidence interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(bootstrap_diffs, lower_pct))
    ci_upper = float(np.percentile(bootstrap_diffs, upper_pct))

    # One-sided p-value against H0: theta <= 0
    p_value = float(np.mean(bootstrap_diffs <= 0.0))

    return {
        "point_estimate": round(point_estimate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "standardized_difference": std_diff,
        "n_clusters": int(n_clusters),
        "n_bootstraps": int(n_bootstraps),
        "bootstrap_mean": round(float(np.mean(bootstrap_diffs)), 4),
        "bootstrap_std": round(float(np.std(bootstrap_diffs, ddof=1)), 4),
    }


def two_one_sided_tests_equivalence(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: List[Any],
    delta_equiv: float = 0.025,
    alpha: float = 0.05,
    n_bootstraps: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Two One-Sided Tests (TOST) for practical equivalence within [-delta_equiv, +delta_equiv].

    Args:
        scores_a: Metric scores from model A.
        scores_b: Metric scores from model B.
        cluster_ids: Cluster / group identifier for each observation.
        delta_equiv: Equivalence margin (default 0.025 = 2.5%).
        alpha: Alpha level (default 0.05).
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dictionary with TOST p-values, 90% CI, and equivalence determination.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    clusters = np.asarray(cluster_ids)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    rng = np.random.RandomState(seed)
    bootstrap_diffs = np.empty(n_bootstraps, dtype=np.float64)

    for b_idx in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_idx_list = [cluster_indices[c] for c in sampled_clusters]
        sampled_indices = np.concatenate(sampled_idx_list)
        bootstrap_diffs[b_idx] = np.mean(a[sampled_indices]) - np.mean(b[sampled_indices])

    point_estimate = float(np.mean(a) - np.mean(b))

    # In TOST at level alpha, practical equivalence corresponds to (1 - 2*alpha) CI inside (-delta, +delta)
    lower_pct = 100.0 * alpha
    upper_pct = 100.0 * (1.0 - alpha)
    ci_lower_90 = float(np.percentile(bootstrap_diffs, lower_pct))
    ci_upper_90 = float(np.percentile(bootstrap_diffs, upper_pct))

    # Test 1: H0_1: theta <= -delta_equiv (p_lower = P(diff <= -delta))
    p_lower = float(np.mean(bootstrap_diffs <= -delta_equiv))
    # Test 2: H0_2: theta >= +delta_equiv (p_upper = P(diff >= +delta))
    p_upper = float(np.mean(bootstrap_diffs >= delta_equiv))

    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(ci_lower_90 > -delta_equiv and ci_upper_90 < delta_equiv)

    return {
        "delta_equiv": delta_equiv,
        "point_estimate": round(point_estimate, 4),
        "ci_lower_90": round(ci_lower_90, 4),
        "ci_upper_90": round(ci_upper_90, 4),
        "p_lower": round(p_lower, 4),
        "p_upper": round(p_upper, 4),
        "p_tost": round(p_tost, 4),
        "is_equivalent": is_equivalent,
    }


def holm_bonferroni_correction(
    p_values: Dict[str, float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Applies Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).

    Args:
        p_values: Dict mapping hypothesis name to raw p-value.
        alpha: FWER threshold (default 0.05).

    Returns:
        Structured dictionary detailing rankings, thresholds, adjusted p-values, and rejections.
    """
    k = len(p_values)
    sorted_hypotheses = sorted(p_values.items(), key=lambda item: item[1])

    results = {}
    stopped = False

    cum_max = 0.0
    for rank, (hyp_name, raw_p) in enumerate(sorted_hypotheses):
        divisor = k - rank
        threshold = alpha / divisor
        
        # Step-down rejection: if previously stopped, cannot reject
        if not stopped and raw_p <= threshold:
            rejected = True
        else:
            rejected = False
            stopped = True

        adj_p = min(1.0, raw_p * divisor)
        cum_max = max(cum_max, adj_p)
        adjusted_p = min(1.0, cum_max)

        results[hyp_name] = {
            "rank": rank + 1,
            "raw_p": round(float(raw_p), 5),
            "threshold": round(float(threshold), 5),
            "adjusted_p": round(float(adjusted_p), 5),
            "rejected_null": rejected,
        }

    return {
        "alpha": alpha,
        "num_hypotheses": k,
        "hypotheses": results,
    }
