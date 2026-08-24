"""Statistical testing and metric computation engine for rigorous scientific validation."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive classification performance metrics."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_t)) < 2:
        acc = float(np.mean(y_t == (y_p >= threshold)))
        return {
            "pr_auc": 1.0 if acc == 1.0 else 0.5,
            "roc_auc": 1.0 if acc == 1.0 else 0.5,
            "f1_macro": 1.0 if acc == 1.0 else 0.5,
            "f1_positive": 1.0 if acc == 1.0 else 0.5,
            "precision": 1.0 if acc == 1.0 else 0.5,
            "recall": 1.0 if acc == 1.0 else 0.5,
            "brier_score": float(np.mean((y_p - y_t) ** 2)),
        }

    pr_auc = float(average_precision_score(y_t, y_p))
    roc_auc = float(roc_auc_score(y_t, y_p))
    y_pred = (y_p >= threshold).astype(int)

    f1_macro = float(f1_score(y_t, y_pred, average="macro", zero_division=0))
    f1_positive = float(f1_score(y_t, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_t, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_t, y_pred, pos_label=1, zero_division=0))
    brier = float(brier_score_loss(y_t, y_p))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_macro": f1_macro,
        "f1_positive": f1_positive,
        "precision": prec,
        "recall": rec,
        "brier_score": brier,
    }


def paired_wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Tuple[float, float]:
    """Execute two-sided paired Wilcoxon signed-rank test.

    Returns:
        (W_statistic, p_value)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Arrays must have identical length: {len(a)} vs {len(b)}")

    diffs = a - b
    if np.all(diffs == 0.0) or len(diffs) == 0:
        return 0.0, 1.0

    # Non-zero differences count
    nonzero_diffs = diffs[diffs != 0.0]
    if len(nonzero_diffs) == 0:
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Fallback to wilcoxon with wilcox zero method
        res = stats.wilcoxon(nonzero_diffs, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)


def compute_cliffs_delta(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    num_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Cliff's delta non-parametric effect size with bootstrap 95% confidence intervals."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    n_a = len(a)
    n_b = len(b)

    if n_a == 0 or n_b == 0:
        return {
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "magnitude": "negligible",
        }

    # Vectorized point estimate calculation
    # delta = ( # (a_i > b_j) - # (a_i < b_j) ) / (n_a * n_b)
    comp_matrix = np.sign(a[:, None] - b[None, :])
    point_delta = float(np.mean(comp_matrix))

    # Bootstrap confidence interval
    rng = np.random.RandomState(seed)
    bootstrap_deltas = np.empty(num_bootstraps, dtype=float)

    for i in range(num_bootstraps):
        idx_a = rng.randint(0, n_a, size=n_a)
        idx_b = rng.randint(0, n_b, size=n_b)
        boot_comp = np.sign(a[idx_a, None] - b[None, idx_b])
        bootstrap_deltas[i] = np.mean(boot_comp)

    ci_low = float(np.percentile(bootstrap_deltas, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(bootstrap_deltas, 100.0 * (1.0 - alpha / 2.0)))

    # Magnitude categorization according to Romano et al. / Hess et al. thresholds
    abs_d = abs(point_delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "delta": point_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "magnitude": magnitude,
    }


def holm_bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Execute Holm-Bonferroni step-down procedure with monotonic adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value ascending
    indexed_p = [(i, float(p)) for i, p in enumerate(p_values)]
    indexed_p.sort(key=lambda x: x[1])

    # Compute step-down adjusted p-values enforcing monotonicity
    adjusted_p = [0.0] * m
    rejects = [False] * m

    cum_max = 0.0
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank_idx
        raw_adj = min(1.0, p_val * multiplier)
        cum_max = max(cum_max, raw_adj)
        adjusted_p[orig_idx] = cum_max

    # Step-down rejection logic: reject until first p_val > alpha / (m - i + 1)
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        step_alpha = alpha / (m - rank_idx)
        if p_val <= step_alpha:
            rejects[orig_idx] = True
        else:
            # Stop step-down rejection on first non-significant hypothesis
            break

    results = []
    for i in range(m):
        results.append({
            "original_index": i,
            "raw_p_value": float(p_values[i]),
            "adjusted_p_value": float(adjusted_p[i]),
            "is_significant": bool(rejects[i]),
            "alpha": alpha,
        })

    return results


def evaluate_hypothesis_verdict(
    p_adj: float,
    delta: float,
    alpha: float = 0.05,
    delta_threshold: float = 0.147,
) -> str:
    """Assign three-valued scientific verdict (SUPPORTED, FALSIFIED, INCONCLUSIVE)."""
    if p_adj < alpha and delta >= delta_threshold:
        return "SUPPORTED"
    elif p_adj < alpha and delta <= -delta_threshold:
        return "FALSIFIED"
    else:
        return "INCONCLUSIVE"
