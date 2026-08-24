"""Candidate-relative normalized persistent homology filtration and vectorizer.

Contract C12-01: Standardized [0, 1] normalized temporal filtration on unfilled 1-skeletons (INV-003),
computing H_0 and H_1 persistent homology intervals and generating consistent 372-dimensional
topological representation vectors.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from source.ph.gudhi_backend import compute_persistence_diagrams


def compute_normalized_betti_curve(
    intervals: np.ndarray,
    num_bins: int = 100,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
) -> np.ndarray:
    """Compute discretized 1D Betti curve over a uniform grid in [grid_min, grid_max]."""
    grid = np.linspace(grid_min, grid_max, num_bins, dtype=np.float32)
    betti_curve = np.zeros(num_bins, dtype=np.float32)

    if intervals.size > 0:
        for b, d in intervals:
            active = (grid >= b) & (grid < d)
            betti_curve += active.astype(np.float32)

    return betti_curve


def compute_normalized_landscapes(
    intervals: np.ndarray,
    num_landscapes: int = 3,
    num_bins: int = 25,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
) -> np.ndarray:
    """Compute top-k persistence landscape functions discretized over [0, 1]."""
    grid = np.linspace(grid_min, grid_max, num_bins, dtype=np.float32)
    landscapes = np.zeros((num_landscapes, num_bins), dtype=np.float32)

    if intervals.size > 0:
        tent_values_at_t = [[] for _ in range(num_bins)]
        for b, d in intervals:
            if d <= b:
                continue
            for t_idx, t in enumerate(grid):
                val = max(0.0, min(float(t - b), float(d - t)))
                if val > 0.0:
                    tent_values_at_t[t_idx].append(val)

        for t_idx in range(num_bins):
            sorted_tents = sorted(tent_values_at_t[t_idx], reverse=True)
            for k in range(min(num_landscapes, len(sorted_tents))):
                landscapes[k, t_idx] = sorted_tents[k]

    return landscapes.flatten()


def compute_persistence_entropy(intervals: np.ndarray) -> float:
    """Compute Shannon entropy of persistence interval lifetimes."""
    if intervals.size == 0:
        return 0.0

    lifetimes = np.maximum(0.0, intervals[:, 1] - intervals[:, 0])
    tot = float(np.sum(lifetimes))
    if tot <= 1e-9:
        return 0.0

    probs = lifetimes / tot
    probs = probs[probs > 1e-9]
    return float(-np.sum(probs * np.log2(probs)))


def extract_persistence_statistics(intervals: np.ndarray, raw_intervals: List[Tuple[float, float]]) -> np.ndarray:
    """Extract 10 summary statistics for persistence dimension."""
    stats = np.zeros(10, dtype=np.float32)
    if intervals.size == 0 and not raw_intervals:
        return stats

    finite_count = 0
    essential_count = 0
    births = []
    lifetimes = []

    for b, d in raw_intervals:
        births.append(b)
        if np.isinf(d) or d >= 1e6:
            essential_count += 1
        else:
            finite_count += 1
            lifetimes.append(max(0.0, d - b))

    stats[0] = float(finite_count)
    stats[1] = float(essential_count)
    stats[2] = float(np.sum(lifetimes)) if lifetimes else 0.0
    stats[3] = float(np.max(lifetimes)) if lifetimes else 0.0
    stats[4] = float(np.mean(lifetimes)) if lifetimes else 0.0
    stats[5] = float(np.std(lifetimes)) if len(lifetimes) > 1 else 0.0
    stats[6] = float(np.min(births)) if births else 0.0
    stats[7] = float(np.max(births)) if births else 0.0
    stats[8] = float(np.mean(births)) if births else 0.0
    stats[9] = float(np.std(births)) if len(births) > 1 else 0.0

    return stats


class NormalizedPersistenceVectorizer:
    """Computes candidate-relative normalized persistence diagrams and vectorizes to 372 dimensions."""

    def __init__(
        self,
        num_betti_bins: int = 100,
        num_landscapes: int = 3,
        landscape_bins: int = 25,
    ):
        self.num_betti_bins = num_betti_bins
        self.num_landscapes = num_landscapes
        self.landscape_bins = landscape_bins

    def extract_diagrams(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Build unfilled 1-skeleton filtration and extract normalized H0 and H1 diagrams."""
        nodes = candidate.get("nodes", candidate.get("participants", []))
        txs = candidate.get("transactions", [])

        node_str_list = [str(n) for n in nodes]
        node_map = {n: i for i, n in enumerate(node_str_list)}

        # Temporal bounds
        epochs = [float(tx.get("timestamp_epoch", tx.get("timestamp", 0.0))) for tx in txs]
        valid_epochs = [t for t in epochs if t > 0]
        min_ts = min(valid_epochs) if valid_epochs else 0.0
        max_ts = max(valid_epochs) if valid_epochs else 1.0
        span = max(1.0, max_ts - min_ts)

        # Build filtration list for compute_persistence_diagrams
        # Simplices: (tuple of vertex indices, filtration_val)
        simplices: List[Tuple[Tuple[int, ...], float]] = []

        for i in range(len(node_str_list)):
            simplices.append(((i,), 0.0))

        for tx in txs:
            u = str(tx.get("from_account", ""))
            v = str(tx.get("to_account", ""))
            if u in node_map and v in node_map and u != v:
                ts = float(tx.get("timestamp_epoch", tx.get("timestamp", 0.0)))
                norm_f = max(0.0, min(1.0, (ts - min_ts) / span)) if ts > 0 else 0.0
                simplices.append(((node_map[u], node_map[v]), norm_f))

        res = compute_persistence_diagrams(simplices)
        h0_raw = res["H0"]
        h1_raw = res["H1"]

        h0_capped: List[Tuple[float, float]] = []
        h1_capped: List[Tuple[float, float]] = []

        for b, d in h0_raw:
            norm_b = max(0.0, min(1.0, float(b)))
            is_inf = np.isinf(d) or d >= 1e6
            norm_d_capped = 1.0 if is_inf else max(0.0, min(1.0, float(d)))
            h0_capped.append((norm_b, norm_d_capped))

        for b, d in h1_raw:
            norm_b = max(0.0, min(1.0, float(b)))
            is_inf = np.isinf(d) or d >= 1e6
            norm_d_capped = 1.0 if is_inf else max(0.0, min(1.0, float(d)))
            h1_capped.append((norm_b, norm_d_capped))

        return {
            "h0_raw": h0_raw,
            "h1_raw": h1_raw,
            "h0_intervals": np.array(h0_capped, dtype=np.float32).reshape(-1, 2) if h0_capped else np.zeros((0, 2), dtype=np.float32),
            "h1_intervals": np.array(h1_capped, dtype=np.float32).reshape(-1, 2) if h1_capped else np.zeros((0, 2), dtype=np.float32),
        }

    def vectorize_candidate(self, candidate: Dict[str, Any]) -> np.ndarray:
        """Extract 372-dimensional normalized topological feature vector."""
        diag = self.extract_diagrams(candidate)

        h0_ints = diag["h0_intervals"]
        h1_ints = diag["h1_intervals"]

        # 1. Betti curves (2 x 100 = 200 dims)
        betti_h0 = compute_normalized_betti_curve(h0_ints, num_bins=self.num_betti_bins)
        betti_h1 = compute_normalized_betti_curve(h1_ints, num_bins=self.num_betti_bins)

        # 2. Persistence landscapes (2 x 3 x 25 = 150 dims)
        land_h0 = compute_normalized_landscapes(h0_ints, num_landscapes=self.num_landscapes, num_bins=self.landscape_bins)
        land_h1 = compute_normalized_landscapes(h1_ints, num_landscapes=self.num_landscapes, num_bins=self.landscape_bins)

        # 3. Persistence entropy (2 dims)
        ent_h0 = compute_persistence_entropy(h0_ints)
        ent_h1 = compute_persistence_entropy(h1_ints)
        entropy_vec = np.array([ent_h0, ent_h1], dtype=np.float32)

        # 4. Summary statistics (2 x 10 = 20 dims)
        stats_h0 = extract_persistence_statistics(h0_ints, diag["h0_raw"])
        stats_h1 = extract_persistence_statistics(h1_ints, diag["h1_raw"])

        # Concatenate: 200 + 150 + 2 + 20 = 372
        feature_vector = np.concatenate([
            betti_h0,
            betti_h1,
            land_h0,
            land_h1,
            entropy_vec,
            stats_h0,
            stats_h1,
        ]).astype(np.float32)

        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        assert feature_vector.shape == (372,), f"Expected shape (372,), got {feature_vector.shape}"
        return feature_vector
