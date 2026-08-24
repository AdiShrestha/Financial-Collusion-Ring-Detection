"""Topological feature vectorization module (Betti curves, landscapes, images, entropy)."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from source.ph.persistence_extractor import PersistenceDiagram


def compute_betti_curve(
    intervals: np.ndarray,
    num_bins: int = 20,
    grid_min: float = 0.0,
    grid_max: float = 100.0,
) -> np.ndarray:
    """Compute discretized 1D Betti curve over a uniform grid."""
    if grid_max <= grid_min:
        grid_max = grid_min + 1.0

    grid = np.linspace(grid_min, grid_max, num_bins, dtype=np.float32)
    betti_curve = np.zeros(num_bins, dtype=np.float32)

    if intervals.size > 0:
        for b, d in intervals:
            # Count where b <= t < d
            active = (grid >= b) & (grid < d)
            betti_curve += active.astype(np.float32)

    return betti_curve


def compute_persistence_landscape(
    intervals: np.ndarray,
    num_landscapes: int = 3,
    num_bins: int = 20,
    grid_min: float = 0.0,
    grid_max: float = 100.0,
) -> np.ndarray:
    """Compute top-k persistence landscape functions discretized over a uniform grid.

    Returns:
        Flattened 1D numpy array of shape (num_landscapes * num_bins,).
    """
    if grid_max <= grid_min:
        grid_max = grid_min + 1.0

    grid = np.linspace(grid_min, grid_max, num_bins, dtype=np.float32)
    landscapes = np.zeros((num_landscapes, num_bins), dtype=np.float32)

    if intervals.size > 0:
        tent_values_at_t = [[] for _ in range(num_bins)]
        for b, d in intervals:
            if d <= b:
                continue
            for t_idx, t in enumerate(grid):
                # Tent function Lambda_(b, d)(t) = max(0, min(t - b, d - t))
                val = max(0.0, min(float(t - b), float(d - t)))
                if val > 0.0:
                    tent_values_at_t[t_idx].append(val)

        for t_idx in range(num_bins):
            sorted_tents = sorted(tent_values_at_t[t_idx], reverse=True)
            for k in range(min(num_landscapes, len(sorted_tents))):
                landscapes[k, t_idx] = sorted_tents[k]

    return landscapes.flatten()


def compute_persistence_image(
    intervals: np.ndarray,
    resolution: Tuple[int, int] = (10, 10),
    sigma: float = 1.0,
    b_range: Tuple[float, float] = (0.0, 100.0),
    p_range: Tuple[float, float] = (0.0, 100.0),
) -> np.ndarray:
    """Compute 2D Gaussian Persistence Image on birth-persistence coordinates.

    Returns:
        Flattened 1D numpy array of shape (resolution[0] * resolution[1],).
    """
    res_b, res_p = resolution
    img = np.zeros((res_b, res_p), dtype=np.float32)

    if intervals.size > 0:
        b_min, b_max = b_range
        p_min, p_max = p_range
        if b_max <= b_min:
            b_max = b_min + 1.0
        if p_max <= p_min:
            p_max = p_min + 1.0

        b_grid = np.linspace(b_min, b_max, res_b, dtype=np.float32)
        p_grid = np.linspace(p_min, p_max, res_p, dtype=np.float32)
        B_mesh, P_mesh = np.meshgrid(b_grid, p_grid, indexing="ij")

        denom = 2.0 * (sigma**2)

        for b, d in intervals:
            p = max(0.0, float(d - b))
            if p <= 0.0:
                continue

            # Weighting function w(b, p) = arctan(p) / (pi / 2)
            weight = float(np.arctan(p) / (np.pi / 2.0))

            # 2D Gaussian kernel
            dist_sq = (B_mesh - b) ** 2 + (P_mesh - p) ** 2
            kernel = weight * np.exp(-dist_sq / denom)
            img += kernel

    return img.flatten()


def compute_persistent_entropy(intervals: np.ndarray) -> float:
    """Compute normalized Shannon persistent entropy over interval lifetimes."""
    if intervals.size == 0:
        return 0.0

    lifetimes = intervals[:, 1] - intervals[:, 0]
    lifetimes = lifetimes[lifetimes > 0.0]
    total_life = float(np.sum(lifetimes))

    if total_life <= 1e-12:
        return 0.0

    probs = lifetimes / total_life
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    return max(0.0, entropy)


def compute_summary_stats(intervals: np.ndarray) -> np.ndarray:
    """Compute summary statistics for persistence intervals.

    Returns:
        [count, total_persistence, mean_persistence, max_persistence, std_persistence, entropy]
    """
    if intervals.size == 0:
        return np.zeros(6, dtype=np.float32)

    lifetimes = intervals[:, 1] - intervals[:, 0]
    lifetimes = lifetimes[lifetimes >= 0.0]

    count = float(len(lifetimes))
    tot = float(np.sum(lifetimes)) if count > 0 else 0.0
    mean_val = float(np.mean(lifetimes)) if count > 0 else 0.0
    max_val = float(np.max(lifetimes)) if count > 0 else 0.0
    std_val = float(np.std(lifetimes)) if count > 0 else 0.0
    entropy = compute_persistent_entropy(intervals)

    return np.array([count, tot, mean_val, max_val, std_val, entropy], dtype=np.float32)


class PersistenceVectorizer:
    """Unified topological feature vectorizer mapping PersistenceDiagrams to fixed embeddings."""

    def __init__(
        self,
        betti_bins: int = 20,
        landscape_levels: int = 3,
        landscape_bins: int = 20,
        image_res: Tuple[int, int] = (10, 10),
        image_sigma: float = 1.0,
        grid_min: float = 0.0,
        grid_max: float = 100.0,
    ):
        self.betti_bins = betti_bins
        self.landscape_levels = landscape_levels
        self.landscape_bins = landscape_bins
        self.image_res = image_res
        self.image_sigma = image_sigma
        self.grid_min = grid_min
        self.grid_max = grid_max

        # Calculate fixed dimension per homology dimension
        self.dim_per_homology = (
            self.betti_bins
            + (self.landscape_levels * self.landscape_bins)
            + (self.image_res[0] * self.image_res[1])
            + 6  # summary stats
        )
        self.total_dim = self.dim_per_homology * 2  # H0 + H1

    @property
    def output_dim(self) -> int:
        return self.total_dim

    def vectorize_intervals(self, intervals: np.ndarray, max_filt: float) -> np.ndarray:
        """Vectorize a single homology interval array (H0 or H1)."""
        g_max = max(self.grid_max, max_filt + 1.0)

        betti = compute_betti_curve(intervals, num_bins=self.betti_bins, grid_min=self.grid_min, grid_max=g_max)
        land = compute_persistence_landscape(
            intervals,
            num_landscapes=self.landscape_levels,
            num_bins=self.landscape_bins,
            grid_min=self.grid_min,
            grid_max=g_max,
        )
        img = compute_persistence_image(
            intervals,
            resolution=self.image_res,
            sigma=self.image_sigma,
            b_range=(self.grid_min, g_max),
            p_range=(0.0, g_max),
        )
        stats = compute_summary_stats(intervals)

        return np.concatenate([betti, land, img, stats]).astype(np.float32)

    def vectorize(self, diagram: PersistenceDiagram) -> np.ndarray:
        """Vectorize a complete PersistenceDiagram into a single 1D feature vector."""
        h0_vec = self.vectorize_intervals(diagram.h0_intervals, diagram.max_filtration)
        h1_vec = self.vectorize_intervals(diagram.h1_intervals, diagram.max_filtration)

        z_topo = np.concatenate([h0_vec, h1_vec]).astype(np.float32)

        if len(z_topo) != self.total_dim:
            raise ValueError(f"Vectorizer output dimension mismatch: got {len(z_topo)}, expected {self.total_dim}")

        if np.isnan(z_topo).any():
            raise ValueError(f"NaN detected in vectorized topological features for candidate {diagram.candidate_id}")

        return z_topo
