"""
Topological Vectorization & Persistence Landscapes for Graph Machine Learning.

Transforms variable-length H0 and H1 persistence diagrams into deterministic,
fixed-dimensional float32 feature vectors via:
1. Multi-layer Persistence Landscapes (Bubenik, 2015).
2. Topological Summary Statistics (Total persistence, max, mean, entropy, count).

Upholds Invariants:
- INV-001 (No Mock Data in Production): Valid topological intervals from actual complexes.
- INV-008 (Self-Contained Verification Scripts): Strict numerical stability without NaNs.
- INV-010 (Dependency and Environment Integrity): Pure NumPy implementation.
"""

from dataclasses import dataclass
import os
import sys
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Resolve imports
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("source"))

try:
    from source.src.ph.gudhi_backend import PersistenceDiagram
except ModuleNotFoundError:
    from src.ph.gudhi_backend import PersistenceDiagram


class PersistenceLandscapeVectorizer:
    """Deterministic persistence landscape and topological summary vectorizer."""

    def __init__(
        self,
        num_landscapes: int = 3,
        resolution: int = 50,
        t_min: float = 0.0,
        t_max: float = 2.0,
        infinity_replacement: float = 1.0,
    ):
        """
        Args:
            num_landscapes: Number of landscape layers (default: 3).
            resolution: Number of grid evaluation points per layer (default: 50).
            t_min: Lower bound for grid sampling (default: 0.0).
            t_max: Upper bound for grid sampling (default: 1.0).
            infinity_replacement: Additional margin added to max_time for infinite bars.
        """
        if num_landscapes <= 0 or resolution < 2:
            raise ValueError("num_landscapes must be positive and resolution must be at least 2")
        if not all(math.isfinite(v) for v in (t_min, t_max, infinity_replacement)):
            raise ValueError("Persistence grid and cap parameters must be finite")
        if t_max <= t_min or infinity_replacement <= 0:
            raise ValueError("Require t_max > t_min and a positive infinity replacement margin")
        self.num_landscapes = num_landscapes
        self.resolution = resolution
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.infinity_replacement = float(infinity_replacement)

    @property
    def feature_dim(self) -> int:
        """Returns total output vector dimensionality for combined H0 and H1."""
        per_dim_landscape = self.num_landscapes * self.resolution
        per_dim_stats = 5
        return 2 * (per_dim_landscape + per_dim_stats)

    def _sanitize_intervals(
        self, intervals: List[Tuple[float, float]], max_time: float
    ) -> List[Tuple[float, float]]:
        """Cleanses intervals, filters degenerate bars, and caps infinite deaths."""
        clean: List[Tuple[float, float]] = []
        cap_val = max_time + self.infinity_replacement

        for b, d in intervals:
            b_val = float(b)
            d_val = float(d)

            if np.isnan(b_val) or np.isnan(d_val) or not np.isfinite(b_val):
                raise ValueError("Persistence diagram contains an invalid interval")
            if np.isinf(d_val):
                d_val = cap_val
            if d_val < b_val:
                raise ValueError(f"Persistence interval has death before birth: {(b_val, d_val)}")
            if d_val == b_val:
                continue

            clean.append((b_val, d_val))

        return clean

    def compute_landscape(
        self,
        intervals: List[Tuple[float, float]],
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Evaluates the first `num_landscapes` persistence landscape functions on a uniform grid.
        
        Tent function for interval (b, d):
            Lambda_{(b, d)}(t) = max(0, min(t - b, d - t))
            
        Returns:
            np.ndarray of shape (num_landscapes, resolution).
        """
        grid_min = self.t_min if t_min is None else float(t_min)
        grid_max = self.t_max if t_max is None else float(t_max)

        if grid_max <= grid_min:
            grid_max = grid_min + 1.0

        grid = np.linspace(grid_min, grid_max, self.resolution, endpoint=True, dtype=np.float64)
        landscapes = np.zeros((self.num_landscapes, self.resolution), dtype=np.float64)

        if not intervals:
            return landscapes

        # Vectorized tent function evaluation
        # grid shape: (R,)
        # intervals: N pairs
        all_tents = []
        for b, d in intervals:
            # Lambda(t) = max(0, min(t - b, d - t))
            left = grid - b
            right = d - grid
            tent = np.maximum(0.0, np.minimum(left, right))
            all_tents.append(tent)

        if not all_tents:
            return landscapes

        # Matrix of shape (N, resolution)
        tents_matrix = np.array(all_tents, dtype=np.float64)
        # Sort descending along points axis (axis 0)
        sorted_tents = np.sort(tents_matrix, axis=0)[::-1]

        # Extract top k layers
        k_avail = min(self.num_landscapes, sorted_tents.shape[0])
        landscapes[:k_avail, :] = sorted_tents[:k_avail, :]

        return landscapes

    def compute_summary_stats(self, intervals: List[Tuple[float, float]]) -> np.ndarray:
        """
        Computes summary topological features:
        [total_persistence, max_persistence, mean_persistence, persistence_entropy, point_count]
        
        Returns:
            np.ndarray of shape (5,), dtype=float64.
        """
        if not intervals:
            return np.zeros(5, dtype=np.float64)

        lifetimes = np.array([d - b for b, d in intervals if d > b], dtype=np.float64)
        if len(lifetimes) == 0:
            return np.zeros(5, dtype=np.float64)

        total_pers = float(np.sum(lifetimes))
        max_pers = float(np.max(lifetimes))
        mean_pers = float(np.mean(lifetimes))
        count = float(len(lifetimes))

        # Persistence Entropy: - sum(p_i * log2(p_i)) where p_i = L_i / total_pers
        if total_pers > 1e-12:
            probs = lifetimes / total_pers
            probs = probs[probs > 1e-12]
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        return np.array([total_pers, max_pers, mean_pers, entropy, count], dtype=np.float64)

    def vectorize(self, diagram: PersistenceDiagram) -> np.ndarray:
        """
        Converts a PersistenceDiagram into a single concatenated 1D float32 vector.
        
        Feature vector structure:
        - H0 landscapes: (num_landscapes * resolution,)
        - H0 summary stats: (5,)
        - H1 landscapes: (num_landscapes * resolution,)
        - H1 summary stats: (5,)
        
        Total length = 2 * (num_landscapes * resolution + 5)
        """
        max_t = float(diagram.max_filtration_time)
        if not np.isfinite(max_t) or max_t < 0:
            raise ValueError("Invalid maximum filtration time")
        scale = max(max_t, 1.0)
        normalized_h0 = [(float(b) / scale, float(d) / scale) for b, d in diagram.h0_intervals]
        normalized_h1 = [(float(b) / scale, float(d) / scale) for b, d in diagram.h1_intervals]
        clean_h0 = self._sanitize_intervals(normalized_h0, max_t / scale)
        clean_h1 = self._sanitize_intervals(normalized_h1, max_t / scale)

        # Every candidate uses the same normalized filtration coordinate.
        grid_min = self.t_min
        grid_max = self.t_max

        # H0 components
        h0_landscapes = self.compute_landscape(clean_h0, t_min=grid_min, t_max=grid_max)
        h0_stats = self.compute_summary_stats(clean_h0)

        # H1 components
        h1_landscapes = self.compute_landscape(clean_h1, t_min=grid_min, t_max=grid_max)
        h1_stats = self.compute_summary_stats(clean_h1)

        feature_vector = np.concatenate([
            h0_landscapes.flatten(),
            h0_stats,
            h1_landscapes.flatten(),
            h1_stats,
        ]).astype(np.float32)

        # Guarantee no NaN or inf
        if not np.isfinite(feature_vector).all():
            raise ValueError("Non-finite persistence vector")

        return feature_vector
