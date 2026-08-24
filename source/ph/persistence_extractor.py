"""Persistence diagram and barcode extractor for 1-skeleton graph filtrations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from source.ph.gudhi_backend import compute_persistence_diagrams
from source.ph.ph_graph_view import PHGraphView


@dataclass
class PersistenceDiagram:
    """Structured representation of H0 and H1 persistence intervals."""

    candidate_id: str
    h0_intervals: np.ndarray  # Shape (N0, 2), columns [birth, death]
    h1_intervals: np.ndarray  # Shape (N1, 2), columns [birth, death]
    max_filtration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.h0_intervals = np.asarray(self.h0_intervals, dtype=np.float32)
        self.h1_intervals = np.asarray(self.h1_intervals, dtype=np.float32)

        if self.h0_intervals.ndim == 1 and self.h0_intervals.size == 0:
            self.h0_intervals = np.zeros((0, 2), dtype=np.float32)
        if self.h1_intervals.ndim == 1 and self.h1_intervals.size == 0:
            self.h1_intervals = np.zeros((0, 2), dtype=np.float32)

        # Assert non-negative lifetimes for all finite intervals
        for intervals, dim in ((self.h0_intervals, 0), (self.h1_intervals, 1)):
            if intervals.size > 0:
                births = intervals[:, 0]
                deaths = intervals[:, 1]
                finite_mask = np.isfinite(deaths)
                if np.any(deaths[finite_mask] < births[finite_mask] - 1e-6):
                    raise ValueError(
                        f"Negative lifetime detected in H{dim} diagram for candidate {self.candidate_id}: "
                        f"deaths < births."
                    )

    @property
    def h0_lifetimes(self) -> np.ndarray:
        if self.h0_intervals.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.h0_intervals[:, 1] - self.h0_intervals[:, 0]

    @property
    def h1_lifetimes(self) -> np.ndarray:
        if self.h1_intervals.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.h1_intervals[:, 1] - self.h1_intervals[:, 0]

    @property
    def has_h1_cycle(self) -> bool:
        return len(self.h1_intervals) > 0


class PersistenceExtractor:
    """Computes H0 and H1 persistence diagrams from PHGraphView filtrations."""

    @classmethod
    def compute_diagram(
        cls,
        ph_graph: PHGraphView,
        cap_infinity: bool = True,
        infinity_cap_margin: float = 1.0,
    ) -> PersistenceDiagram:
        """Compute persistent homology intervals for a PHGraphView."""
        # Determine maximum finite filtration value
        filt_vals = [val for _, val in ph_graph.simplices]
        max_filt = float(max(filt_vals)) if filt_vals else 0.0
        cap_val = max_filt + float(infinity_cap_margin)

        # Compute persistence using SimplexTreeNative / GUDHI backend
        res = compute_persistence_diagrams(ph_graph.simplices)
        raw_intervals = res["raw_pairs"]

        h0_list: List[Tuple[float, float]] = []
        h1_list: List[Tuple[float, float]] = []

        for dim, (b, d) in raw_intervals:
            birth = float(b)
            death = float(d)
            if cap_infinity and np.isinf(death):
                death = cap_val

            if dim == 0:
                h0_list.append((birth, death))
            elif dim == 1:
                h1_list.append((birth, death))

        h0_arr = np.array(h0_list, dtype=np.float32) if h0_list else np.zeros((0, 2), dtype=np.float32)
        h1_arr = np.array(h1_list, dtype=np.float32) if h1_list else np.zeros((0, 2), dtype=np.float32)

        return PersistenceDiagram(
            candidate_id=ph_graph.candidate_id,
            h0_intervals=h0_arr,
            h1_intervals=h1_arr,
            max_filtration=max_filt,
            metadata={
                **ph_graph.metadata,
                "filtration_type": ph_graph.filtration_type,
                "cap_infinity": cap_infinity,
                "cap_val": cap_val,
            },
        )
