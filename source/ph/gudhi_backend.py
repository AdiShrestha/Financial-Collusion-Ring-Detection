"""Persistent homology computation backend with GUDHI integration and deterministic fallback."""

from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np


class SimplexTreeNative:
    """Deterministic, pure-Python SimplexTree implementing standard persistent homology reduction.

    Follows the canonical Edelsbrunner-Letscher-Zomorodian standard reduction algorithm over Z2.
    """

    def __init__(self):
        self.simplices: List[Tuple[Tuple[Union[int, str], ...], float]] = []
        self._is_initialized = False

    def insert(self, simplex: Sequence[Union[int, str]], filtration: float = 0.0) -> None:
        """Insert a simplex into the complex."""
        s = tuple(sorted(simplex, key=lambda x: str(x)))
        self.simplices.append((s, float(filtration)))
        self._is_initialized = False

    def initialize_filtration(self) -> None:
        """Sort simplices by filtration value, then by dimension."""
        # Ensure all faces exist in complex
        existing = {s[0]: s[1] for s in self.simplices}
        all_simplices = dict(existing)

        # Ensure 0-simplices and sub-simplices exist
        for s, f_val in list(existing.items()):
            dim = len(s) - 1
            if dim > 0:
                for v in s:
                    v_simplex = (v,)
                    if v_simplex not in all_simplices:
                        all_simplices[v_simplex] = f_val
                    else:
                        # Face filtration must not exceed coface
                        all_simplices[v_simplex] = min(all_simplices[v_simplex], f_val)

        # Sort by (filtration_value, dimension, lexical)
        sorted_list = sorted(
            all_simplices.items(),
            key=lambda item: (item[1], len(item[0]), [str(x) for x in item[0]]),
        )
        self.simplices = [(k, v) for k, v in sorted_list]
        self._is_initialized = True

    def persistence(self) -> List[Tuple[int, Tuple[float, float]]]:
        """Compute persistent homology pairs."""
        if not self._is_initialized:
            self.initialize_filtration()

        n = len(self.simplices)
        simplex_to_idx = {s[0]: i for i, s in enumerate(self.simplices)}

        # Build boundary matrix representation: each column is a set of row indices
        cols: List[set] = []
        for i, (s, _) in enumerate(self.simplices):
            dim = len(s) - 1
            col_boundary = set()
            if dim > 0:
                for v_idx in range(len(s)):
                    face = tuple(s[:v_idx] + s[v_idx + 1 :])
                    if face in simplex_to_idx:
                        col_boundary.add(simplex_to_idx[face])
            cols.append(col_boundary)

        # Standard persistent homology reduction over Z2
        low = {}  # col_idx -> lowest row idx
        reverse_low = {}  # lowest row idx -> col_idx
        intervals = []
        killed_rows = set()

        for j in range(n):
            while cols[j]:
                i = max(cols[j])
                if i in reverse_low:
                    k = reverse_low[i]
                    # Add column k to column j (symmetric difference)
                    cols[j] = cols[j] ^ cols[k]
                else:
                    low[j] = i
                    reverse_low[i] = j
                    killed_rows.add(i)
                    birth_time = self.simplices[i][1]
                    death_time = self.simplices[j][1]
                    dim = len(self.simplices[i][0]) - 1
                    intervals.append((dim, (birth_time, death_time)))
                    break

        # Essential features (never killed)
        for i in range(n):
            if i not in killed_rows and not cols[i]:
                dim = len(self.simplices[i][0]) - 1
                birth_time = self.simplices[i][1]
                intervals.append((dim, (birth_time, float("inf"))))

        return intervals

    def betti_numbers(self) -> List[int]:
        """Compute Betti numbers (count of essential persistence intervals)."""
        pairs = self.persistence()
        max_dim = max([p[0] for p in pairs], default=0)
        bettis = [0] * (max_dim + 1)
        for dim, (birth, death) in pairs:
            if death == float("inf"):
                bettis[dim] += 1
        return bettis


def get_simplex_tree():
    """Return a GUDHI SimplexTree instance if installed, otherwise SimplexTreeNative."""
    try:
        import gudhi
        return gudhi.SimplexTree()
    except ImportError:
        return SimplexTreeNative()


def compute_persistence_diagrams(
    filtered_simplices: Sequence[Tuple[Sequence[Union[int, str]], float]],
) -> Dict[str, Any]:
    """Compute persistence diagrams (H0 and H1) from a list of filtered simplices.

    Args:
        filtered_simplices: List of (simplex_nodes, filtration_value).

    Returns:
        Dictionary with structure:
        {
            "H0": list of (birth, death) tuples for dimension 0,
            "H1": list of (birth, death) tuples for dimension 1,
            "betti_numbers": list of Betti numbers [beta_0, beta_1, ...],
            "raw_pairs": full list of (dimension, (birth, death))
        }
    """
    st = get_simplex_tree()

    for simplex, filtration in filtered_simplices:
        st.insert(simplex, filtration=filtration)

    st.initialize_filtration()
    raw_pairs = st.persistence()

    h0_intervals = []
    h1_intervals = []

    for dim, (birth, death) in raw_pairs:
        if dim == 0:
            h0_intervals.append((float(birth), float(death)))
        elif dim == 1:
            h1_intervals.append((float(birth), float(death)))

    betti_nums = st.betti_numbers()

    return {
        "H0": h0_intervals,
        "H1": h1_intervals,
        "betti_numbers": betti_nums,
        "raw_pairs": raw_pairs,
    }
