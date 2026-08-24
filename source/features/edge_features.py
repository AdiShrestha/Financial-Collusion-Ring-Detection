"""Edge feature extraction and temporal flow attributes."""

from typing import Dict, List, Optional, Sequence
import numpy as np

from source.data.candidate_extractor import CandidateExample


def extract_edge_features(
    candidate: CandidateExample,
) -> np.ndarray:
    """Extract edge feature matrix for candidate subgraph edges.

    Features per edge: [amount, timestamp, rel_time_delta, log_amount]
    Returns:
        |E| x d_e float32 numpy array.
    """
    edges = candidate.edges
    t_start, _ = candidate.temporal_bounds

    feature_rows: List[List[float]] = []
    for _, _, attr in edges:
        amt = float(attr.get("amount", attr.get("amount_paid", attr.get("amount_received", 0.0))))
        ts = float(attr.get("timestamp", 0.0))
        rel_ts = max(0.0, ts - t_start)
        log_amt = float(np.log1p(max(0.0, amt)))

        row = [amt, ts, rel_ts, log_amt]
        feature_rows.append(row)

    if feature_rows:
        return np.array(feature_rows, dtype=np.float32)
    return np.zeros((0, 4), dtype=np.float32)
