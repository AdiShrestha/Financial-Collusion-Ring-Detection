"""Node feature extraction and graph-local structural statistics."""

from typing import Dict, List, Optional, Sequence
import numpy as np

from source.data.candidate_extractor import CandidateExample


def extract_node_features(
    candidate: CandidateExample,
    base_dim: int = 56,
    include_structural_stats: bool = True,
) -> np.ndarray:
    """Extract node feature matrix for candidate subgraph nodes.

    Returns:
        |V| x (base_dim + structural_dim) float32 numpy array.
    """
    nodes = list(candidate.participant_ids)
    node_map = {n: i for i, n in enumerate(nodes)}

    # Compute in-degree and out-degree from candidate edges
    in_degrees = {n: 0 for n in nodes}
    out_degrees = {n: 0 for n in nodes}

    for u, v, _ in candidate.edges:
        u_str, v_str = str(u), str(v)
        if u_str in out_degrees:
            out_degrees[u_str] += 1
        if v_str in in_degrees:
            in_degrees[v_str] += 1

    feature_rows: List[List[float]] = []
    for n in nodes:
        # Base features
        if n in candidate.node_features and candidate.node_features[n]:
            base_feats = list(candidate.node_features[n])
            if len(base_feats) < base_dim:
                base_feats.extend([0.0] * (base_dim - len(base_feats)))
            elif len(base_feats) > base_dim:
                base_feats = base_feats[:base_dim]
        else:
            base_feats = [0.0] * base_dim

        if include_structural_stats:
            in_d = float(in_degrees[n])
            out_d = float(out_degrees[n])
            tot_d = in_d + out_d
            ratio = in_d / (out_d + 1.0)
            struct_feats = [in_d, out_d, tot_d, ratio]
            row = base_feats + struct_feats
        else:
            row = base_feats

        feature_rows.append(row)

    if feature_rows:
        return np.array(feature_rows, dtype=np.float32)
    dim = base_dim + (4 if include_structural_stats else 0)
    return np.zeros((0, dim), dtype=np.float32)
