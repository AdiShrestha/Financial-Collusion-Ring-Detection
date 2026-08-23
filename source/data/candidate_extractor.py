"""Candidate subgraph representation schemas and label-blind extractor interfaces."""

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx


@dataclass
class CandidateExample:
    """Standardized representation of an extracted candidate subgraph.

    In accordance with project_description.md §2.2 and architecture.md §3,
    one CandidateExample is the primary supervised unit of classification.
    """

    candidate_id: str
    dataset_track: str  # "amlworld" or "elliptic_actors"
    temporal_bounds: Tuple[float, float]
    participant_ids: List[str]
    edges: List[Tuple[str, str, Dict[str, Any]]]
    node_features: Dict[str, List[float]]
    target_y: int  # 1 = positive laundering / illicit ring, 0 = negative candidate
    typology_label: str  # e.g. "FAN-OUT", "CYCLE", "BIPARTITE", "HARD-NEGATIVE"
    group_id: str  # Identifier of the originating pattern group or connected cluster
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate example to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateExample":
        """Deserialize candidate example from dictionary."""
        data_copy = dict(data)
        if isinstance(data_copy.get("temporal_bounds"), list):
            data_copy["temporal_bounds"] = tuple(data_copy["temporal_bounds"])
        if "edges" in data_copy:
            data_copy["edges"] = [
                (e[0], e[1], dict(e[2])) if len(e) == 3 else tuple(e)
                for e in data_copy["edges"]
            ]
        return cls(**data_copy)


class CandidateExtractor:
    """Base interface for label-blind candidate subgraph extractors.

    Contract invariant:
        Extractors must operate strictly on graph topology and temporal windows
        WITHOUT inspecting target labels or laundering status.
    """

    def __init__(self, dataset_track: str, max_candidate_nodes: int = 100):
        self.dataset_track = dataset_track
        self.max_candidate_nodes = max_candidate_nodes

    def extract_candidates(
        self,
        graph: Any,
        context_rules: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateExample]:
        """Extract candidate subgraphs from raw transaction network.

        Must be overridden by domain-specific extractors.
        """
        raise NotImplementedError("Subclasses must implement extract_candidates()")


class LabelBlindCandidateExtractor(CandidateExtractor):
    """Production candidate subgraph extractor for directed cycles and collusion rings.

    Strictly label-blind: operates purely on graph connectivity and transaction timestamps.
    """

    def __init__(
        self,
        dataset_track: str = "amlworld",
        min_cycle_length: int = 3,
        max_cycle_length: int = 6,
        max_temporal_span: Optional[float] = None,
        include_1hop_context: bool = True,
        max_nodes_per_candidate: int = 50,
        feature_dim: int = 56,
    ):
        super().__init__(dataset_track=dataset_track, max_candidate_nodes=max_nodes_per_candidate)
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.max_temporal_span = max_temporal_span
        self.include_1hop_context = include_1hop_context
        self.feature_dim = feature_dim

    def find_directed_cycles(
        self,
        graph: nx.DiGraph,
        min_k: Optional[int] = None,
        max_k: Optional[int] = None,
    ) -> List[List[str]]:
        """Find all elementary directed simple cycles of length between min_k and max_k."""
        min_len = min_k if min_k is not None else self.min_cycle_length
        max_len = max_k if max_k is not None else self.max_cycle_length

        # Find simple cycles using bounded DFS
        valid_cycles: List[List[str]] = []
        raw_cycles = list(nx.simple_cycles(graph))

        for cycle in raw_cycles:
            k = len(cycle)
            if min_len <= k <= max_len:
                # Canonical rotation (smallest element first)
                min_elem = min(cycle)
                min_idx = cycle.index(min_elem)
                canonical_cycle = cycle[min_idx:] + cycle[:min_idx]
                valid_cycles.append([str(node) for node in canonical_cycle])

        # Remove duplicate rotated cycles
        unique_cycles = []
        seen = set()
        for c in valid_cycles:
            c_key = tuple(c)
            if c_key not in seen:
                seen.add(c_key)
                unique_cycles.append(c)

        return unique_cycles

    def extract_candidate_subgraph(
        self,
        graph: nx.DiGraph,
        cycle_nodes: Sequence[str],
        node_features_map: Optional[Dict[str, List[float]]] = None,
    ) -> CandidateExample:
        """Extract induced candidate subgraph for a detected cycle, including 1-hop context."""
        cycle_node_set = set(str(n) for n in cycle_nodes)
        subgraph_nodes = set(cycle_node_set)

        if self.include_1hop_context:
            for node in cycle_nodes:
                if node in graph:
                    # Add in-neighbors and out-neighbors
                    for succ in graph.successors(node):
                        if len(subgraph_nodes) < self.max_candidate_nodes:
                            subgraph_nodes.add(str(succ))
                    for pred in graph.predecessors(node):
                        if len(subgraph_nodes) < self.max_candidate_nodes:
                            subgraph_nodes.add(str(pred))

        # Extract induced subgraph edges
        subgraph_edges: List[Tuple[str, str, Dict[str, Any]]] = []
        edge_timestamps: List[float] = []

        for u in subgraph_nodes:
            if u in graph:
                for v in graph.successors(u):
                    if str(v) in subgraph_nodes:
                        edge_data = dict(graph.get_edge_data(u, v) or {})
                        ts = float(edge_data.get("timestamp", 0.0))
                        edge_timestamps.append(ts)
                        subgraph_edges.append((str(u), str(v), edge_data))

        min_t = min(edge_timestamps) if edge_timestamps else 0.0
        max_t = max(edge_timestamps) if edge_timestamps else 0.0

        # Node features
        cand_node_feats = {}
        for n in subgraph_nodes:
            if node_features_map and n in node_features_map:
                cand_node_feats[n] = list(node_features_map[n])
            else:
                cand_node_feats[n] = [0.0] * self.feature_dim

        # Stable hash identifier
        node_str = "_".join(sorted(subgraph_nodes))
        hash_digest = hashlib.sha256(node_str.encode("utf-8")).hexdigest()[:12]
        cand_id = f"cand_{hash_digest}"

        return CandidateExample(
            candidate_id=cand_id,
            dataset_track=self.dataset_track,
            temporal_bounds=(min_t, max_t),
            participant_ids=sorted(list(subgraph_nodes)),
            edges=subgraph_edges,
            node_features=cand_node_feats,
            target_y=0,  # Unlabeled during label-blind extraction
            typology_label="UNLABELED",
            group_id=f"group_{cand_id}",
            metadata={
                "cycle_nodes": list(cycle_nodes),
                "cycle_length": len(cycle_nodes),
                "subgraph_node_count": len(subgraph_nodes),
                "subgraph_edge_count": len(subgraph_edges),
            },
        )

    def extract_candidates(
        self,
        graph: nx.DiGraph,
        node_features_map: Optional[Dict[str, List[float]]] = None,
        context_rules: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateExample]:
        """Run label-blind cycle detection and candidate subgraph packaging."""
        cycles = self.find_directed_cycles(graph)
        candidates: List[CandidateExample] = []

        for cycle in cycles:
            cand = self.extract_candidate_subgraph(
                graph=graph,
                cycle_nodes=cycle,
                node_features_map=node_features_map,
            )
            # Filter by temporal span if specified
            if self.max_temporal_span is not None:
                span = cand.temporal_bounds[1] - cand.temporal_bounds[0]
                if span > self.max_temporal_span:
                    continue
            candidates.append(cand)

        return candidates

    @staticmethod
    def match_ground_truth(
        candidates: Sequence[CandidateExample],
        true_patterns: Sequence[Any],
        iou_threshold: float = 0.5,
    ) -> Tuple[List[CandidateExample], Dict[str, Any]]:
        """Post-hoc labeling: match candidate participant sets to known pattern ground truth."""
        labeled_candidates: List[CandidateExample] = []
        matched_pattern_ids = set()

        for cand in candidates:
            cand_nodes = set(cand.participant_ids)
            best_match = None
            best_iou = 0.0

            for p in true_patterns:
                if isinstance(p, dict):
                    p_nodes = set(p.get("participant_ids", []))
                    p_pat_id = p.get("pattern_id", 0)
                    p_typology = p.get("typology", "CYCLE")
                else:
                    p_nodes = set(getattr(p, "participant_ids", []))
                    p_pat_id = getattr(p, "pattern_id", 0)
                    p_typology = getattr(p, "typology", "CYCLE")

                intersection = len(cand_nodes.intersection(p_nodes))
                union = len(cand_nodes.union(p_nodes))
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_match = (p_pat_id, p_typology)

            if best_match is not None and best_iou >= iou_threshold:
                pat_id, typology = best_match
                matched_pattern_ids.add(pat_id)

                labeled_cand = CandidateExample(
                    candidate_id=cand.candidate_id,
                    dataset_track=cand.dataset_track,
                    temporal_bounds=cand.temporal_bounds,
                    participant_ids=cand.participant_ids,
                    edges=cand.edges,
                    node_features=cand.node_features,
                    target_y=1,
                    typology_label=typology,
                    group_id=f"pattern_{pat_id}",
                    metadata={**cand.metadata, "match_iou": best_iou, "matched_pattern_id": pat_id},
                )
            else:
                labeled_cand = CandidateExample(
                    candidate_id=cand.candidate_id,
                    dataset_track=cand.dataset_track,
                    temporal_bounds=cand.temporal_bounds,
                    participant_ids=cand.participant_ids,
                    edges=cand.edges,
                    node_features=cand.node_features,
                    target_y=0,
                    typology_label="NEGATIVE_CANDIDATE",
                    group_id=cand.group_id,
                    metadata={**cand.metadata, "match_iou": best_iou},
                )
            labeled_candidates.append(labeled_cand)

        total_true = len(true_patterns)
        recall = len(matched_pattern_ids) / total_true if total_true > 0 else 1.0

        match_stats = {
            "total_candidates": len(candidates),
            "positive_candidates": sum(1 for c in labeled_candidates if c.target_y == 1),
            "negative_candidates": sum(1 for c in labeled_candidates if c.target_y == 0),
            "total_true_patterns": total_true,
            "matched_true_patterns": len(matched_pattern_ids),
            "pattern_recall": recall,
        }

        return labeled_candidates, match_stats
