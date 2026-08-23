"""Candidate subgraph representation schemas and label-blind extractor interfaces."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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
        # Convert temporal_bounds to tuple if needed
        data_copy = dict(data)
        if isinstance(data_copy.get("temporal_bounds"), list):
            data_copy["temporal_bounds"] = tuple(data_copy["temporal_bounds"])
        # Reconstruct edges if needed
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
