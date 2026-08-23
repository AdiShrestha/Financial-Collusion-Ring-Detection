"""Data structures representing parsed IBM AMLworld laundering pattern groups."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class LaunderingPatternGroup:
    """Represents a documented laundering attempt pattern group from IBM AMLworld."""

    pattern_id: int
    typology: str  # e.g. "CYCLE", "FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "BIPARTITE", "STACK", "RANDOM"
    participant_ids: List[str]
    transactions: List[Dict[str, Any]]
    start_timestamp: float
    end_timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize pattern group to dictionary."""
        return asdict(self)

    @property
    def num_transactions(self) -> int:
        return len(self.transactions)

    @property
    def num_participants(self) -> int:
        return len(self.participant_ids)
