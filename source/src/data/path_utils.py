"""
Path resolution for the legacy AMLworld and Elliptic++ datasets.

Resolution order, with no other fallbacks:
1. an explicit path argument (absolute, or relative to the current directory);
2. a directory given by an environment variable
   (``AMLWORLD_DATA_DIR`` or ``ELLIPTIC_DATA_DIR``);
3. the repository-relative default under ``data/raw/``.

Nothing outside these locations is searched, so data can never be picked up
from an unrelated directory on the machine.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import os

PathLike = Union[str, Path]


def get_project_root() -> Path:
    """Return the repository root (the directory that contains ``source/``)."""
    # This file is located at <root>/source/src/data/path_utils.py.
    return Path(__file__).resolve().parents[3]


def _explicit(path: Optional[PathLike], what: str) -> Optional[Path]:
    if path is None or not str(path).strip():
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return p.resolve()


def _data_dir(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var, "").strip()
    return Path(value).expanduser().resolve() if value else default


def _optional_sibling(explicit: Optional[PathLike], directory: Path, name: str, what: str) -> Optional[Path]:
    found = _explicit(explicit, what)
    if found is not None:
        return found
    candidate = directory / name
    return candidate.resolve() if candidate.is_file() else None


def resolve_amlworld_paths(
    trans_csv_path: Optional[PathLike] = None,
    patterns_txt_path: Optional[PathLike] = None,
) -> Tuple[Path, Optional[Path]]:
    """Resolve the AMLworld transaction CSV (required) and patterns file (optional)."""
    directory = _data_dir("AMLWORLD_DATA_DIR", get_project_root() / "data" / "raw" / "ibm_amlworld")
    trans = _explicit(trans_csv_path, "AMLworld transaction CSV")
    if trans is None:
        candidate = directory / "HI-Small_Trans.csv"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"AMLworld transactions not found at {candidate}. Pass the path explicitly "
                "or set AMLWORLD_DATA_DIR. No synthetic fallback is used."
            )
        trans = candidate.resolve()
    patterns = _optional_sibling(patterns_txt_path, trans.parent, "HI-Small_Patterns.txt",
                                 "AMLworld pattern file")
    return trans, patterns


def resolve_elliptic_paths(
    edgelist_path: Optional[PathLike] = None,
    features_path: Optional[PathLike] = None,
    classes_path: Optional[PathLike] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve the Elliptic++ actor edge list (required) and feature/class files (optional)."""
    directory = _data_dir("ELLIPTIC_DATA_DIR", get_project_root() / "data" / "raw" / "elliptic_actors")
    edgelist = _explicit(edgelist_path, "Elliptic++ edgelist")
    if edgelist is None:
        candidate = directory / "AddrAddr_edgelist.csv"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Elliptic++ edgelist not found at {candidate}. Pass the path explicitly "
                "or set ELLIPTIC_DATA_DIR."
            )
        edgelist = candidate.resolve()
    features = _optional_sibling(features_path, edgelist.parent, "wallets_features.csv", "Elliptic++ features")
    classes = _optional_sibling(classes_path, edgelist.parent, "wallets_classes.csv", "Elliptic++ classes")
    return edgelist, features, classes
