"""
Path resolution utilities for raw datasets across the Toporing Engine.

Guarantees deterministic, robust resolution of real datasets regardless of current working directory:
- Project root (the directory containing source/ and data/)
- Subdirectory (e.g. source/, source/scripts/, etc.)
- Custom paths or environment variables
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import os


def get_project_root() -> Path:
    """Returns the absolute path to the project root (Toporing Engine)."""
    # This file is located at <project_root>/source/src/data/path_utils.py
    # parents: [0]=src/data, [1]=src, [2]=source, [3]=Toporing Engine
    return Path(__file__).resolve().parents[3]


def resolve_amlworld_paths(
    trans_csv_path: Optional[Union[str, Path]] = None,
    patterns_txt_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Resolves paths to the IBM AMLworld transaction CSV and patterns file.
    Raises FileNotFoundError if trans_csv_path is not found.
    """
    root = get_project_root()
    
    # 1. Resolve transaction CSV
    if trans_csv_path is not None and str(trans_csv_path).strip():
        p = Path(trans_csv_path)
        if p.is_file():
            resolved_trans = p.resolve()
        elif (root / p).is_file():
            resolved_trans = (root / p).resolve()
        elif (Path.cwd() / p).is_file():
            resolved_trans = (Path.cwd() / p).resolve()
        elif (Path.cwd().parent / p).is_file():
            resolved_trans = (Path.cwd().parent / p).resolve()
        else:
            raise FileNotFoundError(
                f"[INV-001 / INV-012 Violation] Specified transaction CSV not found: {trans_csv_path}"
            )
    else:
        candidates = [
            root / "data" / "raw" / "ibm_amlworld" / "HI-Small_Trans.csv",
            Path.cwd() / "data" / "raw" / "ibm_amlworld" / "HI-Small_Trans.csv",
            Path.cwd().parent / "data" / "raw" / "ibm_amlworld" / "HI-Small_Trans.csv",
            Path.cwd().parent.parent / "data" / "raw" / "ibm_amlworld" / "HI-Small_Trans.csv",
            Path.home() / "Desktop" / "Test" / "data" / "raw" / "ibm_amlworld" / "HI-Small_Trans.csv",
            Path.home() / "Desktop" / "Toporing Engine" / "data" / "raw" / "ibm_amlworld" / "HI-Small_Trans.csv",
        ]
        resolved_trans = None
        for c in candidates:
            if c.is_file():
                resolved_trans = c.resolve()
                break
        if resolved_trans is None:
            raise FileNotFoundError(
                "[INV-001 / INV-012 Violation] Real IBM AMLworld transactions not found. "
                "Expected at 'data/raw/ibm_amlworld/HI-Small_Trans.csv'. "
                "No mock data fallback is permitted in production."
            )

    # 2. Resolve patterns file
    resolved_patterns = None
    if patterns_txt_path is not None and str(patterns_txt_path).strip():
        p = Path(patterns_txt_path)
        if p.is_file():
            resolved_patterns = p.resolve()
        elif (root / p).is_file():
            resolved_patterns = (root / p).resolve()
        elif (Path.cwd() / p).is_file():
            resolved_patterns = (Path.cwd() / p).resolve()
        elif (Path.cwd().parent / p).is_file():
            resolved_patterns = (Path.cwd().parent / p).resolve()
        else:
            raise FileNotFoundError(f"Specified AMLworld pattern file not found: {patterns_txt_path}")
    else:
        # Default alongside the trans csv or in raw dir
        sibling = resolved_trans.parent / "HI-Small_Patterns.txt"
        if sibling.is_file():
            resolved_patterns = sibling.resolve()
        else:
            candidates = [
                root / "data" / "raw" / "ibm_amlworld" / "HI-Small_Patterns.txt",
                Path.cwd() / "data" / "raw" / "ibm_amlworld" / "HI-Small_Patterns.txt",
                Path.cwd().parent / "data" / "raw" / "ibm_amlworld" / "HI-Small_Patterns.txt",
            ]
            for c in candidates:
                if c.is_file():
                    resolved_patterns = c.resolve()
                    break

    return resolved_trans, resolved_patterns


def resolve_elliptic_paths(
    edgelist_path: Optional[Union[str, Path]] = None,
    features_path: Optional[Union[str, Path]] = None,
    classes_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Resolves paths to the Elliptic++ Actor files.
    Raises FileNotFoundError if edgelist_path is not found.
    """
    root = get_project_root()
    base_dir = root / "data" / "raw" / "elliptic_actors"

    # 1. Resolve edgelist
    if edgelist_path is not None and str(edgelist_path).strip():
        p = Path(edgelist_path)
        if p.is_file():
            resolved_edgelist = p.resolve()
        elif (root / p).is_file():
            resolved_edgelist = (root / p).resolve()
        elif (Path.cwd() / p).is_file():
            resolved_edgelist = (Path.cwd() / p).resolve()
        elif (Path.cwd().parent / p).is_file():
            resolved_edgelist = (Path.cwd().parent / p).resolve()
        else:
            raise FileNotFoundError(f"Elliptic++ edgelist not found: {edgelist_path}")
    else:
        candidates = [
            base_dir / "AddrAddr_edgelist.csv",
            Path.cwd() / "data" / "raw" / "elliptic_actors" / "AddrAddr_edgelist.csv",
            Path.cwd().parent / "data" / "raw" / "elliptic_actors" / "AddrAddr_edgelist.csv",
            Path.home() / "Desktop" / "Test" / "data" / "raw" / "elliptic_actors" / "AddrAddr_edgelist.csv",
            Path.home() / "Desktop" / "Toporing Engine" / "data" / "raw" / "elliptic_actors" / "AddrAddr_edgelist.csv",
        ]
        resolved_edgelist = None
        for c in candidates:
            if c.is_file():
                resolved_edgelist = c.resolve()
                break
        if resolved_edgelist is None:
            raise FileNotFoundError(
                "Elliptic++ edgelist not found. Expected at 'data/raw/elliptic_actors/AddrAddr_edgelist.csv'."
            )

    # 2. Resolve features
    resolved_features = None
    if features_path is not None:
        p = Path(features_path)
        if p.is_file():
            resolved_features = p.resolve()
    else:
        default_feat = resolved_edgelist.parent / "wallets_features.csv"
        if default_feat.is_file():
            resolved_features = default_feat.resolve()

    # 3. Resolve classes
    resolved_classes = None
    if classes_path is not None:
        p = Path(classes_path)
        if p.is_file():
            resolved_classes = p.resolve()
    else:
        default_classes = resolved_edgelist.parent / "wallets_classes.csv"
        if default_classes.is_file():
            resolved_classes = default_classes.resolve()

    return resolved_edgelist, resolved_features, resolved_classes
