"""Algebraic topology oracles for boundary nilpotence, Betti numbers, and Hodge Laplacians."""

from typing import Optional, Tuple
import numpy as np


def verify_boundary_nilpotence(b1: np.ndarray, b2: np.ndarray, tol: float = 1e-12) -> bool:
    """Verify nilpotence of boundary operators: B1 @ B2 == 0.

    Returns True if max(|B1 @ B2|) <= tol, else False.
    """
    if b1.shape[1] != b2.shape[0]:
        raise ValueError(
            f"Shape mismatch for matrix multiplication: B1 shape {b1.shape}, B2 shape {b2.shape}"
        )
    prod = b1 @ b2
    max_err = float(np.max(np.abs(prod))) if prod.size > 0 else 0.0
    return max_err <= tol


def compute_betti_numbers(
    b1: np.ndarray,
    b2: Optional[np.ndarray] = None,
    tol: float = 1e-10,
) -> Tuple[int, int]:
    """Compute Betti numbers (beta_0, beta_1) via matrix rank computations.

    Formula:
        beta_0 = |V| - rank(B1)
        dim(ker(B1)) = |E| - rank(B1)
        beta_1 = dim(ker(B1)) - rank(B2)
    """
    num_nodes, num_edges = b1.shape

    # SVD-based rank computation for numerical stability
    s_b1 = np.linalg.svd(b1, compute_uv=False)
    rank_b1 = int(np.sum(s_b1 > tol))

    beta_0 = num_nodes - rank_b1
    dim_ker_b1 = num_edges - rank_b1

    if b2 is not None and b2.size > 0:
        s_b2 = np.linalg.svd(b2, compute_uv=False)
        rank_b2 = int(np.sum(s_b2 > tol))
    else:
        rank_b2 = 0

    beta_1 = dim_ker_b1 - rank_b2
    return max(0, beta_0), max(0, beta_1)


def verify_hodge_laplacian_properties(
    laplacian: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[bool, bool, np.ndarray]:
    """Verify symmetry and positive semi-definiteness (PSD) of a Laplacian matrix.

    Returns:
        (is_symmetric, is_psd, eigenvalues)
    """
    # Check symmetry
    sym_diff = np.max(np.abs(laplacian - laplacian.T))
    is_symmetric = bool(sym_diff <= tol)

    # Compute eigenvalues (using eigvalsh for symmetric matrices)
    eigenvalues = np.linalg.eigvalsh((laplacian + laplacian.T) / 2.0)
    min_eig = float(np.min(eigenvalues))
    is_psd = bool(min_eig >= -tol)

    return is_symmetric, is_psd, eigenvalues
