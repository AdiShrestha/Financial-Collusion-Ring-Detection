"""Test suite for persistent homology filtration monotonicity and GUDHI backend oracles."""

import pytest

from source.ph.graph_filtration import (
    build_temporal_ph_graph,
    validate_filtration_monotonicity,
)
from source.ph.gudhi_backend import (
    SimplexTreeNative,
    compute_persistence_diagrams,
    get_simplex_tree,
)


def test_filtration_monotonicity():
    """Verify that build_temporal_ph_graph clamps edge filtration to >= endpoint discovery times."""
    # Nodes first seen at t=5.0 and t=10.0
    nodes = {1: 5.0, 2: 10.0, 3: 15.0}
    # Edge with transaction timestamp t=2.0 (earlier than endpoints)
    edges = [(1, 2, 2.0), (2, 3, 12.0)]

    simplices = build_temporal_ph_graph(nodes, edges)

    # Convert to dict for inspection
    s_dict = {tuple(s[0]): s[1] for s in simplices}

    # Node 1 at 5.0, Node 2 at 10.0
    assert s_dict[(1,)] == 5.0
    assert s_dict[(2,)] == 10.0
    # Edge (1, 2) must be clamped to max(5.0, 10.0, 2.0) = 10.0
    assert s_dict[(1, 2)] == 10.0
    # Edge (2, 3) must be max(10.0, 15.0, 12.0) = 15.0
    assert s_dict[(2, 3)] == 15.0

    # Ensure validation passes
    assert validate_filtration_monotonicity(simplices) is True


def test_unfilled_cycle_h1_persistence():
    """Construct 4-cycle entering at t=1,2,3,4; assert H1 feature born at t=4 with death inf."""
    nodes = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    edges = [
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 3.0),
        (3, 0, 4.0),  # Closing edge
    ]

    simplices = build_temporal_ph_graph(nodes, edges)
    diagrams = compute_persistence_diagrams(simplices)

    # Check H1 persistence
    h1 = diagrams["H1"]
    assert len(h1) >= 1, "Must have at least one H1 feature for 4-cycle"

    # Find the essential H1 feature
    essential_h1 = [p for p in h1 if p[1] == float("inf")]
    assert len(essential_h1) == 1, "Unfilled 4-cycle must have exactly one infinite H1 feature"
    birth_time, death_time = essential_h1[0]
    assert birth_time == 4.0, f"H1 feature must be born when cycle closes at t=4.0, got {birth_time}"
    assert death_time == float("inf")

    # Betti numbers: beta_0 = 1, beta_1 = 1
    assert diagrams["betti_numbers"][0] == 1
    assert diagrams["betti_numbers"][1] == 1


def test_filled_simplex_kills_h1():
    """Verify that inserting a 2-simplex into the complex closes the 1-hole and kills H1."""
    # 1. Unfilled 3-cycle (triangle)
    unfilled = [
        ([0], 1.0),
        ([1], 1.0),
        ([2], 1.0),
        ([0, 1], 1.0),
        ([1, 2], 2.0),
        ([0, 2], 3.0),
    ]
    diag_unfilled = compute_persistence_diagrams(unfilled)
    assert any(p[1] == float("inf") for p in diag_unfilled["H1"]), "Unfilled triangle must have beta_1=1"
    assert diag_unfilled["betti_numbers"][1] == 1

    # 2. Filled 2-simplex entered at t=5.0
    filled = list(unfilled) + [([0, 1, 2], 5.0)]
    diag_filled = compute_persistence_diagrams(filled)

    # The H1 feature born at 3.0 must die at 5.0 when the 2-simplex is added
    h1_filled = diag_filled["H1"]
    killed_feature = [p for p in h1_filled if p[0] == 3.0 and p[1] == 5.0]
    assert len(killed_feature) == 1, (
        f"2-simplex entering at t=5.0 must kill H1 feature born at 3.0. Found: {h1_filled}"
    )

    # Persistent beta_1 is now 0
    assert diag_filled["betti_numbers"][1] == 0


def test_no_2cells_in_ph_graph_invariant():
    """Pass complex with attempted 2-cell to build_temporal_ph_graph and assert validation/invariant."""
    # Attempting to pass a 3-element simplex tuple to 1-skeleton builder must raise ValueError
    nodes = {0: 1.0, 1: 1.0, 2: 1.0}
    edges_with_illegal_triangle = [(0, 1, 2, 1.0)]  # invalid edge format / 2-simplex attempt

    with pytest.raises(ValueError):
        build_temporal_ph_graph(nodes, edges_with_illegal_triangle)


def test_native_simplex_tree_parity():
    """Assert SimplexTreeNative correctly computes persistence on multi-cycle graphs."""
    st = SimplexTreeNative()
    for v in [1, 2, 3, 4, 5]:
        st.insert([v], filtration=1.0)

    # Build 5-cycle: (1,2), (2,3), (3,4), (4,5), (5,1)
    st.insert([1, 2], filtration=2.0)
    st.insert([2, 3], filtration=3.0)
    st.insert([3, 4], filtration=4.0)
    st.insert([4, 5], filtration=5.0)
    st.insert([5, 1], filtration=6.0)

    pairs = st.persistence()
    h1_pairs = [p[1] for p in pairs if p[0] == 1]
    assert len(h1_pairs) == 1
    assert h1_pairs[0] == (6.0, float("inf"))

    bettis = st.betti_numbers()
    assert bettis[0] == 1
    assert bettis[1] == 1
