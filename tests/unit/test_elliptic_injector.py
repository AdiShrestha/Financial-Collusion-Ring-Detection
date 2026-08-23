"""Unit tests for Elliptic++ Actors loader and semi-synthetic ring injector."""

import pytest

from source.data.elliptic_actors_loader import EllipticActorsLoader
from source.data.ring_injector import RingInjector


MOCK_FEATURES_CSV = """wallet_id,timestep,feat_0,feat_1,feat_2
w_1,1,0.1,0.2,0.3
w_2,1,0.4,0.5,0.6
w_3,2,0.7,0.8,0.9
w_4,2,1.0,1.1,1.2
"""

MOCK_CLASSES_CSV = """wallet_id,class
w_1,1
w_2,2
w_3,3
w_4,unknown
"""

MOCK_EDGELIST_CSV = """input_address,output_address,tx_hash,timestep
w_1,w_2,tx_hash_1,1
w_3,w_4,tx_hash_2,2
"""


def test_elliptic_loader_features_classes_edges():
    """Verify loading of wallet features, classes, and edgelist."""
    loader = EllipticActorsLoader(expected_num_features=56)

    feats = loader.load_features(MOCK_FEATURES_CSV)
    assert len(feats) == 4
    assert "w_1" in feats
    assert feats["w_1"]["timestep"] == 1
    assert len(feats["w_1"]["features"]) == 56
    assert feats["w_1"]["features"][0] == 0.1

    classes = loader.load_classes(MOCK_CLASSES_CSV)
    assert classes["w_1"] == "illicit"
    assert classes["w_2"] == "licit"
    assert classes["w_3"] == "unknown"
    assert classes["w_4"] == "unknown"

    edges = loader.load_edgelist(MOCK_EDGELIST_CSV)
    assert len(edges) == 2
    assert edges[0][0] == "w_1"
    assert edges[0][1] == "w_2"
    assert edges[0][2]["tx_hash"] == "tx_hash_1"


def test_injection_determinism():
    """Verify that running RingInjector with identical seed produces identical manifests."""
    wallets = [f"wallet_{i}" for i in range(20)]
    base_edges = [("wallet_0", "wallet_1", {})]

    inj1 = RingInjector(num_injections=6, cycle_lengths=[3, 4, 5, 6], seed=42)
    cands1, edges1, manifest1 = inj1.inject_rings(wallets, base_edges)

    inj2 = RingInjector(num_injections=6, cycle_lengths=[3, 4, 5, 6], seed=42)
    cands2, edges2, manifest2 = inj2.inject_rings(wallets, base_edges)

    assert manifest1 == manifest2
    assert len(cands1) == 6
    assert len(cands2) == 6
    for c1, c2 in zip(cands1, cands2):
        assert c1.candidate_id == c2.candidate_id
        assert c1.participant_ids == c2.participant_ids
        assert len(c1.edges) == len(c2.edges)


def test_cycle_length_distribution():
    """Verify injected rings match declared cycle lengths k in {3, 4, 5, 6}."""
    wallets = [f"wallet_{i}" for i in range(30)]
    inj = RingInjector(num_injections=8, cycle_lengths=[3, 4, 5, 6], seed=99)
    candidates, edges, manifest = inj.inject_rings(wallets, [])

    observed_lengths = [len(c.participant_ids) for c in candidates]
    expected_pattern = [3, 4, 5, 6, 3, 4, 5, 6]
    assert observed_lengths == expected_pattern

    for c in candidates:
        assert c.target_y == 1
        assert c.typology_label == "CYCLE"
        # Each cycle has k edges
        k = len(c.participant_ids)
        assert len(c.edges) == k


def test_split_boundary_protection():
    """Verify that injected rings never cross partition boundaries."""
    # 10 wallets in train, 10 wallets in test
    train_wallets = [f"train_w_{i}" for i in range(10)]
    test_wallets = [f"test_w_{i}" for i in range(10)]
    all_wallets = train_wallets + test_wallets

    split_partitions = {w: "train" for w in train_wallets}
    split_partitions.update({w: "test" for w in test_wallets})

    inj = RingInjector(num_injections=10, cycle_lengths=[3, 4, 5, 6], seed=1234)
    candidates, edges, manifest = inj.inject_rings(
        all_wallets,
        base_edgelist=[],
        split_partitions=split_partitions,
    )

    for c in candidates:
        parts = {split_partitions[node] for node in c.participant_ids}
        assert len(parts) == 1, f"Injected ring {c.candidate_id} crosses partition boundary: {parts}"

    assert manifest["split_boundary_violations"] == 0
