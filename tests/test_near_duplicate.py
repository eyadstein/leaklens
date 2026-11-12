"""The shared near-duplicate primitive: banding, thresholding, stable keys."""

from __future__ import annotations

import pytest

from leaklens import near_duplicate as nd


def test_band_key_is_stable_and_distinguishes_bands():
    assert nd.band_key((1, 2, 3, 4)) == nd.band_key((1, 2, 3, 4))
    assert nd.band_key((1, 2, 3, 4)) != nd.band_key((1, 2, 3, 5))


def test_bucket_groups_ids_that_share_a_band():
    signatures = {
        "a": (1, 2, 3, 4, 5, 6, 7, 8),
        "b": (1, 2, 3, 4, 9, 9, 9, 9),
        "c": (0, 0, 0, 0, 0, 0, 0, 0),
    }
    buckets = nd.bucket(signatures, rows=4)
    assert any(list(members) == ["a", "b"] for members in buckets.values())


def test_candidate_pairs_are_sorted_unique_and_never_self_pairs():
    buckets = {
        "k": ("a", "b", "c"),
        "j": ("c", "a"),
    }
    assert nd.candidate_pairs(buckets) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_oversized_buckets_can_be_skipped():
    buckets = {"k": ("a", "b", "c")}
    assert nd.candidate_pairs(buckets, max_bucket=2) == []


@pytest.mark.parametrize(("rows", "signature"), [(0, (1, 2)), (4, (1, 2, 3))])
def test_bucket_rejects_bad_band_geometry(rows, signature):
    with pytest.raises(ValueError):
        nd.bucket({"a": signature}, rows=rows)


def test_jaccard_estimates_set_overlap():
    assert nd.jaccard((1, 2, 3, 4), (1, 2, 3, 4)) == 1.0
    assert nd.jaccard((1, 2, 3, 4), (5, 6, 7, 8)) == 0.0
    assert nd.jaccard((1, 2, 3, 4), (1, 2, 5, 6)) == 0.5
    assert nd.jaccard((1, 2), ()) == 0.0


def test_cluster_keeps_pairs_above_the_threshold():
    signatures = {
        "train:0": (1, 2, 3, 4, 5, 6, 7, 8),
        "test:0": (1, 2, 3, 4, 5, 6, 7, 8),
        "test:1": (0, 0, 0, 0, 9, 9, 9, 9),
    }
    clusters = nd.cluster(signatures, threshold=0.8, rows=4)
    assert len(clusters) == 1
    assert clusters[0].members == ("test:0", "train:0")
    assert clusters[0].key == nd.cluster_key(clusters[0].members)


def test_candidate_pairs_below_the_threshold_are_dropped():
    signatures = {
        "a": (1, 2, 3, 4, 5, 6, 7, 8),
        "b": (1, 2, 3, 4, 9, 10, 11, 12),
    }
    assert nd.cluster(signatures, threshold=0.8, rows=4) == []


def test_cluster_is_independent_of_input_order():
    signatures = {
        "train:0": (1, 2, 3, 4, 5, 6, 7, 8),
        "test:0": (1, 2, 3, 4, 5, 6, 7, 8),
        "test:1": (2, 2, 3, 4, 5, 6, 7, 8),
        "train:1": (8, 7, 6, 5, 4, 3, 2, 1),
    }
    forward = nd.cluster(signatures, threshold=0.8, rows=4)
    reverse = nd.cluster(
        dict(reversed(list(signatures.items()))), threshold=0.8, rows=4
    )
    assert forward == reverse


def test_singletons_do_not_cluster():
    assert nd.cluster({"a": (1, 2, 3, 4, 5, 6, 7, 8)}, rows=4) == []
    assert nd.cluster({}, rows=4) == []


def test_cluster_key_is_order_independent_and_bounded():
    assert nd.cluster_key(("b", "a")) == nd.cluster_key(("a", "b"))
    assert nd.cluster_key(("a", "b")) != nd.cluster_key(("a", "c"))
    assert len(nd.cluster_key(("a",))) == 16
