"""Phase 17 M2: exact image duplicates, perceptual clusters and their contract."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

Image = pytest.importorskip("PIL.Image")

from leaklens import images
from leaklens.cli import main
from leaklens.images import (
    ImageBudget,
    analyze_images,
    file_digest,
    hamming,
    image_overlap,
    sign_dir,
)


def _picture(seed, size=256, grid=6):
    """Smooth deterministic shapes: structured enough for dHash, no real data."""
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray(rng.integers(0, 256, (grid, grid, 3), dtype=np.uint8))
    return coarse.resize((size, size), Image.Resampling.BICUBIC)


def _saved(tmp_path, seed, name, size=256):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _picture(seed, size).save(path)
    return path


def test_file_digest_is_sha256_of_the_bytes(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"not an image, just bytes")
    assert file_digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_file_digest_changes_with_content(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"payload one")
    second.write_bytes(b"payload two")
    assert file_digest(first) != file_digest(second)
    assert len(file_digest(first)) == 64


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (0, 0, 0),
        (0, 1, 1),
        (0b1011, 0b0010, 2),
        (2**64 - 1, 0, 64),
        (0, 2**64 - 1, 64),
    ],
)
def test_hamming_is_the_bit_distance(left, right, expected):
    assert hamming(left, right) == expected
    assert hamming(right, left) == expected


def test_sign_dir_returns_digests_for_decoded_images_only(tmp_path):
    good = _saved(tmp_path, 1, "ok.png")
    (tmp_path / "broken.png").write_bytes(b"this is not an image")
    hashes, digests, unreadable = sign_dir(tmp_path)
    assert list(hashes) == ["ok.png"]
    assert digests == {"ok.png": file_digest(good)}
    assert unreadable == 1


def test_sign_dir_skips_a_refused_image_entirely(tmp_path, monkeypatch):
    _saved(tmp_path, 2, "ok.png")
    _saved(tmp_path, 3, "big.png")
    monkeypatch.setattr(images, "MAX_IMAGE_PIXELS", 10_000)
    budget = ImageBudget()
    hashes, digests, _ = sign_dir(tmp_path, budget=budget)
    assert hashes == {} and digests == {}
    assert budget.refused == {"too_many_pixels": 2}


def test_image_overlap_output_is_unchanged(tmp_path):
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    _saved(train, 4, "a.png")
    _saved(train, 5, "b.png")
    _picture(4, 192).save(test / "copy.png")
    _saved(test, 9, "other.png")
    findings, pairs, stats = image_overlap(train, test)
    assert [finding.kind for finding in findings] == ["image_overlap"]
    assert [(pair["test"], pair["train"]) for pair in pairs] == [("copy.png", "a.png")]
    assert set(stats) == {
        "train_images",
        "test_images",
        "unreadable",
        "refused",
        "decoded_pixels",
        "near_duplicates",
    }


def _two_splits(tmp_path, train_names, test_names):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    for name, seed in train_names:
        _saved(train, seed, name)
    for name, seed in test_names:
        _saved(test, seed, name)
    return train, test


def test_exact_duplicates_are_byte_equal_only(tmp_path):
    source = _saved(tmp_path, 20, "source.png")
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    (train / "photo.png").write_bytes(source.read_bytes())
    # Same pixels, different encoding: perceptual, never exact.
    _picture(20).save(test / "reencoded.jpg", quality=90)
    audit = analyze_images(train, test)
    assert audit.exact_duplicates == 0
    assert audit.exact_groups == ()
    assert len(audit.clusters) == 1
    assert audit.clusters[0].test == 1 and audit.clusters[0].train == 1
    assert audit.clusters[0].exact_pairs == 0
    assert audit.pairs[0]["distance"] == 0


def test_byte_identical_copies_are_an_exact_group(tmp_path):
    source = _saved(tmp_path, 21, "source.png")
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    (train / "one.png").write_bytes(source.read_bytes())
    (train / "two.png").write_bytes(source.read_bytes())
    (test / "copy.png").write_bytes(source.read_bytes())
    audit = analyze_images(train, test)
    assert audit.exact_duplicates == 1
    assert audit.exact_groups == (
        {
            "digest": file_digest(source),
            "train": ["one.png", "two.png"],
            "test": ["copy.png"],
        },
    )
    assert len(audit.clusters) == 1
    cluster = audit.clusters[0]
    assert (cluster.train, cluster.test, cluster.exact_pairs) == (2, 1, 2)
    assert cluster.members == ("test:copy.png", "train:one.png", "train:two.png")


def test_exact_detection_does_not_use_the_pixel_budget(tmp_path):
    """A refused file is not decoded, so it is never part of an exact group."""
    source = _saved(tmp_path, 22, "source.png")
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    (train / "photo.png").write_bytes(source.read_bytes())
    (test / "copy.png").write_bytes(source.read_bytes())
    budget = ImageBudget(max_pixels=1)
    audit = analyze_images(train, test, budget=budget)
    assert audit.exact_groups == ()
    assert audit.clusters == ()
    assert audit.stats["refused"] == {"budget_exhausted": 2}


def test_a_cluster_is_a_transitive_component_not_a_clique():
    clusters, exact, uninformative = images.cluster_signatures(
        {"a": 0b0001},
        {"b": 0b0011, "c": 0b0111},
        {"a": "d1"},
        {"b": "d2", "c": "d3"},
        1,
    )
    assert len(clusters) == 1
    assert clusters[0].members == ("test:b", "test:c", "train:a")
    assert hamming(0b0001, 0b0011) == 1
    assert hamming(0b0011, 0b0111) == 1
    assert hamming(0b0001, 0b0111) == 2  # not a direct pair
    assert exact == () and uninformative == 0


def test_the_threshold_is_a_hamming_distance_boundary():
    train = {"a": 0b0000_0001}
    inside, _, _ = images.cluster_signatures(
        train, {"b": 0b0000_0011}, {"a": "d1"}, {"b": "d2"}, 1
    )
    outside, _, _ = images.cluster_signatures(
        train, {"b": 0b0000_0111}, {"a": "d1"}, {"b": "d2"}, 1
    )
    assert [(c.train, c.test) for c in inside] == [(1, 1)]
    assert outside == ()


def test_uniform_images_are_excluded_from_clusters_only():
    """Different flat colours hash alike; they are never a perceptual cluster."""
    signatures = {"a": 0, "b": 0, "c": 2**64 - 1}
    clusters, _, uninformative = images.cluster_signatures(
        {"a": 0}, {"b": 0, "c": 2**64 - 1}, {"a": "d1"}, {"b": "d2", "c": "d3"}, 6
    )
    assert signatures["a"] == signatures["b"]
    assert clusters == () and uninformative == 3


def test_unrelated_images_do_not_merge(tmp_path):
    train, test = _two_splits(
        tmp_path,
        [("a.png", 31), ("b.png", 32)],
        [("a-copy.png", 31), ("b-copy.png", 32)],
    )
    # Same content, smaller encoding: perceptual copies, never exact groups.
    for name in ("a-copy.png", "b-copy.png"):
        with Image.open(test / name) as image:
            image.resize((192, 192), Image.Resampling.LANCZOS).save(
                test / name, format="PNG"
            )
    audit = analyze_images(train, test)
    assert [(c.train, c.test) for c in audit.clusters] == [(1, 1), (1, 1)]
    assert audit.exact_groups == ()
    assert all(pair["distance"] <= 6 for pair in audit.pairs)


def test_singletons_are_not_clusters(tmp_path):
    train, test = _two_splits(tmp_path, [("a.png", 41)], [("b.png", 42)])
    audit = analyze_images(train, test)
    assert audit.clusters == () and audit.pairs == []


def test_empty_collections_are_clean(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    audit = analyze_images(train, test)
    assert audit.findings == [] and audit.pairs == [] and audit.clusters == ()
    assert audit.exact_duplicates == 0
    assert audit.stats["train_images"] == 0 and audit.stats["test_images"] == 0


def test_pairs_within_matches_brute_force_across_block_boundaries():
    rng = np.random.default_rng(7)
    hashes = {
        f"i{index}": int(value)
        for index, value in enumerate(rng.integers(0, 2**64, 40, dtype=np.uint64))
    }
    # High-bit values and a tight cluster straddle the block boundary.
    hashes.update({"a0": 2**63, "a1": 2**63 + 1, "a2": 2**63 - 1, "a3": 0})
    found = list(images.pairs_within(hashes, 4, chunk=7))
    names = sorted(hashes)
    expected = sorted(
        (left, right)
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if hamming(hashes[left], hashes[right]) <= 4
    )
    assert found == expected
    assert len(found) == len(set(found))
    assert ("a0", "a1") in found and ("a1", "a3") in found
    assert hamming(hashes["a0"], hashes["a2"]) == 64
    assert ("a0", "a2") not in found


def test_analysis_is_deterministic_and_order_independent(tmp_path):
    train, test = _two_splits(
        tmp_path,
        [("a.png", 51), ("b.png", 52), ("c.png", 53)],
        [("a-copy.png", 51), ("b-copy.png", 52), ("extra.png", 59)],
    )
    first = analyze_images(train, test)
    second = analyze_images(train, test)
    assert [c.to_dict() for c in first.clusters] == [
        c.to_dict() for c in second.clusters
    ]
    assert first.pairs == second.pairs
    assert first.exact_groups == second.exact_groups


def test_same_relative_name_in_both_splits_keeps_one_member_each(tmp_path):
    source = _saved(tmp_path, 61, "source.png")
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()
    (train / "same.png").write_bytes(source.read_bytes())
    (test / "same.png").write_bytes(source.read_bytes())
    audit = analyze_images(train, test)
    assert audit.clusters[0].members == ("test:same.png", "train:same.png")
    assert len(set(audit.clusters[0].members)) == 2


def test_cluster_payload_does_not_change_findings_or_stats(tmp_path):
    train, test = _two_splits(
        tmp_path, [("a.png", 71)], [("a-copy.png", 71), ("b.png", 72)]
    )
    findings, pairs, stats = image_overlap(train, test)
    audit = analyze_images(train, test)
    assert [(f.kind, f.severity, f.detail) for f in audit.findings] == [
        (f.kind, f.severity, f.detail) for f in findings
    ]
    assert audit.pairs == pairs and audit.stats == stats


def test_pairs_within_rejects_an_out_of_range_distance():
    with pytest.raises(ValueError):
        list(images.pairs_within({}, 65))
    with pytest.raises(ValueError):
        list(images.pairs_within({}, -1))


def test_cli_reports_exact_and_cluster_views(tmp_path, capsys):
    train, test = _two_splits(
        tmp_path, [("a.png", 81), ("b.png", 82)], [("a-copy.png", 81)]
    )
    assert main(["images", str(train), str(test)]) == 1
    out = capsys.readouterr().out
    assert "[HIGH] image_overlap" in out
    assert "near-duplicate" in out
    assert "exact" in out and "cluster" in out

    assert main(["images", str(train), str(test), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["clustering"]["max_distance"] == 6
    assert payload["clustering"]["exact_duplicates"] == 1
    assert len(payload["clustering"]["clusters"]) == 1
    assert payload["trust_score"] == 65
