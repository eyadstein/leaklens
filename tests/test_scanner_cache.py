"""Bounded parsed-source cache: correctness, invalidation and limits."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from leaklens import scanner
from leaklens.scanner import _parse_cache, scan_file, scan_path, scan_source

LEAKY = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "X = StandardScaler().fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
)
CLEAN = "value = 1\n"


def _clear():
    scanner._parse_cache.clear()
    scanner._parse_cache_chars = 0


def _payload(findings):
    return [finding.to_dict() for finding in findings]


def test_cached_and_uncached_outputs_are_identical(monkeypatch):
    _clear()
    monkeypatch.setattr(scanner, "_PARSE_CACHE_MAX_SOURCE", 0)
    uncached = _payload(scan_source(LEAKY, "first.py"))
    assert not _parse_cache

    monkeypatch.setattr(scanner, "_PARSE_CACHE_MAX_SOURCE", 1_000_000)
    first = _payload(scan_source(LEAKY, "first.py"))
    assert len(_parse_cache) == 1

    second = _payload(scan_source(LEAKY, "first.py"))
    assert second == first == uncached
    assert len(_parse_cache) == 1


def test_changed_source_is_a_cache_miss():
    _clear()
    leaky = _payload(scan_source(LEAKY, "file.py"))
    clean = _payload(scan_source(CLEAN, "file.py"))

    assert clean != leaky
    assert len(_parse_cache) == 2


def test_same_content_different_labels_stays_label_aware():
    _clear()
    one = _payload(scan_source(LEAKY, "one.py"))
    two = _payload(scan_source(LEAKY, "two.py"))

    assert len(_parse_cache) == 1
    assert one[0]["column"] == "one.py:3"
    assert two[0]["column"] == "two.py:3"
    assert [item["kind"] for item in one] == [item["kind"] for item in two]


def test_invalid_source_is_not_cached():
    _clear()
    result = scan_source("def broken(:\n", "broken.py")

    assert [finding.kind for finding in result] == ["parse_error"]
    assert not _parse_cache


def test_cache_respects_entry_limit(monkeypatch):
    _clear()
    monkeypatch.setattr(scanner, "_PARSE_CACHE_MAX_ENTRIES", 1)
    scan_source(LEAKY, "one.py")
    scan_source(CLEAN, "two.py")

    assert len(_parse_cache) <= 1


def test_cache_respects_source_size_limit(monkeypatch):
    _clear()
    monkeypatch.setattr(scanner, "_PARSE_CACHE_MAX_SOURCE", 8)
    scan_source(LEAKY, "large.py")

    assert not _parse_cache


def test_cache_respects_character_budget(monkeypatch):
    _clear()
    monkeypatch.setattr(scanner, "_PARSE_CACHE_MAX_CHARS", 8)
    scan_source(LEAKY, "one.py")

    assert scanner._parse_cache_chars == 0
    assert not _parse_cache


def test_cache_is_thread_safe_and_deterministic():
    _clear()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _payload(scan_source(LEAKY)), range(16)))

    assert all(result == results[0] for result in results)
    assert len(_parse_cache) == 1


def test_file_changes_and_replacements_are_not_stale(tmp_path):
    path = tmp_path / "file.py"
    path.write_text(LEAKY, encoding="utf-8")
    leaky = _payload(scan_file(path))
    assert leaky

    path.write_text(CLEAN, encoding="utf-8")
    assert _payload(scan_file(path)) == []

    path.write_text(LEAKY, encoding="utf-8")
    assert _payload(scan_file(path)) == leaky


def test_unavailable_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_file(tmp_path / "missing.py")


def test_scan_path_is_stable_across_cached_runs(tmp_path):
    path = tmp_path / "file.py"
    path.write_text(LEAKY, encoding="utf-8")

    first = _payload(scan_path(path))
    second = _payload(scan_path(path))

    assert second == first


@pytest.mark.parametrize("source", [LEAKY, CLEAN, "x = [i for i in range(10)]\n"])
def test_cache_does_not_change_fingerprints(source):
    _clear()
    first = _payload(scan_source(source))
    second = _payload(scan_source(source))

    assert [item["fingerprint"] for item in second] == [
        item["fingerprint"] for item in first
    ]
