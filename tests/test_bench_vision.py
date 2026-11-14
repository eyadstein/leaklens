"""The Phase 17 M2 benchmark: pinned corpus, deterministic output, no holdout tuning."""

from __future__ import annotations

import pytest

from leaklens import bench_vision


@pytest.fixture(scope="module")
def metrics():
    return bench_vision.evaluate(measure_runtime=False)


def test_corpus_digest_is_pinned_and_stable(metrics):
    assert metrics["corpus_digest"] == bench_vision.PINNED_CORPUS_DIGEST
    assert metrics["digest_pinned"] is True
    assert metrics["cases"] == 180
    assert metrics["calibration_cases"] == metrics["holdout_cases"] == 90


def test_evaluation_is_deterministic_and_passes(metrics):
    again = bench_vision.evaluate(measure_runtime=False)
    assert again["corpus_digest"] == metrics["corpus_digest"]
    assert again["thresholds"] == metrics["thresholds"]
    assert again["families"] == metrics["families"]
    assert again["clusters"] == metrics["clusters"]
    assert metrics["ok"] is True


def test_selected_threshold_and_perfect_partitions(metrics):
    assert metrics["selected"] == 6
    assert metrics["calibration"]["negative_merges"] == 0
    assert metrics["holdout"]["negative_merges"] == 0
    assert metrics["holdout"]["supported_recall"] == 1.0
    # Each partition carries its own five exact cases, not the whole corpus.
    assert metrics["exact_calibration_cases"] == 5
    assert metrics["exact_holdout_cases"] == 5
    assert metrics["exact_calibration"] == 5
    assert metrics["exact_holdout"] == 5


def test_threshold_selection_never_scores_the_holdout(monkeypatch):
    seen: list[bool] = []
    original = bench_vision._score

    def spy(rows, threshold):
        seen.append(all(row["case"].partition == "calibration" for row in rows))
        return original(rows, threshold)

    monkeypatch.setattr(bench_vision, "_score", spy)
    result = bench_vision.evaluate(measure_runtime=False)
    # Every threshold candidate is calibrated on calibration rows only; the
    # single holdout scoring happens after the threshold is already chosen.
    assert seen[: len(bench_vision.THRESHOLDS)] == [True] * len(bench_vision.THRESHOLDS)
    assert seen[len(bench_vision.THRESHOLDS)] is False
    assert result["selected"] == 6


def test_a_missing_pillow_skips_instead_of_failing(monkeypatch, capsys):
    def unavailable():
        raise ImportError("image benchmark needs the 'images' extra (Pillow)")

    monkeypatch.setattr(bench_vision, "_pil", unavailable)
    assert bench_vision.main([]) == 0
    assert "skipped" in capsys.readouterr().out


def test_markdown_reports_partition_scoped_exact_counts(metrics):
    text = bench_vision.markdown(metrics)
    assert "| Calibration | 1.00 |" in text
    assert "5/5 |" in text
    assert "known collision" in text
