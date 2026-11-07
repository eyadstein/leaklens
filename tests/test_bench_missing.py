"""Labeled missingness benchmark."""

from leaklens.bench_missing import evaluate, markdown


def test_missingness_benchmark_has_no_errors():
    metrics = evaluate()
    assert metrics["errors"] == [], metrics["errors"]


def test_missingness_benchmark_metrics():
    metrics = evaluate()
    assert metrics["n"] == 10
    assert (metrics["leaky"], metrics["clean"]) == (6, 4)
    assert metrics["tp"] == 6
    assert metrics["fp"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fp_rate"] == 0.0


def test_missingness_benchmark_covers_every_detector():
    kinds = {row["kinds"] for row in evaluate()["rows"]}
    assert "missingness_target_signal" in kinds
    assert "missingness_split_shift" in kinds
    assert "missingness_group_signal" in kinds


def test_missingness_benchmark_markdown():
    text = markdown(evaluate())
    assert "Missingness benchmark" in text
    assert "binary target association" in text
    assert "False positives" in text
