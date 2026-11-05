"""Dataset-level detector benchmark."""

from leaklens.bench_data import evaluate, markdown


def test_dataset_benchmark_has_no_errors():
    metrics = evaluate()
    assert metrics["errors"] == [], metrics["errors"]


def test_dataset_benchmark_metrics():
    metrics = evaluate()
    assert metrics["n"] == 12
    assert (metrics["leaky"], metrics["clean"]) == (9, 3)
    assert metrics["tp"] == 9
    assert metrics["fp"] == 0
    assert metrics["tn"] == 3
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_dataset_benchmark_runtime_is_recorded():
    metrics = evaluate()
    assert metrics["runtime_total_ms"] > 0
    assert metrics["runtime_max_ms"] >= metrics["runtime_median_ms"]
    assert all("ms" in row for row in metrics["rows"])


def test_dataset_benchmark_markdown_contains_cases():
    text = markdown(evaluate())
    assert "Dataset detector benchmark" in text
    assert "target copied into a feature" in text
    assert "False positives" in text
