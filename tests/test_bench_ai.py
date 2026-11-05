"""The AI benchmark must report measured numbers, not an improvement claim."""

from __future__ import annotations

from leaklens import bench_ai


def test_benchmark_reports_measured_metrics_and_limitations():
    data = bench_ai.evaluate()

    assert data["corpus"]["cases"] == 72
    assert data["external_cases"] > 100
    assert data["threshold"] == 0.65
    cv = data["cross_validation"]
    assert cv["precision"] >= 0.6
    assert cv["recall"] >= 0.6
    assert cv["f1"] >= 0.7
    assert 0.0 <= cv["false_positive_rate"] <= 1.0
    assert data["external"]["f1"] >= 0.6
    assert data["external_threshold_sweep"]
    assert data["top_features"]
    assert data["limitations"]
    assert "Not measured as an improvement" in data["deterministic_comparison"]


def test_benchmark_semantics_have_no_misses_or_false_alarms():
    semantics = bench_ai.evaluate_semantics()

    assert semantics["cases"] == 15
    assert semantics["case_accuracy"] == 1.0
    assert semantics["expected_recall"] == 1.0
    assert semantics["unexpected_concepts"] == 0
    assert semantics["misses"] == []
    assert semantics["false_alarms"] == []


def test_benchmark_markdown_states_the_limits(tmp_path, capsys):
    path = tmp_path / "AI_BENCHMARK.md"

    assert bench_ai.main(["--write", str(path)]) == 0
    assert "wrote" in capsys.readouterr().out
    text = path.read_text(encoding="utf-8")

    assert "# AI intelligence benchmark" in text
    assert "not an improvement claim" in text
    assert "External cases are **not** used for training" in text
    assert "False-positive analysis" in text
    assert "never change deterministic findings" in text


def test_benchmark_rejects_a_write_without_a_path(capsys):
    assert bench_ai.main(["--write"]) == 2
    assert "--write needs a path" in capsys.readouterr().err
