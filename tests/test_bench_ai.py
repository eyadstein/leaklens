"""The AI benchmark must report measured numbers, not an improvement claim."""

from __future__ import annotations

from leaklens import bench_ai


def test_benchmark_reports_measured_metrics_and_limitations():
    data = bench_ai.evaluate()

    assert data["corpus"]["cases"] == 263
    assert data["corpus"]["train_cases"] == 223
    assert data["corpus"]["validation_cases"] == 40
    assert data["corpus"]["holdout_cases"] == 12
    assert data["external_cases"] > 100
    assert 0.05 <= data["model"]["threshold"] <= 0.95
    assert len(data["candidates"]) >= 6
    assert data["model"]["name"] in {row["name"] for row in data["candidates"]}

    validation = data["validation"]
    assert validation["precision"] >= 0.9
    assert validation["recall"] >= 0.9
    assert validation["f1"] >= 0.9
    assert validation["false_positive_rate"] <= 0.05

    locked = data["locked_test"]
    assert locked["cases"] == 47
    assert locked["metrics"]["accuracy"] >= 0.94
    assert locked["metrics"]["precision"] >= 0.94
    assert locked["metrics"]["recall"] >= 0.90
    assert locked["metrics"]["f1"] >= 0.92
    assert locked["metrics"]["false_positive_rate"] <= 0.05
    assert locked["per_category"]
    assert locked["per_source"]

    assert data["external"]["f1"] >= 0.7
    assert data["external_threshold_sweep"]
    assert data["top_features"]
    assert data["limitations"]
    assert data["overfitting_checks"]
    assert data["baseline_phase4"]["cross_validation"]["f1"] == 0.7595
    assert data["split_integrity"]["shared_sources"] == []
    assert data["split_integrity"]["shared_groups"] == []
    assert data["split_integrity"]["exact_duplicate_codes"] == 0
    assert "Not measured as an improvement" in data["deterministic_comparison"]

    holdout = data["phase7_holdout"]
    assert holdout["cases"] == 12
    assert holdout["leaky"] == 6
    assert holdout["metrics"]["recall"] >= 0.9
    assert holdout["metrics"]["precision"] >= 0.8
    assert holdout["metrics"]["false_positive_rate"] <= 0.2
    assert holdout["per_family"]
    assert "frozen before the final fit" in holdout["provenance"]

    gate = data["runtime_gate"]
    assert gate["cases"] == 263
    assert gate["metrics"]["precision"] >= 0.9
    assert gate["metrics"]["recall"] >= 0.9
    assert gate["false_positives"] == []

    attribution = data["local_attribution"]
    assert len(attribution["rows"]) == 2
    labels = {row["label"] for row in attribution["rows"]}
    assert labels == {"leaky", "legitimate"}
    assert all(
        any(item["feature"] for item in row["supporting"])
        for row in attribution["rows"]
    )


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
    assert "## Protocol and data provenance" in text
    assert "## Split integrity" in text
    assert "## Baseline comparison" in text
    assert "## Locked independent test (evaluated once)" in text
    assert "## Phase 7 frozen holdout (authored, evaluated once)" in text
    assert "## Runtime hypothesis gate (development)" in text
    assert "## Local attribution example" in text
    assert "These cases are not used for fitting, calibration" in text
    assert "used for Phase 6 error analysis" in text
    assert "development result, not independent" in text
    assert "## External false-positive analysis" in text
    assert "## Overfitting checks" in text
    assert "never change deterministic findings" in text


def test_benchmark_rejects_a_write_without_a_path(capsys):
    assert bench_ai.main(["--write"]) == 2
    assert "--write needs a path" in capsys.readouterr().err
