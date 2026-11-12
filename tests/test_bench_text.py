"""The text benchmark is deterministic, covers every catalogue case and gates."""

from __future__ import annotations

from leaklens import bench_text


def test_expected_catalogue_cases_exist():
    names = {case.name for case in bench_text._build_cases()}
    assert {
        "exact duplicate across splits",
        "near-duplicate paraphrase",
        "eval fingerprint contamination",
        "shared RAG index",
    } <= names


def test_curated_cases_are_deterministic_and_within_the_gate():
    first = bench_text.evaluate(measure_runtime=False)
    second = bench_text.evaluate(measure_runtime=False)
    assert [(row["name"], row["actual"]) for row in first["rows"]] == [
        (row["name"], row["actual"]) for row in second["rows"]
    ]
    assert first["ok"] is True
    assert first["fp"] == 0
    assert first["recall"] >= bench_text.RECALL_FLOOR
    assert first["misses"] == second["misses"]


def test_hard_paraphrase_miss_is_published():
    metrics = bench_text.evaluate(measure_runtime=False)
    assert any("hard paraphrase" in miss for miss in metrics["misses"])


def test_markdown_reports_metrics_misses_and_limits():
    text = bench_text.markdown(bench_text.evaluate(measure_runtime=False))
    assert "Text benchmark" in text
    assert "Precision" in text
    assert "hard paraphrase" in text
    assert "not real-world accuracy" in text
    assert "Embedding similarity" in text


def test_main_writes_the_report(tmp_path, capsys):
    target = tmp_path / "TEXT_BENCHMARK.md"
    assert bench_text.main(["--write", str(target)]) == 0
    assert "precision=" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8").startswith("# Text benchmark")
