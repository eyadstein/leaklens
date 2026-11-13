"""Phase 16 benchmark contract: labelled cases, published metrics and misses."""

from __future__ import annotations

from leaklens import bench_dl


def test_at_least_twenty_labelled_repositories():
    cases = bench_dl._build_cases()
    assert len(cases) >= 20
    assert len({case.name for case in cases}) == len(cases)
    assert {case.family for case in cases} >= {
        "reproducibility",
        "protocol",
        "split",
        "config",
        "artifact",
        "adversarial",
    }


def test_build_cases_is_pure_and_needs_no_filesystem():
    assert bench_dl._build_cases() == bench_dl._build_cases()
    assert all(
        case.files or case.name == "empty repository"
        for case in bench_dl._build_cases()
    )


def test_the_curated_set_passes_its_declared_floor():
    metrics = bench_dl.evaluate(measure_runtime=False)
    assert metrics["fp"] == 0, metrics["misses"]
    assert metrics["recall"] >= bench_dl.RECALL_FLOOR
    assert metrics["ok"] is True


def test_every_case_and_every_family_is_published():
    metrics = bench_dl.evaluate(measure_runtime=False)
    text = bench_dl.markdown(metrics)
    for case in bench_dl._build_cases():
        assert case.name in text
    for family in metrics["families"]:
        assert f"| {family} |" in text
    assert "## Misses" in text
    assert "regression guard, never an accuracy claim" in text
