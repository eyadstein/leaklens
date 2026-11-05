"""Advanced benchmark set (0.14.0 detection phase)."""

from leaklens.bench import evaluate
from leaklens.bench_advanced import ADVANCED_CASES
from leaklens.bench_cases import EXTRA_CASES


def test_advanced_benchmark_has_no_errors():
    metrics = evaluate()
    assert metrics["errors"] == [], metrics["errors"]


def test_historical_counts_are_preserved():
    metrics = evaluate()
    assert metrics["n"] == 109
    assert metrics["tp"] == 58 and metrics["fp"] == 0
    assert len(EXTRA_CASES) == 36


def test_advanced_case_counts():
    metrics = evaluate()["advanced"]
    assert metrics["n"] == len(ADVANCED_CASES) == 32
    assert metrics["leaky"] == 16
    assert metrics["clean"] == 16
    assert metrics["tp"] == 16
    assert metrics["fp"] == 0
    assert metrics["tn"] == 16
    assert metrics["fn"] == 0


def test_advanced_categories_cover_requested_gaps():
    categories = evaluate()["advanced_categories"]
    assert categories["tuning"]["n"] == 12
    assert categories["encoding"]["n"] == 9
    assert categories["feature_selection"]["n"] == 5
    assert categories["group_cv"]["n"] == 2
    assert categories["temporal"]["n"] == 4
    assert all(not item["errors"] for item in categories.values())


def test_advanced_case_names_are_unique():
    names = [case.name for case in ADVANCED_CASES]
    snippets = [case.code for case in ADVANCED_CASES]
    assert len(names) == len(set(names))
    assert len(snippets) == len(set(snippets))
