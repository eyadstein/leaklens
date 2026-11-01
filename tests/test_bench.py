from leaklens.bench import evaluate
from leaklens.bench_cases import EXTRA_CASES


def test_benchmark_has_no_errors():
    metrics = evaluate()
    assert metrics["errors"] == [], metrics["errors"]


def test_legacy_baseline_is_preserved():
    metrics = evaluate()["legacy"]

    assert metrics["n"] == 69
    assert metrics["leaky"] == 37
    assert metrics["clean"] == 32
    assert metrics["tp"] == 37
    assert metrics["fp"] == 0
    assert metrics["tn"] == 32
    assert metrics["fn"] == 0


def test_expanded_benchmark_counts():
    metrics = evaluate()

    assert metrics["n"] == 105
    assert metrics["leaky"] == 55
    assert metrics["clean"] == 50
    assert metrics["tp"] == 55
    assert metrics["fp"] == 0
    assert metrics["tn"] == 50
    assert metrics["fn"] == 0


def test_additional_categories_are_complete():
    metrics = evaluate()
    categories = metrics["categories"]

    assert sum(item["n"] for item in categories.values()) == 36
    assert categories["scaling"]["n"] == 16
    assert categories["imputation"]["n"] == 15
    assert categories["future_target"]["n"] == 4
    assert categories["encoding"]["n"] == 1
    assert all(not item["errors"] for item in categories.values())


def test_additional_cases_have_unique_names_and_snippets():
    names = [case.name for case in EXTRA_CASES]
    snippets = [case.code for case in EXTRA_CASES]

    assert len(names) == len(set(names))
    assert len(snippets) == len(set(snippets))


def test_low_severity_cases_are_scored_separately():
    metrics = evaluate()

    assert metrics["low_n"] == 14
    assert metrics["low_tp"] + metrics["low_tn"] == 14
    assert metrics["low_fp"] == 0
    assert metrics["low_fn"] == 0