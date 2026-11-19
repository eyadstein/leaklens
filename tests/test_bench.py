import ast
from pathlib import Path

from leaklens import bench
from leaklens.bench import evaluate
from leaklens.bench_cases import EXTRA_CASES


def test_benchmark_has_no_errors():
    metrics = evaluate()
    assert metrics["errors"] == [], metrics["errors"]


def test_legacy_baseline_is_preserved():
    metrics = evaluate()["legacy"]

    assert metrics["n"] == 73
    assert metrics["leaky"] == 40
    assert metrics["clean"] == 33
    assert metrics["tp"] == 40
    assert metrics["fp"] == 0
    assert metrics["tn"] == 33
    assert metrics["fn"] == 0


def test_expanded_benchmark_counts():
    metrics = evaluate()

    assert metrics["n"] == 117
    assert metrics["leaky"] == 63
    assert metrics["clean"] == 54
    assert metrics["tp"] == 63
    assert metrics["fp"] == 0
    assert metrics["tn"] == 54
    assert metrics["fn"] == 0


def test_additional_categories_are_complete():
    metrics = evaluate()
    categories = metrics["categories"]

    assert sum(item["n"] for item in categories.values()) == 44
    assert categories["scaling"]["n"] == 18
    assert categories["imputation"]["n"] == 15
    assert categories["future_target"]["n"] == 10
    assert categories["encoding"]["n"] == 1
    assert all(not item["errors"] for item in categories.values())


def test_additional_cases_have_unique_names_and_snippets():
    names = [case.name for case in EXTRA_CASES]
    snippets = [case.code for case in EXTRA_CASES]

    assert len(names) == len(set(names))
    assert len(snippets) == len(set(snippets))


def test_low_severity_cases_are_scored_separately():
    metrics = evaluate()

    assert metrics["low_n"] == 19
    assert metrics["low_tp"] + metrics["low_tn"] == 19
    assert metrics["low_fp"] == 0
    assert metrics["low_fn"] == 0


def test_cases_are_registered_before_the_main_guard():
    """A case batch indented under ``if __name__`` only exists on direct runs."""
    tree = ast.parse(Path(bench.__file__).read_text(encoding="utf-8"))
    guard = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
    )
    late = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"CASES", "LOW_CASES"}
        and node.lineno > guard
    ]
    assert not late, f"case registration after the __main__ guard: lines {late}"
