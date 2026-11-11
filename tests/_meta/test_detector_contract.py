"""Every emitted finding kind must be registered in the detector manifest."""

import ast
from pathlib import Path

import pytest

from leaklens import manifest
from leaklens.fixes import FIXES

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "leaklens"
DOCS = ROOT / "docs" / "DETECTORS.md"

ASSIGNMENT_NAMES = {"kind", "kd"}


def _strings(node):
    if node is None:
        return set()
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def emitted_kinds() -> set[str]:
    """Collect finding kinds from production emitters without executing them."""
    kinds: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name.startswith("bench") or path.name in {"manifest.py"}:
            continue
        # The AI/intelligence layers describe hypotheses, not deterministic
        # findings, and reuse local variables named ``kind``.
        if {"ai", "intelligence"} & set(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name not in {"Finding", "_sql_finding"}:
                    continue
                if (
                    len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    kinds.add(node.args[1].value)
                for keyword in node.keywords:
                    if (
                        keyword.arg == "kind"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        kinds.add(keyword.value.value)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if not any(
                    isinstance(target, ast.Name) and target.id in ASSIGNMENT_NAMES
                    for target in targets
                ):
                    continue
                kinds |= _strings(node.value)
    return kinds


def benchmark_case_names() -> set[str]:
    from leaklens import (
        bench,
        bench_advanced,
        bench_cases,
        bench_data,
        bench_drift,
        bench_missing,
        bench_sql,
    )

    names = {name for name, _, _ in bench.CASES}
    names |= {name for name, _, _ in bench.LOW_CASES}
    names |= {case.name for case in bench_cases.EXTRA_CASES}
    names |= {case.name for case in bench_advanced.ADVANCED_CASES}
    names |= {case.name for case in bench_data._build_cases()}
    names |= {case.name for case in bench_missing._build_cases()}
    names |= {case.name for case in bench_sql._build_cases()}
    names |= {case.name for case in bench_drift._build_cases()}
    return names


def test_every_emitted_kind_is_registered():
    missing = emitted_kinds() - set(manifest.BY_KIND)
    assert not missing, f"unregistered finding kinds: {sorted(missing)}"


def test_manifest_has_no_orphan_kinds():
    extra = set(manifest.BY_KIND) - emitted_kinds()
    assert not extra, f"manifest kinds no detector emits: {sorted(extra)}"


def test_manifest_entries_are_structurally_valid():
    manifest.validate_manifest()
    assert len(manifest.DETECTORS) == len(manifest.BY_KIND)
    for entry in manifest.DETECTORS:
        assert entry.severity in {"high", "medium", "low"}
        assert entry.advice in FIXES
        assert entry.surface in manifest.SURFACES
        assert entry.docs_anchor == f"#{entry.kind}"
        assert entry.introduced
        assert bool(entry.benchmark) ^ bool(entry.no_bench_reason)
        assert isinstance(entry.scored, bool)
        assert entry.modalities <= manifest.MODALITIES
        assert entry.requires in (None, "images", "sql")


def test_diagnostic_kinds_are_not_scored():
    assert {e.kind for e in manifest.DETECTORS if not e.scored} == {
        "distribution_shift",
        "sql_unresolved_column",
    }


def test_optional_kinds_declare_their_extra():
    assert manifest.BY_KIND["image_overlap"].requires == "images"
    for kind in (
        "sql_target_as_feature",
        "sql_split_predicate_missing",
        "sql_unresolved_column",
        "sql_future_window",
    ):
        assert manifest.BY_KIND[kind].requires == "sql"
        assert manifest.BY_KIND[kind].modalities == {"sql"}


def test_manifest_benchmark_cases_exist():
    cases = benchmark_case_names()
    missing = {
        entry.benchmark
        for entry in manifest.DETECTORS
        if entry.benchmark and entry.benchmark not in cases
    }
    assert not missing, f"unknown benchmark cases: {sorted(missing)}"


def test_missing_benchmark_coverage_is_explained():
    for entry in manifest.DETECTORS:
        if entry.benchmark is None:
            assert entry.no_bench_reason
            assert len(entry.no_bench_reason) >= 20


def test_unknown_kind_fails_the_contract():
    with pytest.raises(KeyError):
        manifest.require_kind("deliberately_unknown_kind")
    with pytest.raises(ValueError):
        manifest.validate_kinds({"deliberately_unknown_kind"})


def test_docs_match_the_manifest():
    assert DOCS.exists(), (
        "docs/DETECTORS.md is missing; run python -m leaklens.manifest"
    )
    assert DOCS.read_text(encoding="utf-8") == manifest.render_docs()
