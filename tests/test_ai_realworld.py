"""Phase 6 real-world holdout: provenance, freeze integrity and evaluation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from leaklens.ai import realworld
from leaklens.ai.corpus import development_cases
from leaklens.ai.corpus_realworld import (
    HOLDOUT_ID,
    REALWORLD_CASES,
    RETRIEVED,
    corpus_stats,
    holdout_cases,
    manifest,
    manifest_digest,
    repositories,
)
from leaklens.ai.features import model_features
from leaklens.ai.learned import reset_model_cache
from leaklens.ai.realworld import (
    FROZEN_DIGEST,
    evaluate_public_notebooks,
    evaluate_realworld,
    public_notebook_cases,
)


def _tokens(code: str) -> set[str]:
    return set(re.findall(r"[a-z_][a-z0-9_]*", code.lower()))


def test_holdout_is_frozen_with_a_recorded_digest():
    assert HOLDOUT_ID == "phase6-realworld-holdout-v1"
    assert RETRIEVED == "2026-11-05"
    assert manifest_digest() == FROZEN_DIGEST, (
        "the real-world holdout changed after it was frozen"
    )
    record = manifest()
    assert record["digest"] == FROZEN_DIGEST
    assert record["stats"] == corpus_stats()
    assert record["role"] == "single_pass_evaluation"


def test_holdout_has_complete_provenance_and_balanced_reviewed_labels():
    cases = holdout_cases()
    stats = corpus_stats()

    assert len(cases) == 48
    assert stats["leaky"] == stats["legitimate"] == 24
    assert stats["repositories"] == len(repositories()) == 12
    assert stats["families"] >= stats["repositories"]
    assert stats["categories"] >= 10

    for case in cases:
        assert case.name and case.code.strip()
        assert case.label_rationale.strip()
        assert case.path.strip()
        assert case.provenance in {"verbatim_excerpt", "reconstructed_excerpt"}
        assert case.source_url.startswith(f"https://github.com/{case.repository}/blob/")
        assert case.path in case.source_url
        assert case.review.startswith("manual-review-")
        assert case.category

    assert len({case.name for case in cases}) == len(cases)
    assert len({case.code for case in cases}) == len(cases)


def test_holdout_repositories_are_licensed_and_pinned():
    for repository in repositories():
        assert repository.slug in {case.repository for case in REALWORLD_CASES}
        assert repository.license_id
        assert repository.license_url.startswith("https://github.com/")
        assert re.fullmatch(r"[0-9a-f]{40}", repository.commit)
        assert repository.branch


def test_holdout_families_never_span_repositories():
    families: dict[str, set[str]] = {}
    for case in REALWORLD_CASES:
        families.setdefault(case.family, set()).add(case.repository)

    assert all(len(owners) == 1 for owners in families.values()), (
        "a pipeline family spans more than one repository"
    )


def test_holdout_is_disjoint_from_the_authored_development_corpus():
    development = development_cases()
    development_codes = {case.code for case in development}
    assert not development_codes & {case.code for case in REALWORLD_CASES}

    development_tokens = [(case.name, _tokens(case.code)) for case in development]
    maximum = 0.0
    worst: tuple[str, str] = ("", "")
    for case in REALWORLD_CASES:
        tokens = _tokens(case.code)
        for name, other in development_tokens:
            union = tokens | other
            if not union:
                continue
            overlap = len(tokens & other) / len(union)
            if overlap > maximum:
                maximum = overlap
                worst = (case.name, name)

    assert maximum < 0.8, f"holdout near-duplicates development case {worst}"


def test_holdout_does_not_reuse_development_repository_strings():
    development_sources = {case.source for case in development_cases()}
    repository_slugs = {repository.slug for repository in repositories()}

    assert development_sources.isdisjoint(repository_slugs), (
        "a holdout repository appears as an authored corpus source"
    )
    for repository in repositories():
        assert "train:" not in repository.slug


def test_every_holdout_case_produces_finite_model_features():
    for case in REALWORLD_CASES:
        vector = model_features(case.code)
        assert vector.shape == (108,)
        assert np.isfinite(vector).all()
        assert vector.any(), f"{case.name} activates no model feature"


def test_frozen_model_scores_the_holdout_within_reported_bounds():
    reset_model_cache()
    data = evaluate_realworld(bootstrap=200)

    assert data["digest"] == FROZEN_DIGEST
    assert data["digest_matches_freeze"] is True
    assert data["cases"] == 48
    assert data["leaky"] + data["legitimate"] == data["cases"]
    confusion = data["confusion"]
    assert sum(confusion.values()) == data["cases"]
    assert (
        abs(
            data["metrics"]["accuracy"]
            - (confusion["true_positives"] + confusion["true_negatives"])
            / data["cases"]
        )
        <= 1e-3
    )

    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate"):
        assert 0.0 <= data["metrics"][key] <= 1.0
    assert data["metrics"]["recall"] >= 0.9
    assert data["metrics"]["precision"] >= 0.6

    assert data["per_category"]
    assert set(data["per_category"]).issubset(
        {case.category for case in REALWORLD_CASES}
    )
    assert set(data["per_repository"]) == {r.slug for r in repositories()}
    assert set(data["per_provenance"]) == {
        "reconstructed_excerpt",
        "verbatim_excerpt",
    }
    assert data["model"]["features"] == 108
    # The shipped threshold is chosen on authored validation only; it is 0.60
    # after the M6 rebuild. The Phase 6 report stays a historical record.
    assert data["model"]["threshold"] == 0.60
    assert data["calibration"]["brier"] <= 0.5


def test_reported_mistakes_are_labelled_and_traceable_to_code():
    data = evaluate_realworld(bootstrap=0)

    assert data["false_positives"] or data["false_negatives"]
    for key, is_leaky in (("false_positives", False), ("false_negatives", True)):
        for item in data[key]:
            assert item["case"]
            assert item["repository"]
            assert item["category"]
            assert item["rationale"]
            assert 0.0 <= item["probability"] <= 1.0
            case = next(case for case in REALWORLD_CASES if case.name == item["case"])
            assert case.leaky is is_leaky
            lines = case.code.splitlines()
            for evidence in item["evidence"]:
                assert 1 <= evidence["line"] <= len(lines)
                assert evidence["code"]
                assert evidence["feature"]
                assert evidence["code"] == lines[evidence["line"] - 1].strip()


def test_public_notebook_check_uses_only_definitive_hand_reviews():
    cases = public_notebook_cases()

    # The fixture corpus is checked into the repository; if it is missing, the
    # benchmark reports the check as unavailable instead of failing.
    if not cases:
        labels = Path(__file__).resolve().parents[1] / "realworld" / "labels.json"
        assert not labels.is_file()
        return

    assert len(cases) == 8
    assert {case["verdict"] for case in cases} == {
        "confirmed_leakage",
        "legitimate",
    }
    assert sum(1 for case in cases if case["leaky"]) == 5
    for case in cases:
        assert case["code"].strip()
        assert case["repository"]
        assert case["source_url"].startswith("https://github.com/")
        assert case["reason"].strip()
        assert case["sha256"]


def test_public_notebook_smoke_check_is_measured_and_traceable():
    data = evaluate_public_notebooks()

    if not data.get("available"):
        return

    assert data["case_count"] == 8
    # The exact predictions can shift with a scikit-learn release; the check is
    # a smoke test, so assert the measured band rather than one bit pattern.
    assert data["metrics"]["accuracy"] >= 0.75
    assert data["metrics"]["precision"] >= 0.7
    assert data["metrics"]["recall"] >= 0.8
    assert len(data["mistakes"]) <= 2
    for item in data["cases"]:
        assert 0.0 <= item["probability"] <= 1.0
        assert item["correct"] == (item["predicted"] == item["leaky"])
        assert item["source_url"].startswith("https://github.com/")
    assert data["accuracy_interval"][0] <= data["metrics"]["accuracy"]
    assert data["accuracy_interval"][1] >= data["metrics"]["accuracy"]


def test_benchmark_report_carries_the_freeze_digest_and_all_checks():
    from leaklens import bench_realworld

    report, data = bench_realworld.build_report(bootstrap=0, profiling=False)

    assert data["manifest_digest"] == FROZEN_DIGEST
    assert "Digest matches the freeze commit: **yes**" in report
    assert "## Public notebook smoke check (real code, not reconstructions)" in report
    assert "## Candidate comparison (development data only)" in report
    assert "## Metrics by category" in report
    assert "## Uncertainty" in report
    assert "## Calibration" in report
    assert "Scanner-benchmark transfer" in report
    assert "development result" in report


def test_training_modules_do_not_import_the_realworld_holdout():
    import inspect

    from leaklens.ai import corpus, learned

    for module in (learned, corpus):
        source = inspect.getsource(module)
        assert "corpus_realworld" not in source
        assert "realworld" not in source

    assert realworld.FROZEN_DIGEST == FROZEN_DIGEST
