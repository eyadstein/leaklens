"""M5 corpus-integrity guards behind the H2 shortfall.

No independent H2 corpus exists, so these tests do not score one. They pin the
counts, label exclusions, duplicate/overlap facts and artifact identity that
``docs/AI_REALWORLD_H2.md`` reports, so the shortfall cannot silently turn into
a number without a reviewed corpus and a freeze digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from leaklens.ai.corpus import development_cases
from leaklens.ai.corpus_realworld import (
    holdout_cases,
    manifest_digest,
    repositories,
)
from leaklens.ai.features import MODEL_FEATURE_NAMES, model_features
from leaklens.ai.learned import MODEL_VERSION, _metrics, get_model
from leaklens.ai.realworld import public_notebook_cases

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "realworld"
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DEFINITIVE = {"confirmed_leakage", "legitimate"}


def _labels() -> dict[str, dict[str, object]]:
    return json.loads((NOTEBOOKS / "labels.json").read_text(encoding="utf-8"))


def _manifest() -> list[dict[str, object]]:
    raw = (NOTEBOOKS / "manifest.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _tokens(code: str) -> set[str]:
    return {match.lower() for match in _WORD.findall(code)}


def _max_overlap(code: str, others: list[set[str]]) -> float:
    tokens = _tokens(code)
    best = 0.0
    for other in others:
        union = tokens | other
        if union:
            best = max(best, len(tokens & other) / len(union))
    return best


def test_shortfall_document_is_published_without_an_h2_corpus():
    document = ROOT / "docs" / "AI_REALWORLD_H2.md"
    assert document.is_file(), "the H2 shortfall must be published, not replaced"
    text = document.read_text(encoding="utf-8").lower()
    assert "not available" in text
    assert "no h2 number" in text or "no number" in text
    assert not (ROOT / "leaklens" / "ai" / "corpus_h2.py").exists(), (
        "an H2 corpus must not appear without reviewers and a freeze record"
    )
    assert not (ROOT / "docs" / "realworld_holdout_h2.json").exists()


def test_notebook_corpus_accounts_for_every_label_and_only_scores_definitive_ones():
    labels = _labels()
    verdicts = Counter(str(item["verdict"]) for item in labels.values())

    assert len(labels) == 60
    assert verdicts == {
        "confirmed_leakage": 5,
        "legitimate": 3,
        "potential_leakage": 2,
        "uncertain": 50,
    }

    cases = public_notebook_cases()
    assert len(cases) == 8
    assert sum(1 for case in cases if case["leaky"]) == 5
    assert {case["verdict"] for case in cases} == _DEFINITIVE
    assert len({case["repository"] for case in cases}) == 8
    assert len({case["code"] for case in cases}) == 8
    assert verdicts["potential_leakage"] + verdicts["uncertain"] == 52, (
        "the 52 potential/uncertain notebooks must stay excluded"
    )


def test_notebook_fixtures_match_their_manifest_hashes_and_are_unique():
    manifest = _manifest()

    assert len(manifest) == 60
    assert len({str(item["sha256"]) for item in manifest}) == 60
    for item in manifest:
        fixture = NOTEBOOKS / "notebooks" / f"{item['id']}.ipynb"
        data = fixture.read_bytes()
        json.loads(data.decode("utf-8"))  # still a readable notebook
        # .gitattributes normalizes line endings, so a Linux checkout can differ
        # in byte count from the recorded one; only re-hash unrewritten content.
        if len(data) == item["bytes"]:
            digest = hashlib.sha256(data).hexdigest()[:16]
            assert digest == item["sha256"], (
                f"{item['id']} fixture changed after freezing"
            )


def test_notebook_repositories_are_disjoint_from_h1_and_the_authored_corpus():
    notebook_repositories = {str(item["source_repo"]) for item in _manifest()}
    h1_repositories = {repository.slug for repository in repositories()}
    development_sources = {str(case.source) for case in development_cases()}

    assert len(notebook_repositories) == 57
    assert notebook_repositories.isdisjoint(h1_repositories)
    assert notebook_repositories.isdisjoint(development_sources)


def test_available_real_corpora_have_no_near_duplicate_units():
    development = [_tokens(case.code) for case in development_cases()]
    holdout = holdout_cases()

    for case in holdout:
        assert _max_overlap(case.code, development) < 0.8, case.name

    holdout_tokens = [_tokens(case.code) for case in holdout]
    for case in public_notebook_cases():
        code = str(case["code"])
        assert _max_overlap(code, development) < 0.8, case["name"]
        assert _max_overlap(code, holdout_tokens) < 0.8, case["name"]


def test_h1_stays_the_single_reviewed_phase6_record():
    holdout = holdout_cases()

    assert len(holdout) == 48
    assert len(repositories()) == 12
    assert sum(1 for case in holdout if case.leaky) == 24
    assert len({case.name for case in holdout}) == 48
    assert {case.review for case in holdout} == {"manual-review-2026-11-05"}
    assert manifest_digest() == (
        "7fde39ce94d34ae7f05243163e2761d5410f4b3a01f9770eabc2a24ea45d5742"
    )


def test_shipped_model_identity_is_pinned_for_any_future_h2_run():
    _evaluation, model = get_model(compare_candidates=False)

    assert model.card.version == MODEL_VERSION == "leaklens-ai-p7-2026.11"
    assert len(MODEL_FEATURE_NAMES) == 108
    assert model.threshold == 0.53
    assert model.card.artifact_digest.startswith("sha256:")


def test_metric_denominators_match_an_independently_checked_example():
    # labels [1,1,1,0,0,0,0] vs predictions [1,1,0,1,0,0,1]
    # TP=2 FP=2 FN=1 TN=2 (7 units, 3 leaky, 4 legitimate)
    metrics = _metrics([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1])

    assert metrics["true_positives"] == 2.0
    assert metrics["false_positives"] == 2.0
    assert metrics["false_negatives"] == 1.0
    assert metrics["true_negatives"] == 2.0
    assert metrics["precision"] == 0.5  # 2 / (2 + 2)
    assert metrics["recall"] == 0.6667  # 2 / (2 + 1)
    assert metrics["f1"] == 0.5714
    assert metrics["false_positive_rate"] == 0.5  # 2 / (2 + 2)
    assert metrics["specificity"] == 0.5
    assert metrics["accuracy"] == 0.5714  # 4 / 7


def test_metric_zero_divisions_report_zero_instead_of_raising():
    metrics = _metrics([0, 0], [0, 0])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

    empty = _metrics([], [])
    assert empty["accuracy"] == 0.0


def test_rescoring_uses_the_same_model_and_leaves_its_identity_unchanged():
    cases = holdout_cases()[:4]
    first = [model_features(case.code) for case in cases]
    second = [model_features(case.code) for case in cases]
    for left, right in zip(first, second):
        assert (left == right).all()

    _evaluation, model = get_model(compare_candidates=False)
    digest = model.card.artifact_digest
    threshold = model.threshold
    assert (model.predict_matrix(first) == model.predict_matrix(second)).all()

    assert model.card.artifact_digest == digest
    assert model.threshold == threshold
