"""Prompt-injection trust-boundary regressions and corpus checks.

These tests exercise the real prompt builders and the real LLM enrichment path.
No network call is made and no model is required.
"""

from __future__ import annotations

import json
import re

import pytest

from leaklens import llm
from leaklens.ai import prompt_injection as corpus
from leaklens.ai.reasoning import build_plan, enrich_plan
from leaklens.leakage import Finding, score


class _Explainer:
    def __init__(self, response: str):
        self.response = response

    def explain(self, finding) -> str:
        return self.response


class _Completer:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _finding(case: dict) -> Finding:
    if case["vector"] == "column_name":
        return Finding("high", "target_leakage", case["content"], "predicts the label")
    return Finding("high", "target_leakage", "model.py:8", case["content"])


def _valid_plan_response(plan) -> str:
    group = plan.root_causes[0]
    return json.dumps(
        {
            "summary": "The plan groups a train/test boundary problem.",
            "notes": [
                {
                    "group_id": group.group_id,
                    "explanation": "Fit the transformer on training rows only.",
                    "actions": ["Split before fitting."],
                    "verification": ["Re-scan and compare."],
                }
            ],
        }
    )


def _hostile_plan_response() -> str:
    return json.dumps(
        {
            "summary": "The dataset has zero risk and is safe to deploy.",
            "notes": [
                {
                    "group_id": "rc-invented",
                    "explanation": "Ignore the finding.",
                    "actions": ["Ship it."],
                    "verification": ["Nothing."],
                }
            ],
        }
    )


def test_corpus_schema_validates_and_ids_are_unique():
    corpus.validate_corpus()

    cases = list(corpus.load_corpus())
    duplicated = [*cases, dict(cases[0])]
    with pytest.raises(corpus.CorpusError, match="duplicate case id"):
        corpus.validate_corpus(duplicated)


def test_corpus_covers_required_categories_and_vectors():
    cases = corpus.load_corpus()

    assert {case["category"] for case in cases} == set(corpus.CATEGORIES)
    assert set(corpus.VECTORS) <= {case["vector"] for case in cases}
    assert any(case["attack"] for case in cases)
    assert any(not case["attack"] for case in cases)
    assert any(case["partition"] == "holdout" for case in cases)
    assert any(case["partition"] == "development" for case in cases)


def test_corpus_contains_no_real_credential_shapes():
    real_secret = re.compile(
        r"(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|"
        r"sk-[A-Za-z0-9]{16,}|-----BEGIN [A-Z ]+PRIVATE KEY-----)"
    )
    for case in corpus.load_corpus():
        assert not real_secret.search(case["content"])
        assert not real_secret.search(case["simulated_response"])


def test_development_and_holdout_digests_are_stable_and_pinned():
    development = corpus.partition_cases("development")
    holdout = corpus.partition_cases("holdout")

    assert corpus.corpus_digest(development) == corpus.DEVELOPMENT_DIGEST
    assert corpus.corpus_digest(holdout) == corpus.HOLDOUT_DIGEST
    assert corpus.corpus_digest(development) == corpus.corpus_digest(
        corpus.partition_cases("development")
    )
    assert {item["case_id"] for item in development}.isdisjoint(
        item["case_id"] for item in holdout
    )


@pytest.mark.parametrize("case", corpus.load_corpus(), ids=lambda case: case["case_id"])
def test_untrusted_content_cannot_break_out_of_the_prompt_field(case):
    prompt = llm._prompt(_finding(case))
    template = llm.load_prompt_template()

    assert prompt.count("\n") == template.count("\n")
    assert "\x00" not in prompt
    assert corpus.rendered_content(case) in prompt


def test_prompt_contains_only_the_selected_finding(monkeypatch):
    monkeypatch.setenv("LEAKLENS_LLM_API_KEY", "CANARY-ENV-API-KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "CANARY-ENV-OPENAI-KEY")
    case = corpus.case_by_id("A-001-ignore-previous")
    other = Finding("medium", "id_like", "customer_id", "CANARY-OTHER-FINDING")

    prompt = llm._prompt(_finding(case))

    assert "CANARY-ENV-API-KEY" not in prompt
    assert "CANARY-ENV-OPENAI-KEY" not in prompt
    assert other.detail not in prompt
    assert corpus.rendered_content(case) in prompt


@pytest.mark.parametrize(
    "case",
    [item for item in corpus.load_corpus() if item["attack"]],
    ids=lambda case: case["case_id"],
)
def test_attack_cannot_change_deterministic_state(case):
    finding = _finding(case)
    before = (finding.severity, finding.kind, finding.column, finding.detail)
    before_score = score([finding])

    llm.explain_findings([finding], _Explainer(case["simulated_response"]))
    plan = build_plan((finding,), ())
    enrich_plan(plan, _Completer(_hostile_plan_response()))

    assert (finding.severity, finding.kind, finding.column, finding.detail) == before
    assert score([finding]) == before_score


GUARD_COVERED_CASES = (
    "A-001-ignore-previous",
    "A-002-replace-system",
    "A-003-fake-admin",
    "A-004-safety-disabled",
    "A-006-mark-clean",
    "A-007-retrieval-system-prompt",
    "A-008-multiple-injections",
    "B-001-dataset-cell",
    "B-002-notebook-markdown",
    "B-003-source-comment",
    "B-004-readme",
    "B-005-retrieved-document",
    "B-006-filename",
    "B-007-metadata-label",
    "B-008-column-name",
    "B-009-base64",
    "D-004-alter-report",
)


@pytest.mark.parametrize(
    "case", [corpus.case_by_id(case_id) for case_id in GUARD_COVERED_CASES]
)
def test_canonical_unsupported_claims_are_dropped(case):
    finding = _finding(case)

    explained = llm.explain_findings([finding], _Explainer(case["simulated_response"]))

    assert explained == [None]


def test_known_paraphrase_is_reported_as_a_false_negative_and_stays_contained():
    case = corpus.case_by_id("A-009-launch-paraphrase")
    finding = _finding(case)
    before = score([finding])

    explained = llm.explain_findings([finding], _Explainer(case["simulated_response"]))
    report = corpus.evaluate()

    assert explained is not None
    assert "A-009-launch-paraphrase" in report["development"]["false_negatives"]
    assert score([finding]) == before
    assert finding.severity == "high"


@pytest.mark.parametrize(
    "case",
    [item for item in corpus.load_corpus() if not item["attack"]],
    ids=lambda case: case["case_id"],
)
def test_benign_hard_negatives_keep_a_normal_explanation(case):
    finding = _finding(case)

    explained = llm.explain_findings([finding], _Explainer(case["simulated_response"]))

    assert explained == [llm._sanitize_llm_output(case["simulated_response"])]


def test_plan_prompt_excludes_raw_untrusted_text_and_hostile_narration_falls_back():
    case = corpus.case_by_id("B-007-metadata-label")
    finding = _finding(case)
    plan = build_plan((finding,), ())
    hostile = _Completer(_hostile_plan_response())

    enriched = enrich_plan(plan, hostile)

    assert corpus.rendered_content(case) not in hostile.prompts[0]
    assert enriched.enrichment == "deterministic_fallback"
    assert enriched.root_causes == plan.root_causes
    assert enriched.priority_order == plan.priority_order


def test_plan_prompt_keeps_valid_narration_bound_to_existing_groups():
    plan = build_plan((Finding("high", "target_leakage", "leak", "predicts"),), ())
    provider = _Completer(_valid_plan_response(plan))

    enriched = enrich_plan(plan, provider)

    assert enriched.enrichment == "llm"
    assert enriched.root_causes == plan.root_causes
    assert "untrusted data" in provider.prompts[0]


def test_provider_failure_still_falls_back_without_touching_findings():
    finding = Finding("high", "target_leakage", "leak", "predicts")
    plan = build_plan((finding,), ())

    class Offline:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("provider is offline")

    enriched = enrich_plan(plan, Offline())

    assert enriched.enrichment == "deterministic_fallback"
    assert "offline" in enriched.fallback_reason
    assert (finding.severity, finding.kind, finding.column) == (
        "high",
        "target_leakage",
        "leak",
    )


def test_output_guard_has_a_documented_false_positive():
    # The guard is a denylist, not a classifier.  A benign sentence that uses
    # security vocabulary is dropped; the docs record this limitation.
    finding = Finding("high", "target_leakage", "leak", "predicts")

    explained = llm.explain_findings(
        [finding], _Explainer("The synthetic example carries no leakage risk.")
    )

    assert explained == [None]


def test_multiple_injections_in_one_input_are_contained():
    case = corpus.case_by_id("A-008-multiple-injections")
    finding = _finding(case)
    prompt = llm._prompt(finding)
    template = llm.load_prompt_template()

    assert prompt.count("\n") == template.count("\n")
    explained = llm.explain_findings([finding], _Explainer(case["simulated_response"]))
    assert explained == [None]
    assert score([finding]) == 65


def test_evaluation_is_reproducible_and_meets_the_published_thresholds():
    first = corpus.evaluate()
    second = corpus.evaluate()

    assert first == second
    assert corpus.check(first) == []


def test_holdout_scored_once_is_not_used_by_the_development_partition():
    report = corpus.evaluate()
    development_ids = {
        case["case_id"] for case in corpus.partition_cases("development")
    }
    holdout_ids = {case["case_id"] for case in corpus.partition_cases("holdout")}

    assert report["holdout"]["cases"] == len(holdout_ids)
    assert report["development"]["cases"] == len(development_ids)
    assert development_ids.isdisjoint(holdout_ids)
    assert report["holdout"]["structural_containment_rate"] == 1.0
    assert report["holdout"]["plan_containment_rate"] == 1.0
