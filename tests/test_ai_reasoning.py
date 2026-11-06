"""Evidence-bound reasoning, remediation and LLM fallback behavior."""

from __future__ import annotations

import json

from leaklens.ai.reasoning import build_plan, enrich_plan, note_for
from leaklens.ai.schemas import EvidenceSpan, Hypothesis
from leaklens.leakage import Finding


class StaticProvider:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class BrokenProvider:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("provider is offline")


def hyp(
    hypothesis_id: str,
    concept: str,
    evidence_id: str,
    *,
    confidence: float = 0.7,
    location: str = "model.py:3",
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        concept=concept,
        title="A semantic observation",
        explanation="A feature looks derived from the label.",
        confidence=confidence,
        evidence=(
            EvidenceSpan(
                evidence_id=evidence_id,
                source="code",
                kind="data_flow",
                location=location,
                excerpt="df['score'] = df['target'] * 2",
            ),
        ),
        location=location,
        detector="semantics",
        severity_hint="medium",
        limitations=("A heuristic can be wrong.",),
    )


def plan_with_groups():
    findings = (
        Finding("high", "target_leakage", "leak", "predicts the target"),
        Finding("medium", "id_like", "customer_id", "98% unique values"),
    )
    hypotheses = (
        hyp("hyp-target", "target_proxy_name", "ev-target"),
        hyp("hyp-learned", "learned_leakage_pattern", "ev-learned", confidence=0.6),
    )
    return build_plan(findings, hypotheses, target="leak")


def test_build_plan_connects_related_observations_without_changing_findings():
    findings = (
        Finding("high", "target_leakage", "leak", "predicts the target"),
        Finding("medium", "id_like", "customer_id", "98% unique values"),
    )
    hypotheses = (
        hyp("hyp-target", "target_proxy_name", "ev-target"),
        hyp("hyp-learned", "learned_leakage_pattern", "ev-learned", confidence=0.6),
    )
    plan = build_plan(findings, hypotheses, target="leak")

    assert [group.group_id for group in plan.root_causes] == list(plan.priority_order)
    categories = {group.category for group in plan.root_causes}
    assert {"target_leakage", "identifier", "learned"} <= categories
    target_group = next(
        group for group in plan.root_causes if group.category == "target_leakage"
    )
    assert target_group.source == "combined"
    assert target_group.priority == "high"
    assert "det-0" in target_group.evidence_ids
    assert "hyp-target" in target_group.hypothesis_ids
    assert target_group.remediation.actions
    assert target_group.remediation.verification
    assert target_group.remediation.code_hint
    assert "did not change" in plan.summary
    assert any("target column 'leak'" in step for step in plan.verification)
    assert findings[0].severity == "high"


def test_phase6_finding_kinds_get_specific_root_cause_and_remediation():
    findings = (
        Finding("medium", "tuning_on_test_data", "model.py:9", "search on test"),
        Finding(
            "low",
            "notebook_out_of_order_execution",
            "nb.ipynb",
            "cells ran out of order",
        ),
        Finding(
            "medium", "group_cv_missing_groups", "model.py:12", "GroupKFold no groups"
        ),
    )
    plan = build_plan(findings, ())

    categories = {group.category for group in plan.root_causes}
    assert {"tuning", "notebook_order", "grouping"} <= categories
    notebook = next(
        group for group in plan.root_causes if group.category == "notebook_order"
    )
    assert notebook.remediation.code_hint
    assert "train_test_split" in notebook.remediation.code_hint
    assert "Restart the kernel" in " ".join(notebook.remediation.actions)


def test_build_plan_handles_clean_inputs_without_claiming_safety():
    plan = build_plan((), ())

    assert plan.root_causes == ()
    assert plan.priority_order == ()
    assert "does not prove" in plan.summary
    assert plan.verification


def valid_response(plan) -> str:
    group = plan.root_causes[0]
    return json.dumps(
        {
            "summary": "Split first, then fit transformers on training rows only.",
            "notes": [
                {
                    "group_id": group.group_id,
                    "explanation": "This group describes a train/test boundary issue.",
                    "actions": ["Split before fitting."],
                    "verification": ["Re-scan and compare scores."],
                }
            ],
        }
    )


def test_valid_llm_enrichment_is_bound_to_existing_group_ids():
    plan = plan_with_groups()
    provider = StaticProvider(valid_response(plan))

    enriched = enrich_plan(plan, provider)

    assert enriched.enrichment == "llm"
    assert enriched.llm_summary.startswith("Split first")
    assert enriched.root_causes == plan.root_causes
    assert enriched.priority_order == plan.priority_order
    assert enriched.fallback_reason is None
    note = note_for(enriched, enriched.root_causes[0])
    assert note is not None
    assert note.actions == ("Split before fitting.",)
    assert "df['score']" not in provider.prompts[0]
    assert "untrusted data" in provider.prompts[0]


def test_llm_cannot_invent_a_group_id():
    plan = plan_with_groups()
    response = json.dumps(
        {
            "summary": "A summary.",
            "notes": [
                {
                    "group_id": "rc-invented",
                    "explanation": "Invented group.",
                    "actions": ["Do something."],
                    "verification": ["Check."],
                }
            ],
        }
    )

    enriched = enrich_plan(plan, StaticProvider(response))

    assert enriched.enrichment == "deterministic_fallback"
    assert "unknown group" in enriched.fallback_reason
    assert enriched.root_causes == plan.root_causes
    assert enriched.llm_notes == ()
    assert enriched.llm_summary is None


def test_llm_cannot_claim_zero_risk_or_reassign_severity():
    plan = plan_with_groups()
    response = json.dumps(
        {
            "summary": "The dataset has zero risk and is safe to deploy.",
            "notes": [
                {
                    "group_id": plan.root_causes[0].group_id,
                    "explanation": "Severity should be low now.",
                    "actions": ["Ship it."],
                    "verification": ["Nothing to check."],
                }
            ],
        }
    )

    enriched = enrich_plan(plan, StaticProvider(response))

    assert enriched.enrichment == "deterministic_fallback"
    assert "unsupported safety" in enriched.fallback_reason
    assert enriched.root_causes == plan.root_causes


def test_malformed_llm_output_falls_back_to_the_deterministic_plan():
    plan = plan_with_groups()

    for response in ("not json at all", '{"notes": []}', '{"summary": "ok"}'):
        enriched = enrich_plan(plan, StaticProvider(response))
        assert enriched.enrichment == "deterministic_fallback"
        assert enriched.root_causes == plan.root_causes
        assert enriched.summary == plan.summary


def test_provider_failure_falls_back_instead_of_breaking_the_scan():
    plan = plan_with_groups()
    enriched = enrich_plan(plan, BrokenProvider())

    assert enriched.enrichment == "deterministic_fallback"
    assert "offline" in enriched.fallback_reason
    assert enriched.root_causes == plan.root_causes
    assert enriched.verification == plan.verification


def test_no_groups_is_explicitly_marked_as_a_fallback():
    plan = build_plan((), ())
    enriched = enrich_plan(plan, StaticProvider("{}"))

    assert enriched.enrichment == "deterministic_fallback"
    assert "No root-cause groups" in enriched.fallback_reason
