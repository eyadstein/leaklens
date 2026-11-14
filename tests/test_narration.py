"""Phase 18 M4: narration is optional, grounded and removable byte-for-byte."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from leaklens.ai import prompt_injection
from leaklens.ai.reasoning import narrate_investigation
from leaklens.cli import _narrate, main
from leaklens.intelligence import build_investigation
from leaklens.leakage import Finding


class _Provider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _findings(count: int = 3) -> list[Finding]:
    return [
        Finding("high", "target_leakage", f"feature_{index}", "copied target")
        for index in range(count)
    ]


def _group_ids(plan) -> list[str]:
    return [slot.evidence_ids[0] for slot in plan.narration_slots()]


def _response(plan, text: str = "Restate the cited evidence.") -> str:
    return json.dumps(
        {
            "summary": "Deterministic findings grouped into investigation steps.",
            "notes": [
                {"group_id": group_id, "explanation": text}
                for group_id in _group_ids(plan)
            ],
        }
    )


def _canonical(plan) -> str:
    return json.dumps(plan.to_dict(), sort_keys=True)


def _without_texts(plan) -> list[dict]:
    steps = []
    for step in plan.to_dict()["steps"]:
        step = dict(step)
        step.pop("text", None)
        steps.append(step)
    return steps


def test_removing_narration_restores_the_plan_byte_for_byte():
    plan = build_investigation(_findings())
    assert plan.narration == "unavailable"
    assert "text" not in json.dumps(plan.to_dict())

    narrated = narrate_investigation(plan, _Provider(_response(plan)))
    assert narrated.narration == "llm"
    assert [step.text for step in narrated.narration_slots()] == [
        "Restate the cited evidence."
    ] * len(narrated.narration_slots())

    assert _canonical(narrated.without_narration()) == _canonical(plan)


def test_narration_never_changes_steps_findings_or_order():
    plan = build_investigation(_findings())
    narrated = narrate_investigation(plan, _Provider(_response(plan)))

    assert narrated.finding_count == plan.finding_count
    assert narrated.omitted_tests == plan.omitted_tests
    assert _without_texts(narrated) == _without_texts(plan)
    assert [step.ordinal for step in narrated.steps] == [
        step.ordinal for step in plan.steps
    ]


def test_unknown_evidence_id_falls_back_to_the_deterministic_plan():
    plan = build_investigation(_findings())
    hostile = json.dumps(
        {
            "summary": "Invented.",
            "notes": [{"group_id": "ev-not-in-plan", "explanation": "Invented risk."}],
        }
    )
    narrated = narrate_investigation(plan, _Provider(hostile))

    assert narrated.narration == "deterministic_fallback"
    assert _canonical(narrated) == _canonical(
        plan.with_narration("deterministic_fallback")
    )
    assert all(step.text == "" for step in narrated.narration_slots())


@pytest.mark.parametrize(
    "response",
    [
        "not json at all",
        json.dumps({"notes": []}),
        json.dumps({"summary": "", "notes": []}),
    ],
)
def test_malformed_responses_fall_back(response):
    plan = build_investigation(_findings())
    narrated = narrate_investigation(plan, _Provider(response))

    assert narrated.narration == "deterministic_fallback"
    assert _canonical(narrated.without_narration()) == _canonical(plan)


def test_unsupported_safety_claim_is_rejected():
    plan = build_investigation(_findings())
    claimed = _response(plan, "This dataset has no leakage, so it is safe to deploy.")
    narrated = narrate_investigation(plan, _Provider(claimed))

    assert narrated.narration == "deterministic_fallback"
    assert _canonical(narrated.without_narration()) == _canonical(plan)


def test_injection_corpus_cannot_change_the_plan():
    cases = list(prompt_injection.load_corpus())
    assert len(cases) == 47

    for case in cases:
        plan = build_investigation(_findings())
        narrated = narrate_investigation(
            plan, _Provider(prompt_injection.rendered_content(case))
        )
        assert _canonical(narrated.without_narration()) == _canonical(plan)
        assert narrated.narration in {"llm", "deterministic_fallback"}


def test_plan_without_slots_reports_the_fallback():
    plan = build_investigation([])
    narrated = narrate_investigation(plan, _Provider("{}"))

    assert plan.narration_slots() == ()
    assert narrated.narration == "deterministic_fallback"
    assert _canonical(narrated.without_narration()) == _canonical(plan)


def test_narrate_marks_an_unreachable_provider_as_ai_unavailable(monkeypatch):
    plan = build_investigation(_findings())
    args = SimpleNamespace(llm_provider="ollama", llm_base_url=None, llm_model=None)

    def _broken(*_args, **_kwargs):
        raise RuntimeError("no provider")

    monkeypatch.setattr("leaklens.llm.create_provider", _broken)
    narrated = _narrate(plan, args)

    assert narrated.narration == "ai_unavailable"
    assert _canonical(narrated.without_narration()) == _canonical(plan)


def test_cli_intelligence_plan_is_unchanged_without_a_llm(tmp_path, capsys):
    data = tmp_path / "data.csv"
    data.write_text(
        "id,target,leak\n"
        + "\n".join(f"{index},{index % 2},{index % 2}" for index in range(40))
        + "\n",
        encoding="utf-8",
    )

    main([str(data), "--target", "target", "--json"])
    plain = json.loads(capsys.readouterr().out)
    main([str(data), "--target", "target", "--json", "--intelligence"])
    planned = json.loads(capsys.readouterr().out)

    assert planned["findings"] == plain["findings"]
    assert planned["trust_score"] == plain["trust_score"]
    assert planned["investigation"]["narration"] == "unavailable"
    assert "text" not in json.dumps(planned["investigation"])
