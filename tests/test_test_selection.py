"""Phase 18 M2: proposed tests cite a registered detector and are deterministic."""

from __future__ import annotations

import json

import pytest

from leaklens import manifest
from leaklens.cli import main
from leaklens.intelligence import SuggestedTest, build_investigation
from leaklens.leakage import Finding


def _findings(count: int = 3) -> list[Finding]:
    return [
        Finding("high", "target_leakage", f"feature_{index}", "copied target")
        for index in range(count)
    ]


def _tests(plan) -> list[SuggestedTest]:
    return [step for step in plan.steps if isinstance(step, SuggestedTest)]


def test_every_suggested_test_uses_a_registered_detector():
    plan = build_investigation(_findings())
    tests = _tests(plan)
    assert tests
    for step in tests:
        assert step.detector in manifest.BY_KIND
        assert step.question
        assert step.evidence_ids


def test_selection_and_question_are_deterministic():
    first = build_investigation(_findings())
    second = build_investigation(_findings())
    assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())
    assert [step.detector for step in _tests(first)] == [
        step.detector for step in _tests(second)
    ]


def test_question_uses_the_anchor_kind_and_severity():
    plan = build_investigation(
        [
            Finding("medium", "id_like", "order_id", "identifier column"),
            Finding(
                "high", "preprocess_outside_cv", "src/train.py", "fit before split"
            ),
        ]
    )
    questions = {step.detector: step.question for step in _tests(plan)}
    assert "on feature 'dataset:order_id'" in questions["id_like"]
    assert "medium" in questions["id_like"]
    assert "in file 'file:src/train.py'" in questions["preprocess_outside_cv"]


def test_unknown_kind_is_omitted_with_a_reason_at_runtime():
    plan = build_investigation([Finding("high", "deliberately_unknown_kind", "x", "d")])
    assert _tests(plan) == []
    assert len(plan.omitted_tests) == 1
    omitted = plan.omitted_tests[0]
    assert omitted.kinds == ("deliberately_unknown_kind",)
    assert omitted.reason == "unregistered_detector"
    assert omitted.finding_ids == ("det-0",)
    assert [step.ordinal for step in plan.steps] == list(range(1, len(plan.steps) + 1))


def test_constructing_an_unknown_detector_is_a_validation_error():
    with pytest.raises(ValueError, match="unregistered detector kind"):
        SuggestedTest(
            ordinal=1,
            evidence_ids=("scanner-000000",),
            finding_ids=("det-0",),
            rationale="check",
            detector="deliberately_unknown_kind",
            question="does it still fire?",
        )


def test_payload_marks_the_omitted_test(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--scan", "clean.py", "--intelligence", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["investigation"]["omitted_tests"] == []
