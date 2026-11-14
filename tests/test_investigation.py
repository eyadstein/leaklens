"""Phase 18 M1: the investigation plan is deterministic, grounded and AI-free."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from leaklens.ai.schemas import ReasoningPlan as AiReasoningPlan
from leaklens.cli import main
from leaklens.evidence import Evidence
from leaklens.intelligence import (
    InspectEvidence,
    MissingEvidence,
    NarrationSlot,
    SuggestedTest,
    build_evidence_graph,
    build_investigation,
)
from leaklens.intelligence.investigation import _ground
from leaklens.intelligence.schemas import ReasoningPlan
from leaklens.leakage import Finding


def _findings(count: int = 3) -> list[Finding]:
    return [
        Finding("high", "target_leakage", f"feature_{index}", "copied target")
        for index in range(count)
    ]


def test_plan_is_deterministic_and_ordered():
    findings = _findings()
    first = build_investigation(findings)
    second = build_investigation(findings)
    assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())
    assert [step.ordinal for step in first.steps] == list(
        range(1, len(first.steps) + 1)
    )
    assert first.narration == "unavailable"
    assert first.engine == "leaklens.intelligence.investigation"


def test_every_step_cites_existing_ids():
    findings = _findings()
    graph = build_evidence_graph(findings)
    plan = build_investigation(findings, graph=graph)
    known = set(graph.evidence_by_id)
    for step in plan.steps:
        assert step.evidence_ids
        assert set(step.evidence_ids) <= known
        assert all(
            finding_id.startswith("det-")
            and 0 <= int(finding_id.removeprefix("det-")) < len(findings)
            for finding_id in step.finding_ids
        )


def test_each_cluster_gets_inspect_test_and_narration_steps():
    findings = _findings()
    plan = build_investigation(findings)
    types = [type(step) for step in plan.steps]
    assert types.count(InspectEvidence) == 3
    assert types.count(SuggestedTest) == 3
    assert types.count(NarrationSlot) == 3
    assert types.count(MissingEvidence) == 3


def test_detector_evidence_suppresses_the_missing_evidence_step():
    findings = [
        Finding(
            "high",
            "split_overlap",
            "*",
            "shared",
            evidence=(
                Evidence(source="split", kind="overlap", description="shared row"),
            ),
        )
    ]
    plan = build_investigation(findings)
    assert MissingEvidence not in {type(step) for step in plan.steps}
    assert InspectEvidence in {type(step) for step in plan.steps}


def test_empty_findings_produce_an_empty_plan():
    plan = build_investigation([])
    assert plan.steps == ()
    assert plan.finding_count == 0


def test_step_count_is_capped():
    plan = build_investigation(_findings(60))
    assert len(plan.steps) == 200
    assert plan.steps[-1].ordinal == 200


def test_grounding_rejects_unknown_evidence_and_finding_ids():
    findings = tuple(_findings(1))
    bad_evidence = InspectEvidence(
        ordinal=1,
        evidence_ids=("scanner-999999",),
        finding_ids=("det-0",),
        rationale="inspect",
    )
    with pytest.raises(ValueError, match="ungrounded evidence id"):
        _ground([bad_evidence], findings, set())
    bad_finding = InspectEvidence(
        ordinal=1,
        evidence_ids=("scanner-000000",),
        finding_ids=("det-9",),
        rationale="inspect",
    )
    with pytest.raises(ValueError, match="ungrounded finding id"):
        _ground([bad_finding], findings, {"scanner-000000"})


def test_plan_builds_with_the_ai_layer_unimportable():
    code = (
        "import sys\n"
        "sys.modules['leaklens.ai'] = None\n"
        "from leaklens.leakage import Finding\n"
        "from leaklens.intelligence.investigation import build_investigation\n"
        "plan = build_investigation([Finding('high', 'target_leakage', 'x', 'd')])\n"
        "assert plan.steps, plan\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_ai_schemas_still_reexports_the_plan_type():
    assert AiReasoningPlan is ReasoningPlan


def test_cli_surfaces_the_plan_additively(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--scan", "clean.py", "--intelligence", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["investigation"] == {
        "engine": "leaklens.intelligence.investigation",
        "narration": "unavailable",
        "finding_count": 0,
        "steps": [],
        "omitted_tests": [],
    }
