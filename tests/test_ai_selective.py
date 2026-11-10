"""Selective reporting: fit-time decision, insufficiency and honest claims."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from leaklens.ai import artifact, learned
from leaklens.ai.analysis import analyze
from leaklens.ai.corpus import development_cases
from leaklens.ai.schemas import (
    EXCHANGEABILITY_LIMITATION,
    SCHEMA_VERSION,
    SelectionReport,
    sufficiency_label,
)
from leaklens.ai.semantics import CodeSource
from leaklens.report import _ai_section

LEAKY = (
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "X = StandardScaler().fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
)


def _selection(**overrides) -> SelectionReport:
    values = {
        "applied": True,
        "threshold": 0.9,
        "coverage": 0.9,
        "unit_coverage": 0.5,
        "accepted": 90,
        "emitted": 100,
        "examples": 200,
        "selective_risk": 0.01,
        "risk_interval": (0.0, 0.03),
        "unselective_error_rate": 0.06,
        "protocol": (
            "Selection fitted from grouped out-of-fold development probabilities."
        ),
        "limitations": (
            "Measured on the authored development corpus, not an independent one.",
            EXCHANGEABILITY_LIMITATION,
            "Below-threshold hypotheses stay visible as ranking-only.",
        ),
    }
    values.update(overrides)
    return SelectionReport(**values)


def _ai_payload(*, sufficiency: str = "insufficient", selection=None) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine": "leaklens.ai.experimental",
        "scope": "code",
        "target": None,
        "deterministic_findings": 1,
        "deterministic_trust_score": 90,
        "changes_deterministic_findings": False,
        "hypotheses": [
            {
                "hypothesis_id": "hyp-learned-abc",
                "concept": "learned_leakage_pattern",
                "title": "<script>alert('x')</script> learned pattern",
                "explanation": "Evidence bound and unverified.",
                "confidence": 0.62,
                "confidence_basis": "calibrated_probability",
                "sufficiency": sufficiency,
                "severity_hint": "low",
                "location": "model.py:3",
                "detector": "learned",
                "corroborates": None,
                "limitations": ["Statistical hint, not a verified finding."],
                "evidence": [
                    {
                        "evidence_id": "ev-1",
                        "source": "code",
                        "kind": "pattern",
                        "location": "model.py:3",
                        "excerpt": "scaler.fit_transform(X)",
                        "supports": True,
                        "graph_node_id": None,
                    }
                ],
            }
        ],
        "plan": {
            "root_causes": [],
            "priority_order": [],
            "summary": "",
            "verification": ["Re-scan."],
            "enrichment": "deterministic",
            "fallback_reason": None,
            "llm_summary": None,
            "llm_notes": [],
        },
        "models": [],
        "limitations": ["AI hypotheses are experimental."],
    }
    if selection is not None:
        payload["selection"] = dataclasses.asdict(selection)
    return payload


def test_selective_risk_does_not_increase_as_the_threshold_rises():
    probabilities = np.array([0.05, 0.20, 0.35, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    points = [
        learned._selective_risk(labels, probabilities, threshold)
        for threshold in (0.30, 0.60, 0.75, 0.90)
    ]

    accepted = [count for count, _false_positives, _risk in points]
    risks = [risk for _count, _false_positives, risk in points]
    assert accepted == sorted(accepted, reverse=True)
    assert risks == sorted(risks, reverse=True)
    assert risks[0] > 0
    assert risks[-1] == 0


def test_wilson_upper_bound_is_cautious_on_small_samples():
    assert learned._wilson_upper(0, 10) > learned._wilson_upper(0, 1000)
    assert learned._wilson_upper(3, 10) >= 3 / 10
    assert learned._wilson_upper(0, 0) == 1.0


def test_fitted_selection_meets_the_acceptance_criteria():
    _evaluation, model = learned.get_model(compare_candidates=False)
    selection = model.selection

    assert isinstance(selection, SelectionReport)
    assert selection.examples == len(development_cases())
    assert selection.emitted >= selection.accepted > 0
    assert selection.coverage == pytest.approx(selection.accepted / selection.emitted)
    assert selection.unit_coverage == pytest.approx(
        selection.accepted / selection.examples
    )
    assert 0 <= selection.risk_interval[0] <= selection.risk_interval[1] <= 1
    assert EXCHANGEABILITY_LIMITATION in selection.limitations
    if selection.applied:
        assert selection.threshold >= model.threshold
        assert selection.coverage >= 0.5
        # The Wilson rule only accepts a threshold whose upper bound is at or
        # below the unselective error rate, so the point estimate must clear it.
        assert selection.selective_risk <= selection.unselective_error_rate
    else:
        assert selection.threshold == pytest.approx(model.threshold)
        assert selection.coverage == pytest.approx(1.0)


def test_selection_round_trips_through_the_artifact():
    evaluation, model = learned.get_model(compare_candidates=False)
    payload = artifact.build_payload(evaluation, model)

    assert payload["model"]["selection"]["threshold"] == pytest.approx(
        model.selection.threshold
    )
    _loaded_evaluation, loaded = artifact._from_payload(payload)
    assert loaded.selection == model.selection


def test_a_legacy_artifact_without_selection_still_loads():
    evaluation, model = learned.get_model(compare_candidates=False)
    payload = artifact.build_payload(evaluation, model)
    payload["model"].pop("selection")
    payload["artifact_digest"] = artifact._digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )

    _loaded_evaluation, loaded = artifact._from_payload(payload)
    assert loaded.selection is None


def test_a_malformed_selection_is_rejected_not_ignored():
    evaluation, model = learned.get_model(compare_candidates=False)
    payload = artifact.build_payload(evaluation, model)
    payload["model"]["selection"]["risk_interval"] = [0.5]
    payload["artifact_digest"] = artifact._digest(
        {key: value for key, value in payload.items() if key != "artifact_digest"}
    )

    with pytest.raises(artifact.ArtifactError, match="selection"):
        artifact._from_payload(payload)


def test_legacy_report_versions_remain_constructible():
    report = analyze(code=(CodeSource(label="model.py", source=LEAKY),))

    legacy = dataclasses.replace(report, schema_version="1.1")
    assert legacy.to_dict()["schema_version"] == "1.1"

    with pytest.raises(ValueError, match="schema version"):
        dataclasses.replace(report, schema_version="9.9")


def test_insufficient_hypotheses_stay_visible_and_render_as_ranking_only():
    ai = _ai_payload(selection=_selection())

    rendered = _ai_section(ai)

    # Not silent suppression: the title, evidence and location all survive.
    assert "learned pattern" in rendered
    assert "scaler.fit_transform(X)" in rendered
    assert "model.py:3" in rendered
    assert "insufficient calibration data" in rendered
    assert "ranking only" in rendered
    # Every value is escaped, including adversarial AI text.
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_supported_hypotheses_do_not_claim_insufficiency():
    ai = _ai_payload(sufficiency="supported", selection=_selection())

    rendered = _ai_section(ai)

    assert "insufficient calibration data" not in rendered
    assert "supported by calibration data" in rendered


def test_the_selection_card_never_claims_a_guarantee():
    rendered = _ai_section(_ai_payload(selection=_selection()))

    assert "no distribution-free guarantee" in rendered
    assert "guaranteed" not in rendered.lower()
    assert "Selective reporting" in rendered


def test_docs_publish_the_no_guarantee_statement():
    root = Path(__file__).resolve().parents[1]
    for name in ("AI_INTELLIGENCE.md", "AI_PERFORMANCE.md"):
        text = (root / "docs" / name).read_text(encoding="utf-8").lower()
        assert "no distribution-free guarantee" in text, name


def test_a_legacy_report_without_selection_still_renders():
    rendered = _ai_section(_ai_payload(sufficiency="weak"))

    assert "AI hypotheses" in rendered
    assert "Selective reporting" not in rendered


def test_empty_reports_render_without_a_selection_decision():
    assert _ai_section({}) == ""


def test_insufficient_learned_hypotheses_are_downgraded_but_kept(monkeypatch):
    _evaluation, real = learned.get_model(compare_candidates=False)

    class _Model:
        threshold = 0.5
        selection = _selection()

        def predict_proba(self, _source):
            return 0.7

        def local_contributions(self, source, limit=5):
            return real.local_contributions(source, limit)

    monkeypatch.setattr(learned, "get_model", lambda **_kwargs: (None, _Model()))

    hypotheses = learned.learned_hypotheses(
        (CodeSource(label="model.py", source=LEAKY),)
    )

    assert hypotheses
    item = hypotheses[0]
    assert item.sufficiency == "insufficient"
    assert item.severity_hint == "low"
    assert item.evidence
    assert "ranking only" in item.title.lower()
    assert "insufficient" in item.explanation.lower()


def test_supported_learned_hypotheses_keep_their_evidence_and_severity(monkeypatch):
    _evaluation, real = learned.get_model(compare_candidates=False)

    class _Model:
        threshold = 0.5
        selection = _selection(threshold=0.6)

        def predict_proba(self, _source):
            return 0.99

        def local_contributions(self, source, limit=5):
            return real.local_contributions(source, limit)

    monkeypatch.setattr(learned, "get_model", lambda **_kwargs: (None, _Model()))

    item = learned.learned_hypotheses((CodeSource(label="model.py", source=LEAKY),))[0]

    assert item.sufficiency == "supported"
    assert item.severity_hint == "medium"
    assert item.evidence


def test_sufficiency_labels_cover_every_machine_value():
    assert "supported" in sufficiency_label("supported")
    assert "weak" in sufficiency_label("weak")
    assert "insufficient" in sufficiency_label("insufficient")
    assert sufficiency_label("unknown") == sufficiency_label("weak")


def test_selective_reporting_does_not_mutate_the_complete_analysis(monkeypatch):
    from leaklens.leakage import Finding

    finding = Finding(
        "high", "preprocess_before_split", "model.py:3", "fit before split"
    )
    source = (CodeSource(label="model.py", source=LEAKY),)

    first = analyze((finding,), code=source)
    second = analyze((finding,), code=source)

    assert first.to_dict() == second.to_dict()
    assert first.changes_deterministic_findings is False
    assert first.deterministic_findings == 1
    assert finding.severity == "high"
    for hypothesis in first.hypotheses:
        assert hypothesis.evidence


def test_cli_prints_the_selection_decision(tmp_path, capsys):
    from leaklens import cli

    path = tmp_path / "model.py"
    path.write_text(LEAKY, encoding="utf-8")

    assert cli.main(["--scan", str(path), "--ai", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    selection = payload["ai"]["selection"]

    assert payload["ai"]["schema_version"] == SCHEMA_VERSION
    assert selection["applied"] in {True, False}
    assert "no distribution-free guarantee" in selection["limitations"][1].lower()

    assert cli.main(["--scan", str(path), "--ai"]) == 1
    out = capsys.readouterr().out
    assert "AI selective reporting:" in out
    assert "no distribution-free guarantee" in out
    assert "AI sufficiency:" in out


def test_development_selection_is_computed_from_grouped_out_of_fold_scores():
    cases = development_cases()
    _evaluation, model = learned.get_model(compare_candidates=False)
    probabilities = np.array(
        [model.predict_proba(case.code) for case in cases], dtype=float
    )

    assert len(probabilities) == len(cases) == model.selection.examples
    assert model.selection.threshold >= model.threshold
