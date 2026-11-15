"""Phase 18 M7: every governance field is provenance-labelled and explicit.

The pack must never claim compliance, must name what it does not know, and the
JSON and Markdown exports must describe the same payload.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from leaklens import assemble, governance
from leaklens.cli import main
from leaklens.governance import cards, export
from leaklens.leakage import Finding
from leaklens.paths import PathOutsideRoot

PACKAGE = Path(__file__).resolve().parents[1] / "leaklens" / "governance"


def _findings() -> list[Finding]:
    findings = [
        Finding("high", "target_leakage", "amount", "copied target"),
        Finding("medium", "split_overlap", "*", "shared rows"),
        Finding("low", "id_like_feature", "row_id", "identifier column"),
        Finding("low", "monitoring_drift_report", "amount", "drift", diagnostic=True),
    ]
    return findings


def _payload(*, plan: bool = True, gaps: bool = False) -> dict:
    sections = assemble.build(_findings())
    payload = {
        "trust_score": sections.trust_score,
        "findings": sections.findings,
    }
    if plan:
        payload["investigation"] = {
            "engine": "leaklens.intelligence.investigation",
            "narration": "unavailable",
            "finding_count": 4,
            "steps": [{"step_type": "inspect_evidence"}],
        }
    if gaps:
        payload["missing_evidence"] = {"entries": [], "counts": {}, "total": 2}
    return payload


def test_every_field_is_labelled_and_sourced():
    payload = governance.build_card(
        _payload(), train="train.csv", test=None, target="label"
    ).to_dict()
    assert payload["fields"]
    for field in payload["fields"]:
        assert field["label"] in governance.LABELS
        assert field["source"].strip()
        if field["label"] == governance.UNKNOWN:
            assert field["value"] is None
        else:
            assert field["value"] is not None


def test_unknowns_are_explicit_and_listed():
    payload = governance.build_card(
        _payload(), train="train.csv", target="label"
    ).to_dict()
    unknown = [f["field"] for f in payload["fields"] if f["label"] == "unknown"]
    assert payload["unknowns"] == unknown
    assert "dataset.test" in unknown
    assert "model.name" in unknown
    assert "data.license" in unknown
    assert all(
        f["source"].strip()
        for f in payload["fields"]
        if f["label"] == governance.UNKNOWN
    )


def test_measured_fields_count_payload_findings_like_the_assembler():
    sections = assemble.build(_findings())
    card = governance.build_card(
        {"trust_score": sections.trust_score, "findings": sections.findings},
        train="train.csv",
        target="label",
    ).to_dict()
    fields = {item["field"]: item for item in card["fields"]}
    counts = assemble.severity_counts(_findings())
    for severity in ("high", "medium", "low", "total"):
        assert fields[f"audit.findings.{severity}"]["value"] == counts[severity]
        assert fields[f"audit.findings.{severity}"]["label"] == "measured"
    assert fields["audit.trust_score"]["value"] == sections.trust_score


def test_absent_optional_sections_become_unknown_not_zero():
    card = governance.build_card(
        _payload(plan=False, gaps=False), train="train.csv", target="label"
    ).to_dict()
    fields = {item["field"]: item for item in card["fields"]}
    for name in (
        "investigation.steps",
        "investigation.narration",
        "audit.missing_evidence",
    ):
        assert fields[name]["label"] == "unknown"
        assert fields[name]["value"] is None


def test_measured_optional_sections_are_reported():
    card = governance.build_card(
        _payload(gaps=True), train="train.csv", target="label"
    ).to_dict()
    fields = {item["field"]: item for item in card["fields"]}
    assert fields["investigation.steps"]["value"] == 1
    assert fields["investigation.narration"]["value"] == "unavailable"
    assert fields["audit.missing_evidence"]["value"] == 2


def test_a_plan_without_a_narration_state_stays_unknown():
    payload = _payload()
    payload["investigation"] = {"steps": []}
    fields = {
        item["field"]: item
        for item in governance.build_card(payload, train="a.csv").to_dict()["fields"]
    }
    assert fields["investigation.steps"]["value"] == 0
    assert fields["investigation.narration"]["label"] == "unknown"
    assert fields["investigation.narration"]["value"] is None


def test_a_payload_without_trust_score_is_rejected():
    with pytest.raises(governance.GovernanceError, match="trust_score"):
        governance.build_card({"findings": []})


def test_a_payload_without_findings_is_rejected():
    with pytest.raises(governance.GovernanceError, match="findings"):
        governance.build_card({"trust_score": 100})


def test_unknown_field_cannot_carry_a_value():
    with pytest.raises(ValueError, match="unknown field"):
        cards.Field("model.name", cards.UNKNOWN, "not supplied", "gpt")


def test_measured_field_needs_a_value():
    with pytest.raises(ValueError, match="needs a value"):
        cards.Field("audit.trust_score", cards.MEASURED, "leakage.score()")


def test_labels_and_sources_are_validated():
    with pytest.raises(ValueError, match="provenance label"):
        cards.Field("audit.trust_score", "guessed", "somewhere", 100)
    with pytest.raises(ValueError, match="source"):
        cards.Field("audit.trust_score", cards.MEASURED, "  ", 100)
    with pytest.raises(ValueError, match="dotted identifiers"):
        cards.Field("audit trust", cards.MEASURED, "leakage.score()", 100)
    with pytest.raises(ValueError, match="unique"):
        governance.GovernanceCard(
            fields=(
                cards.Field("dataset.train", cards.ASSERTED, "cli.dataset", "a.csv"),
                cards.Field("dataset.train", cards.ASSERTED, "cli.dataset", "b.csv"),
            )
        )


def test_json_and_markdown_render_the_same_payload():
    payload = governance.build_card(
        _payload(gaps=True), train="train.csv", test="test.csv", target="label"
    ).to_dict()
    document = json.loads(export.to_json(payload))
    markdown = export.to_markdown(payload)
    assert document == payload
    for field in payload["fields"]:
        assert f"`{field['field']}`" in markdown
        assert f"`{field['label']}`" in markdown
        assert field["source"] in markdown
        if field["value"] is None:
            assert "| - |" in markdown
        else:
            assert str(field["value"]) in markdown
    assert f"## Unknowns ({len(payload['unknowns'])})" in markdown
    assert "## Non-claims" in markdown
    for name in payload["unknowns"]:
        assert f"- `{name}`" in markdown


def test_markdown_cannot_be_split_by_a_pipe_or_newline_in_a_value():
    card = governance.build_card(
        {"trust_score": 100, "findings": []},
        train="train|with\npipe.csv",
        target="label",
    )
    markdown = export.to_markdown(card.to_dict())
    row = next(line for line in markdown.splitlines() if "dataset.train" in line)
    assert row.count("|") == 6  # five column delimiters plus the escaped value pipe
    assert "\\n" in row  # the newline stays inside the JSON string, not the table


def test_no_compliance_claim_in_any_field():
    payload = governance.build_card(
        _payload(), train="train.csv", target="label"
    ).to_dict()
    words = ("certified", "compliant", "compliance", "guarantee", "approved")
    fields = json.dumps(payload["fields"]).lower()
    assert not any(word in fields for word in words)
    assert any("not a certification" in claim for claim in payload["non_claims"])


def test_the_pack_is_deterministic():
    first = governance.build_card(
        _payload(), train="train.csv", target="label"
    ).to_dict()
    second = governance.build_card(
        _payload(), train="train.csv", target="label"
    ).to_dict()
    assert json.dumps(first) == json.dumps(second)


def test_write_json_and_markdown_inside_the_root(tmp_path):
    payload = governance.build_card(
        _payload(), train="train.csv", target="label"
    ).to_dict()
    written = export.write(payload, "pack.json", root=tmp_path)
    assert json.loads(written.read_text(encoding="utf-8")) == payload
    written = export.write(payload, "nested/pack.md", root=tmp_path)
    assert "`dataset.train`" in written.read_text(encoding="utf-8")
    assert not list(tmp_path.rglob(".governance-*"))


def test_write_refuses_an_out_of_root_path_and_an_unknown_suffix(tmp_path):
    payload = governance.build_card(
        _payload(), train="train.csv", target="label"
    ).to_dict()
    with pytest.raises(PathOutsideRoot):
        export.write(payload, "../escape.md", root=tmp_path)
    with pytest.raises(governance.GovernanceError, match="suffix"):
        export.write(payload, "pack.txt", root=tmp_path)


def test_governance_never_imports_the_ai_layer():
    imported: set[str] = set()
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not [name for name in imported if name.split(".")[0] == "ai"]
    assert not [name for name in imported if name.startswith("leaklens.ai")]


def test_governance_loads_with_the_ai_layer_unimportable():
    code = (
        "import sys\n"
        "sys.modules['leaklens.ai'] = None\n"
        "from leaklens import governance\n"
        "card = governance.build_card({'trust_score': 100, 'findings': []}, train='a.csv')\n"
        "assert card.to_dict()['fields']\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_cli_adds_the_pack_without_changing_the_audit(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--scan", "clean.py", "--json"]) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert "governance" not in baseline
    assert main(["--scan", "clean.py", "--governance", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    pack = payload.pop("governance")
    assert pack["engine"] == "leaklens.governance"
    labels = {item["label"] for item in pack["fields"]}
    assert labels <= {"asserted", "measured", "unknown"}
    assert "model.name" in pack["unknowns"]
    assert "dataset.train" in pack["unknowns"]  # a code scan supplies no dataset
    assert payload == baseline


def test_cli_labels_the_declared_dataset_as_asserted(tmp_path, monkeypatch, capsys):
    (tmp_path / "train.csv").write_text(
        "x,label\n1,0\n2,1\n3,0\n4,1\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert main(["train.csv", "--target", "label", "--governance", "--json"]) in (0, 1)
    pack = json.loads(capsys.readouterr().out)["governance"]
    fields = {item["field"]: item for item in pack["fields"]}
    assert fields["dataset.train"]["label"] == "asserted"
    assert fields["dataset.train"]["value"] == "train.csv"
    assert fields["dataset.target"]["value"] == "label"
    assert fields["dataset.test"]["label"] == "unknown"


def test_cli_writes_both_formats_from_one_payload(tmp_path, monkeypatch, capsys):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "--scan",
                "clean.py",
                "--governance",
                "--governance-out",
                "pack.json",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads((tmp_path / "pack.json").read_text(encoding="utf-8"))
    assert written == payload["governance"]
    assert main(["--scan", "clean.py", "--governance-out", "pack.md", "--json"]) == 0
    capsys.readouterr()
    assert "`audit.trust_score`" in (tmp_path / "pack.md").read_text(encoding="utf-8")


def test_cli_refuses_a_governance_output_outside_the_working_directory(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["--scan", "clean.py", "--governance-out", "../escape.md"])
    assert excinfo.value.code == 2
    capsys.readouterr()
