import json

from leaklens.cli import main


def test_intelligence_json_is_opt_in_and_structured(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")

    assert main(["--scan", str(source), "--intelligence", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)

    intelligence = result["intelligence"]
    assert intelligence["engine"] == "deterministic"
    assert intelligence["ai_enriched"] is False
    assert intelligence["finding_count"] == 0
    assert "score" in intelligence["confidence"]
    assert "level" in intelligence["confidence"]
    assert intelligence["root_causes"] == []
    assert intelligence["recommendations"] == []


def test_intelligence_terminal_summary_is_opt_in(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")

    assert main(["--scan", str(source), "--intelligence"]) == 0
    output = capsys.readouterr().out

    assert "Intelligence confidence:" in output
