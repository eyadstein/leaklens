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


LEAKY = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\n"
    "a = train_test_split(X)\n"
)


def test_intelligence_json_includes_evidence_clusters(tmp_path, capsys):
    source = tmp_path / "leaky.py"
    source.write_text(LEAKY, encoding="utf-8")

    assert main(["--scan", str(source), "--intelligence", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)

    clusters = result["intelligence"]["clusters"]
    assert clusters
    assert clusters[0]["anchor"].endswith("leaky.py")
    assert "preprocess_before_split" in clusters[0]["finding_kinds"]
    assert clusters[0]["evidence_ids"]
    assert result["intelligence"]["links"]


def test_intelligence_summary_prints_cluster_and_next_step(tmp_path, capsys):
    source = tmp_path / "leaky.py"
    source.write_text(LEAKY, encoding="utf-8")

    assert main(["--scan", str(source), "--intelligence"]) == 1
    output = capsys.readouterr().out

    assert "Evidence clusters:" in output
    assert "Cluster [high]" in output
    assert "leaky.py" in output
    assert "Next step:" in output
    assert "Verify:" in output
