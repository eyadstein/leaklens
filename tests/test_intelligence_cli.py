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


def test_graph_out_json_implies_intelligence(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")
    graph = tmp_path / "graph.json"

    assert main(["--scan", str(source), "--graph-out", str(graph)]) == 0

    assert json.loads(graph.read_text(encoding="utf-8")) == {
        "evidence": [],
        "clusters": [],
        "links": [],
    }
    assert "Intelligence confidence:" in capsys.readouterr().out


def test_graph_out_infers_mermaid_and_dot_from_extension(tmp_path):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")

    mermaid = tmp_path / "graph.mmd"
    dot = tmp_path / "graph.gv"

    assert main(["--scan", str(source), "--graph-out", str(mermaid)]) == 0
    assert main(["--scan", str(source), "--graph-out", str(dot)]) == 0

    assert mermaid.read_text(encoding="utf-8").startswith("flowchart LR\n")
    assert dot.read_text(encoding="utf-8").startswith("digraph leaklens {\n")


def test_graph_format_overrides_extension(tmp_path):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")
    graph = tmp_path / "graph.txt"

    assert (
        main(
            [
                "--scan",
                str(source),
                "--graph-out",
                str(graph),
                "--graph-format",
                "json",
            ]
        )
        == 0
    )

    assert json.loads(graph.read_text(encoding="utf-8"))["links"] == []


def test_graph_out_rejects_unknown_extension(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")

    code = main(["--scan", str(source), "--graph-out", str(tmp_path / "graph.svg")])

    assert code == 2
    assert "cannot infer graph format" in capsys.readouterr().err


def test_graph_out_rejects_a_missing_directory(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")
    graph = tmp_path / "missing" / "graph.json"

    code = main(["--scan", str(source), "--graph-out", str(graph)])

    assert code == 2
    assert "cannot write evidence graph" in capsys.readouterr().err


def test_graph_out_does_not_change_findings_or_exit_code(tmp_path, capsys):
    source = tmp_path / "leaky.py"
    source.write_text(LEAKY, encoding="utf-8")
    graph = tmp_path / "graph.mmd"
    before = source.read_bytes()

    assert main(["--scan", str(source), "--graph-out", str(graph), "--json"]) == 1
    with_graph = json.loads(capsys.readouterr().out)
    assert main(["--scan", str(source), "--json"]) == 1
    plain = json.loads(capsys.readouterr().out)

    assert with_graph["trust_score"] == plain["trust_score"]
    assert with_graph["findings"] == plain["findings"]
    assert "intelligence" in with_graph
    assert "intelligence" not in plain
    assert source.read_bytes() == before
    assert graph.read_text(encoding="utf-8").startswith("flowchart LR\n")
