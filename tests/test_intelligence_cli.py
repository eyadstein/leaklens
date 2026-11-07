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


def test_graph_snapshot_round_trip_and_unchanged_comparison(tmp_path, capsys):
    source = tmp_path / "leaky.py"
    source.write_text(LEAKY, encoding="utf-8")
    snapshot = tmp_path / "baseline.json"

    assert main(["--scan", str(source), "--graph-snapshot", str(snapshot)]) == 1
    capsys.readouterr()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert payload["schema"] == "leaklens-evidence-graph"
    assert payload["version"] == 1
    assert payload["complete"] is True
    assert payload["finding_fingerprints"]
    assert payload["evidence"]

    assert main(["--scan", str(source), "--graph-baseline", str(snapshot)]) == 1
    output = capsys.readouterr().out
    assert "Graph baseline:" in output
    assert "0 new," in output
    assert "0 resolved" in output

    assert (
        main(["--scan", str(source), "--graph-baseline", str(snapshot), "--json"]) == 1
    )
    result = json.loads(capsys.readouterr().out)
    counts = result["graph_baseline"]["counts"]
    assert counts["new"] == 0
    assert counts["resolved"] == 0
    assert counts["unchanged"] == len(payload["evidence"])


def test_graph_comparison_reports_new_and_resolved(tmp_path, capsys):
    leaky = tmp_path / "leaky.py"
    leaky.write_text(LEAKY, encoding="utf-8")
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    snapshot = tmp_path / "baseline.json"

    assert main(["--scan", str(leaky), "--graph-snapshot", str(snapshot)]) == 1
    capsys.readouterr()

    # Same code again: nothing new.
    assert main(["--scan", str(leaky), "--graph-baseline", str(snapshot)]) == 1
    assert "0 new," in capsys.readouterr().out

    # Baseline had evidence; current has none -> resolved, not new.
    assert main(["--scan", str(clean), "--graph-baseline", str(snapshot)]) == 0
    output = capsys.readouterr().out
    assert "0 new," in output
    assert "resolved" in output

    # Reverse direction: current evidence is new against an empty baseline.
    empty = tmp_path / "empty.json"
    assert main(["--scan", str(clean), "--graph-snapshot", str(empty)]) == 0
    capsys.readouterr()
    assert main(["--scan", str(leaky), "--graph-baseline", str(empty)]) == 1
    assert "new," in capsys.readouterr().out


def test_graph_baseline_incomplete_scan_is_uncertain(tmp_path, capsys):
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    snapshot = tmp_path / "baseline.json"

    assert main(["--scan", str(broken), "--graph-snapshot", str(snapshot)]) == 0
    capsys.readouterr()
    assert json.loads(snapshot.read_text(encoding="utf-8"))["complete"] is False

    assert (
        main(["--scan", str(clean), "--graph-baseline", str(snapshot), "--json"]) == 0
    )
    result = json.loads(capsys.readouterr().out)["graph_baseline"]
    assert result["counts"]["resolved"] == 0
    assert result["counts"]["uncertain"] == 1
    assert result["resolution_complete"] is False
    assert any("incomplete" in warning for warning in result["warnings"])


def test_graph_snapshot_refuses_overwrite_without_explicit_update(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")
    snapshot = tmp_path / "baseline.json"

    assert main(["--scan", str(source), "--graph-snapshot", str(snapshot)]) == 0
    capsys.readouterr()

    assert main(["--scan", str(source), "--graph-snapshot", str(snapshot)]) == 2
    assert "already exists" in capsys.readouterr().err

    assert (
        main(
            [
                "--scan",
                str(source),
                "--graph-snapshot",
                str(snapshot),
                "--update-graph-snapshot",
            ]
        )
        == 0
    )


def test_graph_baseline_errors_exit_two(tmp_path, capsys):
    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")

    assert (
        main(
            [
                "--scan",
                str(source),
                "--graph-baseline",
                str(tmp_path / "missing.json"),
            ]
        )
        == 2
    )
    assert "not found" in capsys.readouterr().err

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert main(["--scan", str(source), "--graph-baseline", str(corrupt)]) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_graph_gate_is_opt_in_and_only_fails_on_new(tmp_path, capsys):
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    clean = tmp_path / "clean.py"
    clean.write_text("value = 1\n", encoding="utf-8")
    empty = tmp_path / "empty.json"
    assert main(["--scan", str(clean), "--graph-snapshot", str(empty)]) == 0
    capsys.readouterr()

    # Low-severity parse error: the default gate stays at 0.
    assert main(["--scan", str(broken)]) == 0
    assert main(["--scan", str(broken), "--graph-baseline", str(empty)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "--scan",
                str(broken),
                "--graph-baseline",
                str(empty),
                "--fail-on-graph-new",
            ]
        )
        == 1
    )
    capsys.readouterr()

    # A baseline built from the same file has no new evidence.
    same = tmp_path / "same.json"
    assert main(["--scan", str(broken), "--graph-snapshot", str(same)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "--scan",
                str(broken),
                "--graph-baseline",
                str(same),
                "--fail-on-graph-new",
            ]
        )
        == 0
    )


def test_graph_flags_do_not_change_scores_findings_or_exit_codes(tmp_path, capsys):
    source = tmp_path / "leaky.py"
    source.write_text(LEAKY, encoding="utf-8")
    snapshot = tmp_path / "baseline.json"
    assert main(["--scan", str(source), "--graph-snapshot", str(snapshot)]) == 1
    capsys.readouterr()

    with_graph = main(
        ["--scan", str(source), "--graph-baseline", str(snapshot), "--json"]
    )
    compared = json.loads(capsys.readouterr().out)
    plain = main(["--scan", str(source), "--json"])
    baseline_less = json.loads(capsys.readouterr().out)

    assert with_graph == plain == 1
    assert compared["trust_score"] == baseline_less["trust_score"]
    assert compared["findings"] == baseline_less["findings"]
    assert "graph_baseline" in compared
    assert "graph_baseline" not in baseline_less


def test_graph_baseline_appears_in_html_report(tmp_path, capsys):
    source = tmp_path / "leaky.py"
    source.write_text(LEAKY, encoding="utf-8")
    snapshot = tmp_path / "baseline.json"
    report = tmp_path / "report.html"
    assert main(["--scan", str(source), "--graph-snapshot", str(snapshot)]) == 1
    capsys.readouterr()

    assert (
        main(
            [
                "--scan",
                str(source),
                "--graph-baseline",
                str(snapshot),
                "--html",
                str(report),
            ]
        )
        == 1
    )
    content = report.read_text(encoding="utf-8")
    assert "Evidence-graph baseline comparison" in content
    assert "not proof of leakage" in content


def test_graph_flag_validation_requires_companions(tmp_path, capsys):
    import pytest

    source = tmp_path / "clean.py"
    source.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["--scan", str(source), "--update-graph-snapshot"])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        main(["--scan", str(source), "--fail-on-graph-new"])
    assert exc.value.code == 2


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
