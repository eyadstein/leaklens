import json

from leaklens.cli import main
from leaklens.leakage import Finding
from leaklens.sarif import to_sarif

BAD = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
)


def test_sarif_structure(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text(BAD, encoding="utf-8")
    out = tmp_path / "o.sarif"
    assert main(["--scan", str(src), "--sarif", str(out)]) == 1
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["version"] == "2.1.0"
    run = d["runs"][0]
    ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    res = run["results"][0]
    assert res["ruleId"] in ids and res["level"] == "error"
    loc = res["locations"][0]["physicalLocation"]
    assert (
        isinstance(loc["region"]["startLine"], int)
        and "\\" not in loc["artifactLocation"]["uri"]
    )


def test_sarif_keeps_the_existing_shape_and_adds_fingerprint_and_evidence():
    finding = Finding("medium", "id_like", "uid", "98% unique values")
    doc = to_sarif([finding], "dataset.csv")
    assert doc["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
    assert doc["version"] == "2.1.0"
    run = doc["runs"][0]
    assert set(run) == {"tool", "results"}
    result = run["results"][0]
    assert set(result) == {
        "ruleId",
        "level",
        "message",
        "locations",
        "partialFingerprints",
        "properties",
    }
    assert result["properties"] == {"leaklens": {"evidence": []}}
    assert result["partialFingerprints"] == {"leaklensFinding/v1": finding.fingerprint}
    physical = result["locations"][0]["physicalLocation"]
    assert physical["artifactLocation"]["uri"] == "dataset.csv"
    assert physical["region"] == {"startLine": 1}


def test_sarif_handles_empty_and_unusual_findings():
    empty = to_sarif([])
    assert empty["version"] == "2.1.0"
    assert empty["runs"][0]["tool"]["driver"]["rules"] == []
    assert empty["runs"][0]["results"] == []

    odd = Finding("unexpected", "third_party_kind", "", "")
    result = to_sarif([odd])["runs"][0]["results"][0]
    assert result["ruleId"] == "third_party_kind"
    assert result["level"] == "note"
    assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "."
    assert result["properties"]["leaklens"]["evidence"] == []
