import json

from leaklens.cli import main

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
