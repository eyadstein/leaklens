import json

from leaklens.cli import main


def test_badge(tmp_path):
    f = tmp_path / "g.py"
    f.write_text("x = 1\n", encoding="utf-8")
    b = tmp_path / "b.json"
    assert main(["--scan", str(f), "--badge", str(b)]) == 0
    d = json.loads(b.read_text(encoding="utf-8"))
    assert d["message"] == "100/100" and d["color"] == "brightgreen"


def test_fail_under(tmp_path):
    f = tmp_path / "leak.py"
    f.write_text(
        "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nscaler.fit(X_test)\n",
        encoding="utf-8",
    )
    assert main(["--scan", str(f), "--fail-under", "90"]) == 1
    assert main(["--scan", str(f), "--fail-under", "50"]) == 1
