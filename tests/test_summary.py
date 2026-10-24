from leaklens.cli import main

BAD = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
)


def test_summary_line_counts_by_severity(tmp_path, capsys):
    f = tmp_path / "b.py"
    f.write_text(BAD, encoding="utf-8")
    main(["--scan", str(f)])
    out = capsys.readouterr().out.splitlines()
    findings = [l for l in out if l.startswith("[")]
    parts = out[-1].split(", ")
    assert findings
    assert all(p.split()[1] in ("high", "medium", "low") for p in parts)
    assert sum(int(p.split()[0]) for p in parts) == len(findings)


def test_no_summary_when_clean(tmp_path, capsys):
    f = tmp_path / "ok.py"
    f.write_text("x = 1\n", encoding="utf-8")
    main(["--scan", str(f)])
    assert capsys.readouterr().out.splitlines()[-1] == "No leakage found."
