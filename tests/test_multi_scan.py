from leaklens.cli import main

BAD = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
)


def test_scan_multiple_files(tmp_path):
    g = tmp_path / "g.py"
    g.write_text("x = 1\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text(BAD, encoding="utf-8")
    assert main(["--scan", str(g)]) == 0
    assert main(["--scan", str(g), str(b)]) == 1
