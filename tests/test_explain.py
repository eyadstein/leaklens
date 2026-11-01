from leaklens.cli import main
from leaklens.fixes import advice

BAD = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
)


def test_explain_prints_fix(tmp_path, capsys):
    f = tmp_path / "b.py"
    f.write_text(BAD, encoding="utf-8")
    assert main(["--scan", str(f), "--explain"]) == 1
    assert "fix:" in capsys.readouterr().out


def test_no_fix_without_flag(tmp_path, capsys):
    f = tmp_path / "b.py"
    f.write_text(BAD, encoding="utf-8")
    main(["--scan", str(f)])
    assert "fix:" not in capsys.readouterr().out


def test_every_kind_has_advice():
    for k in ("preprocess_before_split", "transformer_before_split", "resample_before_split", "fit_on_eval_data",
              "encoder_before_split", "target_leakage", "id_like", "split_overlap", "temporal_leakage",
              "group_overlap", "group_leakage_risk", "get_dummies_before_split"):
        assert advice(k)