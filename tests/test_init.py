from pathlib import Path

import pytest

from leaklens.cli import PRE_COMMIT_SNIPPET, main


def _flag_text(text):
    return text.split("# Project policy", 1)[0]


def _keys(text):
    return [
        l[2:].split(" = ")[0]
        for l in _flag_text(text).splitlines()
        if l.startswith("# ") and " = " in l
    ]


def _policy_toml(text):
    block = text.split("# Project policy", 1)[1]
    lines = []
    for line in block.splitlines():
        if not line.startswith("# "):
            continue
        value = line[2:]
        if value.startswith(("[", "[[")) or " = " in value:
            lines.append(value)
    return "\n".join(lines)


def test_init_writes_starter_config(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    cfg = tmp_path / ".leaklens.toml"
    assert _keys(cfg.read_text(encoding="utf-8"))
    out = capsys.readouterr().out
    assert str(cfg) in out
    assert PRE_COMMIT_SNIPPET in out


def test_starter_keys_are_real_flags(tmp_path, capsys):
    main(["init", str(tmp_path)])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["--help"])
    help_text = capsys.readouterr().out
    for key in _keys((tmp_path / ".leaklens.toml").read_text(encoding="utf-8")):
        assert "--" + key.replace("_", "-") in help_text


def test_starter_config_is_valid_toml(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    main(["init", str(tmp_path)])
    text = (tmp_path / ".leaklens.toml").read_text(encoding="utf-8")
    assert tomllib.loads(text) == {}
    uncommented = "\n".join(
        l[2:]
        for l in _flag_text(text).splitlines()
        if l.startswith("# ") and " = " in l
    )
    assert len(tomllib.loads(uncommented)) == len(_keys(text))
    policy = tomllib.loads(_policy_toml(text))
    assert set(policy) == {"baseline", "suppressions", "missingness"}
    assert set(policy["baseline"]) == {"path", "fail_on_new"}
    assert set(policy["missingness"]) == {
        "enabled",
        "min_rows",
        "min_group_rows",
        "min_missing",
        "min_rate_delta",
        "min_group_delta",
        "min_correlation",
    }
    assert policy["suppressions"][0]["reason"]


def test_snippet_matches_readme():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert PRE_COMMIT_SNIPPET in readme.read_text(encoding="utf-8")


def test_init_does_not_overwrite(tmp_path):
    cfg = tmp_path / ".leaklens.toml"
    cfg.write_text("target = 'y'\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["init", str(tmp_path)])
    assert cfg.read_text(encoding="utf-8") == "target = 'y'\n"
    assert main(["init", str(tmp_path), "--force"]) == 0
    assert cfg.read_text(encoding="utf-8") != "target = 'y'\n"
