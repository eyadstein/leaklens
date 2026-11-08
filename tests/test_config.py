"""Strict .leaklens.toml parsing, path policy, precedence and init safety."""

from pathlib import Path

import pytest

from leaklens.cli import main
from leaklens.config import CONFIG_NAME, ConfigError, load_config
from leaklens.missingness import DEFAULT_SETTINGS


def _project(tmp_path: Path) -> Path:
    (tmp_path / "data.csv").write_text("y\n1\n0\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    return tmp_path


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / CONFIG_NAME
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_config_returns_defaults(tmp_path):
    config = load_config(cwd=tmp_path)
    assert config.path is None
    assert config.missingness == DEFAULT_SETTINGS
    assert config.suppressions == ()


def test_valid_config_is_config_relative(tmp_path):
    _project(tmp_path)
    config = load_config(
        _write(
            tmp_path,
            'target = "y"\ntest = "data.csv"\nscan = ["src"]\n'
            "threshold = 0.9\nfail_under = 80\n"
            "[baseline]\npath = 'base.json'\nfail_on_new = true\n"
            "[missingness]\nmin_rows = 100\nmin_missing = 10\n",
        )
    )
    assert config.target == "y"
    assert config.test == (tmp_path / "data.csv").resolve()
    assert config.scan == ((tmp_path / "src").resolve(),)
    assert config.threshold == 0.9
    assert config.fail_under == 80.0
    assert config.baseline_path == (tmp_path / "base.json").resolve()
    assert config.fail_on_new is True
    assert config.missingness.min_rows == 100
    assert config.missingness.min_missing == 10
    assert config.missingness.min_group_delta == DEFAULT_SETTINGS.min_group_delta


def test_config_is_discovered_in_the_working_directory(tmp_path, monkeypatch):
    _project(tmp_path)
    _write(tmp_path, 'target = "y"\n')
    monkeypatch.chdir(tmp_path)
    assert load_config().target == "y"


@pytest.mark.parametrize(
    "text, message",
    [
        ("unknown = 1\n", "unknown configuration keys: unknown"),
        ("target = 5\n", "target must be a string"),
        ('target = ""\n', "target must not be empty"),
        ("threshold = true\n", "threshold must be a number"),
        ("threshold = 2\n", "threshold must be between"),
        ("fail_under = -1\n", "fail_under must be between"),
        ('explain = "yes"\n', "explain must be true or false"),
        ("scan = []\n", "scan must be a non-empty"),
        ('scan = ["missing"]\n', "scan path not found"),
        ('test = "missing.csv"\n', "test file not found"),
        ("baseline = 1\n", "baseline must be a table"),
        ("[baseline]\nwat = 1\n", "unsupported baseline keys: wat"),
        ("[baseline]\nfail_on_new = true\n", "needs baseline.path"),
        ("[missingness]\nmin_rows = 5\n", "min_rows must be >= 10"),
        ("[missingness]\nmin_correlation = 2\n", "must be between"),
        ("[missingness]\nwat = 1\n", "unsupported missingness keys: wat"),
        ("[missingness]\nenabled = 1\n", "enabled must be true or false"),
        (
            "[[suppressions]]\nfingerprint = 'abc'\nreason = 'x'\n",
            "64 lowercase hexadecimal",
        ),
    ],
)
def test_invalid_configuration_fails_closed(tmp_path, text, message):
    _project(tmp_path)
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, text))


def test_graph_config_is_opt_in_and_project_relative(tmp_path):
    _project(tmp_path)
    baseline = tmp_path / "base.json"
    baseline.write_text("{}", encoding="utf-8")
    config = load_config(
        _write(
            tmp_path,
            "[graph]\nenabled = true\nbaseline = 'base.json'\n"
            "snapshot = 'current.json'\nupdate_snapshot = true\n"
            "fail_on_new = true\n",
        )
    )

    assert config.graph_enabled is True
    assert config.graph_baseline_path == baseline.resolve()
    assert config.graph_snapshot_path == (tmp_path / "current.json").resolve()
    assert config.graph_update_snapshot is True
    assert config.graph_fail_on_new is True


@pytest.mark.parametrize(
    "text, message",
    [
        ("graph = 1\n", "graph must be a table"),
        ("[graph]\nwat = 1\n", "unsupported graph keys: wat"),
        ("[graph]\nsnapshot = 'base.json'\n", "graph.enabled must be true"),
        ("[graph]\nenabled = true\nupdate_snapshot = true\n", "needs graph.snapshot"),
        ("[graph]\nenabled = true\nfail_on_new = true\n", "needs graph.baseline"),
        ("[graph]\nenabled = true\nbaseline = 'missing.json'\n", "not found"),
    ],
)
def test_invalid_graph_configuration_fails_closed(tmp_path, text, message):
    _project(tmp_path)
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, text))


def test_output_paths_must_stay_inside_the_config_directory(tmp_path):
    _project(tmp_path)
    absolute = (tmp_path.parent / "outside.html").as_posix()
    with pytest.raises(ConfigError, match="relative to the configuration directory"):
        load_config(_write(tmp_path, f"html = '{absolute}'\n"))
    with pytest.raises(ConfigError, match="must stay inside the project directory"):
        load_config(_write(tmp_path, 'sarif = "../report.sarif"\n'))


def test_output_path_parent_must_exist(tmp_path):
    _project(tmp_path)
    with pytest.raises(ConfigError, match="parent directory does not exist"):
        load_config(_write(tmp_path, 'badge = "nested/badge.json"\n'))


def test_explicit_config_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "custom.toml")


def test_config_directory_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="is a directory"):
        load_config(tmp_path)


def test_invalid_toml_reports_the_file(tmp_path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(_write(tmp_path, "target = [\n"))


def test_cli_flags_override_config_values(tmp_path, monkeypatch):
    _project(tmp_path)
    src = tmp_path / "src" / "clean.py"
    src.write_text("x = 1\n", encoding="utf-8")
    _write(tmp_path, 'target = "wrong"\nfail_under = 99\nexplain = true\n')
    monkeypatch.chdir(tmp_path)
    assert main(["--scan", str(src), "--fail-under", "0", "--json"]) == 0


def test_config_supplies_scan_and_output_paths(tmp_path, monkeypatch):
    _project(tmp_path)
    src = tmp_path / "src" / "clean.py"
    src.write_text("x = 1\n", encoding="utf-8")
    _write(tmp_path, 'scan = ["src"]\nhtml = "report.html"\nbadge = "badge.json"\n')
    monkeypatch.chdir(tmp_path)
    assert main(["--json"]) == 0
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "badge.json").is_file()


def test_invalid_config_exits_two(tmp_path, monkeypatch, capsys):
    _project(tmp_path)
    _write(tmp_path, "nope = true\n")
    monkeypatch.chdir(tmp_path)
    assert main(["--scan", "src"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_init_preserves_an_existing_configuration(tmp_path):
    existing = _write(tmp_path, 'target = "y"\n')
    with pytest.raises(SystemExit):
        main(["init", str(tmp_path)])
    assert existing.read_text(encoding="utf-8") == 'target = "y"\n'
