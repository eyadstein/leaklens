from dataclasses import dataclass

from leaklens import cli


@dataclass
class Finding:
    severity: str
    kind: str = "example"
    column: str = "feature"
    detail: str = "example finding"

    def to_dict(self):
        return {
            "severity": self.severity,
            "kind": self.kind,
            "column": self.column,
            "detail": self.detail,
        }


def _run_scan(monkeypatch, capsys, severity="high"):
    monkeypatch.setattr(cli, "scan_path", lambda _path: [Finding(severity)])
    return cli.main(["--scan", "example.py"]), capsys.readouterr().out


def test_severity_labels_use_color_for_tty(monkeypatch, capsys):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    _, output = _run_scan(monkeypatch, capsys, "high")
    assert "\033[31m[HIGH]\033[0m" in output


def test_severity_colors_follow_level(monkeypatch, capsys):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    for severity, code in (("high", "31"), ("medium", "33"), ("low", "32")):
        assert cli._severity_label(severity) == f"\033[{code}m[{severity.upper()}]\033[0m"


def test_no_color_environment_disables_color(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    _, output = _run_scan(monkeypatch, capsys)
    assert "[HIGH]" in output
    assert "\033[" not in output


def test_piped_output_disables_color(monkeypatch, capsys):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    _, output = _run_scan(monkeypatch, capsys, "medium")
    assert "[MEDIUM]" in output
    assert "\033[" not in output


def test_empty_no_color_is_ignored(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli._severity_label("high") == "\033[31m[HIGH]\033[0m"
