"""Phase 16 CLI surface: `leaklens dl` is opt-in and never changes a gate."""

from __future__ import annotations

import json

import pytest

from leaklens import cli

CLEAN = (
    "import torch\nfrom lightning.pytorch import Trainer\n"
    "torch.manual_seed(0)\ntrainer = Trainer(max_epochs=3)\n"
)
MISSING_SEED = (
    "from lightning.pytorch import Trainer\n\ntrainer = Trainer(max_epochs=3)\n"
)


def _repo(tmp_path, source: str = MISSING_SEED):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "train.py"
    path.write_text(source, encoding="utf-8")
    return tmp_path


def test_a_clean_repo_prints_no_leakage_and_exits_zero(tmp_path, capsys):
    root = _repo(tmp_path, CLEAN)
    assert cli.main(["dl", str(root)]) == 0
    output = capsys.readouterr().out
    assert "No deep-learning leakage found." in output


def test_diagnostics_stay_scored_false_in_json(tmp_path, capsys):
    root = _repo(tmp_path)
    assert cli.main(["dl", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trust_score"] == 100
    assert payload["counts"]["low"] == 1
    assert payload["findings"][0]["kind"] == "no_random_state"
    assert payload["dl"]["status"] == "ok"
    assert payload["dl"]["signals"]


def test_fail_on_findings_is_the_only_nonzero_exit(tmp_path, capsys):
    root = _repo(tmp_path)
    assert cli.main(["dl", str(root), "--fail-on-findings"]) == 1
    capsys.readouterr()
    assert (
        cli.main(["dl", str(_repo(tmp_path / "clean", CLEAN)), "--fail-on-findings"])
        == 0
    )


def test_a_high_artifact_finding_still_exits_zero_by_default(tmp_path, capsys):
    root = _repo(tmp_path)
    (root / "model.pkl").write_bytes(b"\x80\x04payload")
    assert cli.main(["dl", str(root)]) == 0
    output = capsys.readouterr().out
    assert "artifact_unsafe_format" in output
    assert cli.main(["dl", str(root), "--fail-on-findings"]) == 1


def test_missing_yaml_support_is_surfaced_with_an_install_hint(
    tmp_path, capsys, monkeypatch
):
    root = _repo(tmp_path)
    (root / "train_config.yaml").write_text(
        "trainer:\n  max_epochs: 3\n", encoding="utf-8"
    )
    from leaklens.analysis.dl import config

    def unavailable(*args, **kwargs):
        raise config.DlUnavailable("no yaml")

    monkeypatch.setattr(config, "load_config", unavailable)
    assert cli.main(["dl", str(root)]) == 0
    assert "pip install" in capsys.readouterr().err


def test_the_report_never_contains_the_absolute_root(tmp_path, capsys):
    root = _repo(tmp_path)
    cli.main(["dl", str(root), "--json"])
    output = capsys.readouterr().out
    assert str(root) not in output
    assert "\\" not in json.loads(output)["findings"][0]["column"]


def test_the_directory_is_required(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(["dl", str(tmp_path / "missing")])
    assert error.value.code == 2


def test_the_budgets_must_be_positive(tmp_path):
    root = _repo(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["dl", str(root), "--max-files", "0"])
    with pytest.raises(SystemExit):
        cli.main(["dl", str(root), "--max-artifact-mb", "0"])
