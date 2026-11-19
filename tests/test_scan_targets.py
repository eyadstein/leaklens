"""The scan-target contract: a missing path is an error, an empty one is not.

The CLI, the Python API and the composite Action all rely on the same rule:
``--scan`` must name something that exists. Before this contract, a typo
returned an empty finding list, so the run printed ``100/100`` and exited 0.
"""

import pytest

from leaklens.cli import main
from leaklens.scanner import scan_path, scan_targets

CLEAN = "x = 1\n"

HIGH = """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_test)
"""

MISSING = "scan target not found"


def test_missing_file_is_an_error(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.py"
    assert main(["--scan", str(missing)]) == 2
    out, err = capsys.readouterr()
    assert MISSING in err
    assert str(missing) in err
    assert "No leakage found" not in out
    assert "100/100" not in out


def test_missing_directory_is_an_error(tmp_path, capsys):
    missing = tmp_path / "no-such-folder"
    assert main(["--scan", str(missing)]) == 2
    assert MISSING in capsys.readouterr().err


def test_one_missing_target_fails_the_whole_scan(tmp_path, capsys):
    clean = tmp_path / "clean.py"
    clean.write_text(CLEAN, encoding="utf-8")
    missing = tmp_path / "gone.py"
    assert main(["--scan", str(clean), str(missing)]) == 2
    assert MISSING in capsys.readouterr().err


def test_missing_target_fails_even_with_a_dataset(tmp_path, capsys):
    dataset = tmp_path / "train.csv"
    dataset.write_text("a,label\n1,0\n2,1\n", encoding="utf-8")
    missing = tmp_path / "gone.py"
    assert main([str(dataset), "--target", "label", "--scan", str(missing)]) == 2
    assert MISSING in capsys.readouterr().err


def test_existing_file_still_scans(tmp_path, capsys):
    clean = tmp_path / "clean.py"
    clean.write_text(CLEAN, encoding="utf-8")
    assert main(["--scan", str(clean)]) == 0
    assert "100/100" in capsys.readouterr().out


def test_directory_with_supported_inputs_still_scans(tmp_path):
    (tmp_path / "leaky.py").write_text(HIGH, encoding="utf-8")
    assert main(["--scan", str(tmp_path)]) == 1


def test_empty_directory_is_a_valid_empty_scan(tmp_path, capsys):
    assert main(["--scan", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "100/100" in out
    assert "No leakage found" in out


def test_directory_with_only_unsupported_files_is_a_valid_empty_scan(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("not source\n", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    assert main(["--scan", str(tmp_path)]) == 0
    assert "No leakage found" in capsys.readouterr().out


def test_clean_scan_without_findings_exits_zero(tmp_path, capsys):
    clean = tmp_path / "clean.py"
    clean.write_text(CLEAN, encoding="utf-8")
    assert main(["--scan", str(clean)]) == 0
    assert "No leakage found" in capsys.readouterr().out


def test_finding_producing_scan_still_exits_one(tmp_path):
    leaky = tmp_path / "leaky.py"
    leaky.write_text(HIGH, encoding="utf-8")
    assert main(["--scan", str(leaky)]) == 1


def test_python_api_rejects_a_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError, match=MISSING):
        scan_path(tmp_path / "missing.py")
    with pytest.raises(FileNotFoundError, match=MISSING):
        scan_targets(tmp_path / "missing.py")


def test_python_api_treats_an_empty_directory_as_an_empty_scan(tmp_path):
    assert scan_targets(tmp_path) == []
    assert scan_path(tmp_path) == []
