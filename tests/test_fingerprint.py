"""Stable finding fingerprints: canonicalization, versioning and output surfaces."""

import json
import os
import subprocess
import sys
from pathlib import Path

from leaklens.fingerprint import (
    SCHEMA_VERSION,
    canonical_form,
    fingerprint_for,
    normalize_path,
)
from leaklens.leakage import Finding
from leaklens.sarif import to_sarif

ROOT = Path(__file__).resolve().parents[1]


def _finding(**overrides):
    values = {
        "severity": "medium",
        "kind": "no_random_state",
        "column": "src/train.py:10",
        "detail": "train_test_split() has no random_state",
    }
    values.update(overrides)
    return Finding(**values)


def test_fingerprint_is_deterministic_hex_and_versioned():
    finding = _finding()
    assert SCHEMA_VERSION == 1
    assert finding.fingerprint == fingerprint_for(finding)
    assert len(finding.fingerprint) == 64
    assert set(finding.fingerprint) <= set("0123456789abcdef")
    assert canonical_form(finding).startswith("schema=1\n")


def test_line_movement_keeps_identity():
    first = _finding(column="src/train.py:10")
    second = _finding(column="src/train.py:987")
    assert first.fingerprint == second.fingerprint


def test_notebook_cell_and_line_are_ignored():
    first = _finding(column="notebooks/run.ipynb[cell 3]:10")
    second = _finding(column="notebooks/run.ipynb[cell 7]:99")
    assert first.fingerprint == second.fingerprint


def test_kind_and_feature_and_path_change_identity():
    base = _finding()
    assert base.fingerprint != _finding(kind="temporal_split_shuffle").fingerprint
    assert base.fingerprint != _finding(column="src/test.py:10").fingerprint
    assert base.fingerprint != _finding(column="feature_a").fingerprint
    assert base.fingerprint != _finding(column="feature_b").fingerprint


def test_volatile_detail_does_not_change_identity():
    first = _finding(detail="train ends at 2024-01-01, 3 rows overlap")
    second = _finding(detail="train ends at 2026-11-07, 999 rows overlap")
    assert first.fingerprint == second.fingerprint


def test_windows_and_posix_paths_normalize_to_the_same_relative_form():
    assert normalize_path(r"C:\repo\src\train.py", root=r"C:\repo") == "src/train.py"
    assert normalize_path("/repo/src/train.py", root="/repo") == "src/train.py"
    windows = _finding(column=r"C:\repo\src\train.py:10")
    posix = _finding(column="src/train.py:10")
    assert fingerprint_for(windows, root=r"C:\repo") == fingerprint_for(posix)


def test_file_move_changes_identity():
    before = _finding(column="src/old.py:10")
    after = _finding(column="src/new.py:10")
    assert before.fingerprint != after.fingerprint


def test_optional_stable_key_is_supported():
    base = _finding()
    keyed = _finding(stable_key="scaler-before-split")
    other = _finding(stable_key="split-without-seed")
    assert base.fingerprint != keyed.fingerprint
    assert keyed.fingerprint != other.fingerprint


def test_fingerprint_is_exposed_in_json_and_sarif():
    finding = _finding()
    payload = finding.to_dict()
    assert payload["fingerprint"] == finding.fingerprint
    result = to_sarif([finding])["runs"][0]["results"][0]
    assert result["partialFingerprints"] == {"leaklensFinding/v1": finding.fingerprint}


def test_fingerprint_is_same_across_working_directories_and_processes(tmp_path):
    code = (
        "from leaklens.leakage import Finding;"
        "f=Finding('medium','no_random_state','src/train.py:10','x');"
        "print(f.fingerprint)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    first = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    second = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert first == second == _finding().fingerprint


def test_json_round_trip_keeps_the_fingerprint():
    finding = _finding()
    payload = json.loads(json.dumps(finding.to_dict()))
    assert payload["fingerprint"] == finding.fingerprint
