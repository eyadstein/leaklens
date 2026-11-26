"""Contract tests for the composite Action and its ``fail-under`` input.

The GitHub Actions runtime itself cannot be booted locally, so this combines
wiring assertions on ``action.yml`` with execution of the real CLI using the
same arguments the composite step builds. It does not simulate the runner.
"""

from pathlib import Path

import pytest

from leaklens.cli import main

ACTION = Path(__file__).resolve().parents[1] / "action.yml"

CLEAN = "x = 1\n"

# One medium and two low findings: trust score 75, no high finding.
MEDIUM = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["city_mean"] = df.groupby("city")["sales"].transform("mean")
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""

HIGH = """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_test)
"""


def _action_args(path, fail_under, baseline=None, fail_on_new=None):
    """Mirror action.yml: append each flag only when its input is set."""
    args = ["--scan", str(path)]
    if fail_under:
        args += ["--fail-under", fail_under]
    if baseline:
        args += ["--baseline", str(baseline)]
    if fail_on_new:
        args += ["--fail-on-new"]
    return args


def test_fail_under_input_is_optional_without_a_default():
    text = ACTION.read_text(encoding="utf-8")
    block = text.split("  fail-under:")[1].split("runs:")[0]
    assert "required: false" in block
    assert "default:" not in block


def test_baseline_and_fail_on_new_inputs_are_optional_without_defaults():
    text = ACTION.read_text(encoding="utf-8")
    for name in ("baseline", "fail-on-new"):
        block = text.split(f"  {name}:")[1].split("runs:")[0]
        assert "required: false" in block, name
        assert "default:" not in block, name


def test_action_wires_inputs_into_the_cli():
    text = ACTION.read_text(encoding="utf-8")
    assert "INPUT_PATH: ${{ inputs.path }}" in text
    assert "LEAKLENS_FAIL_UNDER: ${{ inputs.fail-under }}" in text
    assert "LEAKLENS_BASELINE: ${{ inputs.baseline }}" in text
    assert "LEAKLENS_FAIL_ON_NEW: ${{ inputs.fail-on-new }}" in text
    assert 'if [[ -n "$LEAKLENS_FAIL_UNDER" ]]' in text
    assert 'args+=(--fail-under "$LEAKLENS_FAIL_UNDER")' in text
    assert 'args+=(--baseline "$LEAKLENS_BASELINE")' in text
    assert 'if [[ "$LEAKLENS_FAIL_ON_NEW" == "true" ]]' in text
    assert "args+=(--fail-on-new)" in text
    assert 'leaklens --scan "$INPUT_PATH" "${args[@]}"' in text


def test_path_input_works_without_fail_under(tmp_path):
    (tmp_path / "clean.py").write_text(CLEAN, encoding="utf-8")
    assert main(_action_args(tmp_path, None)) == 0


def test_invalid_path_fails_the_action(tmp_path, capsys):
    missing = tmp_path / "checkout-did-not-happen"
    assert main(_action_args(missing, None)) == 2
    err = capsys.readouterr().err
    assert "scan target not found" in err
    assert str(missing) in err


def test_omitting_fail_under_keeps_the_default_threshold_behavior(tmp_path):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    # Score is 75; a default threshold of e.g. 80 would fail here.
    assert main(_action_args(target, None)) == 0


def test_valid_numeric_input_passes_above_the_threshold(tmp_path):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    assert main(_action_args(target, "70")) == 0


def test_valid_numeric_input_fails_below_the_threshold(tmp_path):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    assert main(_action_args(target, "80")) == 1


def test_unset_fail_under_still_fails_on_a_high_finding(tmp_path):
    target = tmp_path / "high.py"
    target.write_text(HIGH, encoding="utf-8")
    assert main(_action_args(target, None)) == 1


def test_invalid_input_is_rejected_with_a_useful_error(tmp_path, capsys):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        main(_action_args(target, "abc"))
    assert err.value.code == 2
    assert "fail-under" in capsys.readouterr().err


# The baseline path is how a team tracks findings it has already reviewed and
# asks CI to fail only on new ones. The Action has to carry it too.


def _write_baseline(target, baseline):
    main(["--scan", str(target), "--write-baseline", "--baseline", str(baseline)])
    assert baseline.exists()


def test_baseline_comparison_keeps_reviewed_findings_from_counting_as_new(tmp_path):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    baseline = tmp_path / "leaklens-baseline.json"
    _write_baseline(target, baseline)
    assert main(_action_args(target, None, baseline, True)) == 0


def test_fail_on_new_fails_when_a_new_finding_appears(tmp_path):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    baseline = tmp_path / "leaklens-baseline.json"
    _write_baseline(target, baseline)
    (tmp_path / "second.py").write_text(MEDIUM, encoding="utf-8")
    assert main(_action_args(tmp_path, None, baseline, True)) == 1
    # Without --fail-on-new the same tree still passes: the baseline is opt-in.
    assert main(_action_args(tmp_path, None, baseline, None)) == 0


def test_known_high_finding_still_fails_the_default_gate(tmp_path):
    target = tmp_path / "high.py"
    target.write_text(HIGH, encoding="utf-8")
    baseline = tmp_path / "leaklens-baseline.json"
    _write_baseline(target, baseline)
    # A baseline labels findings; it does not suppress an active high finding.
    assert main(_action_args(target, None, baseline, True)) == 1


def test_fail_on_new_without_a_baseline_is_a_usage_error(tmp_path, capsys):
    target = tmp_path / "medium.py"
    target.write_text(MEDIUM, encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        main(_action_args(target, None, None, True))
    assert err.value.code == 2
    assert "fail-on-new" in capsys.readouterr().err
