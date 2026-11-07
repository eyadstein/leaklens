"""End-to-end policy workflow: baselines, suppressions, exit codes, outputs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from leaklens.cli import main


def _dataset(path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=1.5, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {"x": x, "leak": y * 2.0, "uid": [f"u{i}" for i in range(n)], "y": y}
    )
    frame.to_csv(path, index=False)
    return path


def _run(capsys, *args):
    code = main(list(args))
    out = capsys.readouterr().out
    return code, out


def _payload(out: str) -> dict:
    return json.loads(out)


def test_write_then_compare_baseline(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    base = tmp_path / "base.json"
    code, out = _run(
        capsys,
        str(csv),
        "--target",
        "y",
        "--write-baseline",
        "--baseline",
        str(base),
        "--json",
    )
    assert code == 1  # the high finding still fails the default gate
    assert base.is_file()
    written = json.loads(base.read_text(encoding="utf-8"))
    assert written["schema"] == "leaklens-baseline"
    assert written["findings"]

    code, out = _run(
        capsys, str(csv), "--target", "y", "--baseline", str(base), "--json"
    )
    payload = _payload(out)
    assert payload["policy"]["baseline"]["new"] == 0
    assert payload["policy"]["baseline"]["resolved"] == 0
    assert {f["status"] for f in payload["findings"]} == {"unchanged"}
    assert code == 1


def test_baseline_write_refuses_to_overwrite_and_update_is_explicit(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    base = tmp_path / "base.json"
    args = (str(csv), "--target", "y", "--json", "--baseline", str(base))
    assert _run(capsys, *args, "--write-baseline")[0] == 1
    original = base.read_text(encoding="utf-8")
    code, _ = _run(capsys, *args, "--write-baseline")
    assert code == 2
    assert base.read_text(encoding="utf-8") == original
    assert _run(capsys, *args, "--update-baseline")[0] == 1


def test_baseline_is_deterministic(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for path in (first, second):
        _run(
            capsys,
            str(csv),
            "--target",
            "y",
            "--json",
            "--baseline",
            str(path),
            "--write-baseline",
        )
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_incompatible_baseline_fails_closed(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    base = tmp_path / "base.json"
    doc = {
        "schema": "leaklens-baseline",
        "version": 1,
        "fingerprint_schema": 99,
        "tool_version": "0.13.0",
        "findings": [],
    }
    base.write_text(json.dumps(doc), encoding="utf-8")
    code, _ = _run(capsys, str(csv), "--target", "y", "--json", "--baseline", str(base))
    assert code == 2


def test_new_findings_appear_and_are_opt_in_for_ci(tmp_path, capsys):
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=1.5, size=n) > 0).astype(int)
    frame = pd.DataFrame({"x": x, "uid": [f"u{i}" for i in range(n)], "y": y})
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    base = tmp_path / "base.json"
    # Medium-only findings keep the default gate green.
    assert (
        _run(
            capsys,
            str(csv),
            "--target",
            "y",
            "--json",
            "--baseline",
            str(base),
            "--write-baseline",
        )[0]
        == 0
    )
    code, out = _run(
        capsys,
        str(csv),
        "--target",
        "y",
        "--json",
        "--baseline",
        str(base),
        "--fail-on-new",
    )
    assert _payload(out)["policy"]["baseline"]["new"] == 0
    assert code == 0

    frame["uid2"] = [f"v{i}" for i in range(n)]
    frame.to_csv(csv, index=False)
    code, out = _run(
        capsys, str(csv), "--target", "y", "--json", "--baseline", str(base)
    )
    assert _payload(out)["policy"]["baseline"]["new"] >= 1
    assert code == 0  # new findings alone are not a failure by default
    code, out = _run(
        capsys,
        str(csv),
        "--target",
        "y",
        "--json",
        "--baseline",
        str(base),
        "--fail-on-new",
    )
    assert code == 1


def test_suppression_clears_the_default_gate_but_keeps_the_finding(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    code, out = _run(capsys, str(csv), "--target", "y", "--json")
    high = next(f for f in _payload(out)["findings"] if f["severity"] == "high")
    config = tmp_path / ".leaklens.toml"
    config.write_text(
        "[[suppressions]]\n"
        f'fingerprint = "{high["fingerprint"]}"\n'
        'reason = "accepted by the ML owner"\n',
        encoding="utf-8",
    )
    code, out = _run(
        capsys, str(csv), "--target", "y", "--json", "--config", str(config)
    )
    payload = _payload(out)
    suppressed = [f for f in payload["findings"] if f.get("status") == "suppressed"]
    assert [f["fingerprint"] for f in suppressed] == [high["fingerprint"]]
    assert suppressed[0]["suppression"]["reason"] == "accepted by the ML owner"
    assert suppressed[0]["severity"] == high["severity"]
    assert payload["trust_score"] > 0
    assert "high" not in {
        f["severity"] for f in payload["findings"] if f.get("status") != "suppressed"
    }
    assert code == 0


def test_expired_suppression_is_reported_and_not_applied(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    _, out = _run(capsys, str(csv), "--target", "y", "--json")
    high = next(f for f in _payload(out)["findings"] if f["severity"] == "high")
    config = tmp_path / ".leaklens.toml"
    config.write_text(
        "[[suppressions]]\n"
        f'fingerprint = "{high["fingerprint"]}"\n'
        'reason = "temporary"\n'
        'expires = "2000-01-01"\n',
        encoding="utf-8",
    )
    code, out = _run(
        capsys, str(csv), "--target", "y", "--json", "--config", str(config)
    )
    payload = _payload(out)
    assert all(f.get("status") != "suppressed" for f in payload["findings"])
    assert "expired" in " ".join(payload["policy"]["diagnostics"])
    assert code == 1


def test_resolved_findings_are_distinct_from_suppressed(tmp_path, capsys):
    csv = _dataset(tmp_path / "data.csv")
    base = tmp_path / "base.json"
    _run(
        capsys,
        str(csv),
        "--target",
        "y",
        "--json",
        "--baseline",
        str(base),
        "--write-baseline",
    )
    frame = pd.read_csv(csv).drop(columns=["leak"])
    frame.to_csv(csv, index=False)
    code, out = _run(
        capsys, str(csv), "--target", "y", "--json", "--baseline", str(base)
    )
    payload = _payload(out)
    resolved = {entry["kind"] for entry in payload["resolved"]}
    assert "target_leakage" in resolved
    assert all(f.get("status") != "suppressed" for f in payload["findings"])
    assert payload["policy"]["baseline"]["resolved"] >= 1
    assert code in (0, 1)


def test_missingness_findings_reach_the_cli(tmp_path, capsys):
    rng = np.random.default_rng(5)
    n = 800
    y = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    x[rng.random(n) < np.where(y == 1, 0.7, 0.03)] = np.nan
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})
    frame.to_csv(tmp_path / "data.csv", index=False)
    code, out = _run(capsys, str(tmp_path / "data.csv"), "--target", "y", "--json")
    payload = _payload(out)
    assert any(f["kind"] == "missingness_target_signal" for f in payload["findings"])
    assert payload["missingness"]["status"] == "ok"
    assert code in (0, 1)
