"""Complete-response parity across the CLI, API, public API and workspace.

Phase 13 M2 promised one payload path. ``tests/test_assemble.py`` proves the
sections are shared; this file proves the whole response agrees for the same
dataset input on every surface that accepts one. Envelopes differ by design
(the workspace is root-relative, the live API adds ``fix`` text), so the
comparison is the identity of every finding plus trust, counts and policy.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from leaklens import cli, public_api
from leaklens.api import app
from leaklens.fingerprint import SCHEMA_VERSION
from leaklens.upload_guard import GuardLimits, guard_upload
from leaklens.workspace import create_workspace_app

CORE = (
    "fingerprint",
    "kind",
    "severity",
    "column",
    "detail",
    "evidence",
    "status",
)


def _frame() -> pd.DataFrame:
    """A deterministic dataset with target leakage and target-signal missingness."""
    rng = np.random.default_rng(7)
    rows = 400
    target = rng.integers(0, 2, rows)
    leak = target * 10.0 + rng.normal(scale=0.01, size=rows)
    leak[target == 0] = np.nan
    return pd.DataFrame({"y": target, "x": leak, "z": rng.normal(size=rows)})


def _csv_bytes() -> bytes:
    return _frame().to_csv(index=False).encode("utf-8")


def _core(findings: list[dict]) -> list[dict]:
    """The shared finding identity every surface must agree on."""
    return sorted(
        ({key: item.get(key) for key in CORE} for item in findings),
        key=lambda item: (
            str(item["kind"]),
            str(item["column"]),
            str(item["fingerprint"]),
        ),
    )


def _baseline_text(thumbs: list[dict]) -> str:
    return json.dumps(
        {
            "schema": "leaklens-baseline",
            "version": 1,
            "fingerprint_schema": SCHEMA_VERSION,
            "tool_version": "test",
            "findings": [
                {
                    "fingerprint": item["fingerprint"],
                    "kind": item["kind"],
                    "severity": item["severity"],
                    "column": item["column"],
                }
                for item in thumbs
            ],
        }
    )


def _cli_json(monkeypatch, capsys, path, *args) -> dict:
    monkeypatch.chdir(path.parent)
    code = cli.main([str(path), "--target", "y", "--json", *args])
    assert code in (0, 1)
    return json.loads(capsys.readouterr().out)


def _api_audit(client: TestClient, **data) -> dict:
    response = client.post(
        "/audit",
        files={"file": ("d.csv", _csv_bytes(), "text/csv")},
        data={"target": "y", **data},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _live_audit() -> dict:
    upload = guard_upload("d.csv", _csv_bytes(), GuardLimits())
    return public_api.run_audit_job(upload, GuardLimits(), "y", 0.95)


def test_the_same_dataset_yields_the_same_findings_on_every_surface(
    tmp_path, monkeypatch, capsys
):
    data_path = tmp_path / "d.csv"
    data_path.write_bytes(_csv_bytes())
    client = TestClient(app)

    cli_payload = _cli_json(monkeypatch, capsys, data_path)
    api_payload = _api_audit(client)
    live_payload = _live_audit()

    expected_kinds = {"missingness_target_signal", "target_leakage"}
    for payload in (cli_payload, api_payload, live_payload):
        assert {item["kind"] for item in payload["findings"]} == expected_kinds

    assert _core(cli_payload["findings"]) == _core(api_payload["findings"])
    assert _core(api_payload["findings"]) == _core(live_payload["findings"])

    trust = cli_payload["trust_score"]
    assert trust == api_payload["trust_score"] == live_payload["trust_score"]
    assert trust == 50  # one high (35) and one medium (15) off a perfect score

    counts = live_payload["counts"]
    assert counts == {"high": 1, "medium": 1, "low": 0, "total": 2}
    severities = [item["severity"] for item in api_payload["findings"]]
    assert severities.count("high") == 1 and severities.count("medium") == 1


def test_the_workspace_reports_the_same_finding_identities(tmp_path):
    frame = _frame()
    frame.to_csv(tmp_path / "d.csv", index=False)
    client = TestClient(create_workspace_app(tmp_path))

    started = client.post("/api/datasets", json={"path": "d.csv", "target": "y"})
    assert started.status_code == 202, started.text
    dataset_id = started.json()["dataset_id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = client.get(f"/api/datasets/{dataset_id}").json()["state"]
        if state not in ("queued", "auditing"):
            break
        time.sleep(0.02)
    assert state == "done", state
    results = client.get(f"/api/datasets/{dataset_id}/results").json()

    live = _live_audit()
    mine = {(item["fingerprint"], item["kind"]) for item in results["findings"]}
    theirs = {(item["fingerprint"], item["kind"]) for item in live["findings"]}
    assert mine == theirs
    assert results["trust_score"] == live["trust_score"] == 50


def test_baseline_policy_effects_match_the_cli_and_the_api(
    tmp_path, monkeypatch, capsys
):
    data_path = tmp_path / "d.csv"
    data_path.write_bytes(_csv_bytes())
    client = TestClient(app)
    plain = _api_audit(client)

    baseline = _baseline_text(plain["findings"][:1])
    baseline_path = tmp_path / "base.json"
    baseline_path.write_text(baseline, encoding="utf-8")

    cli_payload = _cli_json(
        monkeypatch, capsys, data_path, "--baseline", str(baseline_path)
    )
    api_payload = _api_audit(client, baseline=baseline)

    assert _core(cli_payload["findings"]) == _core(api_payload["findings"])
    assert {item["status"] for item in cli_payload["findings"]} == {
        "new",
        "unchanged",
    }
    for payload in (cli_payload, api_payload):
        counts = payload["policy"]["baseline"]
        assert counts["unchanged"] == 1 and counts["new"] == 1
        assert counts["resolved"] == 0
        assert payload.get("resolved") is None


def test_a_shrinking_baseline_serializes_the_same_resolved_entry(
    tmp_path, monkeypatch, capsys
):
    data_path = tmp_path / "d.csv"
    data_path.write_bytes(_csv_bytes())
    client = TestClient(app)
    plain = _api_audit(client)
    assert len(plain["findings"]) == 2

    baseline = _baseline_text(plain["findings"])
    baseline_path = tmp_path / "base.json"
    baseline_path.write_text(baseline, encoding="utf-8")
    config_path = _write_suppression_config(
        tmp_path, plain["findings"][0]["fingerprint"]
    )

    api_payload = _api_audit(
        client,
        baseline=baseline,
        suppressions=json.dumps(
            [
                {
                    "fingerprint": plain["findings"][0]["fingerprint"],
                    "reason": "accepted for this comparison test",
                }
            ]
        ),
    )
    cli_payload = _cli_json(
        monkeypatch,
        capsys,
        data_path,
        "--baseline",
        str(baseline_path),
        "--config",
        config_path,
    )

    # Suppression labels one finding; it never deletes it, so both surfaces keep
    # two findings, agree on every identity and on both policy statuses.
    assert _core(cli_payload["findings"]) == _core(api_payload["findings"])
    expected = {"suppressed", "unchanged"}
    assert {item["status"] for item in api_payload["findings"]} == expected
    assert {item["status"] for item in cli_payload["findings"]} == expected
    assert cli_payload["trust_score"] == api_payload["trust_score"]
    assert cli_payload["trust_score"] > plain["trust_score"]


def _write_suppression_config(tmp_path, fingerprint: str) -> str:
    path = tmp_path / "config.toml"
    path.write_text(
        "[[suppressions]]\n"
        f'fingerprint = "{fingerprint}"\n'
        'reason = "accepted for this comparison test"\n',
        encoding="utf-8",
    )
    return str(path)
