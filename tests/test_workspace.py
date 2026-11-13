"""Interactive workspace: scan and dataset-audit lifecycles, safety, previews."""

import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from leaklens.baseline import BASELINE_SCHEMA, BASELINE_VERSION
from leaklens.fingerprint import SCHEMA_VERSION
from leaklens.workspace import (
    GRAPH_MAX_LIMIT,
    ScanSession,
    WorkspacePathError,
    WorkspaceScanManager,
    _graph_payload,
    _relative,
    create_workspace_app,
)

LEAKY = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\n"
    "a = train_test_split(X)\n"
)


def _project(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "leaky.py").write_text(LEAKY, encoding="utf-8")
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _dataset(tmp_path):
    """A small leaky training frame plus a held-out slice of the same rows."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    frame = pd.DataFrame({"x": x, "y": (x + rng.normal(size=120) > 0).astype(int)})
    frame["leak"] = frame["y"] * 2.0
    frame.to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[:30].to_csv(tmp_path / "test.csv", index=False)
    return frame


def _client(tmp_path):
    return TestClient(create_workspace_app(tmp_path))


POLICY_LEAKY = (
    "import pandas as pd\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.impute import SimpleImputer\n"
    "from sklearn.model_selection import (\n"
    "    train_test_split, cross_val_score, StratifiedKFold)\n"
    "from sklearn.linear_model import LogisticRegression\n"
    'df = pd.read_csv("data.csv")\n'
    'X = df.drop(columns=["y"])\n'
    'y = df["y"]\n'
    "sc = StandardScaler()\n"
    "X = sc.fit_transform(X)\n"
    "imp = SimpleImputer()\n"
    "X = imp.fit_transform(X)\n"
    "a = train_test_split(X)\n"
    "cross_val_score(LogisticRegression(), X, y,\n"
    "    cv=StratifiedKFold(5, shuffle=True, random_state=0))\n"
)


def _policy_project(tmp_path):
    (tmp_path / "leaky.py").write_text(POLICY_LEAKY, encoding="utf-8")
    return tmp_path


def _baseline_entries(findings, *, omit=(), severity=None):
    """Baseline entries from workspace findings, optionally changed or omitted."""
    entries = {}
    for finding in findings:
        key = finding["fingerprint"]
        if key in omit:
            continue
        entries[key] = {
            "fingerprint": key,
            "kind": finding["kind"],
            "severity": (severity or {}).get(key, finding["severity"]),
            "column": finding["column"],
        }
    return list(entries.values())


def _write_baseline(path, entries):
    path.write_text(
        json.dumps(
            {
                "schema": BASELINE_SCHEMA,
                "version": BASELINE_VERSION,
                "fingerprint_schema": SCHEMA_VERSION,
                "tool_version": "test",
                "findings": entries,
            }
        ),
        encoding="utf-8",
    )


def _write_config(tmp_path, *, baseline=None, suppressions=()):
    lines = []
    if baseline:
        lines += ["[baseline]", f'path = "{baseline}"']
    for rule in suppressions:
        lines += [
            "[[suppressions]]",
            f'fingerprint = "{rule["fingerprint"]}"',
            f'reason = "{rule["reason"]}"',
        ]
    (tmp_path / ".leaklens.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wait(client, scan_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/scans/{scan_id}").json()
        if status["state"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("scan did not finish")


def _run(client, path=".", sql=False, baseline=None, include_ai=False):
    response = client.post(
        "/api/scans",
        json={
            "path": path,
            "sql": sql,
            "baseline": baseline,
            "include_ai": include_ai,
        },
    )
    assert response.status_code == 202, response.text
    scan_id = response.json()["scan_id"]
    status = _wait(client, scan_id)
    return scan_id, status


def test_page_and_info_expose_the_root(tmp_path):
    client = _client(_project(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert "LeakLens workspace" in page.text
    for control in (
        'id="baseline"',
        'id="f-status"',
        'id="ds-status"',
        'id="ai"',
        'id="ds-ai"',
        'id="ai-summary"',
        'id="ds-ai-summary"',
    ):
        assert control in page.text
    info = client.get("/api/info").json()
    assert info["root"] == tmp_path.resolve().as_posix()
    assert info["max_files"] >= 1
    assert info["source_context"] >= 1
    assert set(info["config"]) == {"path", "baseline", "suppressions", "error"}
    assert info["config"]["error"] is None
    assert info["config"]["baseline"] is None


def test_scan_lifecycle_reports_progress_and_evidence(tmp_path):
    client = _client(_project(tmp_path))
    scan_id, status = _run(client)

    assert status["state"] == "done"
    assert status["total"] == 2 and status["done"] == 2
    assert status["current"] is None

    results = client.get(f"/api/scans/{scan_id}/results").json()
    assert results["scan_id"] == scan_id
    assert results["state"] == "done"
    assert results["files"] == 2
    assert results["trust_score"] < 100
    assert results["findings"]
    assert sum(results["counts"].values()) == len(results["findings"])
    assert results["duration_ms"] >= 0

    for finding in results["findings"]:
        assert finding["severity"] in {"high", "medium", "low"}
        assert len(finding["fingerprint"]) == 64
        assert isinstance(finding["index"], int)
        assert finding["file"].endswith(".py")
        assert finding["line"] >= 1
        assert isinstance(finding["evidence"], list)

    covered = {
        index for cluster in results["clusters"] for index in cluster["finding_indices"]
    }
    assert covered == {finding["index"] for finding in results["findings"]}
    assert results["links"] and all("relation" in link for link in results["links"])
    assert 0 <= results["confidence"]["score"] <= 1
    assert results["recommendations"]


def test_scanning_one_file_only_returns_that_file(tmp_path):
    client = _client(_project(tmp_path))
    scan_id, status = _run(client, path="pkg/leaky.py")

    assert status["total"] == 1
    results = client.get(f"/api/scans/{scan_id}/results").json()
    assert results["findings"]
    assert {finding["file"] for finding in results["findings"]} == {"pkg/leaky.py"}


def test_paths_outside_the_root_are_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.py").write_text(LEAKY, encoding="utf-8")
    client = _client(root)

    assert client.post("/api/scans", json={"path": "../outside.py"}).status_code == 422
    assert (
        client.post(
            "/api/scans", json={"path": (tmp_path / "outside.py").as_posix()}
        ).status_code
        == 422
    )
    assert (
        client.get("/api/source", params={"path": "../outside.py"}).status_code == 422
    )


def test_relative_labels_never_leak_an_absolute_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert _relative(tmp_path / "elsewhere.py", root) == "elsewhere.py"


def test_a_symlinked_source_outside_the_root_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(LEAKY, encoding="utf-8")
    try:
        (root / "link.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    client = _client(root)
    assert client.get("/api/source", params={"path": "link.py"}).status_code == 422


def test_a_symlinked_dataset_outside_the_root_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("x,y\n1,0\n", encoding="utf-8")
    try:
        (root / "link.csv").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    client = _client(root)
    response = client.post("/api/datasets/inspect", json={"path": "link.csv"})
    assert response.status_code == 422
    assert "outside the workspace root" in response.json()["detail"]


def test_empty_folder_is_rejected(tmp_path):
    root = _project(tmp_path)
    (root / "empty").mkdir()
    client = _client(root)
    response = client.post("/api/scans", json={"path": "empty"})
    assert response.status_code == 422
    assert "no .py" in response.json()["detail"]


def test_unknown_scan_and_finished_cancel(tmp_path):
    client = _client(_project(tmp_path))
    assert client.get("/api/scans/missing").status_code == 404
    assert client.get("/api/scans/missing/results").status_code == 404
    assert client.post("/api/scans/missing/cancel").status_code == 404

    scan_id, _ = _run(client)
    assert client.post(f"/api/scans/{scan_id}/cancel").status_code == 409


def test_second_scan_is_refused_while_one_is_running(tmp_path, monkeypatch):
    from leaklens import workspace

    entered = threading.Event()
    release = threading.Event()

    def blocking_scan(path):
        entered.set()
        release.wait(timeout=10)
        return []

    monkeypatch.setattr(workspace, "scan_file", blocking_scan)
    client = _client(_project(tmp_path))
    first = client.post("/api/scans", json={"path": "."})
    assert first.status_code == 202
    assert entered.wait(timeout=10)

    second = client.post("/api/scans", json={"path": "."})
    assert second.status_code == 409

    release.set()
    _wait(client, first.json()["scan_id"])


def test_cancel_stops_at_the_next_file_boundary(tmp_path, monkeypatch):
    from leaklens import workspace

    entered = threading.Event()
    release = threading.Event()
    calls = []

    def slow_scan(path):
        calls.append(path)
        entered.set()
        release.wait(timeout=10)
        return []

    monkeypatch.setattr(workspace, "scan_file", slow_scan)
    client = _client(_project(tmp_path))
    scan_id = client.post("/api/scans", json={"path": "."}).json()["scan_id"]
    assert entered.wait(timeout=10)

    assert client.post(f"/api/scans/{scan_id}/cancel").status_code == 200
    release.set()
    status = _wait(client, scan_id)

    assert status["state"] == "cancelled"
    assert len(calls) == 1


def test_source_preview_is_windowed_and_bounded(tmp_path):
    root = _project(tmp_path)
    body = "\n".join(f"value_{number} = {number}" for number in range(1, 31))
    (root / "clean.py").write_text(body, encoding="utf-8")
    client = _client(root)

    data = client.get("/api/source", params={"path": "clean.py", "line": 10}).json()
    assert data["start"] == 2 and data["line"] == 10 and data["total"] == 30
    assert [line["number"] for line in data["lines"]] == list(range(2, 19))
    assert data["lines"][8]["text"] == "value_10 = 10"

    assert (
        client.get("/api/source", params={"path": "clean.py", "line": 9999}).json()[
            "line"
        ]
        == 30
    )
    assert client.get("/api/source", params={"path": "missing.py"}).status_code == 404


def test_report_reuses_the_static_renderer(tmp_path):
    client = _client(_project(tmp_path))
    scan_id, _ = _run(client)
    report = client.get(f"/api/scans/{scan_id}/report")
    assert report.status_code == 200
    assert "LeakLens workspace report" in report.text
    assert "Findings" in report.text


# --- dataset audits -------------------------------------------------------


def _wait_dataset(client, dataset_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/datasets/{dataset_id}").json()
        if status["state"] not in ("queued", "auditing"):
            return status
        time.sleep(0.02)
    raise AssertionError("dataset audit did not finish")


def _run_dataset(client, **body):
    body.setdefault("path", "train.csv")
    body.setdefault("target", "y")
    response = client.post("/api/datasets", json=body)
    assert response.status_code == 202, response.text
    dataset_id = response.json()["dataset_id"]
    return dataset_id, _wait_dataset(client, dataset_id)


def test_dataset_inspect_lists_columns_and_target_candidates(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    data = client.post("/api/datasets/inspect", json={"path": "train.csv"}).json()
    assert data["file"] == "train.csv"
    assert data["rows"] == 120
    assert [column["name"] for column in data["columns"]] == ["x", "y", "leak"]
    assert data["target_candidates"] == ["y", "leak"]
    assert data["preview"] and set(data["preview"][0]) == {"x", "y", "leak"}
    assert data["limits"]["max_rows"] >= 120


def test_dataset_audit_lifecycle_and_serialisation(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    dataset_id, status = _run_dataset(client, test_path="test.csv")
    assert status["state"] == "done"
    assert status["error"] is None
    assert status["request"] == {
        "file": "train.csv",
        "test": "test.csv",
        "target": "y",
        "threshold": 0.95,
        "time_column": None,
        "group_column": None,
    }
    assert status["elapsed_ms"] >= 0

    results = client.get(f"/api/datasets/{dataset_id}/results").json()
    json.dumps(results)  # bounded, JSON-serialisable payload
    assert results["dataset_id"] == dataset_id
    assert results["state"] == "done"
    assert results["live"] is True
    assert results["engine"] == "leaklens.python"
    assert results["trust_score"] < 100
    assert results["summary"]["file"] == "train.csv"
    assert results["summary"]["target"] == "y"
    assert results["summary"]["threshold"] == 0.95
    assert results["summary"]["rows"] == 120
    assert results["summary"]["test_rows"] == 30
    assert results["audit"]["rows"] == 120
    assert results["limits"]["formats"] == ["csv", "json"]
    assert results["report_filename"] == "leaklens-report.html"

    kinds = {finding["kind"] for finding in results["findings"]}
    assert "target_leakage" in kinds and "split_overlap" in kinds
    assert sum(results["counts"][key] for key in ("high", "medium", "low")) == len(
        results["findings"]
    )
    for index, finding in enumerate(results["findings"]):
        assert finding["index"] == index
        assert len(finding["fingerprint"]) == 64
        assert finding["severity"] in {"high", "medium", "low"}
        assert finding["evidence"]
        assert finding["advice"]

    # Clusters cover every finding and come from the shared intelligence layer.
    covered = {
        index
        for cluster in results["intelligence"]["clusters"]
        for index in cluster["finding_indices"]
    }
    assert covered == set(range(len(results["findings"])))
    assert results["intelligence"]["recommendations"]


def test_dataset_audit_reuses_the_public_audit_engine(tmp_path):
    from leaklens.public_api import run_audit_job
    from leaklens.upload_guard import guard_upload
    from leaklens.workspace import DATASET_LIMITS

    _dataset(tmp_path)
    client = _client(tmp_path)
    dataset_id, _ = _run_dataset(client, test_path="test.csv")
    results = client.get(f"/api/datasets/{dataset_id}/results").json()

    train = guard_upload(
        "train.csv", (tmp_path / "train.csv").read_bytes(), DATASET_LIMITS
    )
    test = guard_upload(
        "test.csv", (tmp_path / "test.csv").read_bytes(), DATASET_LIMITS
    )
    expected = run_audit_job(
        train,
        DATASET_LIMITS,
        "y",
        0.95,
        test,
        None,
        None,
        include_intelligence=True,
    )
    expected.pop("report_html")

    def engine_view(payload):
        data = json.loads(json.dumps(payload))
        for key in (
            "dataset_id",
            "state",
            "request",
            "elapsed_ms",
            "report_html",
            "policy",
            "resolved",
        ):
            data.pop(key, None)
        data["summary"].pop("duration_ms", None)
        for finding in data["findings"]:
            finding.pop("index", None)
            finding.pop("advice", None)
            finding.pop("requires_attention", None)
        return data

    assert engine_view(results) == engine_view(expected)


def test_dataset_audit_matches_the_cli_trust_score(tmp_path, capsys):
    from leaklens import cli

    _dataset(tmp_path)
    client = _client(tmp_path)
    dataset_id, _ = _run_dataset(client, test_path="test.csv")
    results = client.get(f"/api/datasets/{dataset_id}/results").json()

    code = cli.main(
        [
            str(tmp_path / "train.csv"),
            "--target",
            "y",
            "--test",
            str(tmp_path / "test.csv"),
        ]
    )
    printed = capsys.readouterr().out
    assert f"LeakLens trust score: {results['trust_score']}/100" in printed
    assert code in (0, 1)


def test_dataset_audit_passes_the_request_to_the_engine(tmp_path, monkeypatch):
    from leaklens import workspace

    calls = []
    real = workspace.run_audit_job

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(workspace, "run_audit_job", spy)
    _dataset(tmp_path)
    client = _client(tmp_path)
    _run_dataset(client, threshold=0.8, test_path="test.csv")

    (args, kwargs) = calls[0]
    upload, limits, target, threshold, test_upload = args[:5]
    assert upload.filename == "train.csv" and test_upload.filename == "test.csv"
    assert limits is workspace.DATASET_LIMITS
    assert (target, threshold) == ("y", 0.8)
    assert kwargs == {"include_intelligence": True, "include_ai": False}


def test_dataset_threshold_is_validated_and_rounded(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    for bad in (0.4, 0.49, 1.01, 5):
        response = client.post(
            "/api/datasets", json={"path": "train.csv", "target": "y", "threshold": bad}
        )
        assert response.status_code == 422
        assert "0.5 and 1.0" in response.json()["detail"]

    dataset_id, status = _run_dataset(client, threshold=0.95321)
    assert status["request"]["threshold"] == 0.9532
    assert (
        client.get(f"/api/datasets/{dataset_id}/results").json()["summary"]["threshold"]
        == 0.9532
    )


def test_dataset_target_is_required_and_must_exist(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    response = client.post("/api/datasets", json={"path": "train.csv", "target": " "})
    assert response.status_code == 422
    assert "target column" in response.json()["detail"]

    dataset_id, status = _run_dataset(client, target="missing")
    assert status["state"] == "failed"
    assert status["error"]["code"] == "unknown_target"
    unfinished = client.get(f"/api/datasets/{dataset_id}/results")
    assert unfinished.status_code == 409
    assert "target column was not found" in unfinished.json()["detail"]
    assert client.get(f"/api/datasets/{dataset_id}/report").status_code == 409


def test_dataset_held_out_file_drives_the_impact_section(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    dataset_id, status = _run_dataset(client, test_path="test.csv")
    results = client.get(f"/api/datasets/{dataset_id}/results").json()
    impact = results["impact"]
    assert impact["train_rows"] == 120
    assert impact["test_rows"] == 30
    assert impact["duplicate_test_rows"] == 30
    assert impact["duplicate_test_pct"] == 100.0

    bad = client.post(
        "/api/datasets",
        json={"path": "train.csv", "target": "y", "test_path": "missing.csv"},
    )
    assert bad.status_code == 422

    # A held-out file that lacks the requested option column fails explicitly.
    (tmp_path / "small.csv").write_text("x,y\n1,0\n2,1\n", encoding="utf-8")
    dataset_id, status = _run_dataset(client, test_path="small.csv", time_col="when")
    assert status["state"] == "failed"
    assert status["error"]["code"] == "unknown_option_column"


def test_dataset_formats_and_optional_parsers(tmp_path):
    _dataset(tmp_path)
    (tmp_path / "train.txt").write_text("x,y\n1,0\n", encoding="utf-8")
    (tmp_path / "train.xlsx").write_bytes(b"not really a workbook")
    (tmp_path / "train.parquet").write_bytes(b"not really parquet")
    client = _client(tmp_path)

    unsupported = client.post("/api/datasets/inspect", json={"path": "train.txt"})
    assert unsupported.status_code == 415
    assert ".csv" in unsupported.json()["detail"]

    for name in ("train.xlsx", "train.parquet"):
        response = client.post("/api/datasets/inspect", json={"path": name})
        assert response.status_code == 415
        assert "enabled" in response.json()["detail"]

    formats = {
        item["extension"]: item
        for item in client.get("/api/info").json()["dataset_formats"]
    }
    assert formats[".csv"]["enabled"] is True
    assert formats[".json"]["enabled"] is True
    for item in formats.values():
        if not item["configured"]:
            assert item["enabled"] is False
        assert set(item) == {
            "extension",
            "label",
            "enabled",
            "configured",
            "installed",
            "extra",
        }


def test_dataset_paths_outside_the_root_are_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _dataset(root)
    outside = tmp_path / "outside.csv"
    outside.write_text("x,y\n1,0\n", encoding="utf-8")
    client = _client(root)

    for path in (
        "../outside.csv",
        outside.as_posix(),
        "..\\outside.csv",
        "..\\..\\outside.csv",
    ):
        for endpoint in ("/api/datasets/inspect", "/api/datasets"):
            response = client.post(endpoint, json={"path": path, "target": "y"})
            assert response.status_code == 422, (path, endpoint)
            assert "outside the workspace root" in response.json()["detail"]


def test_dataset_paths_with_windows_separators_resolve_inside_the_root(tmp_path):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    _dataset(root / "nested")
    client = _client(root)

    inspected = client.post("/api/datasets/inspect", json={"path": "nested\\train.csv"})
    assert inspected.status_code == 200, inspected.text
    assert inspected.json()["rows"] == 120

    started = client.post(
        "/api/datasets", json={"path": "nested\\train.csv", "target": "y"}
    )
    assert started.status_code == 202, started.text


def test_workspace_resolve_normalizes_windows_separators(tmp_path):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    manager = WorkspaceScanManager(root)

    expected = (root / "nested" / "train.csv").resolve()
    assert manager.resolve("nested\\train.csv") == expected
    with pytest.raises(WorkspacePathError, match="outside the workspace root"):
        manager.resolve("..\\outside.csv")


def test_dataset_empty_and_malformed_files_are_rejected(tmp_path):
    _dataset(tmp_path)
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "header.csv").write_text("x,y\n", encoding="utf-8")
    (tmp_path / "broken.json").write_text('{"a": [1, 2', encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
    client = _client(tmp_path)

    empty = client.post("/api/datasets/inspect", json={"path": "empty.csv"})
    assert empty.status_code == 422 and "empty" in empty.json()["detail"]

    header = client.post("/api/datasets", json={"path": "header.csv", "target": "y"})
    assert header.status_code == 415
    assert "header line" in header.json()["detail"]

    for name in ("broken.json", "list.json"):
        response = client.post("/api/datasets/inspect", json={"path": name})
        assert response.status_code in (415, 422), response.text
        assert response.json()["detail"]

    assert (
        client.post("/api/datasets/inspect", json={"path": "missing.csv"}).status_code
        == 422
    )


def test_dataset_cancel_while_the_engine_runs_is_refused(tmp_path, monkeypatch):
    from leaklens import workspace

    entered, release = threading.Event(), threading.Event()
    real = workspace.run_audit_job

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=15)
        return real(*args, **kwargs)

    monkeypatch.setattr(workspace, "run_audit_job", blocking)
    _dataset(tmp_path)
    client = _client(tmp_path)

    started = client.post("/api/datasets", json={"path": "train.csv", "target": "y"})
    assert started.status_code == 202
    dataset_id = started.json()["dataset_id"]
    assert entered.wait(timeout=15)

    status = client.get(f"/api/datasets/{dataset_id}").json()
    assert status["state"] == "auditing"
    assert client.get(f"/api/datasets/{dataset_id}/results").status_code == 409

    cancel = client.post(f"/api/datasets/{dataset_id}/cancel")
    assert cancel.status_code == 409
    assert "cannot be interrupted" in cancel.json()["detail"]

    release.set()
    assert _wait_dataset(client, dataset_id)["state"] == "done"
    assert client.post(f"/api/datasets/{dataset_id}/cancel").status_code == 409


def test_dataset_cancel_before_the_engine_starts(tmp_path):
    from leaklens.workspace import DatasetSession

    app = create_workspace_app(tmp_path)
    manager = app.state.manager
    session = DatasetSession(
        dataset_id="queued", path=tmp_path / "train.csv", target="y", threshold=0.95
    )
    manager._datasets[session.dataset_id] = session

    assert manager.cancel_dataset("queued").state == "cancelled"
    assert session.cancel.is_set()


def test_dataset_and_source_scan_share_one_slot(tmp_path, monkeypatch):
    from leaklens import workspace

    entered, release = threading.Event(), threading.Event()
    real = workspace.run_audit_job

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=15)
        return real(*args, **kwargs)

    monkeypatch.setattr(workspace, "run_audit_job", blocking)
    _dataset(tmp_path)
    client = _client(_project(tmp_path))

    started = client.post("/api/datasets", json={"path": "train.csv", "target": "y"})
    assert started.status_code == 202
    dataset_id = started.json()["dataset_id"]
    assert entered.wait(timeout=15)

    refused = client.post("/api/scans", json={"path": "."})
    assert refused.status_code == 409

    release.set()
    assert _wait_dataset(client, dataset_id)["state"] == "done"

    scan_id, scan_status = _run(client)
    dataset_id, _ = _run_dataset(client)
    assert scan_status["state"] == "done"
    results = client.get(f"/api/scans/{scan_id}/results").json()
    assert results["findings"] and results["clusters"]


def test_dataset_report_is_the_static_html_report(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    dataset_id, _ = _run_dataset(client)
    report = client.get(f"/api/datasets/{dataset_id}/report")
    assert report.status_code == 200
    assert "LeakLens report - train.csv" in report.text
    assert "Evidence clusters" in report.text
    assert "Target leakage" in report.text or "target_leakage" in report.text


def test_unknown_dataset_audit_returns_404(tmp_path):
    client = _client(_project(tmp_path))
    assert client.get("/api/datasets/missing").status_code == 404
    assert client.get("/api/datasets/missing/results").status_code == 404
    assert client.get("/api/datasets/missing/report").status_code == 404
    assert client.post("/api/datasets/missing/cancel").status_code == 404


# --- baseline and suppression visibility ----------------------------------


def _statuses(results):
    return {
        finding["fingerprint"]: finding.get("status") for finding in results["findings"]
    }


def _trust_without(findings, suppressed):
    penalty = {"high": 35, "medium": 15, "low": 5}
    return max(
        0,
        100
        - sum(
            penalty.get(finding["severity"], 0)
            for finding in findings
            if finding["fingerprint"] != suppressed
        ),
    )


def test_baseline_statuses_suppression_and_summary_are_consistent(tmp_path):
    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    initial = client.get(f"/api/scans/{scan_id}/results").json()
    findings = initial["findings"]
    assert len(findings) >= 4
    new, changed, suppressed, unchanged = findings[:4]
    changed_severity = "low" if changed["severity"] != "low" else "high"
    extra = {
        "fingerprint": "a" * 64,
        "kind": "target_leakage",
        "severity": "medium",
        "column": "ghost",
    }
    entries = _baseline_entries(
        findings,
        omit={new["fingerprint"]},
        severity={changed["fingerprint"]: changed_severity},
    )
    _write_baseline(tmp_path / "baseline.json", [*entries, extra])
    _write_config(
        tmp_path,
        baseline="baseline.json",
        suppressions=[
            {"fingerprint": suppressed["fingerprint"], "reason": "accepted risk"}
        ],
    )

    scan_id, _ = _run(client)
    results = client.get(f"/api/scans/{scan_id}/results").json()
    statuses = _statuses(results)

    assert statuses[new["fingerprint"]] == "new"
    assert statuses[changed["fingerprint"]] == "changed"
    assert statuses[suppressed["fingerprint"]] == "suppressed"
    assert statuses[unchanged["fingerprint"]] == "unchanged"
    by_fingerprint = {
        finding["fingerprint"]: finding for finding in results["findings"]
    }
    assert (
        by_fingerprint[changed["fingerprint"]]["previous"]["severity"]
        == changed_severity
    )
    assert by_fingerprint[suppressed["fingerprint"]]["suppression"] == {
        "fingerprint": suppressed["fingerprint"],
        "reason": "accepted risk",
    }
    assert by_fingerprint[suppressed["fingerprint"]]["requires_attention"] is False
    assert (
        by_fingerprint[suppressed["fingerprint"]]["evidence"] == suppressed["evidence"]
    )
    for key in (new["fingerprint"], changed["fingerprint"], unchanged["fingerprint"]):
        assert by_fingerprint[key]["requires_attention"] is True

    policy = results["policy"]
    assert policy["baseline_selected"] is True
    assert policy["baseline_path"] == "baseline.json"
    assert policy["baseline"] == {
        "path": "baseline.json",
        "fail_on_new": False,
        "new": 1,
        "unchanged": 2,
        "changed": 1,
        "resolved": 1,
    }
    assert policy["statuses"] == {
        "changed": 1,
        "new": 1,
        "suppressed": 1,
        "unchanged": 1,
    }
    assert policy["unclassified"] == 0
    assert sum(policy["statuses"].values()) == len(results["findings"])
    assert results["resolved"][0]["fingerprint"] == extra["fingerprint"]
    assert results["trust_score"] == _trust_without(
        results["findings"], suppressed["fingerprint"]
    )

    report = client.get(f"/api/scans/{scan_id}/report")
    assert "Baseline and suppression policy" in report.text
    assert "accepted risk" in report.text
    assert "suppressed" in report.text.lower()


def test_suppression_without_baseline_keeps_evidence_and_invents_no_status(tmp_path):
    client = _client(_project(tmp_path))
    scan_id, _ = _run(client)
    initial = client.get(f"/api/scans/{scan_id}/results").json()
    target = initial["findings"][0]
    _write_config(
        tmp_path,
        suppressions=[{"fingerprint": target["fingerprint"], "reason": "known parser"}],
    )

    scan_id, _ = _run(client)
    results = client.get(f"/api/scans/{scan_id}/results").json()
    by_fingerprint = {
        finding["fingerprint"]: finding for finding in results["findings"]
    }
    suppressed = by_fingerprint[target["fingerprint"]]
    assert suppressed["status"] == "suppressed"
    assert suppressed["suppression"]["reason"] == "known parser"
    assert suppressed["severity"] == target["severity"]
    assert suppressed["detail"] == target["detail"]
    assert "evidence" in suppressed
    assert suppressed["requires_attention"] is False
    assert all(
        "status" not in finding
        for finding in results["findings"]
        if finding["fingerprint"] != target["fingerprint"]
    )

    policy = results["policy"]
    assert policy["baseline_selected"] is False
    assert "baseline" not in policy
    assert policy["statuses"] == {"suppressed": 1}
    assert policy["unclassified"] == len(results["findings"]) - 1
    assert results["trust_score"] == _trust_without(
        results["findings"], target["fingerprint"]
    )


def test_baseline_validation_missing_and_traversal_are_explicit(tmp_path):
    client = _client(_project(tmp_path))
    outside = (tmp_path.parent / "outside-baseline.json").as_posix()

    for value in ("../outside-baseline.json", outside):
        response = client.post("/api/scans", json={"path": ".", "baseline": value})
        assert response.status_code == 422, response.text
        assert "outside the workspace root" in response.json()["detail"]

    missing = client.post("/api/scans", json={"path": ".", "baseline": "missing.json"})
    assert missing.status_code == 422
    assert "baseline not found" in missing.json()["detail"]
    assert "--write-baseline" in missing.json()["detail"]
    assert not (tmp_path / "missing.json").exists()

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    broken = client.post("/api/scans", json={"path": ".", "baseline": "broken.json"})
    assert broken.status_code == 422
    assert "invalid JSON" in broken.json()["detail"]

    (tmp_path / "schema.json").write_text(
        json.dumps({"schema": "not-leaklens"}), encoding="utf-8"
    )
    schema = client.post("/api/scans", json={"path": ".", "baseline": "schema.json"})
    assert schema.status_code == 422
    assert "baseline schema" in schema.json()["detail"]

    (tmp_path / "bytes.json").write_bytes(b"\xff\xfe")
    binary = client.post("/api/scans", json={"path": ".", "baseline": "bytes.json"})
    assert binary.status_code == 422
    assert "cannot read baseline" in binary.json()["detail"]

    _write_config(tmp_path, baseline="configured.json")
    configured = client.post("/api/scans", json={"path": "."})
    assert configured.status_code == 422
    assert "baseline not found" in configured.json()["detail"]
    assert not (tmp_path / "configured.json").exists()


def test_dataset_baseline_validation_and_traversal(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)

    for baseline in ("../outside.json", (tmp_path.parent / "outside.json").as_posix()):
        response = client.post(
            "/api/datasets",
            json={"path": "train.csv", "target": "y", "baseline": baseline},
        )
        assert response.status_code == 422, response.text
        assert "outside the workspace root" in response.json()["detail"]

    missing = client.post(
        "/api/datasets",
        json={"path": "train.csv", "target": "y", "baseline": "missing.json"},
    )
    assert missing.status_code == 422
    assert "baseline not found" in missing.json()["detail"]

    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")
    broken = client.post(
        "/api/datasets",
        json={"path": "train.csv", "target": "y", "baseline": "broken.json"},
    )
    assert broken.status_code == 422
    assert "invalid JSON" in broken.json()["detail"]


def test_info_reports_config_and_invalid_config_fails_closed(tmp_path):
    _dataset(tmp_path)
    client = _client(_project(tmp_path))
    _write_config(
        tmp_path,
        baseline="leaklens-baseline.json",
        suppressions=[{"fingerprint": "b" * 64, "reason": "documented"}],
    )
    info = client.get("/api/info").json()["config"]
    assert info == {
        "path": ".leaklens.toml",
        "baseline": "leaklens-baseline.json",
        "suppressions": 1,
        "error": None,
    }

    (tmp_path / ".leaklens.toml").write_text("unknown = true\n", encoding="utf-8")
    info = client.get("/api/info").json()["config"]
    assert info["error"] == "configuration could not be loaded"
    assert "unknown" not in info["error"]
    for endpoint, body in (
        ("/api/scans", {"path": "."}),
        ("/api/datasets", {"path": "train.csv", "target": "y"}),
    ):
        response = client.post(endpoint, json=body)
        assert response.status_code == 422, response.text
        assert "unknown configuration keys" in response.json()["detail"]


def test_source_scan_policy_matches_the_cli(tmp_path, monkeypatch, capsys):
    from leaklens import cli

    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    initial = client.get(f"/api/scans/{scan_id}/results").json()
    target = initial["findings"][0]
    _write_baseline(tmp_path / "baseline.json", _baseline_entries(initial["findings"]))
    _write_config(
        tmp_path,
        baseline="baseline.json",
        suppressions=[{"fingerprint": target["fingerprint"], "reason": "triaged"}],
    )

    scan_id, _ = _run(client)
    workspace = client.get(f"/api/scans/{scan_id}/results").json()

    monkeypatch.chdir(tmp_path)
    code = cli.main(["--scan", ".", "--json"])
    cli_report = json.loads(capsys.readouterr().out)
    assert code in (0, 1)
    workspace_statuses = _statuses(workspace)
    cli_statuses = {
        finding["fingerprint"]: finding.get("status")
        for finding in cli_report["findings"]
    }
    assert workspace_statuses == cli_statuses, (workspace_statuses, cli_statuses)
    assert workspace["trust_score"] == cli_report["trust_score"]


def test_dataset_audit_policy_matches_the_cli(tmp_path, monkeypatch, capsys):
    from leaklens import cli

    _dataset(tmp_path)
    client = _client(tmp_path)
    dataset_id, _ = _run_dataset(client, test_path="test.csv")
    initial = client.get(f"/api/datasets/{dataset_id}/results").json()
    target = initial["findings"][0]
    _write_baseline(tmp_path / "baseline.json", _baseline_entries(initial["findings"]))
    _write_config(
        tmp_path,
        baseline="baseline.json",
        suppressions=[{"fingerprint": target["fingerprint"], "reason": "accepted"}],
    )

    dataset_id, _ = _run_dataset(client, test_path="test.csv")
    workspace = client.get(f"/api/datasets/{dataset_id}/results").json()

    monkeypatch.chdir(tmp_path)
    code = cli.main(
        [
            str(tmp_path / "train.csv"),
            "--target",
            "y",
            "--test",
            str(tmp_path / "test.csv"),
            "--json",
        ]
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert code in (0, 1)
    workspace_statuses = _statuses(workspace)
    cli_statuses = {
        finding["fingerprint"]: finding.get("status")
        for finding in cli_report["findings"]
    }
    assert workspace_statuses == cli_statuses, (workspace_statuses, cli_statuses)
    assert workspace["trust_score"] == cli_report["trust_score"]


# --- interactive evidence graph navigation --------------------------------


def _graph(client, base, item_id, **params):
    response = client.get(f"/api/{base}/{item_id}/graph", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _assert_graph_integrity(graph, evidence, links):
    """Every returned node/edge must already exist in the engine's output."""
    by_id = {item["evidence_id"]: item for item in evidence}
    node_ids = [node["evidence_id"] for node in graph["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    assert set(node_ids) <= set(by_id)
    for node in graph["nodes"]:
        assert node["evidence_id"] == by_id[node["evidence_id"]]["evidence_id"]
        assert node["kind"] == by_id[node["evidence_id"]]["kind"]
        assert node["source"] == by_id[node["evidence_id"]]["source"]

    expected = {
        (link["source_id"], link["target_id"], link["relation"], link["rationale"])
        for link in links
    }
    returned = {
        (edge["source_id"], edge["target_id"], edge["relation"], edge["rationale"])
        for edge in graph["edges"]
    }
    assert returned <= expected
    nodes = {node["evidence_id"]: node for node in graph["nodes"]}
    for edge in graph["edges"]:
        assert edge["source_id"] in nodes and edge["target_id"] in nodes
        assert edge["source_id"] != edge["target_id"]
        assert edge["rationale"]
        assert edge["relation"] in {"supports", "corroborates", "same_anchor"}
        forward = next(
            item
            for item in nodes[edge["source_id"]]["neighbors"]
            if item["evidence_id"] == edge["target_id"]
            and item["relation"] == edge["relation"]
        )
        backward = next(
            item
            for item in nodes[edge["target_id"]]["neighbors"]
            if item["evidence_id"] == edge["source_id"]
            and item["relation"] == edge["relation"]
        )
        assert forward["direction"] == "out" and backward["direction"] == "in"
        assert forward["rationale"] == backward["rationale"] == edge["rationale"]


def test_page_exposes_the_shared_graph_controls(tmp_path):
    page = _client(_project(tmp_path)).get("/").text
    for control in (
        'id="graph-panel"',
        'id="g-open-scan"',
        'id="g-open-dataset"',
        'id="g-kind"',
        'id="g-relation"',
        'id="g-sev"',
        'id="g-status"',
        'id="g-q"',
        'id="g-prev"',
        'id="g-next"',
    ):
        assert control in page


def test_source_graph_serialises_the_existing_evidence(tmp_path):
    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    results = client.get(f"/api/scans/{scan_id}/results").json()
    graph = _graph(client, "scans", scan_id)

    assert graph["available"] is True
    summary = graph["summary"]
    assert summary["nodes"] == len(results["evidence"])
    assert summary["links"] == len(results["links"])
    assert summary["clusters"] == len(results["clusters"])
    assert summary["findings"] == len(results["findings"])
    assert isinstance(results["graph_links_truncated"], bool)
    assert summary["relationships_truncated"] == results["graph_links_truncated"]
    assert summary["invalid_nodes"] == 0
    assert summary["invalid_links"] == 0
    assert summary["dangling_links"] == 0
    assert graph["page"] == {
        "offset": 0,
        "limit": 100,
        "total": summary["nodes"],
        "returned": len(results["evidence"]),
        "next_offset": None,
    }

    findings = {finding["index"]: finding for finding in results["findings"]}
    for node in graph["nodes"]:
        assert node["finding_index"] in findings
        assert node["finding"]["index"] == node["finding_index"]
        assert (
            node["finding"]["fingerprint"]
            == findings[node["finding_index"]]["fingerprint"]
        )
        assert node["severity"] == findings[node["finding_index"]]["severity"]
        assert node["role"] in {"primary", "detector"}
        assert node["status"] == "unclassified"
        assert node["component"] == 0
        assert node["degree"] == len(node["neighbors"])
        assert node["isolated"] is False
        assert node["metadata"]["severity"] == node["severity"]
    _assert_graph_integrity(graph, results["evidence"], results["links"])


def test_graph_navigates_between_findings_and_evidence(tmp_path):
    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    results = client.get(f"/api/scans/{scan_id}/results").json()
    graph = _graph(client, "scans", scan_id)

    finding = results["findings"][0]
    focused = _graph(client, "scans", scan_id, finding=finding["index"])["focus"]
    assert focused["finding_index"] == finding["index"]
    assert focused["role"] != "detector"
    assert focused["finding"]["fingerprint"] == finding["fingerprint"]

    by_id = {node["evidence_id"]: node for node in graph["nodes"]}
    assert (
        by_id[focused["evidence_id"]]["finding"]["fingerprint"]
        == finding["fingerprint"]
    )

    direct = _graph(client, "scans", scan_id, focus=focused["evidence_id"])["focus"]
    assert direct["evidence_id"] == focused["evidence_id"]
    assert direct["neighbors"]
    for neighbor in direct["neighbors"]:
        back = _graph(client, "scans", scan_id, focus=neighbor["evidence_id"])["focus"]
        assert any(
            item["evidence_id"] == direct["evidence_id"]
            and item["relation"] == neighbor["relation"]
            for item in back["neighbors"]
        )

    for unknown in ("scanner-999999", "../../etc/passwd"):
        result = _graph(client, "scans", scan_id, focus=unknown)
        assert result["available"] is True
        assert result["focus_requested"] == unknown
        assert result["focus"] is None
        assert unknown not in {node["evidence_id"] for node in result["nodes"]}


def test_graph_filters_search_and_pagination_are_bounded(tmp_path):
    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    graph = _graph(client, "scans", scan_id)

    picked = graph["filters"]["kinds"][0]
    filtered = _graph(client, "scans", scan_id, kind=picked)
    assert filtered["nodes"]
    assert all(node["kind"] == picked for node in filtered["nodes"])
    assert filtered["page"]["total"] == len(filtered["nodes"])

    high = _graph(client, "scans", scan_id, severity="high")
    assert high["nodes"] and all(node["severity"] == "high" for node in high["nodes"])
    unclassified = _graph(client, "scans", scan_id, status="unclassified")
    assert unclassified["page"]["total"] == graph["summary"]["nodes"]

    target = graph["nodes"][0]
    searched = _graph(client, "scans", scan_id, q=target["evidence_id"].upper())
    assert [node["evidence_id"] for node in searched["nodes"]] == [
        target["evidence_id"]
    ]

    relation = next(
        name for name, count in graph["summary"]["relations"].items() if count
    )
    related = _graph(client, "scans", scan_id, relation=relation)
    assert related["edges"]
    assert all(edge["relation"] == relation for edge in related["edges"])
    incident = {
        evidence_id
        for edge in related["edges"]
        for evidence_id in (edge["source_id"], edge["target_id"])
    }
    assert {node["evidence_id"] for node in related["nodes"]} == incident

    first = _graph(client, "scans", scan_id, limit=1)
    assert len(first["nodes"]) == 1
    assert first["edges"] == []
    assert first["page"]["next_offset"] == 1
    second = _graph(client, "scans", scan_id, limit=1, offset=1)
    assert second["nodes"][0]["evidence_id"] != first["nodes"][0]["evidence_id"]
    assert _graph(client, "scans", scan_id, limit=100_000)["page"]["limit"] == (
        GRAPH_MAX_LIMIT
    )
    assert _graph(client, "scans", scan_id, limit=0)["page"]["limit"] == 1
    beyond = _graph(client, "scans", scan_id, offset=10_000)
    assert beyond["nodes"] == [] and beyond["page"]["next_offset"] is None


def test_clean_source_scan_has_an_explicit_empty_graph(tmp_path):
    client = _client(_project(tmp_path))
    scan_id, _ = _run(client, path="clean.py")
    graph = _graph(client, "scans", scan_id)

    assert graph["available"] is True
    assert graph["nodes"] == [] and graph["edges"] == []
    assert graph["summary"]["nodes"] == 0
    assert graph["summary"]["isolated"] == 0
    assert graph["filters"]["kinds"] == []
    assert graph["page"]["total"] == 0


def test_graph_skips_missing_and_malformed_records_without_inventing():
    results = {
        "evidence": [
            {
                "evidence_id": "a",
                "kind": "k",
                "source": "scanner",
                "description": "detail",
                "location": "file.py:1",
                "finding_index": 0,
                "metadata": [["severity", "high"], ["role", "primary"]],
            },
            {"kind": "missing id"},
            {"evidence_id": "a"},
            "not a record",
        ],
        "links": [
            {
                "source_id": "a",
                "target_id": "ghost",
                "relation": "supports",
                "rationale": "missing endpoint",
            },
            {
                "source_id": "a",
                "target_id": "a",
                "relation": "supports",
                "rationale": "self",
            },
            {
                "source_id": "a",
                "target_id": "a",
                "relation": "invented",
                "rationale": "x",
            },
            {"source_id": "a", "target_id": "a", "relation": "supports"},
            "not a link",
        ],
        "clusters": [],
        "findings": [
            {
                "index": 0,
                "kind": "k",
                "severity": "high",
                "column": "c",
                "detail": "detail",
                "fingerprint": "f" * 64,
            }
        ],
    }
    payload = _graph_payload(results)
    assert payload["available"] is True
    assert payload["summary"]["nodes"] == 1
    assert payload["summary"]["links"] == 0
    assert payload["summary"]["invalid_nodes"] == 3
    assert payload["summary"]["invalid_links"] == 4
    assert payload["summary"]["dangling_links"] == 1
    node = payload["nodes"][0]
    assert node["evidence_id"] == "a"
    assert node["isolated"] is True and node["neighbors"] == []
    assert node["status"] == "unclassified"

    for missing in ({}, {"evidence": "nope"}, {"evidence": [], "links": "nope"}):
        payload = _graph_payload(missing)
        assert payload["available"] is False
        assert isinstance(payload["reason"], str) and payload["reason"]
    invalid = _graph_payload({"evidence": [{"kind": "x"}], "links": [], "clusters": []})
    assert invalid["available"] is False
    assert "no valid evidence nodes" in invalid["reason"]


def test_graph_endpoints_use_session_boundaries(tmp_path, monkeypatch):
    from leaklens import workspace

    root = _project(tmp_path)
    _dataset(tmp_path)
    client = _client(root)
    assert client.get("/api/scans/missing/graph").status_code == 404
    assert client.get("/api/datasets/missing/graph").status_code == 404

    entered, release = threading.Event(), threading.Event()

    def blocking_scan(path):
        entered.set()
        release.wait(timeout=15)
        return []

    monkeypatch.setattr(workspace, "scan_file", blocking_scan)
    scan_id = client.post("/api/scans", json={"path": "."}).json()["scan_id"]
    assert entered.wait(timeout=10)
    assert client.get(f"/api/scans/{scan_id}/graph").status_code == 409
    release.set()
    assert _wait(client, scan_id)["state"] == "done"

    audit_entered, audit_release = threading.Event(), threading.Event()
    real_audit = workspace.run_audit_job

    def blocking_audit(*args, **kwargs):
        audit_entered.set()
        audit_release.wait(timeout=15)
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(workspace, "run_audit_job", blocking_audit)
    response = client.post("/api/datasets", json={"path": "train.csv", "target": "y"})
    assert response.status_code == 202, response.text
    dataset_id = response.json()["dataset_id"]
    assert audit_entered.wait(timeout=10)
    assert client.get(f"/api/datasets/{dataset_id}/graph").status_code == 409
    audit_release.set()
    assert _wait_dataset(client, dataset_id)["state"] == "done"


def test_graph_preserves_suppression_status_and_evidence(tmp_path):
    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    initial = client.get(f"/api/scans/{scan_id}/results").json()
    target = initial["findings"][0]
    _write_baseline(tmp_path / "baseline.json", _baseline_entries(initial["findings"]))
    _write_config(
        tmp_path,
        baseline="baseline.json",
        suppressions=[
            {"fingerprint": target["fingerprint"], "reason": "accepted risk"}
        ],
    )

    scan_id, _ = _run(client)
    results = client.get(f"/api/scans/{scan_id}/results").json()
    graph = _graph(client, "scans", scan_id)
    suppressed = next(
        finding
        for finding in results["findings"]
        if finding["fingerprint"] == target["fingerprint"]
    )
    assert suppressed["status"] == "suppressed"
    assert suppressed["suppression"]["reason"] == "accepted risk"
    assert suppressed["evidence"] == target["evidence"]

    node = next(
        node
        for node in graph["nodes"]
        if node["finding_index"] == suppressed["index"] and node["role"] == "primary"
    )
    assert node["status"] == "suppressed"
    assert node["finding"]["fingerprint"] == target["fingerprint"]
    assert node["finding"]["requires_attention"] is False
    assert graph["summary"]["statuses"]["suppressed"] >= 1

    only_suppressed = _graph(client, "scans", scan_id, status="suppressed")
    assert only_suppressed["nodes"]
    assert all(node["status"] == "suppressed" for node in only_suppressed["nodes"])
    assert suppressed["index"] in {
        node["finding_index"] for node in only_suppressed["nodes"]
    }

    # Reading the graph changed neither policy output nor the trust score.
    assert client.get(f"/api/scans/{scan_id}/results").json() == results
    assert results["trust_score"] == _trust_without(
        results["findings"], target["fingerprint"]
    )


def test_dataset_graph_reuses_nested_intelligence_and_policy(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)
    dataset_id, _ = _run_dataset(client, test_path="test.csv")
    results = client.get(f"/api/datasets/{dataset_id}/results").json()
    intelligence = results["intelligence"]
    graph = _graph(client, "datasets", dataset_id)

    assert graph["available"] is True
    assert graph["summary"]["nodes"] == len(intelligence["evidence"])
    assert graph["summary"]["links"] == len(intelligence["links"])
    assert graph["summary"]["relationships_truncated"] is None
    _assert_graph_integrity(graph, intelligence["evidence"], intelligence["links"])

    anchors = {cluster["cluster_id"] for cluster in intelligence["clusters"]}
    anchored = [node for node in graph["nodes"] if node["cluster_id"]]
    assert anchored
    assert {node["cluster_id"] for node in anchored} <= anchors
    assert any(node["anchor"] for node in anchored)

    finding = results["findings"][0]
    focused = _graph(client, "datasets", dataset_id, finding=finding["index"])["focus"]
    assert focused["finding_index"] == finding["index"]
    assert focused["finding"]["fingerprint"] == finding["fingerprint"]

    _write_config(
        tmp_path,
        suppressions=[{"fingerprint": finding["fingerprint"], "reason": "accepted"}],
    )
    dataset_id, _ = _run_dataset(client, test_path="test.csv")
    rerun = client.get(f"/api/datasets/{dataset_id}/results").json()
    graph = _graph(client, "datasets", dataset_id)
    suppressed = next(
        item for item in graph["nodes"] if item["finding_index"] == finding["index"]
    )
    assert suppressed["status"] == "suppressed"
    assert suppressed["finding"]["suppression"]["reason"] == "accepted"
    assert rerun["findings"][finding["index"]]["status"] == "suppressed"


# --- live updates, history and comparisons --------------------------------


def _sse_events(client, path):
    events = []
    with client.stream("GET", path) as response:
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1].strip())))
    return events


def _fake_finding(fingerprint, **overrides):
    item = {
        "fingerprint": fingerprint,
        "kind": "target_leakage",
        "severity": "high",
        "column": "a.py:1",
        "detail": "feature is the target",
        "file": "a.py",
        "line": 1,
        "index": 0,
        "evidence": [{"kind": "source", "description": "x"}],
    }
    item.update(overrides)
    return item


def _fake_scan_session(manager, scan_id, findings, *, state="done", target="."):
    items = []
    for index, finding in enumerate(findings):
        item = dict(finding)
        item["index"] = index
        items.append(item)
    session = ScanSession(scan_id=scan_id, files=(), include_sql=False, target=target)
    session.state = state
    session.results = {"findings": items, "state": state}
    manager._scans[scan_id] = session
    manager._order.append(scan_id)
    return session


def test_scan_events_stream_progress_completion_and_reconnect(tmp_path, monkeypatch):
    from leaklens import workspace

    entered, release = threading.Event(), threading.Event()

    def blocking_scan(path):
        entered.set()
        release.wait(timeout=15)
        return []

    monkeypatch.setattr(workspace, "scan_file", blocking_scan)
    client = _client(_project(tmp_path))
    scan_id = client.post("/api/scans", json={"path": "."}).json()["scan_id"]
    assert entered.wait(timeout=15)

    with client.stream("GET", f"/api/scans/{scan_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        first = None
        name = None
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                first = (name, json.loads(line.split(":", 1)[1].strip()))
                break

    assert first[0] == "snapshot"
    assert first[1]["type"] == "snapshot"
    assert first[1]["state"] == "running"
    assert first[1]["scan_id"] == scan_id
    assert first[1]["seq"] == 1

    release.set()
    assert _wait(client, scan_id)["state"] == "done"

    tail = _sse_events(client, f"/api/scans/{scan_id}/events")
    assert [name for name, _ in tail] == ["snapshot", "done"]
    assert tail[0][1]["state"] == "done"
    assert tail[-1][1]["type"] == "done"
    assert tail[0][1]["seq"] < tail[1][1]["seq"]


def test_dataset_events_stream_stages_and_engine_failure(tmp_path, monkeypatch):
    from leaklens import workspace

    _dataset(tmp_path)
    entered, release = threading.Event(), threading.Event()
    real = workspace.run_audit_job

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(timeout=15)
        return real(*args, **kwargs)

    monkeypatch.setattr(workspace, "run_audit_job", blocking)
    client = _client(tmp_path)
    started = client.post("/api/datasets", json={"path": "train.csv", "target": "y"})
    assert started.status_code == 202, started.text
    dataset_id = started.json()["dataset_id"]
    assert entered.wait(timeout=15)

    with client.stream("GET", f"/api/datasets/{dataset_id}/events") as response:
        assert response.status_code == 200
        first = None
        name = None
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                first = (name, json.loads(line.split(":", 1)[1].strip()))
                break

    assert first[0] == "snapshot"
    assert first[1]["state"] in {"queued", "auditing"}
    assert first[1]["request"]["target"] == "y"
    release.set()
    assert _wait_dataset(client, dataset_id)["state"] == "done"

    tail = _sse_events(client, f"/api/datasets/{dataset_id}/events")
    assert [name for name, _ in tail] == ["snapshot", "done"]
    assert tail[-1][1]["state"] == "done"

    failed_id, status = _run_dataset(client, target="missing")
    assert status["state"] == "failed"
    failed = _sse_events(client, f"/api/datasets/{failed_id}/events")
    assert [name for name, _ in failed] == ["snapshot", "failure", "done"]
    assert failed[1][1]["failure"]["code"] == "unknown_target"
    assert failed[1][1]["seq"] < failed[2][1]["seq"]


def test_events_reject_unknown_and_stale_sessions(tmp_path, monkeypatch):
    from leaklens import workspace

    client = _client(_project(tmp_path))
    assert client.get("/api/scans/missing/events").status_code == 404
    assert client.get("/api/datasets/missing/events").status_code == 404

    monkeypatch.setattr(workspace, "MAX_SCANS", 1)
    first, _ = _run(client)
    second, _ = _run(client)
    assert [row["id"] for row in client.get("/api/history").json()["scans"]] == [second]
    assert client.get(f"/api/scans/{first}").status_code == 404
    assert client.get(f"/api/scans/{first}/events").status_code == 404


def test_history_is_bounded_in_memory_and_reports_metadata(tmp_path):
    _dataset(tmp_path)
    client = _client(_project(tmp_path))
    scan_id, _ = _run(client)
    dataset_id, _ = _run_dataset(client)

    history = client.get("/api/history").json()
    assert history["retained"] is True
    assert history["persistent"] is False
    assert "restart" in history["note"]
    assert history["limits"]["scans"] >= 1
    assert history["limits"]["datasets"] >= 1
    assert {row["type"] for row in history["entries"]} == {"scan", "dataset"}

    scan = next(row for row in history["entries"] if row["type"] == "scan")
    assert scan["id"] == scan_id
    assert scan["target"] == "."
    assert scan["started_at"] and scan["completed_at"]
    assert scan["duration_ms"] >= 0
    assert scan["results_available"] is True
    assert scan["report_available"] is True
    results = client.get(f"/api/scans/{scan_id}/results").json()
    assert scan["summary"]["findings"] == len(results["findings"])

    dataset = next(row for row in history["entries"] if row["type"] == "dataset")
    assert dataset["id"] == dataset_id
    assert dataset["target"] == "train.csv"
    assert dataset["summary"]["target"] == "y"
    assert dataset["summary"]["rows"] == 120
    assert dataset["results_available"] is True

    # Newest first; a restarted server has no persisted history.
    assert history["entries"][0]["id"] == dataset_id
    fresh = create_workspace_app(tmp_path)
    assert fresh.state.manager.history()["entries"] == []


def test_page_exposes_live_history_and_comparison_controls(tmp_path):
    page = _client(_project(tmp_path)).get("/").text
    for control in (
        'id="history-refresh"',
        'id="history-list"',
        'id="cmp-previous"',
        'id="cmp-current"',
        'id="cmp-run"',
        'id="cmp-summary"',
        'id="cmp-list"',
    ):
        assert control in page
    assert "/events" in page


def test_source_comparison_classifies_without_reinterpreting_policy(tmp_path):
    app = create_workspace_app(tmp_path)
    client = TestClient(app)
    manager = app.state.manager
    fp = {name: char * 64 for name, char in zip("abcde", "abcde")}

    previous = [
        _fake_finding(fp["a"]),
        _fake_finding(fp["b"], column="b.py:1"),
        _fake_finding(fp["c"], column="c.py:1"),
        _fake_finding(
            fp["d"],
            column="d.py:1",
            status="suppressed",
            suppression={"fingerprint": fp["d"], "reason": "accepted"},
            requires_attention=False,
        ),
    ]
    current = [
        _fake_finding(fp["a"]),
        _fake_finding(fp["b"], severity="low", column="b.py:1"),
        _fake_finding(
            fp["d"],
            column="d.py:1",
            status="suppressed",
            suppression={"fingerprint": fp["d"], "reason": "accepted"},
            requires_attention=False,
        ),
        _fake_finding(fp["e"], column="e.py:1"),
    ]
    _fake_scan_session(manager, "prev", previous)
    _fake_scan_session(manager, "curr", current)

    response = client.get(
        "/api/compare/scans", params={"previous": "prev", "current": "curr"}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["available"] is True and data["compatible"] is True
    assert data["type"] == "scan"
    assert data["checks"] and all(check["compatible"] for check in data["checks"])
    summary = data["summary"]
    assert (
        summary["added"],
        summary["unchanged"],
        summary["changed"],
        summary["no_longer_present"],
        summary["inconclusive"],
    ) == (1, 2, 1, 1, 0)
    assert summary["total"] == 5
    assert summary["suppressed"] == 1
    assert summary["unclassified"] == 3
    assert data["counts_consistent"] is True
    assert data["truncated"] is False

    by_change = {entry["change"]: entry for entry in data["changes"]}
    assert by_change["added"]["fingerprint"] == fp["e"]
    assert by_change["added"]["current"]["index"] == 3
    assert by_change["changed"]["previous"]["severity"] == "high"
    assert by_change["changed"]["current"]["severity"] == "low"
    assert by_change["no_longer_present"]["resolved"] is True
    assert by_change["no_longer_present"]["fingerprint"] == fp["c"]
    unchanged = next(
        entry for entry in data["changes"] if entry["fingerprint"] == fp["a"]
    )
    assert unchanged["change"] == "unchanged"
    assert "status" not in unchanged["current"]
    suppressed = next(
        entry for entry in data["changes"] if entry["fingerprint"] == fp["d"]
    )
    assert suppressed["current"]["status"] == "suppressed"
    assert suppressed["current"]["suppression"]["reason"] == "accepted"
    assert suppressed["current"]["requires_attention"] is False

    assert (
        client.get(
            "/api/compare/scans", params={"previous": "missing", "current": "curr"}
        ).status_code
        == 404
    )


def test_comparison_never_marks_findings_from_partial_scans(tmp_path):
    app = create_workspace_app(tmp_path)
    client = TestClient(app)
    manager = app.state.manager
    a, b, c = "a" * 64, "b" * 64, "c" * 64

    _fake_scan_session(manager, "prev", [_fake_finding(a), _fake_finding(b)])
    _fake_scan_session(manager, "curr", [_fake_finding(a)], state="cancelled")
    data = client.get(
        "/api/compare/scans", params={"previous": "prev", "current": "curr"}
    ).json()
    assert data["available"] is True
    assert data["complete"] == {"previous": True, "current": False}
    assert data["summary"]["no_longer_present"] == 0
    assert data["summary"]["inconclusive"] == 1
    unresolved = next(
        entry for entry in data["changes"] if entry["change"] == "inconclusive"
    )
    assert unresolved["resolved"] is False
    assert unresolved["fingerprint"] == b
    assert any("partial" in text for text in data["limitations"])

    _fake_scan_session(manager, "partial", [_fake_finding(a)], state="cancelled")
    _fake_scan_session(manager, "complete", [_fake_finding(a), _fake_finding(c)])
    data = client.get(
        "/api/compare/scans", params={"previous": "partial", "current": "complete"}
    ).json()
    assert data["available"] is True
    assert data["summary"]["added"] == 0
    assert data["summary"]["inconclusive"] == 1
    unresolved = next(
        entry for entry in data["changes"] if entry["change"] == "inconclusive"
    )
    assert unresolved["resolved"] is False
    assert unresolved["fingerprint"] == c


def test_source_comparison_reuses_stable_fingerprints(tmp_path):
    client = _client(_policy_project(tmp_path))
    first, _ = _run(client)
    second, _ = _run(client)
    data = client.get(
        "/api/compare/scans", params={"previous": first, "current": second}
    ).json()
    assert data["available"] is True and data["compatible"] is True
    assert data["summary"]["added"] == 0
    assert data["summary"]["no_longer_present"] == 0
    assert data["summary"]["unchanged"] == data["summary"]["total"] > 0
    assert data["counts_consistent"] is True
    results = client.get(f"/api/scans/{second}/results").json()
    fingerprints = {finding["fingerprint"]: finding for finding in results["findings"]}
    assert {entry["fingerprint"] for entry in data["changes"]} <= set(fingerprints)
    for entry in data["changes"]:
        current = entry["current"]
        assert current["fingerprint"] == entry["fingerprint"]
        assert current["evidence"] == fingerprints[entry["fingerprint"]]["evidence"]


def test_dataset_comparison_requires_matching_configuration(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)
    first, _ = _run_dataset(client, test_path="test.csv")
    second, _ = _run_dataset(client, test_path="test.csv")

    data = client.get(
        "/api/compare/datasets", params={"previous": first, "current": second}
    ).json()
    assert data["available"] is True
    assert data["summary"]["added"] == 0
    assert data["summary"]["no_longer_present"] == 0
    assert data["summary"]["unchanged"] == data["summary"]["total"] > 0
    assert data["counts_consistent"] is True
    results = client.get(f"/api/datasets/{second}/results").json()
    fingerprints = {finding["fingerprint"] for finding in results["findings"]}
    assert {entry["fingerprint"] for entry in data["changes"]} <= fingerprints
    assert all(entry["current"]["evidence"] for entry in data["changes"])

    mismatch, _ = _run_dataset(client, threshold=0.8, test_path="test.csv")
    incompatible = client.get(
        "/api/compare/datasets", params={"previous": second, "current": mismatch}
    ).json()
    assert incompatible["available"] is False
    assert "threshold" in incompatible["reason"]
    assert any(
        check["name"] == "threshold" and check["compatible"] is False
        for check in incompatible["checks"]
    )
    assert (
        client.get(
            "/api/compare/datasets", params={"previous": "missing", "current": second}
        ).status_code
        == 404
    )


# --- M8: opt-in AI surface ------------------------------------------------


def _ai_payload():
    """A minimal valid report used to exercise workspace rendering only."""

    class _Report:
        def to_dict(self):
            return {
                "schema_version": "1.2",
                "engine": "leaklens.ai.experimental",
                "scope": "code",
                "target": None,
                "deterministic_findings": 1,
                "deterministic_trust_score": 90,
                "changes_deterministic_findings": False,
                "hypotheses": [
                    {
                        "hypothesis_id": "hyp-script",
                        "concept": "learned_leakage_pattern",
                        "title": "<script>alert(1)</script>",
                        "explanation": "<img src=x onerror=alert(1)>",
                        "confidence": 0.9,
                        "confidence_basis": "calibrated_probability",
                        "sufficiency": "supported",
                        "severity_hint": "low",
                        "location": "leaky.py:1",
                        "detector": "learned",
                        "corroborates": None,
                        "limitations": ["<b>limit</b>"],
                        "evidence": [],
                    }
                ],
                "plan": {
                    "root_causes": [],
                    "priority_order": [],
                    "summary": "",
                    "verification": [],
                    "enrichment": "deterministic",
                    "fallback_reason": None,
                    "llm_summary": None,
                    "llm_notes": [],
                },
                "models": [],
                "limitations": ["AI hypotheses are experimental."],
            }

    return _Report()


def test_scan_ai_is_off_by_default(tmp_path, monkeypatch):
    from leaklens import ai

    def boom(*_args, **_kwargs):
        raise AssertionError("AI must not run unless requested")

    monkeypatch.setattr(ai, "analyze", boom)
    client = _client(_project(tmp_path))
    scan_id, status = _run(client)

    assert status["state"] == "done"
    results = client.get(f"/api/scans/{scan_id}/results").json()
    assert "ai" not in results and "ai_unavailable" not in results
    assert "AI intelligence" not in client.get(f"/api/scans/{scan_id}/report").text


def test_scan_ai_never_changes_score_findings_or_policy(tmp_path):
    client = _client(_project(tmp_path))
    plain_id, _ = _run(client)
    plain = client.get(f"/api/scans/{plain_id}/results").json()

    ai_id, status = _run(client, include_ai=True)
    assert status["state"] == "done"
    rich = client.get(f"/api/scans/{ai_id}/results").json()

    assert rich["trust_score"] == plain["trust_score"]
    assert rich["counts"] == plain["counts"]
    assert rich["policy"] == plain["policy"]
    assert [
        (f["fingerprint"], f.get("status"), f["severity"], f["evidence"])
        for f in rich["findings"]
    ] == [
        (f["fingerprint"], f.get("status"), f["severity"], f["evidence"])
        for f in plain["findings"]
    ]
    assert rich["ai_unavailable"] is False
    ai = rich["ai"]
    assert ai["changes_deterministic_findings"] is False
    assert ai["deterministic_trust_score"] == rich["trust_score"]
    assert ai["deterministic_findings"] == sum(
        1 for f in rich["findings"] if f.get("status") != "suppressed"
    )
    assert isinstance(ai["hypotheses"], list)
    assert "AI intelligence" in client.get(f"/api/scans/{ai_id}/report").text


def test_source_scan_ai_only_sees_policy_active_findings(tmp_path):
    client = _client(_policy_project(tmp_path))
    scan_id, _ = _run(client)
    findings = client.get(f"/api/scans/{scan_id}/results").json()["findings"]
    target = next(f for f in findings if f["severity"] == "high")
    _write_config(
        tmp_path,
        suppressions=[{"fingerprint": target["fingerprint"], "reason": "known"}],
    )

    scan_id, _ = _run(client, include_ai=True)
    results = client.get(f"/api/scans/{scan_id}/results").json()
    suppressed = {
        f["fingerprint"] for f in results["findings"] if f.get("status") == "suppressed"
    }
    assert target["fingerprint"] in suppressed
    assert results["ai"]["deterministic_findings"] == sum(
        1 for f in results["findings"] if f.get("status") != "suppressed"
    )
    assert results["ai"]["deterministic_trust_score"] == results["trust_score"]


def test_scan_ai_failure_keeps_the_deterministic_result(tmp_path, monkeypatch):
    from leaklens import ai

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated AI failure")

    monkeypatch.setattr(ai, "analyze", boom)
    client = _client(_project(tmp_path))
    scan_id, status = _run(client, include_ai=True)

    assert status["state"] == "done"
    results = client.get(f"/api/scans/{scan_id}/results").json()
    assert results["ai"] is None and results["ai_unavailable"] is True
    assert results["findings"] and results["trust_score"] >= 0
    report = client.get(f"/api/scans/{scan_id}/report").text
    assert "AI intelligence" not in report


def test_workspace_report_escapes_ai_content(tmp_path, monkeypatch):
    from leaklens import ai

    monkeypatch.setattr(ai, "analyze", lambda *_args, **_kwargs: _ai_payload())
    client = _client(_project(tmp_path))
    scan_id, _ = _run(client, include_ai=True)

    assert client.get(f"/api/scans/{scan_id}/results").json()["ai_unavailable"] is False
    report = client.get(f"/api/scans/{scan_id}/report").text
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


def test_dataset_ai_request_is_forwarded_only_when_asked(tmp_path, monkeypatch):
    from leaklens import workspace

    calls = []
    real = workspace.run_audit_job

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(workspace, "run_audit_job", spy)
    _dataset(tmp_path)
    client = _client(tmp_path)
    _run_dataset(client)
    _run_dataset(client, include_ai=True)

    assert calls[0] == {"include_intelligence": True, "include_ai": False}
    assert calls[1] == {"include_intelligence": True, "include_ai": True}


def test_dataset_ai_failure_keeps_the_audit_result(tmp_path, monkeypatch):
    from leaklens import ai

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated AI failure")

    monkeypatch.setattr(ai, "analyze_dataset", boom)
    _dataset(tmp_path)
    client = _client(tmp_path)
    dataset_id, status = _run_dataset(client, include_ai=True)

    assert status["state"] == "done"
    results = client.get(f"/api/datasets/{dataset_id}/results").json()
    assert results["ai"] is None and results["ai_unavailable"] is True
    assert results["findings"] and results["trust_score"] >= 0
    assert (
        "AI intelligence" not in client.get(f"/api/datasets/{dataset_id}/report").text
    )


def test_dataset_ai_never_changes_the_audit(tmp_path):
    _dataset(tmp_path)
    client = _client(tmp_path)
    plain_id, _ = _run_dataset(client)
    plain = client.get(f"/api/datasets/{plain_id}/results").json()

    ai_id, status = _run_dataset(client, include_ai=True)
    rich = client.get(f"/api/datasets/{ai_id}/results").json()

    assert status["state"] == "done"
    assert rich["trust_score"] == plain["trust_score"]
    assert rich["counts"] == plain["counts"]
    assert rich["policy"] == plain["policy"]
    assert [
        (f["fingerprint"], f.get("status"), f["severity"], f["evidence"])
        for f in rich["findings"]
    ] == [
        (f["fingerprint"], f.get("status"), f["severity"], f["evidence"])
        for f in plain["findings"]
    ]
    assert rich["ai_unavailable"] is False
    assert rich["ai"]["changes_deterministic_findings"] is False
    assert rich["ai"]["deterministic_trust_score"] == rich["trust_score"]
    assert "AI intelligence" in client.get(f"/api/datasets/{ai_id}/report").text


def test_ai_on_scan_still_occupies_one_slot(tmp_path, monkeypatch):
    from leaklens import ai

    entered, release = threading.Event(), threading.Event()

    def blocking(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=15)
        raise RuntimeError("simulated AI failure")

    monkeypatch.setattr(ai, "analyze", blocking)
    _dataset(tmp_path)
    client = _client(_project(tmp_path))

    started = client.post("/api/scans", json={"path": ".", "include_ai": True})
    assert started.status_code == 202
    scan_id = started.json()["scan_id"]
    assert entered.wait(timeout=15)

    refused = client.post("/api/datasets", json={"path": "train.csv", "target": "y"})
    assert refused.status_code == 409

    release.set()
    assert _wait(client, scan_id)["state"] == "done"
    assert client.get(f"/api/scans/{scan_id}/results").json()["ai_unavailable"] is True
