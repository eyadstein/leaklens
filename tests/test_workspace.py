"""Interactive workspace: scan lifecycle, path safety and source preview."""

import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from leaklens.workspace import create_workspace_app

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


def _client(tmp_path):
    return TestClient(create_workspace_app(tmp_path))


def _wait(client, scan_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/scans/{scan_id}").json()
        if status["state"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("scan did not finish")


def _run(client, path=".", sql=False):
    response = client.post("/api/scans", json={"path": path, "sql": sql})
    assert response.status_code == 202, response.text
    scan_id = response.json()["scan_id"]
    status = _wait(client, scan_id)
    return scan_id, status


def test_page_and_info_expose_the_root(tmp_path):
    client = _client(_project(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert "LeakLens workspace" in page.text
    info = client.get("/api/info").json()
    assert info["root"] == tmp_path.resolve().as_posix()
    assert info["max_files"] >= 1
    assert info["source_context"] >= 1


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
