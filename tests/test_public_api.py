"""Contract and security tests for the hardened public live-scan API."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from leaklens import __version__
from leaklens.public_api import (
    MAX_SOURCE_BYTES,
    PAGES_ORIGIN,
    BodyLimitMiddleware,
    PublicSettings,
    ScanBusy,
    ScanRunner,
    ScanTimeout,
    _read_bounded,
    create_app,
)
from leaklens.upload_guard import UploadRejected


def settings(**overrides) -> PublicSettings:
    base = {
        "max_bytes": 64 * 1024,
        "max_json_bytes": 32 * 1024,
        "max_rows": 50,
        "max_columns": 10,
        "max_cells": 500,
        "max_uncompressed_bytes": 256 * 1024,
        "formats": ("csv", "json"),
        "timeout_seconds": 5.0,
        "rate_limit_requests": 100,
        "rate_limit_window_seconds": 60,
        "max_rate_keys": 64,
        "allowed_origins": (PAGES_ORIGIN,),
        "allow_localhost_origins": True,
        "trust_proxy": False,
        "enable_docs": False,
    }
    base.update(overrides)
    return PublicSettings(**base)


def client_for(
    config: PublicSettings | None = None,
    *,
    raise_server_exceptions: bool = True,
    **kwargs,
) -> TestClient:
    app = create_app(config or settings(), **kwargs)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


@pytest.fixture()
def client():
    test_client = client_for()
    yield test_client
    test_client.app.state.runner.close()


def leaky_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(size=rows)
    y = (x + rng.normal(scale=0.2, size=rows) > 0).astype(int)
    y[0], y[1] = 0, 1
    return pd.DataFrame({"x": x, "y": y, "leak": y * 10.0})


def clean_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "z": rng.normal(size=rows),
            "y": rng.integers(0, 2, size=rows),
        }
    )


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def multipart(
    name: str = "data.csv", data: bytes | None = None, mime: str = "text/csv"
):
    return {
        "file": (name, data if data is not None else csv_bytes(leaky_frame()), mime)
    }


# --- service discovery ----------------------------------------------------


def test_health_and_index_identify_a_live_service(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "leaklens-public-api",
        "version": __version__,
        "live": True,
    }
    index = client.get("/")
    assert index.status_code == 200
    assert index.json()["live"] is True
    assert index.json()["capabilities"] == "/capabilities"


def test_capabilities_describe_real_limits_and_formats(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["live"] is True
    assert payload["engine"] == "leaklens.python"
    assert payload["storage"] == "none"
    assert payload["endpoints"] == ["/inspect", "/audit", "/scan"]
    assert payload["limits"]["max_rows"] == 50
    assert payload["limits"]["max_source_bytes"] == MAX_SOURCE_BYTES
    by_extension = {item["extension"]: item for item in payload["formats"]}
    assert by_extension[".csv"]["enabled"] is True
    assert by_extension[".xlsx"]["configured"] is False
    assert "never retained" in payload["privacy"]


def test_legacy_predict_endpoint_is_not_exposed(client):
    response = client.get("/predict")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
    assert client.post("/predict", json={"rows": []}).status_code == 404


def test_api_docs_are_off_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    enabled = client_for(settings(enable_docs=True))
    try:
        assert enabled.get("/docs").status_code == 200
        assert enabled.get("/openapi.json").status_code == 200
    finally:
        enabled.app.state.runner.close()


# --- response hardening ---------------------------------------------------


def test_security_headers_are_present_on_success_and_error(client):
    for response in (client.get("/health"), client.get("/nope")):
        headers = response.headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert len(headers["x-request-id"]) == 16


def test_request_ids_are_unique(client):
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    assert first != second


def test_cors_allows_the_pages_origin_and_localhost_only(client):
    allowed = client.get("/health", headers={"Origin": PAGES_ORIGIN})
    assert allowed.headers["access-control-allow-origin"] == PAGES_ORIGIN
    localhost = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert localhost.headers["access-control-allow-origin"] == "http://localhost:5173"
    denied = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_cors_preflight_is_answered(client):
    response = client.options(
        "/audit",
        headers={
            "Origin": PAGES_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == PAGES_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]


def test_cors_headers_are_added_to_upload_errors(client):
    response = client.post(
        "/inspect",
        files=multipart(name="data.txt", data=b"a,b\n1,2\n"),
        headers={"Origin": PAGES_ORIGIN},
    )
    assert response.status_code == 415
    assert response.headers["access-control-allow-origin"] == PAGES_ORIGIN


# --- inspect --------------------------------------------------------------


def test_inspect_returns_bounded_columns_preview_and_candidates(client):
    frame = leaky_frame()
    response = client.post("/inspect", files=multipart(data=csv_bytes(frame)))
    assert response.status_code == 200
    payload = response.json()
    assert payload["file"] == "data.csv"
    assert payload["rows"] == len(frame)
    assert [column["name"] for column in payload["columns"]] == ["x", "y", "leak"]
    assert len(payload["preview"]) == 5
    assert "y" in payload["target_candidates"]
    assert "x" not in payload["target_candidates"]


def test_inspect_accepts_json_records(client):
    body = b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]'
    response = client.post(
        "/inspect",
        files=multipart(name="data.json", data=body, mime="application/json"),
    )
    assert response.status_code == 200
    assert response.json()["rows"] == 2


def test_inspect_rejects_traversal_and_unsupported_names(client):
    traversal = client.post(
        "/inspect",
        files=multipart(name="../../etc/passwd.csv", data=csv_bytes(leaky_frame())),
    )
    assert traversal.status_code == 400
    assert traversal.json()["error"]["code"] == "invalid_filename"

    unsupported = client.post(
        "/inspect", files=multipart(name="data.exe", data=b"MZ\x00\x00")
    )
    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "unsupported_format"


def test_inspect_rejects_content_that_does_not_match_the_extension(client):
    lying = client.post(
        "/inspect", files=multipart(name="data.csv", data=b'[{"a": 1}, {"a": 2}]')
    )
    assert lying.status_code == 415
    assert lying.json()["error"]["code"] == "content_mismatch"

    disabled = client.post(
        "/inspect", files=multipart(name="data.xlsx", data=b"PK\x03\x04fake")
    )
    assert disabled.status_code == 415
    assert disabled.json()["error"]["code"] == "format_disabled"


def test_inspect_rejects_empty_and_oversized_uploads(client):
    empty = client.post("/inspect", files=multipart(data=b""))
    assert empty.status_code == 422
    assert empty.json()["error"]["code"] == "empty_upload"

    oversized = client.post(
        "/inspect", files=multipart(data=b"a,b\n" + b"1,2\n" * 20_000)
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] in {
        "upload_too_large",
        "request_too_large",
    }


def test_inspect_enforces_row_column_and_cell_limits():
    for overrides, frame, code in (
        ({"max_rows": 10}, leaky_frame(rows=20), "dataset_limit"),
        (
            {"max_columns": 3},
            pd.DataFrame({f"c{i}": [1, 2, 3] for i in range(5)}),
            "too_many_columns",
        ),
        (
            {"max_cells": 10},
            pd.DataFrame({f"c{i}": [1, 2, 3] for i in range(5)}),
            "too_many_cells",
        ),
    ):
        test_client = client_for(settings(**overrides))
        try:
            response = test_client.post(
                "/inspect", files=multipart(data=csv_bytes(frame))
            )
        finally:
            test_client.app.state.runner.close()
        assert response.status_code == 413, response.text
        assert response.json()["error"]["code"] == code


def test_inspect_rejects_a_binary_csv(client):
    response = client.post(
        "/inspect", files=multipart(data=b"\x00\x01\x02PK\x03\x04" + b"\x00" * 20)
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "content_mismatch"


# --- audit ----------------------------------------------------------------


def test_audit_runs_real_detectors_and_returns_a_report(client):
    response = client.post("/audit", files=multipart(), data={"target": "y"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["live"] is True
    assert payload["engine"] == "leaklens.python"
    assert payload["version"] == __version__
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert "target_leakage" in kinds
    expected = max(
        0,
        100
        - sum(
            {"high": 35, "medium": 15, "low": 5}[finding["severity"]]
            for finding in payload["findings"]
        ),
    )
    assert payload["trust_score"] == expected
    assert payload["trust_score"] < 100
    assert payload["counts"]["high"] >= 1
    assert payload["counts"]["total"] == len(payload["findings"])
    assert payload["summary"]["target"] == "y"
    assert payload["summary"]["rows"] == 40
    assert payload["summary"]["threshold"] == 0.95
    assert payload["summary"]["duration_ms"] >= 0
    assert payload["audit"]["rows"] == 40
    assert payload["impact"] is None
    leakage_columns = {
        item["column"]
        for item in payload["findings"]
        if item["kind"] == "target_leakage"
    }
    assert leakage_columns & {"x", "leak"}
    finding = next(
        item for item in payload["findings"] if item["kind"] == "target_leakage"
    )
    assert finding["column"] in {"x", "leak"}
    assert finding["fix"]
    assert payload["report_html"].startswith("<!doctype html>")
    assert "LeakLens report" in payload["report_html"]


def test_audit_can_return_a_clean_dataset_without_inventing_findings(client):
    response = client.post(
        "/audit", files=multipart(data=csv_bytes(clean_frame())), data={"target": "y"}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["trust_score"] == 100
    assert payload["findings"] == []
    assert payload["counts"]["total"] == 0


def test_audit_intelligence_is_opt_in_and_full_graph(client):
    plain = client.post("/audit", files=multipart(), data={"target": "y"})
    assert plain.status_code == 200
    assert "intelligence" not in plain.json()
    assert "Evidence clusters" not in plain.json()["report_html"]

    response = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y", "include_intelligence": "true"},
    )
    assert response.status_code == 200
    payload = response.json()
    graph = payload["intelligence"]

    assert payload["trust_score"] == plain.json()["trust_score"]
    assert payload["findings"] == plain.json()["findings"]
    assert set(graph) >= {"evidence", "clusters", "links"}
    assert graph["clusters"]
    assert "Evidence clusters" in payload["report_html"]
    assert "Evidence links" in payload["report_html"]


def test_audit_measures_held_out_overlap_without_estimating_performance(client):
    train = leaky_frame()
    held_out = train.head(10).copy()
    response = client.post(
        "/audit",
        files=multipart(data=csv_bytes(train)),
        data={"target": "y", "threshold": "0.9"},
        # httpx needs repeated keys for two files.
    )
    assert response.status_code == 200
    assert response.json()["impact"] is None

    files = {
        "file": ("train.csv", csv_bytes(train), "text/csv"),
        "test": ("test.csv", csv_bytes(held_out), "text/csv"),
    }
    response = client.post("/audit", files=files, data={"target": "y"})
    assert response.status_code == 200, response.text
    payload = response.json()
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert "split_overlap" in kinds
    assert payload["impact"]["train_rows"] == len(train)
    assert payload["impact"]["test_rows"] == len(held_out)
    assert payload["impact"]["duplicate_test_rows"] == len(held_out)
    assert payload["impact"]["duplicate_test_pct"] == 100.0
    assert payload["summary"]["test_rows"] == len(held_out)


def test_audit_measures_group_and_temporal_overlap(client):
    train = leaky_frame()
    train["account"] = [f"g{index % 6}" for index in range(len(train))]
    train["when"] = pd.date_range("2024-01-01", periods=len(train), freq="D")
    test = pd.DataFrame(
        {
            "x": 100 + np.arange(10, dtype=float),
            "y": [0, 1] * 5,
            "leak": [0.0, 10.0] * 5,
            "account": [f"g{index % 6}" for index in range(10)],
            "when": pd.date_range("2024-01-05", periods=10, freq="D"),
        }
    )
    files = {
        "file": ("train.csv", csv_bytes(train), "text/csv"),
        "test": ("test.csv", csv_bytes(test), "text/csv"),
    }
    response = client.post(
        "/audit", files=files, data={"target": "y", "group": "account", "time": "when"}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert "group_overlap" in kinds
    assert "group_leakage_risk" in kinds
    assert "temporal_leakage" in kinds
    assert payload["impact"]["group_column"] == "account"
    assert payload["impact"]["shared_group_count"] == 6
    assert payload["impact"]["shared_group_test_rows"] == 10
    assert payload["impact"]["time_column"] == "when"
    assert payload["impact"]["temporal_overlap"] is True


def test_audit_rejects_unknown_or_invalid_options(client):
    unknown = client.post(
        "/audit", files=multipart(), data={"target": "definitely-missing"}
    )
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "unknown_target"

    missing = client.post("/audit", files=multipart(), data={})
    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "invalid_request"

    constant = client.post(
        "/audit",
        files=multipart(data=csv_bytes(pd.DataFrame({"x": [1, 2, 3], "y": [1, 1, 1]}))),
        data={"target": "y"},
    )
    assert constant.status_code == 422
    assert constant.json()["error"]["code"] == "constant_target"

    bad = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y", "threshold": "0.2"},
    )
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "bad_threshold"

    not_a_number = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y", "threshold": "abc"},
    )
    assert not_a_number.status_code == 422
    assert not_a_number.json()["error"]["code"] == "invalid_request"

    bad_group = client.post(
        "/audit", files=multipart(), data={"target": "y", "group": "missing"}
    )
    assert bad_group.status_code == 422
    assert bad_group.json()["error"]["code"] == "unknown_option_column"

    control = client.post(
        "/audit", files=multipart(), data={"target": "y", "group": "bad\nname"}
    )
    assert control.status_code == 422
    assert control.json()["error"]["code"] == "bad_option"


def test_audit_requires_at_least_two_columns(client):
    frame = pd.DataFrame({"y": [0, 1, 0, 1]})
    response = client.post(
        "/audit", files=multipart(data=csv_bytes(frame)), data={"target": "y"}
    )
    assert response.status_code == 415  # the CSV sniffer rejects one column first
    assert response.json()["error"]["code"] == "content_mismatch"


# --- code scanning --------------------------------------------------------


def test_scan_runs_the_real_static_scanner(client):
    source = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "scaler = StandardScaler().fit(X)\n"
    )
    response = client.post("/scan", json={"source": source, "name": "leaky.py"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["live"] is True
    assert payload["engine"] == "leaklens.scanner.scan_source"
    assert payload["version"] == __version__
    assert payload["syntax_error"] is False
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert kinds
    assert payload["trust_score"] == max(
        0,
        100
        - sum(
            {"high": 35, "medium": 15, "low": 5}[f["severity"]]
            for f in payload["findings"]
        ),
    )
    for finding in payload["findings"]:
        assert finding["fix"] or finding["kind"] == "parse_error"
    assert payload["summary"]["file"] == "leaky.py"
    assert payload["summary"]["bytes"] == len(source.encode())


def test_scan_reports_syntax_errors_as_findings(client):
    response = client.post("/scan", json={"source": "def broken(:\n"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["syntax_error"] is True
    assert payload["findings"][0]["kind"] == "parse_error"
    assert payload["findings"][0]["fix"]


def test_scan_intelligence_is_opt_in_and_additive(client):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    )

    plain = client.post("/scan", json={"source": source})
    assert plain.status_code == 200
    assert "intelligence" not in plain.json()

    response = client.post(
        "/scan", json={"source": source, "include_intelligence": True}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["trust_score"] == plain.json()["trust_score"]
    assert payload["findings"] == plain.json()["findings"]
    assert set(payload["intelligence"]) >= {"evidence", "clusters", "links"}
    assert payload["intelligence"]["clusters"]


def test_scan_graph_baseline_is_opt_in_and_additive(client):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    )
    plain = client.post("/scan", json={"source": source}).json()
    snapshot = client.post(
        "/scan", json={"source": source, "include_graph_snapshot": True}
    ).json()["graph_snapshot"]
    compared = client.post(
        "/scan", json={"source": source, "graph_baseline": snapshot}
    ).json()

    assert "graph_baseline" not in plain
    assert "graph_snapshot" not in plain
    assert compared["trust_score"] == plain["trust_score"]
    assert compared["findings"] == plain["findings"]
    assert compared["graph_baseline"]["counts"]["new"] == 0
    assert compared["graph_baseline"]["resolution_complete"] is True


def test_scan_rejects_invalid_or_oversized_graph_baselines(client):
    invalid = client.post(
        "/scan", json={"source": "x = 1\n", "graph_baseline": {"x": 1}}
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_graph_baseline"

    from leaklens import public_api

    oversized = {"evidence": "x" * 32}
    original = public_api.MAX_GRAPH_SNAPSHOT_BYTES
    public_api.MAX_GRAPH_SNAPSHOT_BYTES = 8
    try:
        too_big = client.post(
            "/scan", json={"source": "x = 1\n", "graph_baseline": oversized}
        )
    finally:
        public_api.MAX_GRAPH_SNAPSHOT_BYTES = original
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "graph_baseline_too_large"


def test_audit_graph_baseline_is_opt_in_and_additive(client):
    files = multipart()
    plain = client.post("/audit", files=files, data={"target": "y"}).json()
    snapshot = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y", "include_graph_snapshot": "true"},
    ).json()["graph_snapshot"]
    compared = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y", "graph_baseline": json.dumps(snapshot)},
    ).json()

    assert "graph_baseline" not in plain
    assert compared["trust_score"] == plain["trust_score"]
    assert compared["findings"] == plain["findings"]
    assert compared["graph_baseline"]["counts"]["new"] == 0


def test_scan_rejects_empty_and_oversized_sources():
    test_client = client_for(settings(max_json_bytes=1024 * 1024))
    try:
        empty = test_client.post("/scan", json={"source": ""})
        assert empty.status_code == 422
        assert empty.json()["error"]["code"] == "invalid_request"

        huge = test_client.post("/scan", json={"source": "x = 1\n" * 50_000})
        assert huge.status_code == 413
        assert huge.json()["error"]["code"] == "source_too_large"
    finally:
        test_client.app.state.runner.close()


def test_scan_normalises_a_non_python_label(client):
    response = client.post("/scan", json={"source": "x = 1\n", "name": "notes.txt"})
    assert response.status_code == 200
    assert response.json()["summary"]["file"] == "snippet.py"


# --- request limits and failure modes ------------------------------------


def test_rate_limit_returns_429_with_retry_after():
    test_client = client_for(settings(rate_limit_requests=1))
    try:
        assert test_client.post("/scan", json={"source": "x = 1\n"}).status_code == 200
        limited = test_client.post("/scan", json={"source": "x = 1\n"})
    finally:
        test_client.app.state.runner.close()
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.headers["x-ratelimit-limit"] == "1"


def test_rate_limit_reads_are_not_restricted():
    test_client = client_for(settings(rate_limit_requests=1))
    try:
        assert test_client.post("/scan", json={"source": "x = 1\n"}).status_code == 200
        for _ in range(3):
            assert test_client.get("/capabilities").status_code == 200
    finally:
        test_client.app.state.runner.close()


def test_timeout_returns_504_and_busy_until_the_worker_finishes(monkeypatch):
    release = threading.Event()
    started = threading.Event()

    def slow(*_args, **_kwargs):
        started.set()
        release.wait(timeout=5)
        return {"rows": 1}

    monkeypatch.setattr("leaklens.public_api.run_inspect_job", slow)

    runner = ScanRunner(timeout_seconds=0.05)
    test_client = client_for(runner=runner)
    try:
        timed_out = test_client.post("/inspect", files=multipart())
        assert timed_out.status_code == 504
        assert timed_out.json()["error"]["code"] == "scan_timeout"
        assert started.wait(timeout=1)

        busy = test_client.post("/inspect", files=multipart())
        assert busy.status_code == 503
        assert busy.json()["error"]["code"] == "scan_busy"
        assert busy.headers["retry-after"] == "5"

        release.set()
        deadline = time.time() + 3
        while time.time() < deadline:
            if runner._slot.acquire(blocking=False):  # noqa: SLF001 - asserting release
                runner._slot.release()
                break
            time.sleep(0.01)
        else:  # pragma: no cover - failure path
            pytest.fail("worker slot was not released after the timed-out job finished")

        recovered = test_client.post("/inspect", files=multipart())
        assert recovered.status_code == 200
    finally:
        release.set()
        test_client.app.state.runner.close()


def test_unexpected_errors_are_generic(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret path C:\\private\\model.pkl and token abc123")

    monkeypatch.setattr("leaklens.public_api.run_code_job", boom)
    test_client = client_for(raise_server_exceptions=False)
    try:
        response = test_client.post("/scan", json={"source": "x = 1\n"})
    finally:
        test_client.app.state.runner.close()
    assert response.status_code == 500
    body = response.text
    assert "secret" not in body
    assert "private" not in body
    assert "abc123" not in body
    assert "Traceback" not in body
    assert response.json()["error"]["code"] == "internal_error"


def test_unknown_fields_are_named_without_echoing_values(client):
    response = client.post("/scan", json={"source": "x = 1\n", "unexpected": "value"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


# --- body limit middleware ------------------------------------------------


def _asgi_scope(path: str, headers: dict[str, str] | None = None) -> dict:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "scheme": "https",
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ],
        "client": ("203.0.113.9", 1234),
    }


def _run_asgi(app, scope, chunks):
    sent = []

    async def receive():
        if chunks:
            body = chunks.pop(0)
            return {
                "type": "http.request",
                "body": body,
                "more_body": bool(chunks),
            }
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def _status(sent) -> int:
    return next(
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    )


def test_body_limit_rejects_a_declared_oversized_request():
    config = settings()
    seen = []

    async def inner(scope, receive, send):
        seen.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodyLimitMiddleware(inner, config)
    sent = _run_asgi(
        middleware,
        _asgi_scope("/audit", {"content-length": str(config.max_request_bytes + 1)}),
        [],
    )
    assert _status(sent) == 413
    assert seen == []


def test_body_limit_rejects_invalid_content_length():
    config = settings()

    async def inner(scope, receive, send):  # pragma: no cover - must not run
        raise AssertionError("inner app must not run")

    middleware = BodyLimitMiddleware(inner, config)
    assert (
        _status(
            _run_asgi(middleware, _asgi_scope("/audit", {"content-length": "abc"}), [])
        )
        == 400
    )
    assert (
        _status(
            _run_asgi(middleware, _asgi_scope("/audit", {"content-length": "-4"}), [])
        )
        == 400
    )


def test_body_limit_counts_chunked_bodies_without_a_length():
    config = settings()
    half = config.max_request_bytes // 2 + 1

    async def inner(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodyLimitMiddleware(inner, config)
    sent = _run_asgi(middleware, _asgi_scope("/audit"), [b"x" * half, b"x" * half])
    assert _status(sent) == 413


def test_scan_has_a_smaller_body_limit_than_uploads():
    config = settings()
    middleware = BodyLimitMiddleware(lambda scope, receive, send: None, config)
    assert middleware._limit("/scan") == config.max_json_bytes
    assert middleware._limit("/audit") == config.max_request_bytes
    assert middleware._limit("/inspect") == config.max_request_bytes


# --- ScanRunner -----------------------------------------------------------


def test_scan_runner_returns_results_and_releases_the_slot():
    runner = ScanRunner(timeout_seconds=1)
    try:
        assert runner.run(lambda value: value * 2, 21) == 42
        assert runner.run(lambda: "again") == "again"
    finally:
        runner.close()


def test_scan_runner_releases_the_slot_after_an_exception():
    runner = ScanRunner(timeout_seconds=1)

    def fail():
        raise ValueError("nope")

    try:
        with pytest.raises(ValueError):
            runner.run(fail)
        assert runner.run(lambda: "ok") == "ok"
    finally:
        runner.close()


def test_scan_runner_rejects_overlapping_work():
    runner = ScanRunner(timeout_seconds=1)
    assert runner._slot.acquire(blocking=False)  # noqa: SLF001
    try:
        with pytest.raises(ScanBusy):
            runner.run(lambda: 1)
    finally:
        runner._slot.release()  # noqa: SLF001
        runner.close()


def test_scan_runner_timeout_is_distinct_from_busy():
    runner = ScanRunner(timeout_seconds=0.01)
    try:
        with pytest.raises(ScanTimeout):
            runner.run(time.sleep, 0.2)
    finally:
        runner.close()


# --- upload reading and cleanup ------------------------------------------


class _FakeUpload:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    async def read(self, _size):
        return self._chunks.pop(0) if self._chunks else b""

    async def close(self):
        self.closed = True


def test_read_bounded_stops_at_the_limit_and_always_closes():
    upload = _FakeUpload([b"a" * 10, b"b" * 10])
    assert asyncio.run(_read_bounded(upload, 100)) == b"a" * 10 + b"b" * 10
    assert upload.closed is True

    oversized = _FakeUpload([b"a" * 12, b"b" * 12])
    with pytest.raises(UploadRejected) as excinfo:
        asyncio.run(_read_bounded(oversized, 20))
    assert excinfo.value.status_code == 413
    assert oversized.closed is True


def test_upload_is_parsed_from_memory_not_a_filesystem_path(monkeypatch):
    import leaklens.upload_guard as guard

    seen = []
    original = guard.read_dataset

    def wrapper(source, name, *args, **kwargs):
        seen.append((type(source).__name__, str(name)))
        return original(source, name, *args, **kwargs)

    monkeypatch.setattr(guard, "read_dataset", wrapper)
    test_client = client_for()
    try:
        response = test_client.post("/inspect", files=multipart())
    finally:
        test_client.app.state.runner.close()
    assert response.status_code == 200
    assert seen == [("BytesIO", "data.csv")]


# --- environment settings -------------------------------------------------


def test_settings_from_env_clamps_and_falls_back(monkeypatch):
    monkeypatch.setenv("LEAKLENS_PUBLIC_MAX_BYTES", "1")
    monkeypatch.setenv("LEAKLENS_PUBLIC_MAX_ROWS", "999999")
    monkeypatch.setenv("LEAKLENS_PUBLIC_MAX_COLUMNS", "999999")
    monkeypatch.setenv("LEAKLENS_PUBLIC_MAX_CELLS", "999999999")
    monkeypatch.setenv("LEAKLENS_PUBLIC_TIMEOUT_SECONDS", "nan")
    monkeypatch.setenv("LEAKLENS_PUBLIC_RATE_LIMIT", "0")
    monkeypatch.setenv("LEAKLENS_PUBLIC_RATE_WINDOW", "1")
    monkeypatch.setenv("LEAKLENS_PUBLIC_MAX_UNCOMPRESSED_BYTES", "1")
    config = PublicSettings.from_env()
    assert config.max_bytes == 64 * 1024
    assert config.max_rows == 20_000
    assert config.max_columns == 500
    assert config.max_cells == 5_000_000
    assert config.timeout_seconds == 30.0
    assert config.rate_limit_requests == 1
    assert config.rate_limit_window_seconds == 10
    assert config.max_uncompressed_bytes == 1024 * 1024


def test_settings_from_env_ignores_bad_numbers(monkeypatch):
    monkeypatch.setenv("LEAKLENS_PUBLIC_MAX_BYTES", "not-a-number")
    monkeypatch.setenv("LEAKLENS_PUBLIC_TIMEOUT_SECONDS", "infinite")
    config = PublicSettings.from_env()
    assert config.max_bytes == 4 * 1024 * 1024
    assert config.timeout_seconds == 30.0


def test_settings_from_env_parses_formats_and_origins(monkeypatch):
    monkeypatch.setenv("LEAKLENS_PUBLIC_FORMATS", "xlsx, csv,docx")
    monkeypatch.setenv(
        "LEAKLENS_ALLOWED_ORIGINS",
        "https://one.example, ftp://bad.example, http://localhost:3000",
    )
    config = PublicSettings.from_env()
    assert config.formats == ("csv", "xlsx")
    assert config.allowed_origins == ("https://one.example", "http://localhost:3000")
    assert config.max_request_bytes == config.max_bytes * 2 + 512 * 1024


def test_settings_reject_an_empty_format_and_origin_list(monkeypatch):
    monkeypatch.setenv("LEAKLENS_PUBLIC_FORMATS", ",")
    monkeypatch.setenv("LEAKLENS_ALLOWED_ORIGINS", "ftp://bad.example")
    config = PublicSettings.from_env()
    assert config.formats == ("csv", "json")
    assert config.allowed_origins == (PAGES_ORIGIN,)


# --- optional AI layer ----------------------------------------------------


def test_ai_is_opt_in_for_dataset_audits(client):
    plain = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y"},
    )
    assert plain.status_code == 200
    assert "ai" not in plain.json()
    assert "AI intelligence (experimental)" not in plain.json()["report_html"]

    response = client.post(
        "/audit",
        files=multipart(),
        data={"target": "y", "include_ai": "true"},
    )
    assert response.status_code == 200
    payload = response.json()
    ai = payload["ai"]

    assert payload["ai_unavailable"] is False
    assert ai["schema_version"] == "1.2"
    assert ai["engine"] == "leaklens.ai.experimental"
    assert ai["scope"] == "dataset"
    assert ai["target"] == "y"
    assert ai["deterministic_trust_score"] == payload["trust_score"]
    assert ai["deterministic_findings"] == len(payload["findings"])
    assert ai["changes_deterministic_findings"] is False
    assert ai["models"]
    assert ai["limitations"]
    assert "AI intelligence (experimental)" in payload["report_html"]
    assert "Deterministic trust score (unchanged)" in payload["report_html"]


def test_ai_is_opt_in_for_code_scans_and_cannot_change_the_score(client):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    )

    plain = client.post("/scan", json={"source": source})
    assert plain.status_code == 200
    assert "ai" not in plain.json()

    response = client.post("/scan", json={"source": source, "include_ai": True})
    assert response.status_code == 200
    payload = response.json()
    ai = payload["ai"]

    assert ai["scope"] == "code"
    assert ai["deterministic_trust_score"] == payload["trust_score"]
    assert ai["deterministic_findings"] == len(payload["findings"])
    assert ai["changes_deterministic_findings"] is False
    assert payload["trust_score"] == plain.json()["trust_score"]
    assert payload["findings"] == plain.json()["findings"]


def test_ai_evidence_links_only_to_real_graph_nodes(client):
    source = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    )

    response = client.post(
        "/scan",
        json={"source": source, "include_ai": True, "include_intelligence": True},
    )

    assert response.status_code == 200
    payload = response.json()
    graph_ids = {item["evidence_id"] for item in payload["intelligence"]["evidence"]}
    linked = [
        span
        for hypothesis in payload["ai"]["hypotheses"]
        for span in hypothesis["evidence"]
        if span["graph_node_id"]
    ]

    assert linked
    assert all(span["graph_node_id"] in graph_ids for span in linked)


def test_capabilities_advertise_ai_as_opt_in_and_offline(client):
    payload = client.get("/capabilities").json()
    ai = payload["ai"]

    assert ai["enabled"] is True
    assert ai["opt_in"] is True
    assert ai["network"] == "none"
    assert ai["llm"] is False
    assert ai["changes_deterministic_findings"] is False


def test_capabilities_advertise_intelligence_as_opt_in(client):
    intelligence = client.get("/capabilities").json()["intelligence"]

    assert intelligence == {
        "enabled": True,
        "opt_in": True,
        "scope": ["dataset", "code"],
        "engine": "leaklens.intelligence.deterministic",
        "graph": ["evidence", "clusters", "links"],
        "changes_deterministic_findings": False,
        "graph_baseline": {
            "enabled": True,
            "opt_in": True,
            "field": "graph_baseline",
            "max_bytes": 2 * 1024 * 1024,
            "snapshot_field": "include_graph_snapshot",
        },
    }
