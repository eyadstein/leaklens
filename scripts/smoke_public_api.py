"""Real-HTTP smoke test for the hardened public scan API.

By default this starts a local ``uvicorn leaklens.public_api:app`` subprocess
with conservative limits, exercises the real endpoints over the network, and
shuts the server down.  Pass ``--url`` to test an already-running deployment
instead (the smoke test never writes anything to that server).

Usage:
    python scripts/smoke_public_api.py
    python scripts/smoke_public_api.py --url https://your-api.example.com
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PAGES_ORIGIN = "https://eyadstein.github.io"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(base: str, timeout: float) -> str | None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/health", timeout=2, trust_env=False)
            if response.status_code == 200 and response.json().get("live") is True:
                return None
            last_error = f"HTTP {response.status_code}: {response.text[:120]}"
        except Exception as exc:  # noqa: BLE001 - report any startup failure
            last_error = type(exc).__name__
        time.sleep(0.25)
    return last_error or "no response"


def dataset(rows: int) -> bytes:
    lines = ["x,y,leak"]
    for index in range(rows):
        label = index % 2
        lines.append(f"{index / rows:.4f},{label},{float(label) * 10:.1f}")
    return ("\n".join(lines) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="test an existing API instead of starting one")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="startup/wake timeout"
    )
    args = parser.parse_args()

    process = None
    log = None
    if args.url:
        base = args.url.rstrip("/")
    else:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        log = tempfile.TemporaryFile(  # noqa: SIM115 - closed in finally
            mode="w+", encoding="utf-8", errors="replace"
        )
        env = {
            **os.environ,
            "LEAKLENS_PUBLIC_MAX_BYTES": str(64 * 1024),
            "LEAKLENS_PUBLIC_MAX_JSON_BYTES": str(1024 * 1024),
            "LEAKLENS_PUBLIC_MAX_ROWS": "200",
            "LEAKLENS_PUBLIC_MAX_COLUMNS": "50",
            "LEAKLENS_PUBLIC_MAX_CELLS": "50000",
            "LEAKLENS_PUBLIC_RATE_LIMIT": "100",
            "LEAKLENS_ENABLE_DOCS": "0",
        }
        process = subprocess.Popen(  # noqa: S603 - fixed local interpreter and arguments
            [
                sys.executable,
                "-m",
                "uvicorn",
                "leaklens.public_api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
                "--no-access-log",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"PASS  {name}")
        else:
            failures.append(name if not detail else f"{name}: {detail}")
            print(f"FAIL  {name}{' - ' + detail if detail else ''}")

    try:
        error = wait_for_server(base, args.timeout)
        if error:
            print(f"FAIL  server did not become ready: {error}")
            if log is not None:
                log.seek(0)
                print("--- server log ---")
                print(log.read()[-4000:])
            return 1

        with httpx.Client(base_url=base, timeout=30, trust_env=False) as client:
            health = client.get("/health")
            check(
                "health is live",
                health.status_code == 200 and health.json().get("live") is True,
            )
            check(
                "security headers",
                health.headers.get("x-content-type-options") == "nosniff"
                and health.headers.get("cache-control") == "no-store",
            )
            check(
                "pages origin allowed",
                client.get("/health", headers={"Origin": PAGES_ORIGIN}).headers.get(
                    "access-control-allow-origin"
                )
                == PAGES_ORIGIN,
            )

            capabilities = client.get("/capabilities")
            caps = capabilities.json()
            check(
                "capabilities describe live engine",
                caps.get("engine") == "leaklens.python",
            )
            check("no permanent storage advertised", caps.get("storage") == "none")
            check("predict is not exposed", client.get("/predict").status_code == 404)

            limits = caps.get("limits", {})
            max_rows = int(limits.get("max_rows", 5000))
            rows = max(12, min(max_rows, 60))
            train = dataset(rows)
            test_rows = max(2, rows // 3)
            held_out = dataset(test_rows)

            inspect = client.post(
                "/inspect", files={"file": ("train.csv", train, "text/csv")}
            )
            check(
                "inspect returns real columns",
                inspect.status_code == 200
                and [column["name"] for column in inspect.json().get("columns", [])]
                == ["x", "y", "leak"],
                inspect.text[:200],
            )

            audit = client.post(
                "/audit",
                files={
                    "file": ("train.csv", train, "text/csv"),
                    "test": ("test.csv", held_out, "text/csv"),
                },
                data={"target": "y"},
            )
            body = (
                audit.json()
                if audit.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            kinds = {finding.get("kind") for finding in body.get("findings", [])}
            check(
                "audit returns real findings",
                audit.status_code == 200
                and body.get("live") is True
                and "target_leakage" in kinds
                and int(body.get("trust_score", 100)) < 100,
                audit.text[:200],
            )
            check(
                "held-out impact is measured",
                isinstance(body.get("impact"), dict)
                and body["impact"].get("duplicate_test_rows") == test_rows,
                str(body.get("impact"))[:200],
            )
            check(
                "HTML report is generated",
                isinstance(body.get("report_html"), str)
                and body["report_html"].startswith("<!doctype html>"),
            )

            scan = client.post(
                "/scan",
                json={
                    "source": (
                        "from sklearn.model_selection import train_test_split\n"
                        "from sklearn.preprocessing import StandardScaler\n"
                        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
                        "StandardScaler().fit(X)\n"
                    )
                },
            )
            scan_body = scan.json()
            check(
                "code scan returns real findings",
                scan.status_code == 200
                and scan_body.get("engine") == "leaklens.scanner.scan_source"
                and bool(scan_body.get("findings")),
                scan.text[:200],
            )

            traversal = client.post(
                "/inspect",
                files={"file": ("../escape.csv", train, "text/csv")},
            )
            check(
                "path traversal filename rejected",
                traversal.status_code == 400
                and traversal.json().get("error", {}).get("code") == "invalid_filename",
                traversal.text[:200],
            )

            max_bytes = int(limits.get("max_bytes", 4 * 1024 * 1024))
            oversized = b"x,y\n" + b"1,2\n" * (max_bytes // 4 + 64)
            too_big = client.post(
                "/inspect", files={"file": ("big.csv", oversized, "text/csv")}
            )
            check(
                "oversized upload rejected",
                too_big.status_code == 413
                and too_big.json().get("error", {}).get("code")
                in {"upload_too_large", "request_too_large"},
                too_big.text[:200],
            )

            missing = client.post(
                "/audit", files={"file": ("train.csv", train, "text/csv")}
            )
            check("missing target rejected", missing.status_code == 422)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log is not None:
            log.close()

    if failures:
        print(f"\n{len(failures)} smoke check(s) failed")
        return 1
    print("\nAll live API smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
