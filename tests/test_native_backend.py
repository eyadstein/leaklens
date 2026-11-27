"""Native, container-free startup for the LeakLens FastAPI apps.

Docker remains supported; these tests prove the same two apps start through
``python -m`` on the current interpreter, with loopback defaults and actionable
errors for a core-only install.
"""

from __future__ import annotations

import importlib
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from leaklens import serve

ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"  # noqa: S104 - asserted, never bound


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base: str, process: subprocess.Popen, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError("the native server exited before answering /health")
        try:
            response = httpx.get(f"{base}/health", timeout=2, trust_env=False)
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            time.sleep(0.1)
    raise AssertionError(f"the native server did not answer {base}/health")


@pytest.mark.parametrize("module", ["leaklens.api", "leaklens.public_api"])
def test_module_entry_point_starts_the_app_and_answers_health(module):
    port = free_port()
    env = {
        **os.environ,
        "LEAKLENS_HOST": "127.0.0.1",
        "LEAKLENS_PORT": str(port),
        "LEAKLENS_LOG_LEVEL": "warning",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        payload = wait_for_health(f"http://127.0.0.1:{port}", process)
        assert payload["status"] == "ok"
        assert payload["version"]
    finally:
        process.terminate()
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def test_native_and_docker_run_the_same_application():
    pairs = [
        ("Dockerfile", "leaklens.api"),
        ("Dockerfile.api", "leaklens.public_api"),
    ]
    for dockerfile, module in pairs:
        text = (ROOT / dockerfile).read_text(encoding="utf-8")
        assert f"{module}:app" in text, dockerfile
        imported = importlib.import_module(module)
        assert imported.app is not None, module
        source = Path(imported.__file__).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in source, module


def test_published_ports_and_base_images_are_pinned_and_loopback():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    published = re.findall(r'^\s*-\s*"([^"]+)"\s*$', compose, re.MULTILINE)
    assert published
    for mapping in published:
        # The API has no client authentication; publishing beyond loopback
        # would expose it to the host's network.
        assert mapping.startswith("127.0.0.1:"), mapping
    for dockerfile in ("Dockerfile", "Dockerfile.api"):
        text = (ROOT / dockerfile).read_text(encoding="utf-8")
        base = re.search(r"^FROM\s+(\S+)", text, re.MULTILINE)
        assert base is not None, dockerfile
        assert re.search(r":\d+\.\d+\.\d+-", base.group(1)), base.group(1)


def test_serve_defaults_to_loopback(monkeypatch, capsys):
    calls: dict[str, object] = {}
    monkeypatch.delenv("LEAKLENS_HOST", raising=False)
    monkeypatch.delenv("LEAKLENS_PORT", raising=False)
    monkeypatch.delenv("LEAKLENS_LOG_LEVEL", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(kwargs))
    assert serve.serve(object(), service="test API") == 0
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000/" in out
    assert "http://127.0.0.1:8000/health" in out


def test_serve_reads_host_port_and_log_level_from_the_environment(monkeypatch, capsys):
    calls: dict[str, object] = {}
    monkeypatch.setenv("LEAKLENS_HOST", WILDCARD)
    monkeypatch.setenv("LEAKLENS_PORT", "9000")
    monkeypatch.setenv("LEAKLENS_LOG_LEVEL", "warning")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(kwargs))
    assert serve.serve(object(), service="test API") == 0
    assert calls["host"] == WILDCARD
    assert calls["port"] == 9000
    assert calls["log_level"] == "warning"
    assert "no client authentication" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["eighty", "0", "70000", "-1"])
def test_serve_rejects_an_invalid_port(monkeypatch, capsys, value):
    monkeypatch.setenv("LEAKLENS_PORT", value)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("started"))
    assert serve.serve(object(), service="test API") == 2
    assert "LEAKLENS_PORT" in capsys.readouterr().err


def test_serve_rejects_an_invalid_log_level(monkeypatch, capsys):
    monkeypatch.setenv("LEAKLENS_LOG_LEVEL", "loud")
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("started"))
    assert serve.serve(object(), service="test API") == 2
    assert "LEAKLENS_LOG_LEVEL" in capsys.readouterr().err


def test_serve_names_the_api_extra_when_uvicorn_is_missing(monkeypatch, capsys):
    # A None entry makes `import uvicorn` fail the same way a core-only install does.
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert serve.serve(object(), service="test API") == 2
    assert 'pip install "leaklens-ml[api]"' in capsys.readouterr().err


@pytest.mark.parametrize("module", ["leaklens.api", "leaklens.public_api"])
def test_api_module_names_the_extra_when_fastapi_is_missing(module):
    script = f"""
import sys
sys.modules["fastapi"] = None
try:
    __import__({module!r})
except ImportError as exc:
    print(exc)
    raise SystemExit(0 if 'leaklens-ml[api]' in str(exc) else 1)
raise SystemExit(2)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_import_does_not_pull_backend_dependencies():
    script = """
import builtins, sys
real_import = builtins.__import__
blocked = {"fastapi", "uvicorn", "starlette", "pydantic"}
def guard(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("core import pulled " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guard
import leaklens, leaklens.cli
print("core ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "core ok" in result.stdout
