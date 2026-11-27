"""Native entry point shared by ``python -m leaklens.api`` and ``python -m leaklens.public_api``.

Both apps run the same way Docker runs them, without a container. Host, port
and log level come from ``LEAKLENS_HOST``, ``LEAKLENS_PORT`` and
``LEAKLENS_LOG_LEVEL``; the default binds loopback only.
"""

from __future__ import annotations

import os
import sys

from .availability import EngineUnavailable, probe

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
ALL_INTERFACES = ("0.0.0.0", "::")  # noqa: S104  # nosec B104 - never the default bind
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


def _env_port() -> int | None:
    raw = os.environ.get("LEAKLENS_PORT", "").strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        port = 0
    if not 0 < port < 65_536:
        print(
            f"LEAKLENS_PORT must be an integer between 1 and 65535, got {raw!r}.",
            file=sys.stderr,
        )
        return None
    return port


def serve(app: object, *, service: str) -> int:
    """Run a FastAPI app from a module ``__main__`` block; return an exit code."""
    try:
        uvicorn = probe("uvicorn", "api", purpose="LeakLens backend commands")
    except EngineUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    port = _env_port()
    if port is None:
        return 2
    level = os.environ.get("LEAKLENS_LOG_LEVEL", "info").strip().lower()
    if level not in LOG_LEVELS:
        print(
            f"LEAKLENS_LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, "
            f"got {level!r}.",
            file=sys.stderr,
        )
        return 2
    host = os.environ.get("LEAKLENS_HOST", "").strip() or DEFAULT_HOST
    shown = "127.0.0.1" if host in ALL_INTERFACES else host
    print(f"LeakLens {service}: http://{shown}:{port}/", flush=True)
    print(f"Health check: http://{shown}:{port}/health", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if host not in LOOPBACK:
        print(
            f"Warning: bound to {host}, outside loopback. This API has no client "
            "authentication; put it behind a proxy you control.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=host, port=port, log_level=level)
    return 0
