"""Hardened public API for live LeakLens scans.

This is intentionally separate from :mod:`leaklens.api`.  The legacy app is a
trusted/local developer surface that includes a pickle-based ``/predict``
endpoint.  The public app exposes only bounded, stateless scans:

* ``GET  /health``       liveness and version
* ``GET  /capabilities`` formats, limits, checks and privacy statement
* ``POST /inspect``      validate an upload and describe its columns
* ``POST /audit``        run the real dataset detectors (optional held-out file)
* ``POST /scan``         run the real static scanner on Python source

Uploads are read into a bounded in-memory buffer for the lifetime of one
request and are never written to disk or retained.  All errors are generic;
parser internals, filesystem paths and stack traces never reach the client.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__, assemble
from .audit import audit
from .fixes import advice
from .intelligence import (
    GraphBaselineError,
    GraphSnapshot,
    analyze_dataset_impact,
    analyze_findings,
    compare_graph_snapshots,
    snapshot_context,
)
from .leakage import (
    feature_overlap,
    group_overlap,
    group_risk,
    near_duplicate_overlap,
    scan,
    split_overlap,
    temporal_overlap,
)
from .missingness import audit_missingness
from .ratelimit import SlidingWindowLimiter
from .report import render
from .scanner import scan_source
from .upload_guard import (
    DEFAULT_FORMATS,
    GuardLimits,
    UploadRejected,
    available_formats,
    describe_dataset,
    guard_upload,
    read_guarded_dataset,
)

PAGES_ORIGIN = "https://eyadstein.github.io"
LOCALHOST_ORIGIN = r"http://(localhost|127\.0\.0\.1)(:\d+)?"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_IP_CHARS = re.compile(r"^[0-9a-fA-F:.]{3,64}$")

MAX_SOURCE_BYTES = 262_144
MAX_GRAPH_SNAPSHOT_BYTES = 2 * 1024 * 1024
_CODE_NAME = "snippet.py"


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(low, min(high, value))


def _env_formats() -> tuple[str, ...]:
    raw = os.environ.get("LEAKLENS_PUBLIC_FORMATS", ",".join(DEFAULT_FORMATS))
    allowed = {
        item.strip().lower().lstrip(".") for item in raw.split(",") if item.strip()
    }
    known = set(DEFAULT_FORMATS) | {"xlsx", "parquet", "feather"}
    selected = tuple(item for item in DEFAULT_FORMATS if item in allowed)
    selected += tuple(sorted((allowed & known) - set(selected)))
    return selected or DEFAULT_FORMATS


def _env_origins() -> tuple[str, ...]:
    raw = os.environ.get("LEAKLENS_ALLOWED_ORIGINS", PAGES_ORIGIN)
    origins = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin != "*" and not origin.startswith(("https://", "http://")):
            continue
        origins.append(origin)
    return tuple(dict.fromkeys(origins)) or (PAGES_ORIGIN,)


@dataclass(frozen=True)
class PublicSettings:
    """Runtime policy for the public app, normally supplied by environment."""

    max_bytes: int = 4 * 1024 * 1024
    max_json_bytes: int = 512 * 1024
    max_rows: int = 5_000
    max_columns: int = 200
    max_cells: int = 750_000
    max_uncompressed_bytes: int = 24 * 1024 * 1024
    formats: tuple[str, ...] = DEFAULT_FORMATS
    timeout_seconds: float = 30.0
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    max_rate_keys: int = 2_048
    allowed_origins: tuple[str, ...] = (PAGES_ORIGIN,)
    allow_localhost_origins: bool = True
    trust_proxy: bool = False
    enable_docs: bool = False

    @property
    def limits(self) -> GuardLimits:
        return GuardLimits(
            max_bytes=self.max_bytes,
            max_rows=self.max_rows,
            max_columns=self.max_columns,
            max_cells=self.max_cells,
            max_uncompressed_bytes=self.max_uncompressed_bytes,
            formats=self.formats,
        )

    @property
    def max_request_bytes(self) -> int:
        # One optional held-out dataset can arrive beside the primary upload.
        return self.max_bytes * 2 + 512 * 1024

    def public_limits(self) -> dict[str, Any]:
        data = self.limits.to_public_dict()
        data.update(
            {
                "max_source_bytes": MAX_SOURCE_BYTES,
                "timeout_seconds": self.timeout_seconds,
                "rate_limit_requests": self.rate_limit_requests,
                "rate_limit_window_seconds": self.rate_limit_window_seconds,
                "max_request_bytes": self.max_request_bytes,
            }
        )
        return data

    @classmethod
    def from_env(cls) -> PublicSettings:
        return cls(
            max_bytes=_env_int(
                "LEAKLENS_PUBLIC_MAX_BYTES",
                4 * 1024 * 1024,
                64 * 1024,
                16 * 1024 * 1024,
            ),
            max_json_bytes=_env_int(
                "LEAKLENS_PUBLIC_MAX_JSON_BYTES", 512 * 1024, 16 * 1024, 2 * 1024 * 1024
            ),
            max_rows=_env_int("LEAKLENS_PUBLIC_MAX_ROWS", 5_000, 10, 20_000),
            max_columns=_env_int("LEAKLENS_PUBLIC_MAX_COLUMNS", 200, 2, 500),
            max_cells=_env_int("LEAKLENS_PUBLIC_MAX_CELLS", 750_000, 100, 5_000_000),
            max_uncompressed_bytes=_env_int(
                "LEAKLENS_PUBLIC_MAX_UNCOMPRESSED_BYTES",
                24 * 1024 * 1024,
                1024 * 1024,
                128 * 1024 * 1024,
            ),
            formats=_env_formats(),
            timeout_seconds=_env_float(
                "LEAKLENS_PUBLIC_TIMEOUT_SECONDS", 30.0, 5.0, 120.0
            ),
            rate_limit_requests=_env_int("LEAKLENS_PUBLIC_RATE_LIMIT", 20, 1, 1_000),
            rate_limit_window_seconds=_env_int(
                "LEAKLENS_PUBLIC_RATE_WINDOW", 60, 10, 3_600
            ),
            allowed_origins=_env_origins(),
            allow_localhost_origins=os.environ.get("LEAKLENS_ALLOW_LOCAL_ORIGINS", "1")
            not in {"0", "false", "False"},
            trust_proxy=os.environ.get("LEAKLENS_TRUST_PROXY", "0")
            in {"1", "true", "True"},
            enable_docs=os.environ.get("LEAKLENS_ENABLE_DOCS", "0")
            in {"1", "true", "True"},
        )


class ScanBusy(RuntimeError):
    """The single scan worker is still finishing a previous request."""


class ScanTimeout(RuntimeError):
    """A scan exceeded its wall-clock budget."""


class ScanRunner:
    """Run scans on one bounded worker with a request-level timeout.

    Python threads cannot be killed.  The hard resource guarantee comes from
    the upload/row/cell limits; this runner adds a single-worker queue so a
    timed-out scan can never overlap another scan and exhaust the free tier.
    The slot is held until the worker actually finishes.
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="leaklens-scan"
        )
        self._slot = threading.BoundedSemaphore(1)

    def run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not self._slot.acquire(blocking=False):
            raise ScanBusy
        future = self._executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=self.timeout_seconds)
        except FutureTimeout:
            future.add_done_callback(lambda _future: self._slot.release())
            raise ScanTimeout from None
        except BaseException:
            self._slot.release()
            raise
        else:
            self._slot.release()
            return result

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cache-Control": "no-store",
}


def _error(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    merged = {**_SECURITY_HEADERS, "X-Request-ID": secrets.token_hex(8)}
    merged.update(headers or {})
    payload = {"error": {"code": code, "message": message}}
    return JSONResponse(status_code=status_code, content=payload, headers=merged)


def _origin_allowed(origin: str, settings: PublicSettings) -> bool:
    if not origin:
        return False
    if origin in settings.allowed_origins or "*" in settings.allowed_origins:
        return True
    return bool(
        settings.allow_localhost_origins and re.fullmatch(LOCALHOST_ORIGIN, origin)
    )


def _cors_headers(scope: Scope, settings: PublicSettings) -> dict[str, str]:
    origin = Headers(scope=scope).get("origin", "")
    if _origin_allowed(origin, settings):
        return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
    return {}


def _client_key(scope: Scope, trust_proxy: bool) -> str:
    headers = Headers(scope=scope)
    if trust_proxy:
        forwarded = headers.get("x-forwarded-for", "")
        candidate = forwarded.split(",")[0].strip()
        if _IP_CHARS.match(candidate):
            return candidate
    client = scope.get("client")
    if client:
        host = str(client[0])
        if _IP_CHARS.match(host):
            return host
    return "unknown"


class _BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Reject oversized request bodies before Starlette parses multipart.

    ``Content-Length`` is checked first.  Chunked bodies without a length are
    counted as they stream so a client cannot bypass the cap.
    """

    def __init__(self, app: ASGIApp, settings: PublicSettings) -> None:
        self.app = app
        self.settings = settings

    def _limit(self, path: str) -> int:
        if path.rstrip("/") == "/scan":
            return self.settings.max_json_bytes
        return self.settings.max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._limit(scope.get("path", ""))
        headers = Headers(scope=scope)
        raw_length = headers.get("content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length.split(",")[0].strip())
            except (TypeError, ValueError):
                await self._reject(scope, send, 400, "invalid_content_length")
                return
            if declared < 0:
                await self._reject(scope, send, 400, "invalid_content_length")
                return
            if declared > limit:
                await self._reject(scope, send, 413, "request_too_large")
                return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(scope, send, 413, "request_too_large")

    async def _reject(self, scope: Scope, send: Send, status: int, code: str) -> None:
        if code == "invalid_content_length":
            message = "The Content-Length header is invalid."
        else:
            message = "The request body is larger than this server allows."
        response = _error(
            status,
            code,
            message,
            headers=_cors_headers(scope, self.settings),
        )
        await response(scope, receive=_empty_receive, send=send)


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


class RateLimitMiddleware:
    """Apply a bounded per-client limit to expensive POST endpoints."""

    _LIMITED: ClassVar[frozenset[str]] = frozenset({"/inspect", "/audit", "/scan"})

    def __init__(self, app: ASGIApp, settings: PublicSettings, limiter: Any) -> None:
        self.app = app
        self.settings = settings
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if scope.get("method") != "POST" or path.rstrip("/") not in self._LIMITED:
            await self.app(scope, receive, send)
            return
        key = _client_key(scope, self.settings.trust_proxy)
        allowed, retry = self.limiter.check(key)
        if not allowed:
            response = _error(
                429,
                "rate_limited",
                "Too many scans from this client. Wait a moment and retry.",
                headers={
                    "Retry-After": str(retry),
                    "X-RateLimit-Limit": str(self.settings.rate_limit_requests),
                    **_cors_headers(scope, self.settings),
                },
            )
            await response(scope, receive=_empty_receive, send=send)
            return
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Add conservative headers to every response, including error responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {
                    key.lower().decode("latin-1")
                    for key, _value in message.get("headers", [])
                }
                additions = {
                    "x-content-type-options": "nosniff",
                    "x-frame-options": "DENY",
                    "referrer-policy": "no-referrer",
                    "permissions-policy": "camera=(), microphone=(), geolocation=()",
                    "content-security-policy": (
                        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
                    ),
                    "cache-control": "no-store",
                    "x-request-id": secrets.token_hex(8),
                }
                if scope.get("scheme") == "https":
                    additions["strict-transport-security"] = (
                        "max-age=15552000; includeSubDomains"
                    )
                for key, value in additions.items():
                    if key not in present:
                        message.setdefault("headers", []).append(
                            (key.encode("latin-1"), value.encode("latin-1"))
                        )
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _clean_option(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > 120 or _CONTROL.search(text):
        raise UploadRejected(f"The {label} name is not valid.", 422, "bad_option")
    return text


def _scan_sections(findings: list[Any]):
    sections = assemble.build(findings)
    for payload in sections.findings:
        payload["fix"] = advice(payload["kind"]) or (
            "Fix the syntax error before relying on this scan."
            if payload["kind"] == "parse_error"
            else ""
        )
    return sections


def _severity_counts(findings: list[Any]) -> dict[str, int]:
    return assemble.severity_counts(findings)


def _form_json(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise UploadRejected(
            "The graph baseline is not valid JSON.", 422, "invalid_graph_baseline"
        ) from None


def parse_graph_baseline(payload: Any) -> GraphSnapshot | None:
    """Validate a client snapshot; never read or write server files."""
    if payload is None:
        return None
    try:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        raise UploadRejected(
            "The graph baseline must be a JSON object.", 422, "invalid_graph_baseline"
        ) from None
    if len(encoded) > MAX_GRAPH_SNAPSHOT_BYTES:
        raise UploadRejected(
            f"The graph baseline is larger than {MAX_GRAPH_SNAPSHOT_BYTES} bytes.",
            413,
            "graph_baseline_too_large",
        )
    try:
        return GraphSnapshot.from_dict(payload)
    except GraphBaselineError:
        raise UploadRejected(
            "The graph baseline is malformed or uses an unsupported schema.",
            422,
            "invalid_graph_baseline",
        ) from None


def _graph_payload(
    findings: list[Any],
    *,
    baseline: GraphSnapshot | None,
    include_snapshot: bool,
    include_intelligence: bool,
    context: dict[str, str],
) -> dict[str, Any]:
    """Derive intelligence, comparison and snapshot from one graph build."""
    payload: dict[str, Any] = {}
    if baseline is None and not include_snapshot and not include_intelligence:
        return payload
    report = analyze_findings(findings)
    parse_error = any(finding.kind == "parse_error" for finding in findings)
    snapshot = GraphSnapshot.from_findings(
        findings,
        graph=report.to_graph(),
        complete=not parse_error,
        incomplete_reasons=(
            ("a source file could not be parsed completely",) if parse_error else ()
        ),
        context=context,
    )
    if include_intelligence:
        payload["intelligence"] = report.to_dict()
    if baseline is not None:
        try:
            payload["graph_baseline"] = compare_graph_snapshots(
                baseline, snapshot
            ).to_dict()
        except GraphBaselineError:
            raise UploadRejected(
                "The graph baseline is incompatible with this scan.",
                422,
                "incompatible_graph_baseline",
            ) from None
    if include_snapshot:
        payload["graph_snapshot"] = snapshot.to_dict()
    return payload


def run_inspect_job(upload: Any, limits: GuardLimits) -> dict[str, Any]:
    df = read_guarded_dataset(upload, limits)
    result = describe_dataset(df, limits)
    result["file"] = upload.filename
    return result


def run_audit_job(
    upload: Any,
    limits: GuardLimits,
    target: str,
    threshold: float,
    test_upload: Any | None = None,
    time_col: str | None = None,
    group_col: str | None = None,
    include_ai: bool = False,
    include_intelligence: bool = False,
    graph_baseline: GraphSnapshot | None = None,
    include_graph_snapshot: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    df = read_guarded_dataset(upload, limits, target, require_target=True)
    test_df = None
    if test_upload is not None:
        test_df = read_guarded_dataset(test_upload, limits)
    for option, label in ((time_col, "time"), (group_col, "group")):
        if option and option not in df.columns:
            raise UploadRejected(
                f"The {label} column was not found in the dataset.",
                422,
                "unknown_option_column",
            )
        if option and test_df is not None and option not in test_df.columns:
            raise UploadRejected(
                f"The {label} column was not found in the held-out dataset.",
                422,
                "unknown_option_column",
            )
    try:
        findings = scan(df, target, threshold)
        if test_df is not None:
            exact = split_overlap(df, test_df)
            findings += exact
            if not exact:
                shared = feature_overlap(df, test_df, target)
                findings += shared
                if not shared:
                    findings += near_duplicate_overlap(df, test_df, target)
            if time_col:
                findings += temporal_overlap(df, test_df, time_col)
        if group_col:
            findings += group_risk(df, group_col)
            if test_df is not None:
                findings += group_overlap(df, test_df, group_col)
    except Exception:  # noqa: BLE001 - never return parser/sklearn internals
        raise UploadRejected(
            "LeakLens could not scan this dataset. Check the target column and try a smaller file.",
            422,
            "scan_failed",
        ) from None

    audit_data = audit(df, target)
    missing = audit_missingness(df, target, test=test_df, group=group_col)
    findings += missing.findings
    impact = None
    if test_df is not None:
        try:
            impact = analyze_dataset_impact(
                df, test_df, group_col=group_col, time_col=time_col
            ).to_dict()
        except (TypeError, ValueError):
            impact = None
    sections = _scan_sections(findings)
    payload = {
        "live": True,
        "engine": "leaklens.python",
        "version": __version__,
        "trust_score": sections.trust_score,
        "findings": sections.findings,
        "counts": sections.counts,
        "audit": audit_data,
        "missingness": missing.to_dict(),
        "impact": impact,
        "summary": {
            "file": upload.filename,
            "target": target,
            "threshold": threshold,
            "rows": len(df),
            "columns": int(df.shape[1]),
            "test_rows": len(test_df) if test_df is not None else None,
            "time_column": time_col,
            "group_column": group_col,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
        "limits": limits.to_public_dict(),
        "report_filename": "leaklens-report.html",
    }
    payload.update(
        _graph_payload(
            findings,
            baseline=graph_baseline,
            include_snapshot=include_graph_snapshot,
            include_intelligence=include_intelligence,
            context=snapshot_context(
                mode="dataset",
                target=target,
                test=test_df is not None,
                group=group_col or "",
                time=time_col or "",
                threshold=threshold,
            ),
        )
    )
    intelligence = payload.get("intelligence")
    ai_payload = None
    if include_ai:
        try:
            from .ai import analyze_dataset

            ai_payload = analyze_dataset(
                df,
                test_df,
                target=target,
                group=group_col,
                time=time_col,
                findings=findings,
                trust_score=payload["trust_score"],
            ).to_dict()
        except Exception:  # noqa: BLE001 - optional AI must not break the scan
            ai_payload = None
        payload["ai"] = ai_payload
        payload["ai_unavailable"] = ai_payload is None
    payload["report_html"] = render(
        {
            "trust_score": payload["trust_score"],
            "findings": payload["findings"],
            "audit": audit_data,
            "impact": impact,
            "intelligence": intelligence,
            "graph_baseline": payload.get("graph_baseline"),
            "ai": ai_payload,
        },
        title=f"LeakLens report - {upload.filename}",
    )
    return payload


def run_code_job(
    source: str,
    name: str,
    include_ai: bool = False,
    include_intelligence: bool = False,
    graph_baseline: GraphSnapshot | None = None,
    include_graph_snapshot: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    findings = scan_source(source, name)
    sections = _scan_sections(findings)
    payload = {
        "live": True,
        "engine": "leaklens.scanner.scan_source",
        "version": __version__,
        "trust_score": sections.trust_score,
        "findings": sections.findings,
        "counts": sections.counts,
        "syntax_error": any(finding.kind == "parse_error" for finding in findings),
        "summary": {
            "file": name,
            "bytes": len(source.encode("utf-8")),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    }
    payload.update(
        _graph_payload(
            findings,
            baseline=graph_baseline,
            include_snapshot=include_graph_snapshot,
            include_intelligence=include_intelligence,
            context=snapshot_context(
                mode="code",
                target=name,
                threshold=0.95,
            ),
        )
    )
    if include_ai:
        try:
            from .ai import CodeSource, analyze_code

            payload["ai"] = analyze_code(
                (CodeSource(label=name, source=source),),
                findings,
            ).to_dict()
        except Exception:  # noqa: BLE001 - optional AI must not break the scan
            payload["ai"] = None
            payload["ai_unavailable"] = True
    return payload


class CodeScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    name: str = Field(default=_CODE_NAME, max_length=120)
    include_ai: bool = Field(default=False)
    include_intelligence: bool = Field(default=False)
    graph_baseline: dict | None = Field(default=None)
    include_graph_snapshot: bool = Field(default=False)


async def _read_bounded(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise UploadRejected(
                    "The upload is larger than this server allows.",
                    413,
                    "upload_too_large",
                )
            chunks.append(chunk)
    finally:
        with contextlib.suppress(Exception):
            await file.close()
    return b"".join(chunks)


async def _guard_upload(file: UploadFile, settings: PublicSettings) -> Any:
    data = await _read_bounded(file, settings.max_bytes)
    return guard_upload(
        file.filename,
        data,
        settings.limits,
        content_type=file.content_type or "",
    )


def _threshold(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise UploadRejected(
            "The threshold must be a number between 0.5 and 1.0.",
            422,
            "bad_threshold",
        ) from None
    if not math.isfinite(number) or not 0.5 <= number <= 1.0:
        raise UploadRejected(
            "The threshold must be between 0.5 and 1.0.", 422, "bad_threshold"
        )
    return round(number, 4)


def create_app(
    settings: PublicSettings | None = None,
    runner: ScanRunner | None = None,
) -> FastAPI:
    settings = settings or PublicSettings.from_env()
    scan_runner = runner or ScanRunner(settings.timeout_seconds)
    rate_limiter = SlidingWindowLimiter(
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
        max_keys=settings.max_rate_keys,
    )
    app = FastAPI(
        title="LeakLens live scan API",
        version=__version__,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    app.state.runner = scan_runner
    app.state.rate_limiter = rate_limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_origin_regex=LOCALHOST_ORIGIN
        if settings.allow_localhost_origins
        else None,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
        allow_credentials=False,
        max_age=600,
    )
    app.add_middleware(BodyLimitMiddleware, settings=settings)
    app.add_middleware(RateLimitMiddleware, settings=settings, limiter=rate_limiter)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.exception_handler(UploadRejected)
    async def _upload_rejected(request: Request, exc: UploadRejected) -> JSONResponse:
        return _error(
            exc.status_code,
            exc.code,
            exc.message,
            headers=_cors_headers(request.scope, settings),
        )

    @app.exception_handler(ScanBusy)
    async def _busy(request: Request, _exc: ScanBusy) -> JSONResponse:
        return _error(
            503,
            "scan_busy",
            "A scan is already running on this small server. Retry in a few seconds.",
            headers={
                "Retry-After": "5",
                **_cors_headers(request.scope, settings),
            },
        )

    @app.exception_handler(ScanTimeout)
    async def _timeout(request: Request, _exc: ScanTimeout) -> JSONResponse:
        return _error(
            504,
            "scan_timeout",
            "The scan exceeded this server's time limit. Try a smaller dataset.",
            headers=_cors_headers(request.scope, settings),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        fields = []
        for error in exc.errors()[:8]:
            location = error.get("loc", ())
            name = ".".join(
                str(part) for part in location if part not in {"body", "query"}
            )
            if name and name not in fields:
                fields.append(name[:80])
        message = "Check the request fields and try again."
        if fields:
            message = f"Check these fields: {', '.join(fields)}."
        return _error(
            422,
            "invalid_request",
            message,
            headers=_cors_headers(request.scope, settings),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = {
            404: "That endpoint does not exist.",
            405: "That method is not allowed here.",
        }.get(exc.status_code, "The request could not be completed.")
        return _error(
            exc.status_code,
            "http_error",
            message,
            headers=_cors_headers(request.scope, settings),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, _exc: Exception) -> JSONResponse:
        return _error(
            500,
            "internal_error",
            "The scan failed unexpectedly. Try again or run LeakLens locally.",
            headers=_cors_headers(request.scope, settings),
        )

    @app.get("/")
    def index() -> dict[str, Any]:
        return {
            "service": "LeakLens live scan API",
            "live": True,
            "health": "/health",
            "capabilities": "/capabilities",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "leaklens-public-api",
            "version": __version__,
            "live": True,
        }

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "live": True,
            "service": "LeakLens live scan API",
            "version": __version__,
            "engine": "leaklens.python",
            "endpoints": ["/inspect", "/audit", "/scan"],
            "limits": settings.public_limits(),
            "formats": available_formats(settings.limits),
            "code_scan": {
                "enabled": True,
                "language": "Python",
                "max_bytes": MAX_SOURCE_BYTES,
            },
            "held_out_dataset": {
                "enabled": True,
                "checks": [
                    "exact row overlap",
                    "feature-vector overlap",
                    "normalised near-duplicate rows",
                    "group overlap",
                    "temporal boundary overlap",
                ],
            },
            "checks": [
                "single-feature target leakage",
                "identifier-like columns",
                "categorical target leakage",
                "group leakage risk",
            ],
            "ai": {
                "enabled": True,
                "opt_in": True,
                "scope": ["dataset", "code"],
                "engine": "leaklens.ai.experimental",
                "llm": False,
                "network": "none",
                "changes_deterministic_findings": False,
            },
            "intelligence": {
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
                    "max_bytes": MAX_GRAPH_SNAPSHOT_BYTES,
                    "snapshot_field": "include_graph_snapshot",
                },
            },
            "privacy": (
                "Uploads are parsed in memory for one request, never written to "
                "disk, and never retained. Responses contain only your findings."
            ),
            "storage": "none",
        }

    @app.post("/inspect")
    async def inspect_endpoint(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
    ) -> JSONResponse:
        upload = await _guard_upload(file, settings)
        result = await run_in_threadpool(
            partial(
                request.app.state.runner.run, run_inspect_job, upload, settings.limits
            )
        )
        return JSONResponse(result)

    @app.post("/audit")
    async def audit_endpoint(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        target: str = Form(...),
        threshold: float = Form(0.95),
        test: UploadFile | None = File(default=None),  # noqa: B008
        time_col: str | None = Form(default=None, alias="time"),
        group_col: str | None = Form(default=None, alias="group"),
        include_ai: bool = Form(default=False),
        include_intelligence: bool = Form(default=False),
        graph_baseline: str | None = Form(default=None),
        include_graph_snapshot: bool = Form(default=False),
    ) -> JSONResponse:
        clean_target = target.strip()
        if not clean_target:
            raise UploadRejected(
                "Choose a target column before scanning.", 422, "missing_target"
            )
        baseline = parse_graph_baseline(_form_json(graph_baseline))
        upload = await _guard_upload(file, settings)
        test_upload = await _guard_upload(test, settings) if test is not None else None
        result = await run_in_threadpool(
            partial(
                request.app.state.runner.run,
                run_audit_job,
                upload,
                settings.limits,
                clean_target,
                _threshold(threshold),
                test_upload,
                _clean_option(time_col, "time"),
                _clean_option(group_col, "group"),
                include_ai,
                include_intelligence,
                baseline,
                include_graph_snapshot,
            )
        )
        return JSONResponse(result)

    @app.post("/scan")
    async def scan_endpoint(request: Request, body: CodeScanRequest) -> JSONResponse:
        size = len(body.source.encode("utf-8"))
        if size > MAX_SOURCE_BYTES:
            raise UploadRejected(
                f"Source is larger than the {MAX_SOURCE_BYTES // 1024} KiB limit.",
                413,
                "source_too_large",
            )
        name = body.name if body.name.lower().endswith(".py") else _CODE_NAME
        baseline = parse_graph_baseline(body.graph_baseline)
        result = await run_in_threadpool(
            partial(
                request.app.state.runner.run,
                run_code_job,
                body.source,
                name,
                body.include_ai,
                body.include_intelligence,
                baseline,
                body.include_graph_snapshot,
            )
        )
        return JSONResponse(result)

    return app


app = create_app()
