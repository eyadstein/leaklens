"""Local, read-only interactive scan workspace.

One bounded scan session at a time: start a scan of a file or folder, poll
progress, browse findings grouped by the existing evidence graph, and read the
source around a finding.  Nothing is written to disk, nothing is executed, and
every path stays inside the workspace root supplied at startup.

This is a thin orchestration layer: findings, fingerprints, trust score,
clusters and fixes all come from the existing deterministic modules.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .fingerprint import LOCATION, normalize_path
from .fixes import advice
from .intelligence import analyze_findings
from .leakage import Finding, score
from .report import render
from .scanner import scan_file, scan_targets
from .sqlscan import SqlSettings, scan_sql_file
from .sqlscan import available as sql_available

MAX_FILES = 5_000
MAX_SOURCE_BYTES = 1_000_000
MAX_SCANS = 20
MAX_ERRORS = 200
SOURCE_CONTEXT = 8
_CELL = re.compile(r"\[cell (\d+)\]")


class WorkspacePathError(ValueError):
    """A requested path is missing, not scannable, or outside the root."""


class WorkspaceBusy(RuntimeError):
    """A scan is already running; this workspace runs one at a time."""


@dataclass
class ScanSession:
    """Mutable progress plus the final payload for one scan."""

    scan_id: str
    files: tuple[Path, ...]
    include_sql: bool
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.perf_counter)
    state: str = "running"
    done: int = 0
    current: str | None = None
    counts: dict[str, int] = field(
        default_factory=lambda: {"high": 0, "medium": 0, "low": 0}
    )
    errors: list[dict[str, str]] = field(default_factory=list)
    results: dict[str, Any] | None = None
    report_html: str | None = None

    def update(self, **changes: Any) -> None:
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def note_findings(self, findings: list[Finding]) -> None:
        counts = {"high": 0, "medium": 0, "low": 0}
        for finding in findings:
            if finding.severity in counts:
                counts[finding.severity] += 1
        with self.lock:
            self.counts = counts

    def add_error(self, path: str, message: str) -> None:
        with self.lock:
            if len(self.errors) < MAX_ERRORS:
                self.errors.append({"path": path, "message": message})

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 1)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "scan_id": self.scan_id,
                "state": self.state,
                "done": self.done,
                "total": len(self.files),
                "current": self.current,
                "counts": dict(self.counts),
                "errors": list(self.errors),
                "elapsed_ms": self.elapsed_ms(),
            }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - resolve() keeps paths inside root
        return path.as_posix()


def _finding_payload(finding: Finding, index: int, root: Path) -> dict[str, Any]:
    """Finding dict plus the file/line needed to open the source preview."""
    data = finding.to_dict(root=root)
    match = LOCATION.match(str(finding.column))
    if match:
        data["file"] = normalize_path(match.group("path"), root=root)
        data["line"] = int(match.group("line"))
        cell = _CELL.search(str(finding.column))
        data["cell"] = int(cell.group(1)) if cell else None
    else:
        data["file"] = ""
        data["line"] = None
        data["cell"] = None
    data["index"] = index
    data["advice"] = advice(finding.kind) or None
    return data


def _report_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The static report table already knows these columns; keep it readable."""
    return [
        {key: item.get(key) for key in ("severity", "kind", "column", "detail")}
        for item in findings
    ]


class WorkspaceScanManager:
    """Owns the workspace root and the single active scan thread."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._scans: dict[str, ScanSession] = {}
        self._order: list[str] = []
        self._active: ScanSession | None = None
        self._lock = threading.Lock()

    def resolve(self, value: str) -> Path:
        text = (value or "").strip()
        if not text:
            raise WorkspacePathError("a path is required")
        candidate = Path(text)
        target = candidate if candidate.is_absolute() else self.root / candidate
        target = target.resolve()
        if target != self.root and not target.is_relative_to(self.root):
            raise WorkspacePathError("path is outside the workspace root")
        return target

    def start(self, path: str = ".", *, sql: bool = False) -> ScanSession:
        target = self.resolve(path)
        if not target.exists():
            raise WorkspacePathError(f"not found: {path}")
        if not (target.is_file() or target.is_dir()):
            raise WorkspacePathError(f"not a file or folder: {path}")
        if sql and not sql_available():
            raise WorkspacePathError(
                'SQL scanning needs the sql extra: pip install "leaklens-ml[sql]"'
            )
        settings = SqlSettings(enabled=True) if sql else None
        files = tuple(scan_targets(target, sql=settings))
        if not files:
            raise WorkspacePathError("no .py, .ipynb or .sql files to scan here")
        if len(files) > MAX_FILES:
            raise WorkspacePathError(
                f"too many files ({len(files)}); scan a folder with fewer than {MAX_FILES}"
            )
        with self._lock:
            if self._active is not None and self._active.state == "running":
                raise WorkspaceBusy("a scan is already running")
            session = ScanSession(
                scan_id=secrets.token_hex(6),
                files=files,
                include_sql=sql,
            )
            self._scans[session.scan_id] = session
            self._order.append(session.scan_id)
            self._active = session
            self._trim()
        threading.Thread(
            target=self._run,
            args=(session,),
            name=f"leaklens-workspace-{session.scan_id}",
            daemon=True,
        ).start()
        return session

    def _trim(self) -> None:
        while len(self._order) > MAX_SCANS:
            for scan_id in list(self._order):
                session = self._scans.get(scan_id)
                if session is not None and session.state == "running":
                    continue
                self._order.remove(scan_id)
                self._scans.pop(scan_id, None)
                break
            else:  # pragma: no cover - only reachable with MAX_SCANS running
                return

    def get(self, scan_id: str) -> ScanSession:
        session = self._scans.get(scan_id)
        if session is None:
            raise KeyError(scan_id)
        return session

    def cancel(self, scan_id: str) -> ScanSession:
        session = self.get(scan_id)
        if session.state != "running":
            raise WorkspaceBusy("scan is not running")
        session.cancel.set()
        return session

    def _run(self, session: ScanSession) -> None:
        findings: list[Finding] = []
        settings = SqlSettings(enabled=True) if session.include_sql else None
        try:
            for index, path in enumerate(session.files):
                if session.cancel.is_set():
                    self._finish(session, "cancelled", findings)
                    return
                relative = _relative(path, self.root)
                session.update(done=index, current=relative)
                try:
                    if path.suffix == ".sql":
                        findings.extend(scan_sql_file(path, settings=settings))
                    else:
                        findings.extend(scan_file(path))
                except Exception as exc:  # noqa: BLE001 - one bad file must not end the scan
                    session.add_error(relative, str(exc)[:300])
                session.note_findings(findings)
                session.update(done=index + 1)
            self._finish(session, "done", findings)
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            session.add_error("", str(exc)[:300])
            self._finish(session, "failed", findings)

    def _finish(
        self, session: ScanSession, state: str, findings: list[Finding]
    ) -> None:
        report = analyze_findings(findings)
        intelligence = report.to_dict()
        payload = {
            "scan_id": session.scan_id,
            "state": state,
            "trust_score": score(findings),
            "counts": dict(session.counts),
            "files": len(session.files),
            "errors": list(session.errors),
            "duration_ms": session.elapsed_ms(),
            # Clusters, links, evidence, confidence, root causes and next
            # actions are the existing deterministic intelligence layer.
            **intelligence,
            "findings": [
                _finding_payload(finding, index, self.root)
                for index, finding in enumerate(findings)
            ],
        }
        report_html = render(
            {
                "trust_score": payload["trust_score"],
                "findings": _report_findings(payload["findings"]),
                "intelligence": intelligence,
            },
            title="LeakLens workspace report",
        )
        with session.lock:
            session.state = state
            session.current = None
            session.results = payload
            session.report_html = report_html


class ScanRequest(BaseModel):
    path: str = Field(default=".", max_length=1000)
    sql: bool = Field(default=False)


def create_workspace_app(root: str | Path | None = None) -> FastAPI:
    manager = WorkspaceScanManager(root or Path.cwd())
    app = FastAPI(
        title="LeakLens workspace",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        from .workspace_ui import PAGE

        return PAGE.replace("@@VERSION@@", __version__).replace(
            "@@ROOT@@", manager.root.as_posix()
        )

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "version": __version__,
            "root": manager.root.as_posix(),
            "max_files": MAX_FILES,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "source_context": SOURCE_CONTEXT,
            "sql_available": sql_available(),
        }

    @app.post("/api/scans", status_code=202)
    def start_scan(req: ScanRequest) -> dict[str, str]:
        try:
            session = manager.start(req.path, sql=req.sql)
        except WorkspacePathError as exc:
            raise HTTPException(422, str(exc)) from None
        except WorkspaceBusy as exc:
            raise HTTPException(409, str(exc)) from None
        return {"scan_id": session.scan_id}

    @app.get("/api/scans/{scan_id}")
    def scan_status(scan_id: str) -> dict[str, Any]:
        try:
            return manager.get(scan_id).snapshot()
        except KeyError:
            raise HTTPException(404, "unknown scan") from None

    @app.post("/api/scans/{scan_id}/cancel")
    def cancel_scan(scan_id: str) -> dict[str, Any]:
        try:
            session = manager.cancel(scan_id)
        except KeyError:
            raise HTTPException(404, "unknown scan") from None
        except WorkspaceBusy as exc:
            raise HTTPException(409, str(exc)) from None
        return session.snapshot()

    @app.get("/api/scans/{scan_id}/results")
    def scan_results(scan_id: str) -> dict[str, Any]:
        try:
            session = manager.get(scan_id)
        except KeyError:
            raise HTTPException(404, "unknown scan") from None
        if session.results is None:
            raise HTTPException(409, "scan has not finished")
        return session.results

    @app.get("/api/scans/{scan_id}/report", response_class=HTMLResponse)
    def scan_report(scan_id: str) -> str:
        try:
            session = manager.get(scan_id)
        except KeyError:
            raise HTTPException(404, "unknown scan") from None
        if session.report_html is None:
            raise HTTPException(409, "scan has not finished")
        return session.report_html

    @app.get("/api/source")
    def source(path: str, line: int | None = None) -> dict[str, Any]:
        try:
            target = manager.resolve(path)
        except WorkspacePathError as exc:
            raise HTTPException(422, str(exc)) from None
        if not target.is_file():
            raise HTTPException(404, "file not found")
        if target.stat().st_size > MAX_SOURCE_BYTES:
            raise HTTPException(413, "file is too large to preview")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        anchor = max(1, min(int(line or 1), max(len(lines), 1)))
        start = max(1, anchor - SOURCE_CONTEXT)
        end = min(len(lines), anchor + SOURCE_CONTEXT)
        return {
            "path": _relative(target, manager.root),
            "start": start,
            "line": anchor,
            "total": len(lines),
            "lines": [
                {"number": number, "text": lines[number - 1]}
                for number in range(start, end + 1)
            ],
        }

    return app
