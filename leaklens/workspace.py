"""Local, read-only interactive scan workspace.

One bounded activity at a time: start a source scan of a file or folder, or a
dataset audit of a CSV/JSON/... file, poll progress, browse findings grouped by
the existing evidence graph, and read the source around a finding.  Nothing is
written to disk, nothing is executed, and every path stays inside the workspace
root supplied at startup.

This is a thin orchestration layer: findings, fingerprints, trust score,
clusters and fixes all come from the existing deterministic modules, and
dataset audits go through the public API's ``run_audit_job``.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .baseline import Baseline, BaselineError, compare
from .config import ConfigError, load_config
from .fingerprint import LOCATION, normalize_path
from .fixes import advice
from .intelligence import analyze_findings
from .leakage import Finding, score
from .policy import PolicyOutcome, apply_policy
from .public_api import GuardLimits, _threshold, run_audit_job, run_inspect_job
from .report import render
from .scanner import scan_file, scan_targets
from .sqlscan import SqlSettings, scan_sql_file
from .sqlscan import available as sql_available
from .suppression import SuppressionRule
from .upload_guard import UploadRejected, available_formats, guard_upload

MAX_FILES = 5_000
MAX_SOURCE_BYTES = 1_000_000
MAX_SCANS = 20
MAX_DATASETS = 20
MAX_ERRORS = 200
SOURCE_CONTEXT = 8
GRAPH_DEFAULT_LIMIT = 100
GRAPH_MAX_LIMIT = 500
# Live updates: poll the in-memory session often enough to feel immediate, but
# only emit an event when the snapshot actually changed.
SSE_POLL_SECONDS = 0.1
SSE_HEARTBEAT_SECONDS = 10.0
SSE_MAX_EVENTS = 5_000
MAX_COMPARISON_CHANGES = 500
HISTORY_NOTE = (
    "Sessions are retained in memory only: they do not survive a server restart, "
    "and the oldest finished sessions are dropped first."
)
_COMPARISON_FIELDS = (
    "index",
    "kind",
    "severity",
    "column",
    "detail",
    "file",
    "line",
    "cell",
    "fingerprint",
    "status",
    "requires_attention",
    "suppression",
    "previous",
    "diagnostic",
)
_CELL = re.compile(r"\[cell (\d+)\]")
# Result states that still hold the single activity slot.
ACTIVE_STATES = frozenset({"running", "queued", "auditing"})
# The same hard limits the public audit API applies to an uploaded dataset.
DATASET_LIMITS = GuardLimits()


class WorkspacePathError(ValueError):
    """A requested path is missing, not scannable, or outside the root."""


class WorkspaceBusy(RuntimeError):
    """A scan or audit is already running; this workspace runs one at a time."""


def _trim(order: list[str], sessions: dict[str, Any], limit: int) -> None:
    """Drop the oldest finished session so memory stays bounded."""
    while len(order) > limit:
        for key in list(order):
            session = sessions.get(key)
            if session is not None and session.state in ACTIVE_STATES:
                continue
            order.remove(key)
            sessions.pop(key, None)
            break
        else:  # pragma: no cover - only reachable with `limit` active sessions
            return


@dataclass
class ScanSession:
    """Mutable progress plus the final payload for one scan."""

    scan_id: str
    files: tuple[Path, ...]
    include_sql: bool
    include_ai: bool = False
    target: str = "."
    baseline: Baseline | None = None
    rules: tuple[SuppressionRule, ...] = ()
    baseline_path: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
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


@dataclass
class DatasetSession:
    """One dataset audit: its request, stage and final engine payload.

    Stage-based rather than item-based progress: the engine call is a single
    opaque ``run_audit_job``, so there is no finer signal to report honestly.
    """

    dataset_id: str
    path: Path
    target: str
    threshold: float
    test_path: Path | None = None
    time_col: str | None = None
    group_col: str | None = None
    include_ai: bool = False
    baseline: Baseline | None = None
    rules: tuple[SuppressionRule, ...] = ()
    baseline_path: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.perf_counter)
    completed_at: float | None = None
    state: str = "queued"
    error: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    report_html: str | None = None

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 1)

    def request(self) -> dict[str, Any]:
        return {
            "file": self.path.name,
            "test": self.test_path.name if self.test_path is not None else None,
            "target": self.target,
            "threshold": self.threshold,
            "time_column": self.time_col,
            "group_column": self.group_col,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "dataset_id": self.dataset_id,
                "state": self.state,
                "request": self.request(),
                "error": dict(self.error) if self.error else None,
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
        {
            key: item.get(key)
            for key in ("severity", "kind", "column", "detail", "status", "diagnostic")
        }
        for item in findings
    ]


@dataclass(frozen=True, slots=True)
class _PolicyFinding:
    """A serialized finding that keeps the engine's exact fingerprint."""

    payload: dict[str, Any]
    fingerprint: str
    severity: str
    kind: str
    column: str
    diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _policy_view(finding: dict[str, Any]) -> _PolicyFinding:
    return _PolicyFinding(
        payload=finding,
        fingerprint=str(finding["fingerprint"]),
        severity=str(finding["severity"]),
        kind=str(finding["kind"]),
        column=str(finding["column"]),
        diagnostic=bool(finding.get("diagnostic", False)),
    )


def _apply_workspace_policy(
    findings: list[dict[str, Any]],
    *,
    baseline: Baseline | None,
    rules: tuple[SuppressionRule, ...],
    baseline_path: str | None,
    partial: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]], PolicyOutcome]:
    """Apply the existing policy engine to already-serialized findings.

    Serialized fingerprints are the ones ``Finding.to_dict(root=...)`` produced,
    so a workspace scan and the CLI from the same root classify identically.
    A cancelled scan is never compared with a baseline: its missing findings
    would otherwise look resolved.
    """
    views = [_policy_view(finding) for finding in findings]
    outcome = apply_policy(views, baseline=None if partial else baseline, rules=rules)
    items = [outcome.finding_payload(view) for view in views]
    previous = (
        {entry.fingerprint: entry.to_dict() for entry in baseline.entries}
        if baseline is not None and not partial
        else {}
    )
    for item in items:
        item["requires_attention"] = item["fingerprint"] not in outcome.suppressed
        if item.get("status") == "changed" and item["fingerprint"] in previous:
            item["previous"] = previous[item["fingerprint"]]
    statuses = Counter(item["status"] for item in items if item.get("status"))
    policy = outcome.summary(baseline_path=baseline_path)
    policy["baseline_selected"] = outcome.diff is not None
    policy["baseline_path"] = baseline_path
    policy["suppressions_configured"] = len(rules)
    policy["statuses"] = dict(sorted(statuses.items()))
    policy["unclassified"] = sum(1 for item in items if not item.get("status"))
    if partial and baseline is not None:
        policy["baseline_skipped"] = (
            "the scan was cancelled; a baseline comparison needs a complete scan"
        )
    resolved = (
        [entry.to_dict() for entry in outcome.diff.resolved]
        if outcome.diff is not None
        else []
    )
    return items, policy, resolved, outcome


def _report_policy(policy: dict[str, Any]) -> dict[str, Any] | None:
    """Only render the report policy card when there is policy content."""
    return policy if policy.get("baseline") or policy.get("suppressions") else None


def _graph_metadata(value: Any) -> dict[str, str]:
    pairs: dict[str, str] = {}
    if isinstance(value, (list, tuple)):
        for item in value:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                pairs[item[0]] = item[1]
    return pairs


def _graph_finding(
    node: dict[str, Any], by_index: dict[int, dict[str, Any]]
) -> dict[str, Any] | None:
    index = node.get("finding_index")
    if not isinstance(index, int):
        return None
    finding = by_index.get(index)
    if not isinstance(finding, dict):
        return None
    return {
        key: finding.get(key)
        for key in (
            "index",
            "kind",
            "severity",
            "column",
            "detail",
            "file",
            "line",
            "status",
            "fingerprint",
            "requires_attention",
            "suppression",
            "previous",
        )
        if key in finding
    }


def _graph_payload(
    results: dict[str, Any],
    *,
    kind: str | None = None,
    relation: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    query: str | None = None,
    offset: int = 0,
    limit: int = GRAPH_DEFAULT_LIMIT,
    focus: str | None = None,
    finding: int | None = None,
) -> dict[str, Any]:
    """Project the already-built evidence graph into a bounded, filterable view.

    This does not build a graph: every node is an existing serialized
    ``Evidence`` record and every edge is an existing serialized
    ``EvidenceLink``. Invalid records are skipped and reported, never repaired
    into an invented relationship.
    """
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = max(1, min(int(limit), GRAPH_MAX_LIMIT))
    except (TypeError, ValueError):
        limit = GRAPH_DEFAULT_LIMIT

    intelligence = results.get("intelligence")
    if not isinstance(intelligence, dict):
        # Source-scan results spread the intelligence report at the top level;
        # dataset results nest it under "intelligence".
        intelligence = (
            results if isinstance(results.get("evidence"), (list, tuple)) else None
        )
    if not isinstance(intelligence, dict):
        return {
            "available": False,
            "reason": "this result has no evidence graph",
        }
    raw_evidence = intelligence.get("evidence")
    raw_links = intelligence.get("links", [])
    raw_clusters = intelligence.get("clusters", [])
    if not isinstance(raw_evidence, (list, tuple)):
        return {
            "available": False,
            "reason": "the evidence graph is malformed: evidence is not a list",
        }
    if not isinstance(raw_links, (list, tuple)) or not isinstance(
        raw_clusters, (list, tuple)
    ):
        return {
            "available": False,
            "reason": "the evidence graph is malformed: links or clusters are not lists",
        }

    nodes: dict[str, dict[str, Any]] = {}
    invalid_nodes = 0
    for item in raw_evidence:
        if not isinstance(item, dict):
            invalid_nodes += 1
            continue
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in nodes:
            invalid_nodes += 1
            continue
        nodes[evidence_id] = item
    if raw_evidence and not nodes:
        return {
            "available": False,
            "reason": "the evidence graph is malformed: no valid evidence nodes",
        }

    findings = results.get("findings")
    by_index: dict[int, dict[str, Any]] = {}
    if isinstance(findings, list):
        for item in findings:
            if (
                isinstance(item, dict)
                and isinstance(item.get("index"), int)
                and item["index"] not in by_index
            ):
                by_index[item["index"]] = item

    cluster_by_evidence: dict[str, dict[str, Any]] = {}
    cluster_rows: list[dict[str, Any]] = []
    for cluster in raw_clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        cluster_rows.append(cluster)
        evidence_ids = cluster.get("evidence_ids")
        if isinstance(evidence_ids, (list, tuple)):
            for evidence_id in evidence_ids:
                if isinstance(evidence_id, str):
                    cluster_by_evidence.setdefault(evidence_id, cluster)

    valid_links: list[dict[str, str]] = []
    dangling = 0
    invalid_links = 0
    for link in raw_links:
        if not isinstance(link, dict):
            invalid_links += 1
            continue
        source_id = link.get("source_id")
        target_id = link.get("target_id")
        link_relation = link.get("relation")
        rationale = link.get("rationale")
        if not all(
            isinstance(value, str) and value
            for value in (source_id, target_id, link_relation, rationale)
        ):
            invalid_links += 1
            continue
        if source_id not in nodes or target_id not in nodes:
            dangling += 1
            continue
        if source_id == target_id:
            invalid_links += 1
            continue
        valid_links.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation": link_relation,
                "rationale": rationale,
            }
        )

    neighbors: dict[str, list[dict[str, Any]]] = {eid: [] for eid in nodes}
    for link in valid_links:
        source_id = link["source_id"]
        target_id = link["target_id"]
        neighbors[source_id].append(
            {
                "evidence_id": target_id,
                "kind": str(nodes[target_id].get("kind") or ""),
                "relation": link["relation"],
                "direction": "out",
                "rationale": link["rationale"],
            }
        )
        neighbors[target_id].append(
            {
                "evidence_id": source_id,
                "kind": str(nodes[source_id].get("kind") or ""),
                "relation": link["relation"],
                "direction": "in",
                "rationale": link["rationale"],
            }
        )
    for items in neighbors.values():
        items.sort(key=lambda item: (item["evidence_id"], item["relation"]))

    component_of: dict[str, int] = {}
    component_count = 0
    for evidence_id in nodes:
        if evidence_id in component_of:
            continue
        stack = [evidence_id]
        component_of[evidence_id] = component_count
        while stack:
            current = stack.pop()
            for item in neighbors[current]:
                other = item["evidence_id"]
                if other not in component_of:
                    component_of[other] = component_count
                    stack.append(other)
        component_count += 1

    views: dict[str, dict[str, Any]] = {}
    haystacks: dict[str, str] = {}
    for evidence_id, item in nodes.items():
        metadata = _graph_metadata(item.get("metadata"))
        finding_view = _graph_finding(item, by_index)
        node_severity = (finding_view or {}).get("severity") or metadata.get("severity")
        node_status = (finding_view or {}).get("status") or "unclassified"
        cluster = cluster_by_evidence.get(evidence_id)
        views[evidence_id] = {
            "evidence_id": evidence_id,
            "kind": str(item.get("kind") or ""),
            "source": str(item.get("source") or ""),
            "description": str(item.get("description") or ""),
            "location": item.get("location"),
            "finding_index": (
                item.get("finding_index")
                if isinstance(item.get("finding_index"), int)
                else None
            ),
            "metadata": metadata,
            "role": metadata.get("role", "primary"),
            "finding": finding_view,
            "severity": node_severity or "low",
            "status": node_status,
            "cluster_id": cluster.get("cluster_id") if cluster else None,
            "anchor": cluster.get("anchor") if cluster else None,
            "anchor_kind": cluster.get("anchor_kind") if cluster else None,
            "component": component_of.get(evidence_id),
            "degree": len(neighbors[evidence_id]),
            "isolated": not neighbors[evidence_id],
            "neighbors": neighbors[evidence_id],
        }
        haystacks[evidence_id] = " ".join(
            str(value)
            for value in (
                evidence_id,
                item.get("kind"),
                item.get("source"),
                item.get("description"),
                item.get("location"),
                (finding_view or {}).get("kind"),
                (finding_view or {}).get("column"),
                (finding_view or {}).get("detail"),
                (finding_view or {}).get("file"),
                (finding_view or {}).get("fingerprint"),
            )
            if value is not None
        ).lower()

    kind_filter = (kind or "").strip()
    severity_filter = (severity or "").strip()
    status_filter = (status or "").strip().lower()
    relation_filter = (relation or "").strip()
    text_filter = (query or "").strip().lower()

    def matches(view: dict[str, Any]) -> bool:
        if kind_filter and view["kind"] != kind_filter:
            return False
        if severity_filter and view["severity"] != severity_filter:
            return False
        if status_filter:
            if status_filter in {"none", "unclassified"}:
                if view["status"] != "unclassified":
                    return False
            elif view["status"] != status_filter:
                return False
        return not text_filter or text_filter in haystacks[view["evidence_id"]]

    matching = [views[eid] for eid in nodes if matches(views[eid])]
    matching_ids = {view["evidence_id"] for view in matching}
    match_edges = [
        link
        for link in valid_links
        if link["source_id"] in matching_ids
        and link["target_id"] in matching_ids
        and (not relation_filter or link["relation"] == relation_filter)
    ]
    if relation_filter:
        incident = {
            evidence_id
            for link in match_edges
            for evidence_id in (link["source_id"], link["target_id"])
        }
        matching = [view for view in matching if view["evidence_id"] in incident]

    page_nodes = matching[offset : offset + limit]
    page_ids = {view["evidence_id"] for view in page_nodes}
    page_edges = [
        link
        for link in match_edges
        if link["source_id"] in page_ids and link["target_id"] in page_ids
    ]

    requested_focus = (focus or "").strip() or None
    if requested_focus is None and isinstance(finding, int):
        for evidence_id, view in views.items():
            if view["finding_index"] == finding and view["role"] != "detector":
                requested_focus = evidence_id
                break
    focus_view = views.get(requested_focus) if requested_focus else None

    def count(values: Any) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    relation_counts = dict(
        sorted(Counter(link["relation"] for link in valid_links).items())
    )

    summary = {
        "nodes": len(nodes),
        "links": len(valid_links),
        "clusters": len(cluster_rows),
        "components": component_count,
        "isolated": sum(1 for view in views.values() if view["isolated"]),
        "findings": len(by_index),
        "relations": relation_counts,
        "kinds": count(view["kind"] for view in views.values()),
        "sources": count(view["source"] for view in views.values()),
        "anchor_kinds": count(
            view["anchor_kind"] or "unclustered" for view in views.values()
        ),
        "severities": count(view["severity"] for view in views.values()),
        "statuses": count(view["status"] for view in views.values()),
        "invalid_nodes": invalid_nodes,
        "invalid_links": invalid_links,
        "dangling_links": dangling,
        "relationships_truncated": results.get("graph_links_truncated"),
        "matching_nodes": len(matching),
        "matching_edges": len(match_edges),
    }
    return {
        "available": True,
        "summary": summary,
        "filters": {
            "kinds": sorted(summary["kinds"]),
            "sources": sorted(summary["sources"]),
            "relations": sorted(relation_counts),
            "severities": sorted(summary["severities"]),
            "statuses": sorted(summary["statuses"]),
            "anchor_kinds": sorted(summary["anchor_kinds"]),
        },
        "nodes": page_nodes,
        "edges": page_edges,
        "page": {
            "offset": offset,
            "limit": limit,
            "total": len(matching),
            "returned": len(page_nodes),
            "next_offset": (offset + limit if offset + limit < len(matching) else None),
        },
        "focus_requested": requested_focus,
        "focus": focus_view,
    }


def _wall_time(value: float | None) -> str | None:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        if isinstance(value, (int, float))
        else None
    )


def _sse_event(name: str, seq: int, payload: dict[str, Any]) -> str:
    return (
        f"id: {seq}\nevent: {name}\n"
        f"data: {json.dumps(payload, default=str, sort_keys=True)}\n\n"
    )


def _scan_change_key(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        snapshot["state"],
        snapshot["done"],
        snapshot["total"],
        snapshot["current"],
        tuple(sorted(snapshot["counts"].items())),
        tuple((item.get("path"), item.get("message")) for item in snapshot["errors"]),
    )


def _dataset_change_key(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    return (
        snapshot["state"],
        json.dumps(snapshot["request"], sort_keys=True, default=str),
        json.dumps(snapshot["error"], sort_keys=True, default=str),
    )


def _scan_failure(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if snapshot["state"] != "failed":
        return None
    errors = list(snapshot.get("errors") or [])
    message = str(errors[-1].get("message")) if errors else "the scan failed"
    return {"message": message, "errors": errors[:10]}


def _dataset_failure(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    if snapshot["state"] != "failed":
        return None
    return snapshot.get("error") or {"message": "the audit failed"}


def _sse_stream(session: Any, change_key, failure) -> Iterator[str]:
    """Stream one session's snapshots until it reaches a terminal state.

    The stream reads the live session instead of queueing events, so a slow or
    disconnected client cannot grow memory. Duplicate snapshots are dropped and
    ``done`` is always the last event.
    """
    seq = 0
    last: tuple[Any, ...] | None = None
    last_beat = time.monotonic()
    yield "retry: 1000\n\n"
    while seq < SSE_MAX_EVENTS:
        snapshot = session.snapshot()
        key = change_key(snapshot)
        if last is None or key != last:
            seq += 1
            last = key
            yield _sse_event(
                "snapshot", seq, {"seq": seq, "type": "snapshot", **snapshot}
            )
            last_beat = time.monotonic()
        if snapshot["state"] not in ACTIVE_STATES:
            problem = failure(snapshot)
            if problem is not None:
                seq += 1
                yield _sse_event(
                    "failure",
                    seq,
                    {"seq": seq, "type": "failure", **snapshot, "failure": problem},
                )
            seq += 1
            yield _sse_event("done", seq, {"seq": seq, "type": "done", **snapshot})
            return
        now = time.monotonic()
        if now - last_beat >= SSE_HEARTBEAT_SECONDS:
            yield ": keep-alive\n\n"
            last_beat = now
        time.sleep(SSE_POLL_SECONDS)
    seq += 1
    yield _sse_event(
        "failure",
        seq,
        {
            "seq": seq,
            "type": "failure",
            "failure": {
                "message": "live updates reached their event limit; reconnect to continue"
            },
        },
    )


def _finding_ref(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    ref = {key: item.get(key) for key in _COMPARISON_FIELDS if key in item}
    evidence = item.get("evidence")
    if isinstance(evidence, (list, tuple)):
        ref["evidence"] = list(evidence)[:20]
        ref["evidence_total"] = len(evidence)
    return ref


def _comparison_baseline(findings: list[dict[str, Any]]) -> Baseline:
    """Reuse ``Baseline.from_findings`` for a previous session's identities."""
    views = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        if not all(
            key in item for key in ("fingerprint", "kind", "severity", "column")
        ):
            continue
        views.append(_policy_view(item))
    return Baseline.from_findings(views)


def _compare_sessions(
    previous: Any,
    current: Any,
    *,
    kind: str,
    root: Path,
    previous_meta: dict[str, Any],
    current_meta: dict[str, Any],
) -> dict[str, Any]:
    """Compare two retained sessions through the existing baseline semantics."""
    with previous.lock:
        previous_state = previous.state
        previous_results = previous.results
        previous_errors = list(previous.errors) if kind == "scan" else []
    with current.lock:
        current_state = current.state
        current_results = current.results
        current_errors = list(current.errors) if kind == "scan" else []

    checks: list[dict[str, Any]] = []
    if kind == "scan":
        fields = [
            ("target", previous.target, current.target),
            ("SQL scanning", previous.include_sql, current.include_sql),
        ]
    else:
        before = previous.request()
        after = current.request()
        fields = [
            ("path", _relative(previous.path, root), _relative(current.path, root)),
            ("file", before["file"], after["file"]),
            ("target", before["target"], after["target"]),
            ("threshold", before["threshold"], after["threshold"]),
            ("held-out file", before["test"], after["test"]),
            ("time column", before["time_column"], after["time_column"]),
            ("group column", before["group_column"], after["group_column"]),
        ]
    mismatched = []
    for name, before, after in fields:
        compatible = before == after
        checks.append(
            {
                "name": name,
                "compatible": compatible,
                "previous": before,
                "current": after,
            }
        )
        if not compatible:
            mismatched.append(name)

    base = {
        "type": kind,
        "checks": checks,
        "previous": previous_meta,
        "current": current_meta,
    }
    if previous_results is None or current_results is None:
        return {
            **base,
            "available": False,
            "compatible": False,
            "reason": "both sessions need finished results before they can be compared",
        }
    if mismatched:
        return {
            **base,
            "available": False,
            "compatible": False,
            "reason": (
                "these sessions differ in "
                + ", ".join(mismatched)
                + "; compare sessions that use the same target and settings"
            ),
        }

    previous_findings = [
        item
        for item in (previous_results.get("findings") or [])
        if isinstance(item, dict) and item.get("fingerprint")
    ]
    current_findings = [
        item
        for item in (current_results.get("findings") or [])
        if isinstance(item, dict) and item.get("fingerprint")
    ]

    def by_fingerprint(findings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for item in sorted(
            findings,
            key=lambda entry: (
                str(entry.get("kind")),
                str(entry.get("column")),
                str(entry.get("fingerprint")),
            ),
        ):
            indexed.setdefault(str(item["fingerprint"]), item)
        return indexed

    previous_by_fp = by_fingerprint(previous_findings)
    current_by_fp = by_fingerprint(current_findings)
    try:
        baseline = _comparison_baseline(previous_findings)
        diff = compare(
            baseline,
            [
                _policy_view(item)
                for item in current_findings
                if all(
                    key in item for key in ("fingerprint", "kind", "severity", "column")
                )
            ],
        )
    except (BaselineError, KeyError, TypeError) as exc:
        return {
            **base,
            "available": False,
            "compatible": False,
            "reason": f"the saved findings cannot be compared: {exc}",
        }

    previous_complete = previous_state == "done" and not previous_errors
    current_complete = current_state == "done" and not current_errors
    limitations: list[str] = []
    if not previous_complete:
        limitations.append(
            "The previous session was partial; findings absent from it are "
            "inconclusive, not newly added."
        )
    if not current_complete:
        limitations.append(
            "The current session was partial; findings absent from it are "
            "inconclusive, not resolved."
        )
    notes = [
        "Comparison reuses the existing stable finding fingerprints and "
        "baseline classification; policy statuses come from each session."
    ]

    counts = {
        "added": 0,
        "unchanged": 0,
        "changed": 0,
        "no_longer_present": 0,
        "inconclusive": 0,
    }
    changes: list[dict[str, Any]] = []

    def add(
        change: str,
        current_item: dict[str, Any] | None,
        previous_item: dict[str, Any] | None,
        fingerprint: str,
        reason: str | None = None,
    ) -> None:
        direction = (
            "both"
            if current_item is not None and previous_item is not None
            else ("current_only" if current_item is not None else "previous_only")
        )
        subject = current_item or previous_item or {}
        changes.append(
            {
                "change": change,
                "direction": direction,
                "fingerprint": fingerprint,
                "kind": subject.get("kind"),
                "severity": subject.get("severity"),
                "column": subject.get("column"),
                "resolved": change == "no_longer_present",
                "current": _finding_ref(current_item),
                "previous": _finding_ref(previous_item),
                "reason": reason,
            }
        )
        counts[change] += 1

    for finding in diff.new:
        fingerprint = finding.fingerprint
        if previous_complete:
            add("added", current_by_fp.get(fingerprint), None, fingerprint)
        else:
            add(
                "inconclusive",
                current_by_fp.get(fingerprint),
                None,
                fingerprint,
                "the previous session was partial; this finding may have been missed",
            )
    for finding in diff.unchanged:
        fingerprint = finding.fingerprint
        add(
            "unchanged",
            current_by_fp.get(fingerprint),
            previous_by_fp.get(fingerprint),
            fingerprint,
        )
    for finding in diff.changed:
        fingerprint = finding.fingerprint
        add(
            "changed",
            current_by_fp.get(fingerprint),
            previous_by_fp.get(fingerprint),
            fingerprint,
        )
    for entry in diff.resolved:
        fingerprint = entry.fingerprint
        previous_item = previous_by_fp.get(fingerprint)
        if current_complete:
            add("no_longer_present", None, previous_item, fingerprint)
        else:
            add(
                "inconclusive",
                None,
                previous_item,
                fingerprint,
                "the current session was partial; absence does not mean this finding is resolved",
            )

    total = len(changes)
    counts_consistent = sum(counts.values()) == total
    suppressed = sum(
        1 for item in current_by_fp.values() if item.get("status") == "suppressed"
    )
    unclassified = sum(1 for item in current_by_fp.values() if not item.get("status"))
    if unclassified:
        notes.append(
            "Findings without a policy status stay unclassified; comparison does "
            "not relabel them."
        )
    return {
        **base,
        "available": True,
        "compatible": True,
        "reason": None,
        "complete": {"previous": previous_complete, "current": current_complete},
        "limitations": limitations,
        "notes": notes,
        "summary": {
            **counts,
            "total": total,
            "suppressed": suppressed,
            "unclassified": unclassified,
        },
        "changes": changes[:MAX_COMPARISON_CHANGES],
        "truncated": total > MAX_COMPARISON_CHANGES,
        "counts_consistent": counts_consistent,
    }


class WorkspaceScanManager:
    """Owns the workspace root and the single active scan or audit thread."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._scans: dict[str, ScanSession] = {}
        self._order: list[str] = []
        self._datasets: dict[str, DatasetSession] = {}
        self._dataset_order: list[str] = []
        self._active: ScanSession | DatasetSession | None = None
        self._lock = threading.Lock()

    def resolve(self, value: str) -> Path:
        text = (value or "").strip()
        if not text:
            raise WorkspacePathError("a path is required")
        # POSIX reads "..\outside.csv" as one filename, so a Windows separator must be
        # normalized before parsing or the traversal never reaches the containment check.
        # shortcut: a POSIX file whose name contains "\" is unreachable from the workspace.
        candidate = Path(text.replace("\\", "/"))
        target = candidate if candidate.is_absolute() else self.root / candidate
        target = target.resolve()
        if target != self.root and not target.is_relative_to(self.root):
            raise WorkspacePathError("path is outside the workspace root")
        return target

    def _policy(
        self, value: str | None
    ) -> tuple[Baseline | None, tuple[SuppressionRule, ...], str | None]:
        """Load the project config and the selected baseline before a scan runs."""
        try:
            cfg = load_config(cwd=self.root)
        except ConfigError as exc:
            raise WorkspacePathError(str(exc)) from None
        selected = (value or "").strip()
        if selected:
            path = self.resolve(selected)
            label = _relative(path, self.root)
        elif cfg.baseline_path is not None:
            path = cfg.baseline_path
            label = _relative(path, self.root)
        else:
            return None, cfg.suppressions, None
        if path != self.root and not path.is_relative_to(self.root):
            raise WorkspacePathError("baseline is outside the workspace root")
        try:
            baseline = Baseline.load(path)
        except (BaselineError, UnicodeError) as exc:
            raise WorkspacePathError(str(exc)) from None
        return baseline, cfg.suppressions, label

    def _claim(self, session) -> None:
        """Take the one activity slot; callers already hold ``self._lock``."""
        if self._active is not None and self._active.state in ACTIVE_STATES:
            raise WorkspaceBusy("a scan or audit is already running")
        self._active = session

    def start(
        self,
        path: str = ".",
        *,
        sql: bool = False,
        baseline: str | None = None,
        include_ai: bool = False,
    ) -> ScanSession:
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
        baseline_doc, rules, baseline_label = self._policy(baseline)
        with self._lock:
            session = ScanSession(
                scan_id=secrets.token_hex(6),
                files=files,
                include_sql=sql,
                include_ai=include_ai,
                target=_relative(target, self.root),
                baseline=baseline_doc,
                rules=rules,
                baseline_path=baseline_label,
            )
            self._claim(session)
            self._scans[session.scan_id] = session
            self._order.append(session.scan_id)
            _trim(self._order, self._scans, MAX_SCANS)
        threading.Thread(
            target=self._run,
            args=(session,),
            name=f"leaklens-workspace-{session.scan_id}",
            daemon=True,
        ).start()
        return session

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
        items, policy, resolved, outcome = _apply_workspace_policy(
            [
                _finding_payload(finding, index, self.root)
                for index, finding in enumerate(findings)
            ],
            baseline=session.baseline,
            rules=session.rules,
            baseline_path=session.baseline_path,
            partial=state != "done",
        )
        payload = {
            "scan_id": session.scan_id,
            "state": state,
            "trust_score": score(outcome.active),
            "counts": dict(session.counts),
            "files": len(session.files),
            "errors": list(session.errors),
            "duration_ms": session.elapsed_ms(),
            # Clusters, links, evidence, confidence, root causes and next
            # actions are the existing deterministic intelligence layer.
            **intelligence,
            "policy": policy,
            "findings": items,
            "graph_links_truncated": report.links_truncated,
        }
        if resolved:
            payload["resolved"] = resolved
        if session.include_ai:
            # Policy runs first so the optional layer never sees a suppressed
            # finding as active; failure stays a normal result, never an error.
            active = [
                finding
                for finding, item in zip(findings, items)
                if item["fingerprint"] not in outcome.suppressed
            ]
            ai_payload = None
            try:
                from .ai import analyze, collect_code_sources

                ai_payload = analyze(
                    active,
                    code=collect_code_sources(
                        path
                        for path in session.files
                        if path.suffix.lower() in {".py", ".ipynb"}
                    ),
                    trust_score=payload["trust_score"],
                ).to_dict()
            except Exception:  # noqa: BLE001 - optional AI must not break the scan
                ai_payload = None
            payload["ai"] = ai_payload
            payload["ai_unavailable"] = ai_payload is None
        report_html = render(
            {
                "trust_score": payload["trust_score"],
                "findings": _report_findings(payload["findings"]),
                "intelligence": intelligence,
                "policy": _report_policy(policy),
                "ai": payload.get("ai"),
            },
            title="LeakLens workspace report",
        )
        with session.lock:
            session.state = state
            session.current = None
            session.completed_at = time.time()
            session.results = payload
            session.report_html = report_html

    def _dataset_bytes(self, path: Path, label: str) -> bytes:
        if not path.is_file():
            raise WorkspacePathError(f"not a dataset file: {label}")
        if path.stat().st_size > DATASET_LIMITS.max_bytes:
            mib = DATASET_LIMITS.max_bytes / (1024 * 1024)
            raise UploadRejected(
                f"The dataset is larger than the {mib:.1f} MiB workspace limit.",
                413,
                "upload_too_large",
            )
        return path.read_bytes()

    def open_dataset(self, value: str):
        """Resolve a workspace file and run the public upload guard on it."""
        target = self.resolve(value)
        return guard_upload(
            target.name, self._dataset_bytes(target, value), DATASET_LIMITS
        )

    def inspect_dataset(self, value: str) -> dict[str, Any]:
        return run_inspect_job(self.open_dataset(value), DATASET_LIMITS)

    def start_dataset(
        self,
        *,
        path: str,
        target: str,
        threshold: float = 0.95,
        test_path: str | None = None,
        time_col: str | None = None,
        group_col: str | None = None,
        baseline: str | None = None,
        include_ai: bool = False,
    ) -> DatasetSession:
        threshold = _threshold(threshold)
        upload = self.open_dataset(path)
        test_upload = self.open_dataset(test_path) if test_path else None
        name = (target or "").strip()
        if not name:
            raise UploadRejected(
                "Choose a target column before scanning.", 422, "missing_target"
            )
        baseline_doc, rules, baseline_label = self._policy(baseline)
        with self._lock:
            session = DatasetSession(
                dataset_id=secrets.token_hex(6),
                path=self.resolve(path),
                target=name,
                threshold=threshold,
                test_path=self.resolve(test_path) if test_path else None,
                time_col=(time_col or "").strip() or None,
                group_col=(group_col or "").strip() or None,
                include_ai=include_ai,
                baseline=baseline_doc,
                rules=rules,
                baseline_path=baseline_label,
            )
            self._claim(session)
            self._datasets[session.dataset_id] = session
            self._dataset_order.append(session.dataset_id)
            _trim(self._dataset_order, self._datasets, MAX_DATASETS)
        threading.Thread(
            target=self._run_dataset,
            args=(session, upload, test_upload),
            name=f"leaklens-dataset-{session.dataset_id}",
            daemon=True,
        ).start()
        return session

    def get_dataset(self, dataset_id: str) -> DatasetSession:
        session = self._datasets.get(dataset_id)
        if session is None:
            raise KeyError(dataset_id)
        return session

    def cancel_dataset(self, dataset_id: str) -> DatasetSession:
        session = self.get_dataset(dataset_id)
        with session.lock:
            if session.state == "queued":
                session.cancel.set()
                session.state = "cancelled"
                session.completed_at = time.time()
                return session
            if session.state == "auditing":
                raise WorkspaceBusy(
                    "the audit engine is already running and cannot be interrupted"
                )
        raise WorkspaceBusy("audit is not running")

    def _scan_history(self, session: ScanSession) -> dict[str, Any]:
        with session.lock:
            results = session.results
            state = session.state
            counts = dict(session.counts)
            completed_at = session.completed_at
            errors = len(session.errors)
            done = session.done
            target = session.target
            created_at = session.created_at
            duration = (
                results.get("duration_ms")
                if isinstance(results, dict)
                else session.elapsed_ms()
            )
            findings = (
                len(results.get("findings") or [])
                if isinstance(results, dict)
                else None
            )
            trust = results.get("trust_score") if isinstance(results, dict) else None
            report = session.report_html is not None
        return {
            "id": session.scan_id,
            "type": "scan",
            "target": target,
            "state": state,
            "started_at": _wall_time(created_at),
            "completed_at": _wall_time(completed_at),
            "duration_ms": duration,
            "results_available": results is not None,
            "report_available": report,
            "summary": {
                "findings": findings,
                "high": counts.get("high", 0),
                "medium": counts.get("medium", 0),
                "low": counts.get("low", 0),
                "trust_score": trust,
                "files": len(session.files),
                "done": done,
                "errors": errors,
            },
        }

    def _dataset_history(self, session: DatasetSession) -> dict[str, Any]:
        with session.lock:
            results = session.results
            state = session.state
            completed_at = session.completed_at
            created_at = session.created_at
            duration = (
                results.get("elapsed_ms")
                if isinstance(results, dict)
                else session.elapsed_ms()
            )
            request = session.request()
            path = _relative(session.path, self.root)
            summary = results.get("summary") if isinstance(results, dict) else None
            summary = summary if isinstance(summary, dict) else {}
            findings = (
                len(results.get("findings") or [])
                if isinstance(results, dict)
                else None
            )
            trust = results.get("trust_score") if isinstance(results, dict) else None
            counts = results.get("counts") if isinstance(results, dict) else None
            counts = counts if isinstance(counts, dict) else {}
            report = session.report_html is not None
        return {
            "id": session.dataset_id,
            "type": "dataset",
            "target": path,
            "state": state,
            "started_at": _wall_time(created_at),
            "completed_at": _wall_time(completed_at),
            "duration_ms": duration,
            "results_available": results is not None,
            "report_available": report,
            "summary": {
                "findings": findings,
                "high": counts.get("high", 0),
                "medium": counts.get("medium", 0),
                "low": counts.get("low", 0),
                "trust_score": trust,
                "rows": summary.get("rows"),
                "columns": summary.get("columns"),
                "held_out_rows": summary.get("test_rows"),
                "target": request.get("target"),
                "threshold": request.get("threshold"),
                "file": request.get("file"),
            },
        }

    def history(self) -> dict[str, Any]:
        """Return the bounded in-memory session list, newest first."""
        with self._lock:
            scans = [self._scans[key] for key in self._order if key in self._scans]
            datasets = [
                self._datasets[key]
                for key in self._dataset_order
                if key in self._datasets
            ]
        scan_rows = [self._scan_history(session) for session in scans]
        dataset_rows = [self._dataset_history(session) for session in datasets]
        entries = sorted(
            [*scan_rows, *dataset_rows],
            key=lambda row: (row["started_at"] or "", row["id"]),
            reverse=True,
        )
        return {
            "retained": True,
            "persistent": False,
            "note": HISTORY_NOTE,
            "limits": {
                "scans": MAX_SCANS,
                "datasets": MAX_DATASETS,
                "total": MAX_SCANS + MAX_DATASETS,
            },
            "scans": scan_rows,
            "datasets": dataset_rows,
            "entries": entries,
        }

    def _history_entry(self, session: Any, kind: str) -> dict[str, Any]:
        if kind == "scan":
            return self._scan_history(session)
        return self._dataset_history(session)

    def compare(self, kind: str, previous: Any, current: Any) -> dict[str, Any]:
        return _compare_sessions(
            previous,
            current,
            kind=kind,
            root=self.root,
            previous_meta=self._history_entry(previous, kind),
            current_meta=self._history_entry(current, kind),
        )

    def _run_dataset(self, session: DatasetSession, upload, test_upload) -> None:
        with session.lock:
            if session.cancel.is_set():
                session.completed_at = session.completed_at or time.time()
                return
            session.state = "auditing"
        try:
            # The public audit job is the whole engine: parse, validate, scan,
            # impact, intelligence payload and HTML report.
            payload = run_audit_job(
                upload,
                DATASET_LIMITS,
                session.target,
                session.threshold,
                test_upload,
                session.time_col,
                session.group_col,
                include_intelligence=True,
                include_ai=session.include_ai,
            )
        except UploadRejected as exc:
            with session.lock:
                session.state = "failed"
                session.completed_at = time.time()
                session.error = {
                    "status": exc.status_code,
                    "code": exc.code,
                    "message": exc.message,
                }
            return
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            with session.lock:
                session.state = "failed"
                session.completed_at = time.time()
                session.error = {
                    "status": 500,
                    "code": "audit_failed",
                    "message": str(exc)[:300],
                }
            return
        payload.pop("report_html", None)
        for index, finding in enumerate(payload["findings"]):
            finding["index"] = index
            finding["advice"] = advice(finding["kind"]) or None
        items, policy, resolved, outcome = _apply_workspace_policy(
            payload["findings"],
            baseline=session.baseline,
            rules=session.rules,
            baseline_path=session.baseline_path,
        )
        payload["findings"] = items
        payload["trust_score"] = score(outcome.active)
        payload["policy"] = policy
        if resolved:
            payload["resolved"] = resolved
        report_html = render(
            {
                "trust_score": payload["trust_score"],
                "findings": items,
                "audit": payload.get("audit"),
                "impact": payload.get("impact"),
                "intelligence": payload.get("intelligence"),
                "graph_baseline": payload.get("graph_baseline"),
                "ai": payload.get("ai"),
                "policy": _report_policy(policy),
            },
            title=f"LeakLens report - {payload['summary']['file']}",
        )
        results = {
            "dataset_id": session.dataset_id,
            "state": "done",
            "request": session.request(),
            "elapsed_ms": session.elapsed_ms(),
            **payload,
        }
        with session.lock:
            session.state = "done"
            session.completed_at = time.time()
            session.results = results
            session.report_html = report_html


def _unfinished(session: DatasetSession) -> str:
    """Why a finished-but-empty audit has no results, without hiding the reason."""
    if session.error:
        return str(session.error.get("message", "the audit failed"))
    return f"audit is {session.state}; results are not available"


class ScanRequest(BaseModel):
    path: str = Field(default=".", max_length=1000)
    sql: bool = Field(default=False)
    baseline: str | None = Field(default=None, max_length=1000)
    include_ai: bool = Field(default=False)


class DatasetPathRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


class DatasetRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    target: str = Field(default="", max_length=200)
    threshold: float = Field(default=0.95)
    test_path: str | None = Field(default=None, max_length=1000)
    time_col: str | None = Field(default=None, max_length=200)
    group_col: str | None = Field(default=None, max_length=200)
    baseline: str | None = Field(default=None, max_length=1000)
    include_ai: bool = Field(default=False)


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
        try:
            cfg = load_config(cwd=manager.root)
            config_info = {
                "path": _relative(cfg.path, manager.root) if cfg.path else None,
                "baseline": (
                    _relative(cfg.baseline_path, manager.root)
                    if cfg.baseline_path is not None
                    else None
                ),
                "suppressions": len(cfg.suppressions),
                "error": None,
            }
        except ConfigError as exc:
            config_info = {
                "path": None,
                "baseline": None,
                "suppressions": 0,
                "error": str(exc),
            }
        return {
            "version": __version__,
            "root": manager.root.as_posix(),
            "max_files": MAX_FILES,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "source_context": SOURCE_CONTEXT,
            "sql_available": sql_available(),
            "dataset_limits": DATASET_LIMITS.to_public_dict(),
            "dataset_formats": available_formats(DATASET_LIMITS),
            "threshold_range": [0.5, 1.0],
            "default_threshold": 0.95,
            "config": config_info,
        }

    @app.get("/api/history")
    def history() -> dict[str, Any]:
        return manager.history()

    def _compare(kind: str, previous: str, current: str) -> dict[str, Any]:
        try:
            if kind == "scan":
                before = manager.get(previous)
                after = manager.get(current)
            else:
                before = manager.get_dataset(previous)
                after = manager.get_dataset(current)
        except KeyError:
            raise HTTPException(404, f"unknown {kind} session") from None
        return manager.compare(kind, before, after)

    @app.get("/api/compare/scans")
    def compare_scans(previous: str, current: str) -> dict[str, Any]:
        return _compare("scan", previous, current)

    @app.get("/api/compare/datasets")
    def compare_datasets(previous: str, current: str) -> dict[str, Any]:
        return _compare("dataset", previous, current)

    @app.post("/api/scans", status_code=202)
    def start_scan(req: ScanRequest) -> dict[str, str]:
        try:
            session = manager.start(
                req.path, sql=req.sql, baseline=req.baseline, include_ai=req.include_ai
            )
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

    @app.get("/api/scans/{scan_id}/events")
    def scan_events(scan_id: str) -> StreamingResponse:
        try:
            session = manager.get(scan_id)
        except KeyError:
            raise HTTPException(404, "unknown scan") from None
        return StreamingResponse(
            _sse_stream(session, _scan_change_key, _scan_failure),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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

    @app.get("/api/scans/{scan_id}/graph")
    def scan_graph(
        scan_id: str,
        kind: str | None = None,
        relation: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        q: str | None = None,
        offset: int = 0,
        limit: int = GRAPH_DEFAULT_LIMIT,
        focus: str | None = None,
        finding: int | None = None,
    ) -> dict[str, Any]:
        try:
            session = manager.get(scan_id)
        except KeyError:
            raise HTTPException(404, "unknown scan") from None
        if session.results is None:
            raise HTTPException(409, "scan has not finished")
        return _graph_payload(
            session.results,
            kind=kind,
            relation=relation,
            severity=severity,
            status=status,
            query=q,
            offset=offset,
            limit=limit,
            focus=focus,
            finding=finding,
        )

    @app.get("/api/scans/{scan_id}/report", response_class=HTMLResponse)
    def scan_report(scan_id: str) -> str:
        try:
            session = manager.get(scan_id)
        except KeyError:
            raise HTTPException(404, "unknown scan") from None
        if session.report_html is None:
            raise HTTPException(409, "scan has not finished")
        return session.report_html

    @app.post("/api/datasets/inspect")
    def inspect_dataset(req: DatasetPathRequest) -> dict[str, Any]:
        try:
            return manager.inspect_dataset(req.path)
        except WorkspacePathError as exc:
            raise HTTPException(422, str(exc)) from None
        except UploadRejected as exc:
            raise HTTPException(exc.status_code, exc.message) from None

    @app.post("/api/datasets", status_code=202)
    def start_dataset(req: DatasetRequest) -> dict[str, str]:
        try:
            session = manager.start_dataset(
                path=req.path,
                target=req.target,
                threshold=req.threshold,
                test_path=req.test_path,
                time_col=req.time_col,
                group_col=req.group_col,
                baseline=req.baseline,
                include_ai=req.include_ai,
            )
        except WorkspacePathError as exc:
            raise HTTPException(422, str(exc)) from None
        except UploadRejected as exc:
            raise HTTPException(exc.status_code, exc.message) from None
        except WorkspaceBusy as exc:
            raise HTTPException(409, str(exc)) from None
        return {"dataset_id": session.dataset_id}

    @app.get("/api/datasets/{dataset_id}")
    def dataset_status(dataset_id: str) -> dict[str, Any]:
        try:
            return manager.get_dataset(dataset_id).snapshot()
        except KeyError:
            raise HTTPException(404, "unknown dataset audit") from None

    @app.get("/api/datasets/{dataset_id}/events")
    def dataset_events(dataset_id: str) -> StreamingResponse:
        try:
            session = manager.get_dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, "unknown dataset audit") from None
        return StreamingResponse(
            _sse_stream(session, _dataset_change_key, _dataset_failure),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/datasets/{dataset_id}/cancel")
    def cancel_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            session = manager.cancel_dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, "unknown dataset audit") from None
        except WorkspaceBusy as exc:
            raise HTTPException(409, str(exc)) from None
        return session.snapshot()

    @app.get("/api/datasets/{dataset_id}/results")
    def dataset_results(dataset_id: str) -> dict[str, Any]:
        try:
            session = manager.get_dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, "unknown dataset audit") from None
        if session.results is None:
            raise HTTPException(409, _unfinished(session))
        return session.results

    @app.get("/api/datasets/{dataset_id}/graph")
    def dataset_graph(
        dataset_id: str,
        kind: str | None = None,
        relation: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        q: str | None = None,
        offset: int = 0,
        limit: int = GRAPH_DEFAULT_LIMIT,
        focus: str | None = None,
        finding: int | None = None,
    ) -> dict[str, Any]:
        try:
            session = manager.get_dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, "unknown dataset audit") from None
        if session.results is None:
            raise HTTPException(409, _unfinished(session))
        return _graph_payload(
            session.results,
            kind=kind,
            relation=relation,
            severity=severity,
            status=status,
            query=q,
            offset=offset,
            limit=limit,
            focus=focus,
            finding=finding,
        )

    @app.get("/api/datasets/{dataset_id}/report", response_class=HTMLResponse)
    def dataset_report(dataset_id: str) -> str:
        try:
            session = manager.get_dataset(dataset_id)
        except KeyError:
            raise HTTPException(404, "unknown dataset audit") from None
        if session.report_html is None:
            raise HTTPException(409, _unfinished(session))
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
