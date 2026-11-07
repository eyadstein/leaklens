"""Deterministic, baseline-aware evidence-graph snapshots and comparison.

A snapshot is a validated JSON view of one evidence graph plus the finding
fingerprints the graph was built from. Comparison matches evidence by stable
finding identity (never by list position or display id), then classifies it as
new, unchanged, changed, resolved or uncertain. It never turns an evidence
change into a leakage claim and never changes the findings or trust score.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leaklens import __version__

from .provenance import EvidenceGraph
from .schemas import (
    ConfidenceAssessment,
    Evidence,
    EvidenceCluster,
    EvidenceLink,
)

SNAPSHOT_SCHEMA = "leaklens-evidence-graph"
SNAPSHOT_VERSION = 1
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_EVIDENCE = 50_000
MAX_SNAPSHOT_CLUSTERS = 50_000
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL = {
    "schema",
    "version",
    "fingerprint_schema",
    "tool_version",
    "complete",
    "context",
    "summary",
    "finding_fingerprints",
    "evidence",
    "clusters",
    "links",
}
_EVIDENCE_FIELDS = {
    "evidence_id",
    "source",
    "kind",
    "description",
    "finding_index",
    "location",
    "supports",
    "metadata",
}
_CLUSTER_FIELDS = {
    "cluster_id",
    "anchor",
    "anchor_kind",
    "finding_kinds",
    "finding_indices",
    "evidence_ids",
    "sources",
    "severity",
    "priority",
    "confidence",
    "measured_impact",
}
_CONFIDENCE_FIELDS = {
    "score",
    "level",
    "reasons",
    "supporting_evidence",
    "contradicting_evidence",
}
_LINK_FIELDS = {"source_id", "target_id", "relation", "rationale"}
_STATES = ("new", "unchanged", "changed", "resolved", "uncertain")
_RELATION_STATES = ("added", "removed")


class GraphBaselineError(ValueError):
    """A snapshot is missing, malformed, incompatible or cannot be written."""


def _text(value: Any, field: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise GraphBaselineError(f"{field} must be a string")
    text = value.strip()
    if required and not text:
        raise GraphBaselineError(f"{field} must not be empty")
    if len(text) > limit:
        raise GraphBaselineError(f"{field} exceeds {limit} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise GraphBaselineError(f"{field} contains control characters")
    return text


def _list(value: Any, field: str) -> list:
    if not isinstance(value, list):
        raise GraphBaselineError(f"{field} must be an array")
    return value


def _exact_keys(value: Any, expected: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise GraphBaselineError(f"{where} must be an object")
    unknown = sorted(set(value) - expected)
    if unknown:
        raise GraphBaselineError(f"{where} has unsupported keys: {', '.join(unknown)}")
    missing = sorted(expected - set(value))
    if missing:
        raise GraphBaselineError(f"{where} is missing: {', '.join(missing)}")
    return value


def _parse_metadata(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, dict):
        items = sorted(value.items())
    elif isinstance(value, list):
        items = []
        for index, pair in enumerate(value):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(item, str) for item in pair)
            ):
                raise GraphBaselineError(f"{where}[{index}] must be a two-string array")
            items.append((pair[0], pair[1]))
    else:
        raise GraphBaselineError(f"{where} must be an object or an array of pairs")
    return tuple(items)


def _evidence(value: Any, where: str) -> Evidence:
    data = _exact_keys(value, _EVIDENCE_FIELDS, where)
    index = data["finding_index"]
    if index is not None and (type(index) is not int or index < 0):
        raise GraphBaselineError(f"{where}.finding_index must be non-negative or null")
    location = data["location"]
    if location is not None:
        location = _text(location, f"{where}.location", 2_000)
    if type(data["supports"]) is not bool:
        raise GraphBaselineError(f"{where}.supports must be a boolean")
    return Evidence(
        evidence_id=_text(data["evidence_id"], f"{where}.evidence_id", 300),
        source=_text(data["source"], f"{where}.source", 40),
        kind=_text(data["kind"], f"{where}.kind", 120),
        description=_text(data["description"], f"{where}.description", 20_000),
        finding_index=index,
        location=location,
        supports=data["supports"],
        metadata=_parse_metadata(data["metadata"], f"{where}.metadata"),
    )


def _strings(value: Any, field: str) -> tuple[str, ...]:
    return tuple(_text(item, field, 2_000) for item in _list(value, field))


def _confidence(value: Any, where: str) -> ConfidenceAssessment:
    data = _exact_keys(value, _CONFIDENCE_FIELDS, where)
    score = data["score"]
    if type(score) not in (int, float):
        raise GraphBaselineError(f"{where}.score must be a number")
    supporting = data["supporting_evidence"]
    contradicting = data["contradicting_evidence"]
    for name, number in (
        ("supporting_evidence", supporting),
        ("contradicting_evidence", contradicting),
    ):
        if type(number) is not int:
            raise GraphBaselineError(f"{where}.{name} must be an integer")
    reasons = _strings(data["reasons"], f"{where}.reasons")
    try:
        return ConfidenceAssessment(
            score=float(score),
            level=_text(data["level"], f"{where}.level", 40),
            reasons=reasons,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
        )
    except ValueError as exc:
        raise GraphBaselineError(f"{where}: {exc}") from None


def _cluster(value: Any, where: str) -> EvidenceCluster:
    data = _exact_keys(value, _CLUSTER_FIELDS, where)
    indices = []
    for index, item in enumerate(
        _list(data["finding_indices"], f"{where}.finding_indices")
    ):
        if type(item) is not int or item < 0:
            raise GraphBaselineError(
                f"{where}.finding_indices[{index}] must be a non-negative integer"
            )
        indices.append(item)
    try:
        return EvidenceCluster(
            cluster_id=_text(data["cluster_id"], f"{where}.cluster_id", 300),
            anchor=_text(data["anchor"], f"{where}.anchor", 1_000),
            anchor_kind=_text(data["anchor_kind"], f"{where}.anchor_kind", 40),
            finding_kinds=_strings(data["finding_kinds"], f"{where}.finding_kinds"),
            finding_indices=tuple(indices),
            evidence_ids=_strings(data["evidence_ids"], f"{where}.evidence_ids"),
            sources=_strings(data["sources"], f"{where}.sources"),
            severity=_text(data["severity"], f"{where}.severity", 40),
            priority=_text(data["priority"], f"{where}.priority", 40),
            confidence=_confidence(data["confidence"], f"{where}.confidence"),
            measured_impact=_strings(
                data["measured_impact"], f"{where}.measured_impact"
            ),
        )
    except ValueError as exc:
        raise GraphBaselineError(f"{where}: {exc}") from None


def _link(value: Any, where: str) -> EvidenceLink:
    data = _exact_keys(value, _LINK_FIELDS, where)
    try:
        return EvidenceLink(
            source_id=_text(data["source_id"], f"{where}.source_id", 300),
            target_id=_text(data["target_id"], f"{where}.target_id", 300),
            relation=_text(data["relation"], f"{where}.relation", 40),
            rationale=_text(data["rationale"], f"{where}.rationale", 20_000),
        )
    except ValueError as exc:
        raise GraphBaselineError(f"{where}: {exc}") from None


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """One validated evidence graph plus the identities needed to compare it."""

    evidence: tuple[Evidence, ...]
    clusters: tuple[EvidenceCluster, ...]
    links: tuple[EvidenceLink, ...]
    finding_fingerprints: tuple[tuple[int, str], ...] = ()
    complete: bool = True
    tool_version: str = __version__
    fingerprint_schema: int = 1
    context: tuple[tuple[str, str], ...] = ()
    version: int = SNAPSHOT_VERSION

    @property
    def graph(self) -> EvidenceGraph:
        return EvidenceGraph(
            evidence=self.evidence, clusters=self.clusters, links=self.links
        )

    @property
    def fingerprints(self) -> dict[int, str]:
        return dict(self.finding_fingerprints)

    @classmethod
    def from_findings(
        cls,
        findings: Any,
        *,
        graph: EvidenceGraph | None = None,
        complete: bool = True,
        context: dict[str, str] | None = None,
        tool_version: str = __version__,
    ) -> GraphSnapshot:
        """Build a snapshot from findings and, optionally, their built graph."""
        findings = tuple(findings)
        if graph is None:
            from .provenance import build_evidence_graph

            graph = build_evidence_graph(findings)
        fingerprints = tuple(
            (index, finding.fingerprint) for index, finding in enumerate(findings)
        )
        return cls(
            evidence=graph.evidence,
            clusters=graph.clusters,
            links=graph.links,
            finding_fingerprints=fingerprints,
            complete=bool(complete),
            tool_version=tool_version,
            context=tuple(sorted((str(k), str(v)) for k, v in (context or {}).items())),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> GraphSnapshot:
        data = _exact_keys(payload, _TOP_LEVEL, "snapshot")
        if data["schema"] != SNAPSHOT_SCHEMA:
            raise GraphBaselineError(
                f"snapshot schema must be {SNAPSHOT_SCHEMA!r}, got {data['schema']!r}"
            )
        version = data["version"]
        if type(version) is not int or version != SNAPSHOT_VERSION:
            raise GraphBaselineError(
                f"unsupported snapshot version {version!r}; this LeakLens "
                f"supports version {SNAPSHOT_VERSION}"
            )
        schema = data["fingerprint_schema"]
        if type(schema) is not int:
            raise GraphBaselineError("fingerprint_schema must be an integer")
        if schema != 1:
            raise GraphBaselineError(
                f"snapshot uses fingerprint schema {schema}, but this LeakLens "
                "computes schema 1; regenerate the snapshot"
            )
        if type(data["complete"]) is not bool:
            raise GraphBaselineError("complete must be a boolean")
        tool_version = _text(data["tool_version"], "tool_version", 40)
        context_data = data["context"]
        if not isinstance(context_data, dict):
            raise GraphBaselineError("context must be an object")
        context = tuple(
            sorted(
                (
                    _text(key, "context key", 120),
                    _text(value, f"context[{key!r}]", 1_000, required=False),
                )
                for key, value in context_data.items()
            )
        )
        summary = data["summary"]
        if not isinstance(summary, dict):
            raise GraphBaselineError("summary must be an object")

        raw_evidence = _list(data["evidence"], "evidence")
        if len(raw_evidence) > MAX_SNAPSHOT_EVIDENCE:
            raise GraphBaselineError(
                f"snapshot has more than {MAX_SNAPSHOT_EVIDENCE} evidence records"
            )
        evidence = tuple(
            _evidence(item, f"evidence[{index}]")
            for index, item in enumerate(raw_evidence)
        )
        ids = [item.evidence_id for item in evidence]
        if len(ids) != len(set(ids)):
            raise GraphBaselineError("evidence ids must be unique")

        raw_clusters = _list(data["clusters"], "clusters")
        if len(raw_clusters) > MAX_SNAPSHOT_CLUSTERS:
            raise GraphBaselineError(
                f"snapshot has more than {MAX_SNAPSHOT_CLUSTERS} clusters"
            )
        clusters = tuple(
            _cluster(item, f"clusters[{index}]")
            for index, item in enumerate(raw_clusters)
        )

        raw_links = _list(data["links"], "links")
        links = tuple(
            _link(item, f"links[{index}]") for index, item in enumerate(raw_links)
        )

        known = set(ids)
        for index, cluster in enumerate(clusters):
            missing = sorted(set(cluster.evidence_ids) - known)
            if missing:
                raise GraphBaselineError(
                    f"clusters[{index}] references unknown evidence ids: "
                    f"{', '.join(missing)}"
                )
        for index, link in enumerate(links):
            for field in ("source_id", "target_id"):
                value = getattr(link, field)
                if value not in known:
                    raise GraphBaselineError(
                        f"links[{index}].{field} references unknown evidence: {value}"
                    )

        raw_fingerprints = _list(data["finding_fingerprints"], "finding_fingerprints")
        fingerprints: list[tuple[int, str]] = []
        seen_indices: set[int] = set()
        for index, pair in enumerate(raw_fingerprints):
            where = f"finding_fingerprints[{index}]"
            if not isinstance(pair, list) or len(pair) != 2:
                raise GraphBaselineError(f"{where} must be a two-item array")
            finding_index, fingerprint = pair
            if type(finding_index) is not int or finding_index < 0:
                raise GraphBaselineError(f"{where}[0] must be a non-negative integer")
            if finding_index in seen_indices:
                raise GraphBaselineError(
                    f"{where} duplicates finding index {finding_index}"
                )
            if not isinstance(fingerprint, str) or not _FINGERPRINT.match(fingerprint):
                raise GraphBaselineError(
                    f"{where}[1] must be 64 lowercase hexadecimal characters"
                )
            seen_indices.add(finding_index)
            fingerprints.append((finding_index, fingerprint))
        fingerprints.sort()

        expected = len(evidence)
        if summary.get("evidence") != expected:
            raise GraphBaselineError("summary.evidence does not match evidence length")
        if summary.get("clusters") != len(clusters):
            raise GraphBaselineError("summary.clusters does not match cluster length")
        if summary.get("links") != len(links):
            raise GraphBaselineError("summary.links does not match link length")

        return cls(
            evidence=evidence,
            clusters=clusters,
            links=links,
            finding_fingerprints=tuple(fingerprints),
            complete=data["complete"],
            tool_version=tool_version,
            fingerprint_schema=schema,
            context=context,
            version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> GraphSnapshot:
        source = Path(path)
        try:
            size = source.stat().st_size
        except FileNotFoundError:
            raise GraphBaselineError(f"graph snapshot not found: {source}") from None
        except OSError as exc:
            raise GraphBaselineError(
                f"cannot inspect graph snapshot {source}: {exc}"
            ) from None
        if size > MAX_SNAPSHOT_BYTES:
            raise GraphBaselineError(
                f"graph snapshot {source} exceeds {MAX_SNAPSHOT_BYTES} bytes"
            )
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise GraphBaselineError(
                f"graph snapshot {source} is not valid UTF-8: {exc.reason}"
            ) from None
        except OSError as exc:
            raise GraphBaselineError(
                f"cannot read graph snapshot {source}: {exc}"
            ) from None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GraphBaselineError(
                f"invalid JSON in graph snapshot {source} at line {exc.lineno} "
                f"column {exc.colno}: {exc.msg}"
            ) from None
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        relations = {"corroborates": 0, "same_anchor": 0, "supports": 0}
        for link in self.links:
            relations[link.relation] = relations.get(link.relation, 0) + 1
        return {
            "schema": SNAPSHOT_SCHEMA,
            "version": self.version,
            "fingerprint_schema": self.fingerprint_schema,
            "tool_version": self.tool_version,
            "complete": self.complete,
            "context": dict(self.context),
            "summary": {
                "evidence": len(self.evidence),
                "clusters": len(self.clusters),
                "links": len(self.links),
                "relations": relations,
            },
            "finding_fingerprints": [
                [index, fingerprint] for index, fingerprint in self.finding_fingerprints
            ],
            "evidence": [item.to_dict() for item in self.evidence],
            "clusters": [item.to_dict() for item in self.clusters],
            "links": [item.to_dict() for item in self.links],
        }

    def to_json(self) -> str:
        """Deterministic serialization in a fixed field order."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def write(self, path: str | Path, *, update: bool = False) -> Path:
        """Atomically write the snapshot; never overwrite without ``update``."""
        target = Path(path)
        if target.is_dir():
            raise GraphBaselineError(f"graph snapshot path is a directory: {target}")
        if target.is_symlink():
            raise GraphBaselineError(
                f"refusing to write a graph snapshot through a symlink: {target}"
            )
        if target.exists() and not update:
            raise GraphBaselineError(
                f"graph snapshot already exists: {target}; pass "
                "--update-graph-snapshot to replace it"
            )
        parent = target.parent if str(target.parent) else Path(".")
        if not parent.is_dir():
            raise GraphBaselineError(
                f"graph snapshot directory does not exist: {parent}"
            )
        data = self.to_json().encode("utf-8")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            with suppress(OSError):
                os.unlink(temporary)
            raise GraphBaselineError(
                f"cannot write graph snapshot {target}: {exc}"
            ) from None
        return target


@dataclass(frozen=True, slots=True)
class EvidenceState:
    """One compared evidence record; ids are references, never identity."""

    state: str
    identity: str
    kind: str
    fingerprint: str | None = None
    baseline_id: str | None = None
    current_id: str | None = None
    changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise GraphBaselineError(f"unsupported evidence state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "identity": self.identity,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "baseline_id": self.baseline_id,
            "current_id": self.current_id,
            "changes": list(self.changes),
        }


@dataclass(frozen=True, slots=True)
class ClusterState:
    state: str
    identity: str
    anchor: str
    changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise GraphBaselineError(f"unsupported cluster state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "identity": self.identity,
            "anchor": self.anchor,
            "changes": list(self.changes),
        }


@dataclass(frozen=True, slots=True)
class RelationshipChange:
    state: str
    source: str
    relation: str
    target: str

    def __post_init__(self) -> None:
        if self.state not in _RELATION_STATES:
            raise GraphBaselineError(f"unsupported relationship state: {self.state}")

    def to_dict(self) -> dict[str, str]:
        return {
            "state": self.state,
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
        }


@dataclass(frozen=True, slots=True)
class GraphDiff:
    """Deterministic comparison of two snapshots."""

    evidence: tuple[EvidenceState, ...]
    clusters: tuple[ClusterState, ...]
    relationships: tuple[RelationshipChange, ...]
    warnings: tuple[str, ...] = ()
    duplicate_evidence: int = 0
    resolution_complete: bool = True

    @property
    def counts(self) -> dict[str, int]:
        counts = {state: 0 for state in _STATES}
        for item in self.evidence:
            counts[item.state] += 1
        return counts

    @property
    def cluster_counts(self) -> dict[str, int]:
        counts = {state: 0 for state in _STATES}
        for item in self.clusters:
            counts[item.state] += 1
        return counts

    @property
    def relationship_counts(self) -> dict[str, int]:
        counts = {state: 0 for state in _RELATION_STATES}
        for item in self.relationships:
            counts[item.state] += 1
        return counts

    @property
    def summary(self) -> dict[str, Any]:
        return {
            **self.counts,
            "clusters": self.cluster_counts,
            "relationships": self.relationship_counts,
            "duplicates": self.duplicate_evidence,
            "resolution_complete": self.resolution_complete,
        }

    @property
    def finding_states(self) -> dict[str, str]:
        """Current or resolved finding state, worst state wins for duplicates."""
        rank = {"new": 0, "changed": 1, "unchanged": 2, "uncertain": 3, "resolved": 4}
        states: dict[str, str] = {}
        for item in sorted(self.evidence, key=lambda entry: entry.identity):
            if item.fingerprint is None:
                continue
            previous = states.get(item.fingerprint)
            if previous is None or rank[item.state] < rank[previous]:
                states[item.fingerprint] = item.state
        return states

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "clusters": self.cluster_counts,
            "relationships": self.relationship_counts,
            "duplicates": self.duplicate_evidence,
            "resolution_complete": self.resolution_complete,
            "warnings": list(self.warnings),
            "evidence": [item.to_dict() for item in self.evidence],
            "cluster_changes": [item.to_dict() for item in self.clusters],
            "relationship_changes": [item.to_dict() for item in self.relationships],
        }


@dataclass(frozen=True, slots=True)
class _Prepared:
    identity: str
    digest: str
    evidence: Evidence
    fingerprint: str | None


def _metadata(item: Evidence) -> dict[str, str]:
    return dict(item.metadata)


def _role(item: Evidence) -> str:
    return "detector" if _metadata(item).get("role") == "detector" else "primary"


def _content_payload(item: Evidence, fingerprint: str | None) -> dict[str, Any]:
    return {
        "source": item.source,
        "kind": item.kind,
        "description": item.description,
        "location": item.location,
        "supports": item.supports,
        "finding_fingerprint": fingerprint or "",
        # finding_index and evidence_id are deliberately excluded: both move
        # when unrelated findings are reordered.
        "metadata": sorted((str(key), str(value)) for key, value in item.metadata),
    }


def _digest(item: Evidence, fingerprint: str | None) -> str:
    payload = json.dumps(
        _content_payload(item, fingerprint),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base_identity(item: Evidence, fingerprint: str | None, content: str) -> str:
    role = _role(item)
    if fingerprint:
        if role == "detector":
            return f"detector:{fingerprint}:{item.kind}"
        return f"primary:{fingerprint}"
    return f"{role}:content:{content[:24]}"


def _prepare(
    snapshot: GraphSnapshot,
) -> tuple[dict[str, _Prepared], int, dict[str, str]]:
    fingerprints = snapshot.fingerprints
    groups: dict[str, list[_Prepared]] = {}
    by_id: dict[str, str] = {}
    duplicates = 0
    for item in snapshot.evidence:
        fingerprint = fingerprints.get(item.finding_index)
        content = _digest(item, fingerprint)
        base = _base_identity(item, fingerprint, content)
        groups.setdefault(base, []).append(
            _Prepared(
                identity=base,
                digest=content,
                evidence=item,
                fingerprint=fingerprint,
            )
        )
    prepared: dict[str, _Prepared] = {}
    for base, items in groups.items():
        items.sort(key=lambda entry: (entry.digest, entry.evidence.evidence_id))
        duplicates += len(items) - 1
        for ordinal, entry in enumerate(items):
            identity = f"{base}#{ordinal}"
            prepared[identity] = _Prepared(
                identity=identity,
                digest=entry.digest,
                evidence=entry.evidence,
                fingerprint=entry.fingerprint,
            )
            by_id[entry.evidence.evidence_id] = identity
    return prepared, duplicates, by_id


def _relationships(
    snapshot: GraphSnapshot, by_id: dict[str, str]
) -> tuple[set[tuple[str, str, str]], list[str]]:
    links: set[tuple[str, str, str]] = set()
    warnings: list[str] = []
    for link in snapshot.links:
        source = by_id.get(link.source_id)
        target = by_id.get(link.target_id)
        if source is None or target is None:
            warnings.append(
                f"skipped link with unknown evidence: {link.source_id} -> "
                f"{link.target_id}"
            )
            continue
        links.add((source, link.relation, target))
    return links, warnings


def _cluster_identity(item: EvidenceCluster) -> str:
    return f"{item.anchor_kind}:{item.anchor}"


def _cluster_prepared(
    snapshot: GraphSnapshot,
    prepared: dict[str, _Prepared],
    by_id: dict[str, str],
) -> dict[str, list[tuple[str, EvidenceCluster, frozenset[str]]]]:
    groups: dict[str, list[tuple[str, EvidenceCluster, frozenset[str]]]] = {}
    for cluster in snapshot.clusters:
        members = []
        identities = set()
        for evidence_id in cluster.evidence_ids:
            identity = by_id.get(evidence_id)
            if identity is None:
                continue
            identities.add(identity)
            members.append((identity, prepared[identity].digest))
        payload = {
            "anchor_kind": cluster.anchor_kind,
            "finding_kinds": list(cluster.finding_kinds),
            "sources": list(cluster.sources),
            "severity": cluster.severity,
            "priority": cluster.priority,
            "confidence": [cluster.confidence.score, cluster.confidence.level],
            "measured_impact": list(cluster.measured_impact),
            "members": sorted(members),
        }
        content = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        groups.setdefault(_cluster_identity(cluster), []).append(
            (content, cluster, frozenset(identities))
        )
    for items in groups.values():
        items.sort(key=lambda entry: (entry[0], entry[1].cluster_id))
    return groups


def _cluster_changes(
    baseline: EvidenceCluster,
    current: EvidenceCluster,
    baseline_members: frozenset[str],
    current_members: frozenset[str],
) -> tuple[str, ...]:
    changes = []
    if baseline.finding_kinds != current.finding_kinds:
        changes.append("finding_kinds")
    if baseline_members != current_members:
        changes.append("membership")
    if baseline.sources != current.sources:
        changes.append("sources")
    if baseline.severity != current.severity:
        changes.append("severity")
    if baseline.priority != current.priority:
        changes.append("priority")
    if (
        baseline.confidence.score != current.confidence.score
        or baseline.confidence.level != current.confidence.level
    ):
        changes.append("confidence")
    if baseline.measured_impact != current.measured_impact:
        changes.append("measured_impact")
    return tuple(changes)


def _evidence_changes(baseline: Evidence, current: Evidence) -> tuple[str, ...]:
    changes = []
    if baseline.source != current.source:
        changes.append("source")
    if baseline.kind != current.kind:
        changes.append("kind")
    if baseline.description != current.description:
        changes.append("description")
    if baseline.location != current.location:
        changes.append("location")
    if baseline.supports != current.supports:
        changes.append("supports")
    if sorted(baseline.metadata) != sorted(current.metadata):
        changes.append("metadata")
    return tuple(changes)


def compare_graph_snapshots(
    baseline: GraphSnapshot, current: GraphSnapshot
) -> GraphDiff:
    """Compare two compatible snapshots without ever inventing a resolution."""
    if baseline.version != current.version:
        raise GraphBaselineError(
            f"cannot compare snapshot versions {baseline.version} and {current.version}"
        )
    if baseline.fingerprint_schema != current.fingerprint_schema:
        raise GraphBaselineError(
            "cannot compare snapshots with different fingerprint schemas "
            f"({baseline.fingerprint_schema} and {current.fingerprint_schema})"
        )

    warnings: list[str] = []
    if baseline.tool_version != current.tool_version:
        warnings.append(
            f"tool versions differ ({baseline.tool_version} -> "
            f"{current.tool_version}); identities still use fingerprint schema "
            f"{current.fingerprint_schema}"
        )
    context_compatible = dict(baseline.context) == dict(current.context)
    if not context_compatible:
        warnings.append(
            "analysis context differs; baseline-only evidence is reported as "
            "uncertain instead of resolved"
        )
    resolution_complete = baseline.complete and current.complete and context_compatible
    if not baseline.complete:
        warnings.append("baseline scan was incomplete")
    if not current.complete:
        warnings.append("current scan was incomplete")

    baseline_evidence, baseline_duplicates, baseline_ids = _prepare(baseline)
    current_evidence, current_duplicates, current_ids = _prepare(current)
    duplicates = baseline_duplicates + current_duplicates

    baseline_links, baseline_link_warnings = _relationships(baseline, baseline_ids)
    current_links, current_link_warnings = _relationships(current, current_ids)
    warnings.extend(baseline_link_warnings)
    warnings.extend(current_link_warnings)
    if baseline_link_warnings or current_link_warnings:
        resolution_complete = False

    removed_links = baseline_links - current_links
    added_links = current_links - baseline_links
    relationship_changes = tuple(
        sorted(
            [RelationshipChange("removed", *link) for link in removed_links]
            + [RelationshipChange("added", *link) for link in added_links],
            key=lambda item: (item.source, item.relation, item.target, item.state),
        )
    )
    affected = {
        identity
        for link in (*removed_links, *added_links)
        for identity in (link[0], link[2])
    }

    states: list[EvidenceState] = []
    for identity in sorted(set(baseline_evidence) | set(current_evidence)):
        before = baseline_evidence.get(identity)
        after = current_evidence.get(identity)
        if before is None:
            item = after.evidence  # type: ignore[union-attr]
            states.append(
                EvidenceState(
                    state="new",
                    identity=identity,
                    kind=item.kind,
                    fingerprint=after.fingerprint,  # type: ignore[union-attr]
                    current_id=item.evidence_id,
                )
            )
        elif after is None:
            item = before.evidence
            states.append(
                EvidenceState(
                    state="resolved" if resolution_complete else "uncertain",
                    identity=identity,
                    kind=item.kind,
                    fingerprint=before.fingerprint,
                    baseline_id=item.evidence_id,
                )
            )
        else:
            changes = list(_evidence_changes(before.evidence, after.evidence))
            if identity in affected:
                changes.append("relationships")
            if before.digest != after.digest and not changes:
                changes.append("content")
            state = "changed" if changes else "unchanged"
            states.append(
                EvidenceState(
                    state=state,
                    identity=identity,
                    kind=after.evidence.kind,
                    fingerprint=after.fingerprint,
                    baseline_id=before.evidence.evidence_id,
                    current_id=after.evidence.evidence_id,
                    changes=tuple(changes),
                )
            )

    baseline_clusters = _cluster_prepared(baseline, baseline_evidence, baseline_ids)
    current_clusters = _cluster_prepared(current, current_evidence, current_ids)
    cluster_states: list[ClusterState] = []
    for base in sorted(set(baseline_clusters) | set(current_clusters)):
        before_items = baseline_clusters.get(base, [])
        after_items = current_clusters.get(base, [])
        for ordinal in range(max(len(before_items), len(after_items))):
            identity = f"{base}#{ordinal}"
            before = before_items[ordinal] if ordinal < len(before_items) else None
            after = after_items[ordinal] if ordinal < len(after_items) else None
            if before is None and after is not None:
                cluster_states.append(ClusterState("new", identity, after[1].anchor))
            elif after is None and before is not None:
                cluster_states.append(
                    ClusterState(
                        "resolved" if resolution_complete else "uncertain",
                        identity,
                        before[1].anchor,
                    )
                )
            elif before is not None and after is not None:
                changes = _cluster_changes(before[1], after[1], before[2], after[2])
                if before[0] != after[0] and not changes:
                    changes = ("content",)
                cluster_states.append(
                    ClusterState(
                        "changed" if changes else "unchanged",
                        identity,
                        after[1].anchor,
                        changes,
                    )
                )

    return GraphDiff(
        evidence=tuple(states),
        clusters=tuple(cluster_states),
        relationships=relationship_changes,
        warnings=tuple(warnings),
        duplicate_evidence=duplicates,
        resolution_complete=resolution_complete,
    )
