"""Deterministic evidence graph over scanner findings.

The graph adds relationships and grouping on top of existing evidence. It
never creates a finding, changes a severity, or claims causality: every node
is an existing evidence record and every edge explains why two records are
shown together.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from leaklens.fingerprint import normalize_location
from leaklens.leakage import Finding
from leaklens.manifest import surface_for

from .confidence import assess_cluster_confidence
from .context import AnalysisContext, build_context
from .impact import DatasetImpact
from .schemas import Evidence, EvidenceCluster, EvidenceLink

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
_MAX_LINKS = 400
_SLUG = re.compile(r"[^a-z0-9]+")
_FEATURE_SURFACES = frozenset({"dataset", "split"})


@dataclass(frozen=True, slots=True)
class EvidenceGraph:
    """Evidence records grouped into anchors, with explicit links."""

    evidence: tuple[Evidence, ...]
    clusters: tuple[EvidenceCluster, ...]
    links: tuple[EvidenceLink, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "clusters": [item.to_dict() for item in self.clusters],
            "links": [item.to_dict() for item in self.links],
        }

    @property
    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}

    def cluster_for(self, finding_index: int) -> EvidenceCluster | None:
        """Return the cluster that contains a finding index, if any."""
        for cluster in self.clusters:
            if finding_index in cluster.finding_indices:
                return cluster
        return None

    def neighbors(self, evidence_id: str) -> tuple[tuple[str, str, Evidence], ...]:
        """Return ``(relation, direction, evidence)`` for one record.

        ``direction`` is ``"out"`` when the record is the link source and
        ``"in"`` when it is the target. Unknown ids return an empty tuple.
        Results are sorted by the neighbouring evidence id, then relation.
        """
        by_id = self.evidence_by_id
        found: list[tuple[str, str, Evidence]] = []
        for link in self.links:
            if link.source_id == evidence_id:
                other, direction = link.target_id, "out"
            elif link.target_id == evidence_id:
                other, direction = link.source_id, "in"
            else:
                continue
            item = by_id.get(other)
            if item is not None:
                found.append((link.relation, direction, item))
        found.sort(key=lambda item: (item[2].evidence_id, item[0], item[1]))
        return tuple(found)

    def relation_counts(self) -> dict[str, int]:
        """Count every link relation, including zeroes, in a stable order."""
        counts = {"corroborates": 0, "same_anchor": 0, "supports": 0}
        for link in self.links:
            counts[link.relation] = counts.get(link.relation, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        return {
            "evidence": len(self.evidence),
            "clusters": len(self.clusters),
            "links": len(self.links),
            "relations": self.relation_counts(),
        }

    def to_mermaid(self) -> str:
        """Render the graph as a deterministic Mermaid flowchart."""
        return _render_graph(self, "mermaid")

    def to_dot(self) -> str:
        """Render the graph as a deterministic Graphviz DOT digraph."""
        return _render_graph(self, "dot")


def _short(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _node_label(item: Evidence) -> str:
    return _short(f"{item.evidence_id} - {item.kind} - {item.description}")


def _dot_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _mermaid_escape(value: str) -> str:
    return (
        value.replace("\\", "/")
        .replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("|", "/")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _render_graph(graph: EvidenceGraph, fmt: str) -> str:
    by_id = graph.evidence_by_id
    node_ids = {
        item.evidence_id: f"n{index}" for index, item in enumerate(graph.evidence)
    }
    in_cluster = {
        evidence_id
        for cluster in graph.clusters
        for evidence_id in cluster.evidence_ids
    }
    if fmt == "mermaid":
        lines = ["flowchart LR"]
        for index, cluster in enumerate(graph.clusters):
            title = _mermaid_escape(
                _short(f"{cluster.anchor} ({cluster.priority})", 100)
            )
            lines.append(f'  subgraph c{index}["{title}"]')
            for evidence_id in cluster.evidence_ids:
                item = by_id.get(evidence_id)
                if item is None:
                    continue
                label = _mermaid_escape(_node_label(item))
                lines.append(f'    {node_ids[evidence_id]}["{label}"]')
            lines.append("  end")
        for item in graph.evidence:
            if item.evidence_id not in in_cluster:
                label = _mermaid_escape(_node_label(item))
                lines.append(f'  {node_ids[item.evidence_id]}["{label}"]')
        for link in graph.links:
            source = node_ids.get(link.source_id)
            target = node_ids.get(link.target_id)
            if source is None or target is None:
                continue
            lines.append(f"  {source} -->|{_mermaid_escape(link.relation)}| {target}")
        return "\n".join(lines) + "\n"

    if fmt == "dot":
        lines = [
            "digraph leaklens {",
            "  rankdir=LR;",
            "  node [shape=box, fontsize=10];",
        ]
        for index, cluster in enumerate(graph.clusters):
            title = _dot_escape(_short(f"{cluster.anchor} ({cluster.priority})", 100))
            lines.append(f"  subgraph cluster_{index} {{")
            lines.append(f'    label="{title}";')
            for evidence_id in cluster.evidence_ids:
                item = by_id.get(evidence_id)
                if item is None:
                    continue
                label = _dot_escape(_node_label(item))
                lines.append(f'    {node_ids[evidence_id]} [label="{label}"];')
            lines.append("  }")
        for item in graph.evidence:
            if item.evidence_id not in in_cluster:
                label = _dot_escape(_node_label(item))
                lines.append(f'  {node_ids[item.evidence_id]} [label="{label}"];')
        for link in graph.links:
            source = node_ids.get(link.source_id)
            target = node_ids.get(link.target_id)
            if source is None or target is None:
                continue
            lines.append(
                f'  {source} -> {target} [label="{_dot_escape(link.relation)}"];'
            )
        lines.append("}")
        return "\n".join(lines) + "\n"

    raise ValueError(f"unsupported graph format: {fmt!r}")


def _slug(value: str, limit: int = 64) -> str:
    slug = _SLUG.sub("-", value.lower()).strip("-")
    return slug[:limit] or "anchor"


def anchor_for(finding: Finding) -> tuple[str, str]:
    """Return the inspectable anchor a finding belongs to.

    Dataset/split findings anchor on the affected feature (or on the whole
    split when the column is ``*``). Code and notebook findings anchor on the
    file, so repeated patterns in one file form one cluster. Unknown kinds get
    a stable per-kind anchor instead of guessing a surface.
    """

    surface = surface_for(finding.kind)
    if surface in _FEATURE_SURFACES:
        _, feature = normalize_location(str(finding.column), surface=surface)
        if feature:
            return (
                f"{surface}:{feature}",
                "feature" if surface == "dataset" else "split",
            )
        return surface, surface
    if surface in {"code", "notebook"}:
        path, _ = normalize_location(str(finding.column), surface=surface)
        return f"file:{path or finding.column}", "file"
    if surface == "image":
        return "image", "image"
    return f"finding:{finding.kind}", "finding"


def _max_severity(severities: Iterable[str]) -> str:
    return min(
        severities,
        key=lambda value: _SEVERITY_RANK.get(value, len(_SEVERITY_RANK)),
        default="low",
    )


def _impact_notes(
    anchor: str, anchor_kind: str, impact: DatasetImpact | None
) -> tuple[str, ...]:
    if impact is None:
        return ()

    notes: list[str] = []
    feature = anchor.split(":", 1)[1] if ":" in anchor else None
    duplicate = (
        f"{impact.duplicate_test_rows} of {impact.test_rows} test rows "
        f"({impact.duplicate_test_pct:.2f}%) duplicate training rows"
    )
    group = (
        f"{impact.shared_group_count} group value(s) appear in both splits; "
        f"{impact.shared_group_test_rows} test rows "
        f"({impact.shared_group_test_pct:.2f}%) affected"
    )
    temporal = (
        f"train reaches {impact.train_max_time} but test starts {impact.test_min_time}"
    )

    if anchor_kind in {"split", "dataset"}:
        if feature is None:
            if impact.duplicate_test_rows:
                notes.append(duplicate)
            if impact.shared_group_count:
                notes.append(group)
            if impact.temporal_overlap:
                notes.append(temporal)
        elif impact.group_column == feature and impact.shared_group_count:
            notes.append(group)
        elif impact.time_column == feature and impact.temporal_overlap:
            notes.append(temporal)
    elif anchor_kind == "feature":
        if impact.group_column == feature and impact.shared_group_count:
            notes.append(group)
        elif impact.time_column == feature and impact.temporal_overlap:
            notes.append(temporal)

    return tuple(dict.fromkeys(notes))


def _cluster(
    *,
    anchor: str,
    anchor_kind: str,
    indices: list[int],
    findings: tuple[Finding, ...],
    by_finding: dict[int, list[Evidence]],
    impact: DatasetImpact | None,
    cluster_id: str,
) -> EvidenceCluster:
    members = [item for index in indices for item in by_finding.get(index, ())]
    kinds = tuple(dict.fromkeys(findings[index].kind for index in indices))
    sources = tuple(dict.fromkeys(item.source for item in members))
    severity = _max_severity(findings[index].severity for index in indices)

    return EvidenceCluster(
        cluster_id=cluster_id,
        anchor=anchor,
        anchor_kind=anchor_kind,
        finding_kinds=kinds,
        finding_indices=tuple(indices),
        evidence_ids=tuple(item.evidence_id for item in members),
        sources=sources,
        severity=severity,
        priority=severity,
        confidence=assess_cluster_confidence(members),
        measured_impact=_impact_notes(anchor, anchor_kind, impact),
    )


def _links_for(
    cluster: EvidenceCluster, by_id: dict[str, Evidence]
) -> list[EvidenceLink]:
    members = [by_id[evidence_id] for evidence_id in cluster.evidence_ids]
    primaries = [
        item for item in members if dict(item.metadata).get("role") != "detector"
    ]
    detector = [
        item for item in members if dict(item.metadata).get("role") == "detector"
    ]
    if not primaries:
        return []

    representative = primaries[0]
    links: list[EvidenceLink] = []
    for item in primaries[1:]:
        relation = "corroborates" if item.kind != representative.kind else "same_anchor"
        links.append(
            EvidenceLink(
                source_id=representative.evidence_id,
                target_id=item.evidence_id,
                relation=relation,
                rationale=(
                    f"findings {representative.kind!r} and {item.kind!r} share "
                    f"the anchor {cluster.anchor!r}"
                ),
            )
        )
    # A detector record explains the finding it was attached to, which is not
    # necessarily the cluster representative.
    primary_by_finding = {
        item.finding_index: item for item in primaries if item.finding_index is not None
    }
    for item in detector:
        target = primary_by_finding.get(item.finding_index)
        if target is None:
            # No verified primary record: do not invent a different target.
            continue
        links.append(
            EvidenceLink(
                source_id=item.evidence_id,
                target_id=target.evidence_id,
                relation="supports",
                rationale=(
                    f"detector evidence explains why finding "
                    f"#{item.finding_index} fired"
                ),
            )
        )
    return links


def build_evidence_graph(
    findings: Iterable[Finding],
    *,
    impact: DatasetImpact | None = None,
    context: AnalysisContext | None = None,
) -> EvidenceGraph:
    """Group findings by anchor and link their evidence deterministically."""

    findings = tuple(findings)
    context = context if context is not None else build_context(findings)

    by_finding: dict[int, list[Evidence]] = {}
    for item in context.evidence:
        if item.finding_index is not None:
            by_finding.setdefault(item.finding_index, []).append(item)

    buckets: dict[tuple[str, str], list[int]] = {}
    for index, finding in enumerate(findings):
        buckets.setdefault(anchor_for(finding), []).append(index)

    used_ids: set[str] = set()
    clusters: list[EvidenceCluster] = []
    for (anchor, anchor_kind), indices in buckets.items():
        base = f"cluster-{_slug(anchor)}"
        cluster_id = base
        suffix = 2
        while cluster_id in used_ids:
            cluster_id = f"{base}-{suffix}"
            suffix += 1
        used_ids.add(cluster_id)
        clusters.append(
            _cluster(
                anchor=anchor,
                anchor_kind=anchor_kind,
                indices=indices,
                findings=findings,
                by_finding=by_finding,
                impact=impact,
                cluster_id=cluster_id,
            )
        )

    clusters.sort(
        key=lambda item: (
            _SEVERITY_RANK.get(item.priority, len(_SEVERITY_RANK)),
            -item.confidence.score,
            item.anchor,
            item.cluster_id,
        )
    )

    by_id = {item.evidence_id: item for item in context.evidence}
    links: list[EvidenceLink] = []
    for cluster in clusters:
        links.extend(_links_for(cluster, by_id))
        if len(links) >= _MAX_LINKS:
            del links[_MAX_LINKS:]
            break

    return EvidenceGraph(
        evidence=context.evidence,
        clusters=tuple(clusters),
        links=tuple(links),
    )
