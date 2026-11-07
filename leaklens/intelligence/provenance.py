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
    for item in detector:
        links.append(
            EvidenceLink(
                source_id=item.evidence_id,
                target_id=representative.evidence_id,
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
