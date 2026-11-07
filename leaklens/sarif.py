from __future__ import annotations

import re

from . import __version__
from .fixes import advice

LEVEL = {"high": "error", "medium": "warning", "low": "note"}
LOC = re.compile(r"^(?P<path>.+?)(?:\[cell (?P<cell>\d+)\])?:(?P<line>\d+)$")


def _graph_evidence_index(snapshot) -> dict[str, list[str]]:
    """Map finding fingerprints to sorted evidence ids for compact SARIF refs."""
    fingerprints = snapshot.fingerprints
    index: dict[str, list[str]] = {}
    for item in snapshot.evidence:
        fingerprint = fingerprints.get(item.finding_index)
        if fingerprint:
            index.setdefault(fingerprint, []).append(item.evidence_id)
    return {fingerprint: sorted(ids) for fingerprint, ids in index.items()}


def to_sarif(
    findings,
    default_uri: str = ".",
    *,
    outcome=None,
    graph_snapshot=None,
    graph_diff=None,
) -> dict:
    """Render SARIF 2.1.0; policy and graph metadata are additive and optional."""
    diff = getattr(outcome, "diff", None)
    suppression = getattr(outcome, "suppression", None)
    status = getattr(outcome, "status", None) or {}
    graph_index = _graph_evidence_index(graph_snapshot) if graph_snapshot else {}
    finding_states = graph_diff.finding_states if graph_diff is not None else {}
    rules, seen, results = [], set(), []
    for f in findings:
        fingerprint = f.fingerprint
        if f.kind not in seen:
            seen.add(f.kind)
            rules.append(
                {
                    "id": f.kind,
                    "name": f.kind,
                    "shortDescription": {"text": f.kind.replace("_", " ")},
                    "help": {"text": advice(f.kind) or f.kind},
                }
            )
        m = LOC.match(str(f.column))
        if m:
            uri = m.group("path").replace("\\", "/")
            region = {"startLine": int(m.group("line"))}
            msg = f.detail
            if m.group("cell"):
                msg += (
                    f" (notebook cell {m.group('cell')}; line is relative to the cell)"
                )
        else:
            uri = str(default_uri).replace("\\", "/")
            region = {"startLine": 1}
            msg = f"{f.column}: {f.detail}"
        properties = {"evidence": [item.to_dict() for item in f.evidence]}
        if getattr(f, "diagnostic", False):
            properties["diagnostic"] = True
        if graph_snapshot is not None:
            evidence_ids = graph_index.get(fingerprint, [])
            graph = {
                "evidence_ids": evidence_ids,
                "evidence_count": len(evidence_ids),
            }
            state = finding_states.get(fingerprint)
            if state is not None:
                graph["state"] = state
            properties["graph"] = graph
        if fingerprint in status:
            properties["status"] = status[fingerprint]
        rule = suppression.rule_for(fingerprint) if suppression is not None else None
        if rule is not None:
            properties["suppression"] = rule.to_dict()
        result = {
            "ruleId": f.kind,
            "level": LEVEL.get(f.severity, "note"),
            "message": {"text": msg},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": uri},
                        "region": region,
                    }
                }
            ],
            "partialFingerprints": {"leaklensFinding/v1": fingerprint},
            "properties": {"leaklens": properties},
        }
        if rule is not None:
            result["suppressions"] = [
                {"kind": "external", "justification": rule.reason}
            ]
        results.append(result)
    run = {
        "tool": {
            "driver": {
                "name": "LeakLens",
                "version": __version__,
                "informationUri": "https://github.com/eyadstein/leaklens",
                "rules": rules,
            }
        },
        "results": results,
    }
    if diff is not None or suppression is not None or graph_snapshot is not None:
        policy: dict = {}
        if diff is not None:
            policy["baseline"] = diff.counts
            if diff.resolved:
                policy["resolved"] = [entry.to_dict() for entry in diff.resolved]
        if suppression is not None:
            policy["suppressions"] = suppression.to_dict()
        if graph_snapshot is not None:
            graph = {
                "schema": "leaklens-evidence-graph",
                "version": graph_snapshot.version,
                "complete": graph_snapshot.complete,
                "summary": graph_snapshot.graph.summary(),
            }
            if graph_diff is not None:
                graph["comparison"] = graph_diff.summary
            policy["graph"] = graph
        run["properties"] = {"leaklens": policy}
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [run],
    }
