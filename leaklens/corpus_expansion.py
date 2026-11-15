"""Real-world expansion corpus: provenance, deduplication and review logic.

Pure functions only: no network, no writes outside explicit ``write_manifest``.
Acquisition lives in ``scripts/collect_expansion.py``; this module owns the
rules that make the resulting corpus auditable:

- stable, content-addressed sample ids
- licence state and redistribution eligibility
- exact, code-level and near-duplicate relationships (shared primitive:
  :mod:`leaklens.near_duplicate`, ADR-022)
- a deterministic review queue that never assigns a label
- human review states with the evidence a reviewer must supply

The corpus is a *candidate* corpus. Nothing here scores the scanner, and no
detector imports this module: collection and review metadata must never become
a detector input.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections import Counter
from datetime import datetime, timezone

from . import near_duplicate

EXPANSION_DIR = pathlib.Path("realworld_expansion")
MANIFEST = EXPANSION_DIR / "manifest.jsonl"
REVIEWS = EXPANSION_DIR / "reviews.jsonl"

REVIEW_STATES = (
    "not_reviewed",
    "confirmed_leakage",
    "legitimate",
    "potential",
    "uncertain",
)
# Only hand-reviewed, sufficiently supported labels may ever be scored.
SCORED_STATES = ("confirmed_leakage", "legitimate")

REDISTRIBUTABLE_SPDX = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "0BSD",
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "MPL-2.0",
        "Unlicense",
    }
)
COPYLEFT_SPDX = frozenset(
    {"GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0", "EUPL-1.2"}
)

REQUIRED_FIELDS = ("source_id", "repo", "path", "file_url", "sha256", "collected_at")
GITHUB_KINDS = frozenset({"github_search", "github_repos"})

SKETCH_SIZE = 64
SKETCH_SHINGLE = 5
NEAR_THRESHOLD = 0.85

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s\w]")


# --- identity ---------------------------------------------------------------


def candidate_id(source_id: str, repo: str, path: str, sha256: str) -> str:
    """Stable id from the sample's provenance and content, never from list order."""
    payload = "\x1f".join((source_id, repo, path, sha256 or "")).encode("utf-8")
    return "rwx_" + hashlib.blake2b(payload, digest_size=6).hexdigest()


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_sha256(text: str) -> str:
    """Hash of the normalised code only, so re-execution is not a new sample."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- licence ----------------------------------------------------------------


def license_status(spdx: str | None) -> str:
    """One of ``permissive``, ``copyleft`` or ``unknown``."""
    if not spdx:
        return "unknown"
    if spdx in REDISTRIBUTABLE_SPDX:
        return "permissive"
    if spdx in COPYLEFT_SPDX:
        return "copyleft"
    return "unknown"


def redistributable(spdx: str | None) -> bool:
    """Only a known permissive licence allows shipping the file in this repo."""
    return license_status(spdx) == "permissive"


# --- content ----------------------------------------------------------------


def code_text(raw: bytes, file_type: str) -> str:
    """Code only: notebooks lose markdown, outputs and execution counts.

    Used for code-level deduplication and near-duplicate sketches. It never
    executes anything and never raises on malformed input.
    """
    text = raw.decode("utf-8", "replace")
    if file_type != "ipynb":
        return text
    try:
        notebook = json.loads(text)
    except ValueError:
        return ""
    parts = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            parts.append(source.rstrip())
    return "\n\n".join(parts) + ("\n" if parts else "")


def sketch(text: str, size: int = SKETCH_SIZE, shingle: int = SKETCH_SHINGLE):
    """Bottom-k sketch of token shingles, for near-duplicate clustering."""
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < shingle:
        return ()
    hashes = {
        int.from_bytes(
            hashlib.blake2b(
                "\x1f".join(tokens[i : i + shingle]).encode("utf-8"), digest_size=8
            ).digest(),
            "big",
        )
        for i in range(len(tokens) - shingle + 1)
    }
    return tuple(sorted(hashes)[:size])


# --- deduplication ----------------------------------------------------------


def _reset_links(records) -> None:
    for rec in records:
        rec["duplicate_of"] = None
        rec["code_duplicate_of"] = None
        rec["near_of"] = None
        rec.pop("near_cluster", None)


def _link(group, key, target) -> None:
    """Mark every member of ``group`` except the canonical id."""
    if len(group) < 2:
        return
    canonical = min(rec["id"] for rec in group)
    for rec in group:
        if rec["id"] != canonical:
            rec[key] = canonical
            rec["duplicate_of"] = rec["duplicate_of"] or canonical


def deduplicate(records) -> None:
    """Set ``duplicate_of`` / ``code_duplicate_of`` / ``near_of`` in place.

    Order-independent and idempotent: the canonical member of a group is the
    smallest id, so re-running on a re-sorted manifest gives the same links.
    A notebook mirrored in several datasets therefore counts once.
    """
    _reset_links(records)
    by_content: dict[str, list] = {}
    for rec in records:
        if rec.get("sha256"):
            by_content.setdefault(rec["sha256"], []).append(rec)
    for group in by_content.values():
        _link(group, "duplicate_of", None)

    by_code: dict[str, list] = {}
    for rec in records:
        if rec.get("code_sha256"):
            by_code.setdefault(rec["code_sha256"], []).append(rec)
    for group in by_code.values():
        _link(group, "code_duplicate_of", None)

    signatures = {
        rec["id"]: tuple(rec["sketch"])
        for rec in records
        if len(rec.get("sketch") or ()) >= SKETCH_SIZE
    }
    if len(signatures) < 2:
        return
    for cluster in near_duplicate.cluster(
        signatures, threshold=NEAR_THRESHOLD, rows=4, max_bucket=200
    ):
        members = [rec for rec in records if rec["id"] in cluster.members]
        canonical = cluster.members[0]
        for rec in members:
            rec["near_cluster"] = cluster.key
        for rec in members:
            if rec["id"] != canonical:
                rec["near_of"] = canonical
                rec["duplicate_of"] = rec["duplicate_of"] or canonical


# --- review queue -----------------------------------------------------------


def _priority(rec) -> tuple:
    """Higher first: scanner signal, then workflow richness, then stable identity."""
    signal = any(f.get("severity") in ("high", "medium") for f in rec["findings"])
    workflow = rec.get("workflow") or {}
    richness = sum(bool(workflow.get(key)) for key in ("split", "fit", "cv"))
    return (0 if signal else 1, -richness, rec["source_id"], rec["repo"], rec["path"])


def queue(records, limit: int = 300, per_repo_cap: int = 2) -> list[str]:
    """Deterministic, diversified review queue of *unreviewed* unique samples.

    Diversity beats volume: at most ``per_repo_cap`` samples per repository, one
    repository before a second pass. Never assigns a review state, so no
    prediction or heuristic can enter the ground truth.
    """
    eligible = [
        rec
        for rec in records
        if not rec["duplicate_of"]
        and rec.get("eligible_for_review")
        and not rec.get("overlaps_frozen")
        and rec["review"]["state"] == "not_reviewed"
    ]
    chosen: list[str] = []
    per_repo: Counter = Counter()
    for rec in sorted(eligible, key=_priority):
        if len(chosen) >= limit:
            break
        if per_repo[rec["repo"]] >= per_repo_cap:
            continue
        per_repo[rec["repo"]] += 1
        chosen.append(rec["id"])
    return chosen


# --- review workflow --------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def review(
    rec,
    *,
    state: str,
    evidence: str,
    explanation: str,
    mechanism: str | None = None,
    confidence: float | None = None,
    reviewer: str = "unassigned",
    when: str | None = None,
) -> dict:
    """Record one human decision; returns a new record, never mutates ``rec``.

    A scored state (``confirmed_leakage``/``legitimate``) requires both the
    evidence and an explanation of why it holds, so a label can never be
    recorded as a bare verdict.
    """
    if state not in REVIEW_STATES:
        raise ValueError(f"unknown review state: {state!r}")
    if state in SCORED_STATES and not (evidence.strip() and explanation.strip()):
        raise ValueError("a scored verdict needs evidence and an explanation")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    updated = dict(rec)
    updated["review"] = {
        "state": state,
        "evidence": evidence,
        "mechanism": mechanism,
        "explanation": explanation,
        "confidence": confidence,
        "reviewer": reviewer,
        "decided_at": when or _now(),
    }
    updated["scored"] = state in SCORED_STATES
    return updated


# --- validation -------------------------------------------------------------


def validate(records) -> list[str]:
    """Provenance problems, one human-readable string each."""
    errors = []
    seen: dict[str, str] = {}
    for rec in records:
        ident = rec.get("id", "<no id>")
        for field in REQUIRED_FIELDS:
            if not str(rec.get(field) or "").strip():
                errors.append(f"{ident}: missing {field}")
        if rec.get("source_kind") in GITHUB_KINDS and not rec.get("commit_sha"):
            errors.append(f"{ident}: github sample without commit_sha")
        if rec.get("license_spdx") is None:
            errors.append(f"{ident}: license not recorded (use '' for none found)")
        if ident in seen:
            errors.append(f"{ident}: duplicate id (also {seen[ident]})")
        seen[ident] = rec.get("path", "?")
        if rec["review"]["state"] not in REVIEW_STATES:
            errors.append(f"{ident}: invalid review state")
    return errors


# --- manifest and statistics ------------------------------------------------


def load_manifest(path=MANIFEST) -> list[dict]:
    path = pathlib.Path(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_manifest(path, records) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")


def summarize(records, queue_ids=None) -> dict:
    """Counts for the report: samples, repositories, licences, review progress."""
    records = list(records)
    licenses = Counter(license_status(rec.get("license_spdx")) for rec in records)
    reviews = Counter(rec["review"]["state"] for rec in records)
    unique = [rec for rec in records if not rec["duplicate_of"]]
    return {
        "candidates": len(records),
        "unique_samples": len(unique),
        "duplicates": len(records) - len(unique),
        "near_duplicates": sum(1 for rec in records if rec.get("near_of")),
        "repositories": len({rec["repo"] for rec in records}),
        # a Kaggle kernel is an independent author's notebook, not a repository
        "github_repositories": len(
            {rec["repo"] for rec in records if rec["source_kind"] in GITHUB_KINDS}
        ),
        "kaggle_kernels": len(
            {rec["repo"] for rec in records if rec["source_kind"] == "kaggle"}
        ),
        "unique_repositories": len({rec["repo"] for rec in unique}),
        "sources": dict(sorted(Counter(rec["source_id"] for rec in records).items())),
        "file_types": dict(
            sorted(Counter(rec["file_type"] for rec in records).items())
        ),
        "licenses": {
            state: licenses.get(state, 0)
            for state in ("permissive", "copyleft", "unknown")
        },
        "redistributable": sum(
            1 for rec in records if redistributable(rec.get("license_spdx"))
        ),
        "unknown_license": licenses.get("unknown", 0),
        "eligible_for_review": sum(
            1 for rec in unique if rec.get("eligible_for_review")
        ),
        "overlaps_frozen": sum(1 for rec in records if rec.get("overlaps_frozen")),
        "bytes": sum(int(rec.get("bytes") or 0) for rec in unique),
        "queue_size": len(queue_ids) if queue_ids is not None else 0,
        "reviews": {state: reviews.get(state, 0) for state in REVIEW_STATES},
        "confirmed_labels": reviews.get("confirmed_leakage", 0),
        "scored_labels": sum(reviews.get(state, 0) for state in SCORED_STATES),
    }
