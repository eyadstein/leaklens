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


# --- review batch (Mission 2A) ----------------------------------------------

# Deliberate source x signal plan: a batch of 30 must not be 30 obvious
# positives, and must not come from one channel. Kaggle is over-sampled relative
# to its 13% share of the queue because it is the scarcest independent channel.
BATCH_PLAN = (
    ("github_repos", {"high": 3, "medium": 2, "low": 2, "none": 1}),
    ("github_search", {"high": 6, "medium": 3, "low": 4, "none": 2}),
    ("kaggle_distilkaggle", {"high": 2, "medium": 2, "low": 2, "none": 1}),
)
SIGNAL_QUOTA = tuple(
    (bucket, sum(plan.get(bucket, 0) for _source, plan in BATCH_PLAN))
    for bucket in ("high", "medium", "low", "none")
)
BATCH_SIZE = sum(size for _name, size in SIGNAL_QUOTA)

HYPOTHESIS_AUTHOR = "ai_unverified"
# Attribution that can never be a real human decision.
PLACEHOLDER_REVIEWERS = frozenset(
    {
        "",
        "unassigned",
        "unknown",
        "none",
        "n/a",
        "na",
        "tbd",
        "todo",
        "ai",
        "a.i.",
        "llm",
        "model",
        "gpt",
        "assistant",
        "bot",
        "claude",
        "chatgpt",
        "openai",
        "anonymous",
    }
)
DECISION_FIELDS = (
    "candidate_id",
    "state",
    "rationale",
    "evidence_refs",
    "reviewer",
    "decided_at",
    "source",
)


def signal_bucket(rec) -> str:
    """Highest scanner severity on the record, or ``none``."""
    severities = {finding.get("severity") for finding in rec.get("findings") or ()}
    for name in ("high", "medium", "low"):
        if name in severities:
            return name
    return "none"


def workflow_signature(rec) -> str:
    """Compact, stable description of which ML workflow markers fired."""
    flags = rec.get("workflow") or {}
    return "+".join(name for name, on in sorted(flags.items()) if on) or "no-markers"


def _batch_pool(records, decisions=()) -> list[dict]:
    decided = {dec["candidate_id"] for dec in decisions}
    return [
        rec
        for rec in records
        if not rec["duplicate_of"]
        and rec.get("eligible_for_review")
        and not rec.get("overlaps_frozen")
        and rec["review"]["state"] == "not_reviewed"
        and rec["id"] not in decided
    ]


def select_batch(records, decisions=(), per_repo: int = 1) -> list[str]:
    """Deterministic, diversity-first batch of candidate ids.

    Fills an explicit source x signal plan (:data:`BATCH_PLAN`), round-robin over
    (licence, workflow signature) strata inside each cell, at most ``per_repo``
    samples per repository. A short cell is topped up from the same source before
    any other source is used, so a thin channel cannot collapse the batch. Every
    choice is a pure function of the manifest, so re-running gives the same batch
    while the manifest and decisions are unchanged.
    """
    pool = _batch_pool(records, decisions)
    chosen: list[dict] = []
    per_repo_count: Counter = Counter()
    licence_counts: Counter = Counter()

    def taken_ids() -> set[str]:
        return {rec["id"] for rec in chosen}

    def fill(source: str | None, bucket: str | None, quota: int) -> int:
        already = taken_ids()
        strata: dict[tuple, list[dict]] = {}
        for rec in pool:
            if rec["id"] in already:
                continue
            if source is not None and rec["source_id"] != source:
                continue
            if bucket is not None and signal_bucket(rec) != bucket:
                continue
            key = (license_status(rec.get("license_spdx")), workflow_signature(rec))
            strata.setdefault(key, []).append(rec)
        groups = {
            key: sorted(group, key=lambda rec: rec["id"])
            for key, group in strata.items()
        }
        picked = 0
        level = 0
        while picked < quota:
            eligible = [
                (key, group[level])
                for key, group in groups.items()
                if level < len(group)
            ]
            if not eligible:
                break
            # least-represented licence first, then a stable order: a small quota
            # must not silently drop a whole licence state from the batch.
            eligible.sort(key=lambda item: (licence_counts[item[0][0]], item[0]))
            for key, rec in eligible:
                if picked >= quota:
                    break
                if per_repo_count[rec["repo"]] >= per_repo:
                    continue
                chosen.append(rec)
                per_repo_count[rec["repo"]] += 1
                licence_counts[key[0]] += 1
                picked += 1
            level += 1
        return picked

    for source, plan in BATCH_PLAN:
        for bucket in ("high", "medium", "low", "none"):
            fill(source, bucket, plan.get(bucket, 0))
        short = sum(plan.values()) - sum(
            1 for rec in chosen if rec["source_id"] == source
        )
        if short > 0:
            fill(source, None, short)
    while len(chosen) < BATCH_SIZE:
        if not fill(None, None, BATCH_SIZE - len(chosen)):
            break
    return [rec["id"] for rec in chosen]


# Vocabulary a dossier hypothesis is drawn from. Closed so forty batches cannot
# drift into forty spellings of the same mechanism, but a candidate is never
# forced into a category: ``none_observed`` and ``legitimate_lookalike`` exist
# precisely because most real notebooks are not leaking.
MECHANISMS = (
    "preprocess_before_split",
    "target_leakage",
    "target_encoding_leakage",
    "feature_selection_on_holdout",
    "duplicate_contamination",
    "temporal_leakage",
    "group_entity_leakage",
    "augmentation_leakage",
    "cv_contamination",
    "test_set_reuse",
    "legitimate_lookalike",
    "none_observed",
)
HYPOTHESIS_CONFIDENCE = ("low", "medium", "high")

# ``cell 128:4`` (one line), ``cell 128:3-6`` (a range), ``line 42`` (script).
CODE_REF = re.compile(r"^(cell \d+:\d+(-\d+)?|line \d+)$")

DOSSIER_FIELDS = (
    "candidate_id",
    "provenance",
    "license",
    "scanner",
    "code_refs",
    "excerpts",
    "workflow",
    "mechanism",
    "why_it_may_apply",
    "evidence_against",
    "questions",
    "hypothesis",
    "decision",
)


def dossier(rec, analysis: dict, excerpts=()) -> dict:
    """One review dossier: manifest provenance + AI analysis, decision empty.

    The hypothesis is always marked ``ai_unverified`` and the decision is always
    empty, so nothing here can be mistaken for ground truth. ``excerpts`` are
    supplied by the caller (materialised from the cache) and must already have
    passed the redistribution check: the licence guard lives in
    :func:`validate_dossier`, which rejects code text on a non-permissive row.
    """
    missing = [
        key
        for key in (
            "mechanism",
            "why_it_may_apply",
            "evidence_against",
            "questions",
            "code_refs",
            "workflow_summary",
        )
        if not analysis.get(key)
    ]
    if missing:
        raise ValueError(f"{rec['id']}: analysis missing {', '.join(missing)}")
    return {
        "candidate_id": rec["id"],
        "provenance": {
            "source_id": rec["source_id"],
            "repo": rec["repo"],
            "repo_url": rec.get("repo_url", ""),
            "commit_sha": rec.get("commit_sha", ""),
            "path": rec["path"],
            "file_url": rec["file_url"],
            "collected_at": rec["collected_at"],
            "sha256": rec["sha256"],
        },
        "license": {
            "spdx": rec.get("license_spdx") or "",
            "status": license_status(rec.get("license_spdx")),
            "redistributable": redistributable(rec.get("license_spdx")),
        },
        "scanner": {
            "signal": signal_bucket(rec),
            "findings": [
                {"severity": f.get("severity"), "kind": f.get("kind")}
                for f in rec.get("findings") or ()
            ],
        },
        "code_refs": list(analysis["code_refs"]),
        "excerpts": [{"ref": item["ref"], "text": item["text"]} for item in excerpts],
        "workflow": {
            "summary": analysis["workflow_summary"],
            "markers": rec.get("workflow") or {},
        },
        "mechanism": analysis["mechanism"],
        "why_it_may_apply": analysis["why_it_may_apply"],
        "evidence_against": list(analysis["evidence_against"]),
        "questions": list(analysis["questions"]),
        "hypothesis": {
            "author": HYPOTHESIS_AUTHOR,
            "text": analysis["why_it_may_apply"],
            "confidence": analysis.get("hypothesis_confidence", "low"),
            "is_ground_truth": False,
        },
        "decision": None,
    }


def validate_dossier(entry, rec=None) -> list[str]:
    """Problems with a dossier, one string each. ``rec`` checks the binding."""
    errors = []
    ident = entry.get("candidate_id", "<no id>")
    for field in DOSSIER_FIELDS:
        if field not in entry:
            errors.append(f"{ident}: missing {field}")
    if entry.get("decision") is not None:
        errors.append(f"{ident}: decision must stay empty until a human records one")
    hypothesis = entry.get("hypothesis") or {}
    if hypothesis.get("author") != HYPOTHESIS_AUTHOR:
        errors.append(f"{ident}: hypothesis must be marked {HYPOTHESIS_AUTHOR}")
    if hypothesis.get("is_ground_truth") is not False:
        errors.append(f"{ident}: hypothesis must not claim ground truth")
    if hypothesis.get("confidence") not in HYPOTHESIS_CONFIDENCE:
        errors.append(f"{ident}: hypothesis confidence must be low, medium or high")
    if entry.get("mechanism") not in MECHANISMS:
        errors.append(f"{ident}: mechanism must be one of MECHANISMS")
    if not entry.get("code_refs"):
        errors.append(f"{ident}: no code references")
    refs = set()
    for ref in entry.get("code_refs") or ():
        value = ref.get("ref") if isinstance(ref, dict) else ref
        if not CODE_REF.match(str(value)):
            errors.append(f"{ident}: malformed code reference {value!r}")
        else:
            refs.add(str(value))
    if not entry.get("questions"):
        errors.append(f"{ident}: no reviewer questions")
    if not entry.get("evidence_against"):
        errors.append(f"{ident}: no counter-evidence recorded")
    if not (entry.get("license") or {}).get("status"):
        errors.append(f"{ident}: licence status missing")
    # Redistribution is a licence rule, not a preference: code text may only be
    # shipped for a permissive row, and only against a reference it cites.
    for excerpt in entry.get("excerpts") or ():
        if not (entry.get("license") or {}).get("redistributable"):
            errors.append(f"{ident}: code excerpt on a non-permissive licence")
            break
        if excerpt.get("ref") not in refs:
            errors.append(
                f"{ident}: excerpt {excerpt.get('ref')!r} is not in code_refs"
            )
        if not str(excerpt.get("text") or "").strip():
            errors.append(f"{ident}: empty code excerpt")
    if rec is not None:
        provenance = entry.get("provenance") or {}
        for entry_key, rec_key in (
            ("source_id", "source_id"),
            ("repo", "repo"),
            ("path", "path"),
            ("sha256", "sha256"),
            ("commit_sha", "commit_sha"),
            ("file_url", "file_url"),
        ):
            if provenance.get(entry_key) != rec.get(rec_key):
                errors.append(
                    f"{ident}: provenance {entry_key} does not match manifest"
                )
    return errors


def validate_decision(decision: dict) -> list[str]:
    """Reject anything that is not a complete, human-attributed decision."""
    errors = []
    ident = decision.get("candidate_id", "<no id>")
    for field in DECISION_FIELDS:
        if field not in decision:
            errors.append(f"{ident}: decision missing {field}")
    if decision.get("source") != "human":
        errors.append(f"{ident}: decision source must be 'human'")
    reviewer = str(decision.get("reviewer") or "").strip()
    if reviewer.lower() in PLACEHOLDER_REVIEWERS:
        errors.append(f"{ident}: reviewer attribution is a placeholder")
    if len(reviewer) > 64:
        errors.append(f"{ident}: reviewer attribution is too long")
    state = decision.get("state")
    if state not in REVIEW_STATES:
        errors.append(f"{ident}: invalid review state {state!r}")
    refs = decision.get("evidence_refs")
    if not isinstance(refs, list) or not all(
        isinstance(ref, str) and ref.strip() for ref in refs or ()
    ):
        errors.append(
            f"{ident}: evidence_refs must be a non-empty list of code references"
        )
    if state in SCORED_STATES:
        if not str(decision.get("rationale") or "").strip():
            errors.append(f"{ident}: a scored verdict needs a rationale")
        if not refs:
            errors.append(
                f"{ident}: a scored verdict needs at least one code reference"
            )
    decided_at = str(decision.get("decided_at") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", decided_at):
        errors.append(f"{ident}: decided_at must be an ISO-8601 UTC timestamp")
    if decision.get("confidence") is not None:
        value = decision["confidence"]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 1
        ):
            errors.append(f"{ident}: confidence must be between 0 and 1")
    return errors


def apply_decisions(records, decisions) -> list[dict]:
    """Merge human decisions into records without touching the manifest on disk.

    Unknown ids or invalid decisions raise: a decision must never be silently
    dropped or partially applied.
    """
    by_id = {rec["id"]: rec for rec in records}
    merged = {rec["id"]: dict(rec) for rec in records}
    for decision in decisions:
        errors = validate_decision(decision)
        if errors:
            raise ValueError("; ".join(errors))
        ident = decision["candidate_id"]
        if ident not in by_id:
            raise ValueError(f"decision for unknown candidate {ident}")
        review_record = {
            "state": decision["state"],
            "evidence": "; ".join(decision["evidence_refs"]),
            "mechanism": decision.get("mechanism"),
            "explanation": decision.get("rationale", ""),
            "confidence": decision.get("confidence"),
            "reviewer": decision["reviewer"],
            "decided_at": decision["decided_at"],
        }
        merged[ident]["review"] = review_record
        merged[ident]["scored"] = decision["state"] in SCORED_STATES
    return [merged[rec["id"]] for rec in records]


def scored_records(records, decisions) -> list[dict]:
    """The only records allowed into a scored set: human, definitive verdicts."""
    return [rec for rec in apply_decisions(records, decisions) if rec.get("scored")]


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
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_decisions(path=REVIEWS) -> list[dict]:
    """Human decisions, oldest first. An absent log means nobody has decided yet."""
    return load_manifest(path)


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
