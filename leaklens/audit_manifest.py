"""Deterministic, unsigned audit-result manifest and offline verifier (M-S3).

The manifest records what one audit saw and concluded: digests of the declared
input files, the options that affect the result, the detector catalogue, the
serialized findings, the policy outcome, the trust score and the gate decision.

It is **evidence, not a certificate**. Verification proves that the document
(the timestamp excepted) and the declared inputs were not changed after the
manifest was written. It does not
prove who wrote the manifest or that the conclusion is correct; anyone who can
rewrite the manifest can also recompute its digest. Record ``manifest_digest``
out of band (a report, a ticket, a CI log) when a deliberate rewrite must be
detected.

Verification reads declared files as bytes only. It never imports, deserializes
or executes project content, and it uses no third-party module.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .paths import PathOutsideRoot, resolve_within

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the verifier light
    from .assemble import Sections

SCHEMA = "leaklens-audit-manifest"
SCHEMA_VERSION = 1
DOMAIN = b"leaklens:audit-manifest:v1\0"

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_INPUTS = 10_000
MAX_TREE_FILES = 200_000

_IGNORED_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "node_modules", "dist", "build"}
)
_TOP_LEVEL = frozenset(
    {
        "schema",
        "schema_version",
        "tool",
        "generated_at",
        "inputs",
        "options",
        "detectors",
        "result",
        "section_digests",
        "manifest_digest",
    }
)
_RESULT_KEYS = (
    "trust_score",
    "counts",
    "findings",
    "policy",
    "resolved",
    "gate",
)


class ManifestError(ValueError):
    """The manifest is unreadable, unsupported or structurally invalid."""


def canonical_bytes(value: Any) -> bytes:
    """Stable bytes for one JSON value: sorted keys, no optional whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """Domain-separated SHA-256 of one manifest section."""
    return "sha256:" + hashlib.sha256(DOMAIN + canonical_bytes(value)).hexdigest()


def dumps(document: dict[str, Any]) -> str:
    """Serialized manifest with a stable key order and a trailing newline."""
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + _file_sha256(path)


def _raise_walk_error(error: OSError) -> None:
    # os.walk silently skips a directory it cannot list; that would let a
    # change inside the declared tree go unreported, so fail instead.
    raise error


def _tree_digest(root: Path) -> str:
    """Digest a directory as sorted ``relative path + file digest`` records.

    ``os.walk`` does not follow directory symlinks; file symlinks are skipped
    so verification cannot read outside the tree it was pointed at.
    """
    sha = hashlib.sha256()
    sha.update(DOMAIN)
    count = 0
    for directory, dirnames, filenames in os.walk(
        root, onerror=_raise_walk_error, followlinks=False
    ):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _IGNORED_DIRS and not name.endswith(".egg-info")
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            count += 1
            if count > MAX_TREE_FILES:
                raise ManifestError(f"input tree is too large to digest: {root}")
            sha.update(path.relative_to(root).as_posix().encode("utf-8"))
            sha.update(b"\0")
            sha.update(bytes.fromhex(_file_sha256(path)))
            sha.update(b"\n")
    return "sha256:" + sha.hexdigest()


def describe_input(path: str | Path, *, root: str | Path) -> dict[str, str]:
    """Digest one declared regular file or directory for the manifest."""
    source = Path(path)
    try:
        mode = source.stat().st_mode
    except OSError as exc:
        raise ManifestError(f"input not found: {path}") from exc
    if stat.S_ISDIR(mode):
        input_kind = "directory"
    elif stat.S_ISREG(mode):
        input_kind = "file"
    else:
        raise ManifestError(f"input is not a regular file or directory: {path}")
    value = _tree_digest(source) if input_kind == "directory" else _file_digest(source)
    resolved = source.resolve()
    try:
        # Paths under the audit root stay portable and resolve against the
        # verifier's default root.
        display = resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {"path": display, "kind": input_kind, "digest": value}


def detector_catalogue() -> list[dict[str, Any]]:
    """The shipped detectors, in a stable order, read from the one catalogue."""
    from .manifest import DETECTORS

    return sorted(
        (
            {
                "kind": entry.kind,
                "surface": entry.surface,
                "severity": entry.severity,
                "scored": entry.scored,
                "introduced": entry.introduced,
                "requires": entry.requires,
            }
            for entry in DETECTORS
        ),
        key=lambda entry: entry["kind"],
    )


def _sections(document: dict[str, Any]) -> dict[str, Any]:
    result = document["result"]
    return {
        "tool": document["tool"],
        "inputs": document["inputs"],
        "options": document["options"],
        "detectors": document["detectors"],
        **{name: result[name] for name in _RESULT_KEYS},
    }


def _digest_view(document: dict[str, Any]) -> dict[str, Any]:
    # generated_at and the digest itself are excluded so two identical runs
    # produce the same manifest digest.
    return {
        key: value
        for key, value in document.items()
        if key not in ("generated_at", "manifest_digest")
    }


def build(
    sections: Sections,
    *,
    inputs: Sequence[str | Path] = (),
    options: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    root: str | Path,
    version: str | None = None,
) -> dict[str, Any]:
    """Build one manifest from the shared payload sections."""
    result = {
        "trust_score": int(sections.trust_score),
        "counts": dict(sections.counts),
        "findings": list(sections.findings),
        "policy": dict(sections.policy),
        "resolved": list(sections.resolved),
        "gate": dict(gate or {}),
    }
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "leaklens", "version": version or __version__},
        "generated_at": _now(),
        "inputs": sorted(
            (describe_input(path, root=root) for path in inputs),
            key=lambda entry: (entry["path"], entry["kind"]),
        ),
        "options": dict(options or {}),
        "detectors": detector_catalogue(),
        "result": result,
    }
    document["section_digests"] = {
        name: digest(value) for name, value in _sections(document).items()
    }
    document["manifest_digest"] = digest(_digest_view(document))
    return document


def _verify_digests(document: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    stored = document.get("manifest_digest")
    if not isinstance(stored, str):
        issues.append("manifest_digest is missing")
    else:
        try:
            actual = digest(_digest_view(document))
        except (RecursionError, TypeError, ValueError) as exc:
            issues.append(f"manifest cannot be canonicalized: {exc}")
            return issues
        if actual != stored:
            issues.append(
                "manifest digest does not match: the manifest was edited after "
                "it was written, or the digest was not recomputed"
            )
    recorded = document.get("section_digests")
    if not isinstance(recorded, dict):
        issues.append("section_digests is missing")
        return issues
    try:
        sections = _sections(document)
        actual_sections = {name: digest(value) for name, value in sections.items()}
    except (KeyError, RecursionError, TypeError, ValueError) as exc:
        issues.append(f"manifest is missing a required section: {exc}")
        return issues
    for name, value in actual_sections.items():
        if recorded.get(name) != value:
            issues.append(f"{name} changed since the manifest was written")
    for name in sorted(set(recorded) - set(sections)):
        issues.append(f"unknown section digest: {name}")
    return issues


def _verify_inputs(entries: list[Any], root: Path) -> list[str]:
    if len(entries) > MAX_INPUTS:
        return [f"too many input entries: {len(entries)}"]
    issues: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            issues.append(f"malformed input entry #{index}")
            continue
        value = entry.get("path")
        input_kind = entry.get("kind")
        recorded = entry.get("digest")
        if (
            not isinstance(value, str)
            or not value.strip()
            or not isinstance(input_kind, str)
            or not isinstance(recorded, str)
        ):
            issues.append(f"malformed input entry #{index}")
            continue
        try:
            target = resolve_within(root, value)
        except PathOutsideRoot:
            issues.append(f"unsafe path: {value}")
            continue
        except (OSError, ValueError) as exc:
            issues.append(f"cannot resolve {value}: {exc}")
            continue
        if input_kind not in ("file", "directory"):
            issues.append(f"unsupported input kind: {input_kind} ({value})")
            continue
        try:
            mode = target.stat().st_mode
        except FileNotFoundError:
            issues.append(f"missing: {value}")
            continue
        except (OSError, ValueError) as exc:
            issues.append(f"cannot read {value}: {exc}")
            continue
        if input_kind == "file" and not stat.S_ISREG(mode):
            issues.append(f"not a regular file: {value}")
            continue
        if input_kind == "directory" and not stat.S_ISDIR(mode):
            issues.append(f"not a directory: {value}")
            continue
        try:
            actual = (
                _tree_digest(target)
                if input_kind == "directory"
                else _file_digest(target)
            )
        except (OSError, ValueError) as exc:
            issues.append(f"cannot digest {value}: {exc}")
            continue
        if actual != recorded:
            issues.append(f"modified: {value}")
    return issues


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # A repeated key would let two readers see two different documents, so
    # refuse the ambiguity instead of silently keeping the last value.
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def verify(path: str | Path, *, root: str | Path | None = None) -> list[str]:
    """Recompute the manifest and input digests; return human-readable issues.

    Raises :class:`ManifestError` when the file cannot be read, is not JSON, or
    declares a schema this verifier does not understand. A returned issue means
    the manifest is readable but does not verify.
    """
    manifest_path = Path(path)
    try:
        # Read one byte past the cap so a lying stat() cannot turn into an
        # unbounded read.
        with open(manifest_path, "rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest: {exc}") from exc
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ManifestError(f"manifest is larger than {MAX_MANIFEST_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"cannot read manifest as UTF-8: {exc}") from exc
    try:
        document = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"manifest is not valid JSON: {exc.msg} (line {exc.lineno})"
        ) from exc
    except (RecursionError, ValueError) as exc:
        raise ManifestError(f"manifest cannot be parsed: {exc}") from exc
    if not isinstance(document, dict):
        raise ManifestError("manifest is not a JSON object")
    if document.get("schema") != SCHEMA:
        raise ManifestError(f"unsupported manifest schema: {document.get('schema')!r}")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestError(f"unsupported manifest schema_version: {version!r}")
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version: "
            f"{version!r} (this verifier reads {SCHEMA_VERSION})"
        )
    if not isinstance(document.get("result"), dict):
        raise ManifestError("manifest has no result object")
    if not isinstance(document.get("inputs"), list):
        raise ManifestError("manifest inputs must be a list")
    root_dir = Path(root) if root is not None else Path.cwd()
    try:
        root_dir = root_dir.resolve()
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot resolve verification root: {exc}") from exc
    if not root_dir.is_dir():
        raise ManifestError(f"verification root is not a directory: {root_dir}")

    issues: list[str] = [
        f"unknown key: {key}" for key in sorted(set(document) - _TOP_LEVEL)
    ]
    issues.extend(f"missing field: {key}" for key in sorted(_TOP_LEVEL - set(document)))
    issues.extend(_verify_digests(document))
    issues.extend(_verify_inputs(document["inputs"], root_dir))
    return issues
