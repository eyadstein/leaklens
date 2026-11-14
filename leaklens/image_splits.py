"""Declared-split disjointness for COCO, YOLO and VOC image datasets (Phase 17 M3).

The audit is read-only. It reads the split *declarations* a dataset already
ships, resolves each declaration to a file inside the dataset root, and compares
the files by SHA-256. Two identical files under different names are therefore
the same evidence as one path declared in two splits; a different encoding of
the same picture is not.

Nothing here decodes an image, so Pillow is not required and the M-S5 decode
budgets are untouched. A digest is computed only after the declaration resolved
to a file inside the root, and every file is size-capped before it is read.

A cross-split duplicate, or one image path declared in two splits, is an
integrity concern worth investigating; it is not proof that a model leaked.
"""

from __future__ import annotations

import importlib.util
import json
import os
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .availability import EngineUnavailable
from .evidence import Evidence
from .images import IMAGE_SUFFIXES, MAX_IMAGE_BYTES, file_digest
from .leakage import Finding, score
from .paths import PathOutsideRoot, resolve_within

PARTITIONS = ("train", "val", "test")

# Alias table shared by the three formats. ``validation`` is the only spelling
# that needs normalising; ``trainval`` is deliberately absent (it is a union,
# not a partition).
ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

MAX_MANIFEST_BYTES = 64 * 1024 * 1024  # one annotation JSON/YAML document
MAX_LIST_BYTES = 8 * 1024 * 1024  # one VOC/YOLO text list
MAX_DECLARATIONS = 100_000  # records read from one dataset audit
MAX_PATH_CHARS = 1_024  # a single declared path
MAX_DIGEST_BYTES = 1024 * 1024 * 1024  # aggregate bytes read for digests
MAX_FINDINGS = 20  # per-group findings before a bounded summary
MAX_EVIDENCE_PATHS = 3  # paths shown per partition in one locator
MAX_LIMITATIONS = 16  # distinct limitation messages kept
MAX_SOURCE_CHARS = 300  # whole locator/provenance field

_YOLO_CONFIGS = ("data.yaml", "dataset.yaml")
_VOC_LISTS = (("train.txt", "train"), ("val.txt", "val"), ("test.txt", "test"))
_SPLIT_TOKEN = re.compile(r"[a-z]+")


class SplitUnusable(ValueError):
    """The dataset declares no split this audit can interpret."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Parsed:
    """One format's declarations, before files are digested."""

    format: str
    members: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: {name: {} for name in PARTITIONS}
    )
    declared: Counter = field(default_factory=Counter)
    unresolved: Counter = field(default_factory=Counter)
    limitations: list[str] = field(default_factory=list)
    total: int = 0

    def limit(self, message: str) -> None:
        if message not in self.limitations and len(self.limitations) < MAX_LIMITATIONS:
            self.limitations.append(message)

    def record(self, partition: str, source: str) -> bool:
        """Count one declaration and say whether the cap still allows work."""
        if self.total >= MAX_DECLARATIONS:
            self.limit(
                f"declaration cap ({MAX_DECLARATIONS}) reached; remaining entries "
                "were not read"
            )
            return False
        self.total += 1
        self.declared[partition] += 1
        return True

    def verify(self, partition: str, identity: str, source: str) -> None:
        self.members[partition].setdefault(identity, set()).add(source)


@dataclass(frozen=True, slots=True)
class SplitAudit:
    """Findings plus the bounded, deterministic record of what was audited."""

    findings: list[Finding]
    stats: dict
    cross_groups: tuple[dict, ...] = ()
    within_groups: tuple[dict, ...] = ()
    identity_conflicts: tuple[dict, ...] = ()

    @property
    def ok(self) -> bool:
        return self.stats.get("status") == "ok"

    def to_dict(self) -> dict:
        return {
            "trust_score": score(self.findings),
            "ok": self.ok,
            **self.stats,
            "cross_split": [dict(group) for group in self.cross_groups],
            "within_split": [dict(group) for group in self.within_groups],
            "identity_conflicts": [dict(group) for group in self.identity_conflicts],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def detect_format(root: str | Path) -> str:
    """The documented detection order: VOC, YOLO, COCO, else ``unknown``."""
    base = Path(root)
    if (base / "ImageSets" / "Main").is_dir():
        return "voc"
    if any((base / name).is_file() for name in _YOLO_CONFIGS):
        return "yolo"
    annotations = base / "annotations"
    if annotations.is_dir() and any(annotations.glob("*.json")):
        return "coco"
    return "unknown"


def _read_bytes(path: Path, limit: int) -> bytes:
    try:
        if path.stat().st_size > limit:
            raise SplitUnusable("too_large")
        return path.read_bytes()
    except OSError:
        raise SplitUnusable("unreadable") from None


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise SplitUnusable("invalid_encoding") from None


def _bounded(text: str) -> str:
    return (
        text if len(text) <= MAX_SOURCE_CHARS else text[: MAX_SOURCE_CHARS - 3] + "..."
    )


def _has_control(text: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in text)


def _rel(root: Path, path: Path) -> str | None:
    """Containment-checked POSIX path relative to ``root``."""
    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return None
    relative = relative.replace("\\", "/")
    if (
        relative in {"", ".", ".."}
        or relative.startswith("../")
        or os.path.isabs(relative)
    ):
        return None
    return posixpath.normpath(relative)


def _relative_identity(
    root: Path, declared: str, *, prefixes: tuple[str, ...] = ()
) -> tuple[str | None, str]:
    """Resolve one declared path to a contained image, or a stable reason."""
    text = declared.strip()
    if not text:
        return None, "empty_path"
    if len(text) > MAX_PATH_CHARS:
        return None, "path_too_long"
    if _has_control(text):
        return None, "control_character"
    if "://" in text:
        # Never fetch a remote declaration; it is reported, not followed.
        return None, "remote_path"

    candidates = [text]
    if not Path(text.replace("\\", "/")).is_absolute():
        candidates += [f"{prefix.rstrip('/')}/{text}" for prefix in prefixes]

    found: list[Path] = []
    for candidate in candidates:
        try:
            resolved = resolve_within(root, candidate)
        except PathOutsideRoot:
            return None, "outside_root"
        except OSError:
            return None, "unreadable"
        try:
            exists = resolved.is_file()
        except OSError:
            return None, "unreadable"
        if exists and resolved not in found:
            found.append(resolved)

    if not found:
        return None, "missing_asset"
    if len(found) > 1:
        return None, "ambiguous_asset"
    resolved = found[0]
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        return None, "unsupported_suffix"
    identity = _rel(root, resolved)
    if identity is None:
        return None, "outside_root"
    if _has_control(identity):
        return None, "control_character"
    return identity, ""


def _split_from_stem(stem: str) -> str | None:
    """First train/val/test token in an annotation file name, if any."""
    for token in _SPLIT_TOKEN.findall(stem.lower()):
        if token in ALIASES:
            return ALIASES[token]
    return None


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def _coco_files(root: Path) -> list[Path]:
    annotations = root / "annotations"
    if not annotations.is_dir():
        return []
    try:
        return sorted(
            path
            for path in annotations.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
    except OSError:
        return []


def _parse_coco(root: Path) -> _Parsed:
    parsed = _Parsed("coco")
    files = _coco_files(root)
    if not files:
        parsed.limit("no COCO annotation JSON was found under annotations/")
        return parsed
    for path in files:
        source = _rel(root, path) or path.name
        partition = _split_from_stem(path.stem)
        if partition is None:
            parsed.limit(
                f"{source}: no train/val/test token in the file name; not treated "
                "as a split declaration"
            )
            continue
        try:
            data = json.loads(_decode(_read_bytes(path, MAX_MANIFEST_BYTES)))
        except SplitUnusable as exc:
            parsed.limit(f"{source}: {exc.reason}")
            parsed.unresolved[exc.reason] += 1
            continue
        except (ValueError, RecursionError):
            parsed.limit(f"{source}: invalid JSON")
            parsed.unresolved["invalid_json"] += 1
            continue
        images = data.get("images") if isinstance(data, dict) else None
        if not isinstance(images, list):
            parsed.limit(f"{source}: no images list; not a COCO image declaration")
            parsed.unresolved["missing_images_list"] += 1
            continue
        for record in images:
            if not parsed.record(partition, source):
                break
            name = record.get("file_name") if isinstance(record, dict) else None
            if not isinstance(name, str) or not name.strip():
                parsed.unresolved["malformed_image_record"] += 1
                continue
            # COCO stores paths relative to the images root; try the dataset
            # root first, then ``images/``, and refuse when both exist.
            identity, reason = _relative_identity(root, name, prefixes=("images",))
            if reason:
                parsed.unresolved[reason] += 1
            else:
                parsed.verify(partition, identity, source)
    return parsed


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


def _yaml_module():
    if importlib.util.find_spec("yaml") is None:
        raise EngineUnavailable(
            'YOLO split configs need PyYAML: pip install "leaklens-ml[yaml]"'
        )
    import yaml

    return yaml


def _yolo_entries(value, parsed: _Parsed, source: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        entries = []
        for item in value:
            if isinstance(item, str):
                entries.append(item)
            else:
                parsed.limit(f"{source}: a non-string path entry was ignored")
                parsed.unresolved["unsupported_declaration"] += 1
        return entries
    parsed.limit(f"{source}: unsupported declaration form ({type(value).__name__})")
    parsed.unresolved["unsupported_declaration"] += 1
    return []


def _declare_directory(
    parsed: _Parsed, root: Path, partition: str, directory: Path, source: str
) -> None:
    try:
        entries = sorted(directory.rglob("*"))
    except OSError:
        parsed.limit(f"{source}: {directory.name}/ cannot be listed")
        parsed.unresolved["unreadable"] += 1
        return
    for entry in entries:
        if entry.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        if not parsed.record(partition, source):
            return
        try:
            resolved = resolve_within(root, str(entry))
        except (PathOutsideRoot, OSError):
            parsed.unresolved["outside_root"] += 1
            continue
        identity = _rel(root, resolved)
        if identity is None or _has_control(identity):
            parsed.unresolved["outside_root"] += 1
        else:
            parsed.verify(partition, identity, source)


def _declare_list(
    parsed: _Parsed, root: Path, partition: str, path: Path, source: str
) -> None:
    try:
        text = _decode(_read_bytes(path, MAX_LIST_BYTES))
    except SplitUnusable as exc:
        parsed.limit(f"{source}: {path.name}: {exc.reason}")
        parsed.unresolved[exc.reason] += 1
        return
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if not parsed.record(partition, source):
            return
        identity, reason = _relative_identity(root, entry)
        if reason:
            parsed.unresolved[reason] += 1
        else:
            parsed.verify(partition, identity, source)


def _declare_yolo_path(
    parsed: _Parsed, root: Path, partition: str, declared: str, source: str
) -> None:
    text = declared.strip()
    if not text:
        return
    if "://" in text:
        if not parsed.record(partition, source):
            return
        parsed.unresolved["remote_path"] += 1
        parsed.limit(f"{source}: remote declarations are reported, never fetched")
        return
    try:
        resolved = resolve_within(root, text)
    except PathOutsideRoot:
        if not parsed.record(partition, source):
            return
        parsed.unresolved["outside_root"] += 1
        return
    except OSError:
        if not parsed.record(partition, source):
            return
        parsed.unresolved["unreadable"] += 1
        return
    try:
        if resolved.is_dir():
            _declare_directory(parsed, root, partition, resolved, source)
            return
        is_file = resolved.is_file()
    except OSError:
        is_file = False
    if not is_file:
        if not parsed.record(partition, source):
            return
        parsed.unresolved["missing_asset"] += 1
        return
    if resolved.suffix.lower() == ".txt":
        _declare_list(parsed, root, partition, resolved, source)
        return
    if not parsed.record(partition, source):
        return
    identity, reason = _relative_identity(root, str(resolved))
    if reason:
        parsed.unresolved[reason] += 1
    else:
        parsed.verify(partition, identity, source)


def _parse_yolo(root: Path) -> _Parsed:
    parsed = _Parsed("yolo")
    config = next(
        (root / name for name in _YOLO_CONFIGS if (root / name).is_file()), None
    )
    if config is None:
        parsed.limit("no data.yaml or dataset.yaml was found")
        return parsed
    source = _rel(root, config) or config.name
    yaml = _yaml_module()
    try:
        data = yaml.safe_load(_decode(_read_bytes(config, MAX_MANIFEST_BYTES)))
    except SplitUnusable as exc:
        parsed.limit(f"{source}: {exc.reason}")
        parsed.unresolved[exc.reason] += 1
        return parsed
    except Exception as exc:  # noqa: BLE001 - any YAML error is a typed limitation
        parsed.limit(
            f"{source}: YAML could not be parsed safely ({type(exc).__name__})"
        )
        parsed.unresolved["yaml_error"] += 1
        return parsed
    if not isinstance(data, dict):
        parsed.limit(f"{source}: the top level is not a mapping")
        parsed.unresolved["malformed_config"] += 1
        return parsed
    for key, value in data.items():
        partition = ALIASES.get(str(key).strip().lower())
        if partition is None:
            continue
        for declared in _yolo_entries(value, parsed, source):
            _declare_yolo_path(parsed, root, partition, declared, source)
    if not parsed.declared:
        parsed.limit(f"{source}: no train/val/test declaration could be read")
    return parsed


# ---------------------------------------------------------------------------
# Pascal VOC
# ---------------------------------------------------------------------------


def _voc_index(parsed: _Parsed, root: Path) -> dict[str, list[str]]:
    directory = root / "JPEGImages"
    index: dict[str, set[str]] = {}
    if not directory.is_dir():
        parsed.limit(
            "no JPEGImages directory; VOC identifiers cannot be resolved to images"
        )
        return {}
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        parsed.limit("JPEGImages cannot be listed")
        parsed.unresolved["unreadable"] += 1
        return {}
    for entry in entries:
        if entry.suffix.lower() not in IMAGE_SUFFIXES or _has_control(entry.name):
            continue
        if not entry.is_file():
            continue
        index.setdefault(entry.stem, set()).add(entry.name)
        if len(index) > MAX_DECLARATIONS:
            parsed.limit(
                f"JPEGImages holds more than {MAX_DECLARATIONS} distinct identifiers; "
                "the index was truncated"
            )
            break
    return {stem: sorted(names) for stem, names in index.items()}


def _resolve_voc(
    root: Path, index: dict[str, list[str]], identifier: str
) -> tuple[str | None, str]:
    text = identifier.strip()
    if not text:
        return None, "empty_path"
    if len(text) > MAX_PATH_CHARS:
        return None, "path_too_long"
    if _has_control(text):
        return None, "control_character"
    if "/" in text or "\\" in text or Path(text).suffix.lower() in IMAGE_SUFFIXES:
        return _relative_identity(root, text, prefixes=("JPEGImages",))
    names = index.get(text)
    if not names:
        return None, "missing_asset"
    if len(names) > 1:
        # The identifier maps to more than one image extension.
        return None, "ambiguous_asset"
    return _relative_identity(root, f"JPEGImages/{names[0]}")


def _parse_voc(root: Path) -> _Parsed:
    parsed = _Parsed("voc")
    main = root / "ImageSets" / "Main"
    if not main.is_dir():
        parsed.limit("no ImageSets/Main directory was found")
        return parsed
    has_train = (main / "train.txt").is_file()
    has_val = (main / "val.txt").is_file()
    if (main / "trainval.txt").is_file():
        if has_train and has_val:
            parsed.limit(
                "ImageSets/Main/trainval.txt is the documented union of train and val; "
                "it is not audited as a separate partition"
            )
        else:
            parsed.limit(
                "only ImageSets/Main/trainval.txt is present; train/val membership "
                "cannot be separated"
            )
    index = _voc_index(parsed, root)
    for name, partition in _VOC_LISTS:
        path = main / name
        if not path.is_file():
            continue
        source = _rel(root, path) or name
        try:
            text = _decode(_read_bytes(path, MAX_LIST_BYTES))
        except SplitUnusable as exc:
            parsed.limit(f"{source}: {exc.reason}")
            parsed.unresolved[exc.reason] += 1
            continue
        for line in text.splitlines():
            identifier = line.strip()
            if not identifier or identifier.startswith("#"):
                continue
            if not parsed.record(partition, source):
                break
            identity, reason = _resolve_voc(root, index, identifier)
            if reason:
                parsed.unresolved[reason] += 1
            else:
                parsed.verify(partition, identity, source)
    if not any(parsed.declared.values()):
        parsed.limit("no train/val/test ImageSets list was read")
    return parsed


# ---------------------------------------------------------------------------
# Disjointness
# ---------------------------------------------------------------------------


def _digest_files(
    root: Path, parsed: _Parsed, max_digest_bytes: int
) -> tuple[dict[str, str], int]:
    identities = sorted(
        {identity for bucket in parsed.members.values() for identity in bucket}
    )
    digests: dict[str, str] = {}
    used = 0
    for identity in identities:
        path = root / identity
        try:
            size = path.stat().st_size
        except OSError:
            parsed.unresolved["unreadable"] += 1
            continue
        if size > MAX_IMAGE_BYTES:
            parsed.unresolved["too_large"] += 1
            continue
        if used + size > max_digest_bytes:
            parsed.limit(
                f"digest budget ({max_digest_bytes} bytes) exhausted; remaining "
                "files were not hashed"
            )
            parsed.unresolved["digest_budget_exhausted"] += 1
            break
        try:
            digests[identity] = file_digest(path)
        except OSError:
            parsed.unresolved["unreadable"] += 1
            continue
        used += size
    return digests, used


def _groups(
    parsed: _Parsed, digests: dict[str, str]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]], dict[str, set[str]]]:
    identity_parts: dict[str, set[str]] = {}
    for partition in PARTITIONS:
        for identity in parsed.members[partition]:
            identity_parts.setdefault(identity, set()).add(partition)

    by_digest: dict[str, dict[str, list[str]]] = {}
    for identity, digest in digests.items():
        for partition in sorted(identity_parts[identity]):
            by_digest.setdefault(digest, {}).setdefault(partition, []).append(identity)
    for buckets in by_digest.values():
        for names in buckets.values():
            names.sort()

    cross = {
        digest: buckets for digest, buckets in by_digest.items() if len(buckets) >= 2
    }
    within = {
        digest: buckets
        for digest, buckets in by_digest.items()
        if len(buckets) == 1 and len(next(iter(buckets.values()))) > 1
    }
    # An identity that was digested is already covered by its digest group.
    conflicts = {
        identity: parts
        for identity, parts in identity_parts.items()
        if len(parts) >= 2 and identity not in digests
    }
    return cross, within, conflicts


def _ordered(partitions) -> list[str]:
    return [name for name in PARTITIONS if name in partitions]


def _locator(buckets: dict[str, list[str]]) -> str:
    parts = []
    for partition in _ordered(buckets):
        for name in buckets[partition][:MAX_EVIDENCE_PATHS]:
            parts.append(f"{partition}:{name}")
    suffix = ""
    if sum(len(names) for names in buckets.values()) > len(parts):
        suffix = "; ..."
    return _bounded("; ".join(parts) + suffix)


def _provenance(parsed: _Parsed, buckets: dict[str, list[str]]) -> str:
    sources = sorted(
        {
            source
            for partition in _ordered(buckets)
            for identity in buckets[partition]
            for source in parsed.members[partition][identity]
        }
    )
    return _bounded("; ".join(sources))


def _finding_detail(digest: str, parts: list[str]) -> str:
    return (
        f"the same image bytes (SHA-256 {digest[:12]}...) are declared in "
        f"{' + '.join(parts)}; duplicate content is an integrity concern, not proof "
        "of leakage"
    )


def _findings(
    parsed: _Parsed,
    cross: dict[str, dict[str, list[str]]],
    conflicts: dict[str, set[str]],
) -> list[Finding]:
    findings: list[Finding] = []
    groups = sorted(cross.items())
    for digest, buckets in groups[:MAX_FINDINGS]:
        parts = _ordered(buckets)
        findings.append(
            Finding(
                "high",
                "image_split_overlap",
                "*",
                _finding_detail(digest, parts),
                evidence=(
                    Evidence(
                        source="image",
                        kind="duplicate",
                        description=(
                            "identical file bytes are declared in more than one "
                            "dataset partition"
                        ),
                        location=_locator(buckets),
                        observed=(
                            f"{len(parts)} partitions, "
                            f"{sum(len(names) for names in buckets.values())} path(s)"
                        ),
                        expected="each image content is declared in exactly one partition",
                    ),
                    Evidence(
                        source="split",
                        kind="parse",
                        description="declaration files that named these members",
                        location=_provenance(parsed, buckets),
                    ),
                ),
                stable_key=f"sha256:{digest}:{'+'.join(parts)}",
            )
        )
    for identity, parts in sorted(conflicts.items())[
        : max(0, MAX_FINDINGS - len(findings))
    ]:
        ordered = _ordered(parts)
        findings.append(
            Finding(
                "high",
                "image_split_overlap",
                "*",
                f"{_bounded(identity)} is declared in {' + '.join(ordered)}",
                evidence=(
                    Evidence(
                        source="split",
                        kind="overlap",
                        description=(
                            "one image path is declared in more than one dataset "
                            "partition"
                        ),
                        location=_bounded(identity),
                        observed=f"declared in {', '.join(ordered)}",
                        expected="each image path is declared in exactly one partition",
                    ),
                ),
                stable_key=_bounded(f"path:{identity}")[:200],
            )
        )
    total = len(groups) + len(conflicts)
    if len(findings) < total:
        findings.append(
            Finding(
                "high",
                "image_split_overlap",
                "*",
                f"{total - len(findings)} further cross-split overlap(s) were not listed",
                evidence=(
                    Evidence(
                        source="split",
                        kind="overlap",
                        description=(
                            "the per-group list is bounded; the counts below are complete"
                        ),
                        observed=f"{total} groups, {len(findings)} listed",
                        expected="group counts are reported without listing every path",
                    ),
                ),
                stable_key="truncated",
            )
        )
    return findings


def _group_payload(cross: dict[str, dict[str, list[str]]]) -> tuple[dict, ...]:
    return tuple(
        {
            "digest": digest,
            "partitions": {
                partition: list(buckets[partition]) for partition in _ordered(buckets)
            },
        }
        for digest, buckets in sorted(cross.items())
    )


def _within_payload(within: dict[str, dict[str, list[str]]]) -> tuple[dict, ...]:
    return tuple(
        {
            "digest": digest,
            "partition": next(iter(buckets)),
            "paths": list(next(iter(buckets.values()))),
        }
        for digest, buckets in sorted(within.items())
    )


def _conflict_payload(conflicts: dict[str, set[str]]) -> tuple[dict, ...]:
    return tuple(
        {"path": identity, "partitions": _ordered(parts)}
        for identity, parts in sorted(conflicts.items())
    )


def audit_splits(
    root: str | Path,
    fmt: str = "auto",
    *,
    max_digest_bytes: int = MAX_DIGEST_BYTES,
) -> SplitAudit:
    """Audit declared splits under ``root``; ``fmt`` is auto, coco, yolo or voc."""
    base = Path(root)
    if fmt not in {"auto", "coco", "yolo", "voc"}:
        raise ValueError(f"unsupported format: {fmt!r}")
    if not base.is_dir():
        return SplitAudit(
            findings=[],
            stats={
                "status": "unusable",
                "format": "unknown",
                "reason": "the dataset path is not a directory",
                "partitions": {
                    name: {"declared": 0, "unique": 0} for name in PARTITIONS
                },
                "declarations": 0,
                "verified_images": 0,
                "digested_images": 0,
                "digest_bytes": 0,
                "unresolved": {},
                "limitations": [],
                "cross_split_groups": 0,
                "within_split_groups": 0,
                "identity_conflicts": 0,
                "truncated": False,
            },
        )
    base = base.resolve()
    chosen = detect_format(base) if fmt == "auto" else fmt
    if chosen == "unknown":
        return SplitAudit(
            findings=[],
            stats={
                "status": "unusable",
                "format": "unknown",
                "reason": (
                    "no supported split declaration was found (looked for "
                    "ImageSets/Main, data.yaml or dataset.yaml, annotations/*.json)"
                ),
                "partitions": {
                    name: {"declared": 0, "unique": 0} for name in PARTITIONS
                },
                "declarations": 0,
                "verified_images": 0,
                "digested_images": 0,
                "digest_bytes": 0,
                "unresolved": {},
                "limitations": [],
                "cross_split_groups": 0,
                "within_split_groups": 0,
                "identity_conflicts": 0,
                "truncated": False,
            },
        )

    parser = {"coco": _parse_coco, "yolo": _parse_yolo, "voc": _parse_voc}[chosen]
    parsed = parser(base)
    digests, digest_bytes = _digest_files(base, parsed, max_digest_bytes)
    cross, within, conflicts = _groups(parsed, digests)
    findings = _findings(parsed, cross, conflicts)
    verified = len(
        {identity for bucket in parsed.members.values() for identity in bucket}
    )
    status = "ok" if verified else "unusable"
    if verified == 0 and not parsed.limitations:
        parsed.limit("no declared image could be verified")
    stats = {
        "status": status,
        "format": chosen,
        "partitions": {
            name: {
                "declared": parsed.declared.get(name, 0),
                "unique": len(parsed.members[name]),
            }
            for name in PARTITIONS
        },
        "declarations": parsed.total,
        "verified_images": verified,
        "digested_images": len(digests),
        "digest_bytes": digest_bytes,
        "unresolved": dict(sorted(parsed.unresolved.items())),
        "limitations": list(parsed.limitations),
        "cross_split_groups": len(cross),
        "within_split_groups": len(within),
        "identity_conflicts": len(conflicts),
        "truncated": len(findings) < len(cross) + len(conflicts),
    }
    if status == "unusable":
        stats["reason"] = (
            parsed.limitations[0]
            if parsed.limitations
            else "no declared image could be verified"
        )
    return SplitAudit(
        findings=findings,
        stats=stats,
        cross_groups=_group_payload(cross),
        within_groups=_within_payload(within),
        identity_conflicts=_conflict_payload(conflicts),
    )
