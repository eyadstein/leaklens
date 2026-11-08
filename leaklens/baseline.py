"""Deterministic, versioned finding baselines.

A baseline records the stable fingerprints of accepted findings so later audits
can distinguish new, unchanged and resolved issues. The format is strict: it is
validated before use, unknown keys are rejected, and a baseline produced with a
different fingerprint schema is refused rather than compared against
incompatible identities.

Baseline files contain only finding identity metadata (fingerprint, kind,
severity and column). They never contain dataset rows, feature values, source
code, credentials or machine-specific paths.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .fingerprint import SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from .leakage import Finding

BASELINE_SCHEMA = "leaklens-baseline"
BASELINE_VERSION = 1
MAX_ENTRIES = 100_000
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL = {
    "schema",
    "version",
    "fingerprint_schema",
    "tool_version",
    "findings",
}
_ENTRY_FIELDS = {"fingerprint", "kind", "severity", "column"}


class BaselineError(ValueError):
    """A baseline is missing, malformed, incompatible or cannot be written."""


def _text(value: Any, field: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise BaselineError(f"{field} must be a string")
    text = value.strip()
    if required and not text:
        raise BaselineError(f"{field} must not be empty")
    if len(text) > limit:
        raise BaselineError(f"{field} exceeds {limit} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise BaselineError(f"{field} contains control characters")
    return text


@dataclass(frozen=True, slots=True)
class BaselineEntry:
    """One recorded finding identity."""

    fingerprint: str
    kind: str
    severity: str
    column: str

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or not _FINGERPRINT.match(
            self.fingerprint
        ):
            raise BaselineError(
                "fingerprint must be 64 lowercase hexadecimal characters"
            )
        _text(self.kind, "kind", 120)
        _text(self.severity, "severity", 20)
        _text(self.column, "column", 300, required=False)

    def to_dict(self) -> dict[str, str]:
        return {
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "severity": self.severity,
            "column": self.column,
        }


@dataclass(frozen=True, slots=True)
class BaselineDiff:
    """Comparison of a baseline with the findings of one audit."""

    new: tuple[Finding, ...] = ()
    unchanged: tuple[Finding, ...] = ()
    changed: tuple[Finding, ...] = ()
    resolved: tuple[BaselineEntry, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {
            "new": len(self.new),
            "unchanged": len(self.unchanged),
            "changed": len(self.changed),
            "resolved": len(self.resolved),
        }


@dataclass(frozen=True, slots=True)
class Baseline:
    """A validated, deterministic set of recorded finding identities."""

    entries: tuple[BaselineEntry, ...] = ()
    tool_version: str = __version__
    fingerprint_schema: int = SCHEMA_VERSION
    version: int = BASELINE_VERSION

    def __post_init__(self) -> None:
        if self.version != BASELINE_VERSION:
            raise BaselineError(
                f"unsupported baseline version {self.version!r}; this LeakLens "
                f"supports version {BASELINE_VERSION}"
            )
        if self.fingerprint_schema != SCHEMA_VERSION:
            raise BaselineError(
                f"baseline uses fingerprint schema {self.fingerprint_schema!r}, "
                f"but this LeakLens computes schema {SCHEMA_VERSION}; "
                "regenerate the baseline with --write-baseline or --update-baseline"
            )
        _text(self.tool_version, "tool_version", 40)
        if len(self.entries) > MAX_ENTRIES:
            raise BaselineError(f"baseline has more than {MAX_ENTRIES} entries")
        seen: set[str] = set()
        for entry in self.entries:
            if entry.fingerprint in seen:
                raise BaselineError(
                    f"duplicate baseline fingerprint: {entry.fingerprint}"
                )
            seen.add(entry.fingerprint)

    @classmethod
    def from_findings(
        cls, findings: Any, *, tool_version: str = __version__
    ) -> Baseline:
        """Build a deterministic baseline, collapsing repeated identities."""
        ordered = sorted(findings, key=lambda f: (f.kind, str(f.column), f.fingerprint))
        entries: dict[str, BaselineEntry] = {}
        for finding in ordered:
            fingerprint = finding.fingerprint
            entries.setdefault(
                fingerprint,
                BaselineEntry(
                    fingerprint=fingerprint,
                    kind=finding.kind,
                    severity=finding.severity,
                    column=str(finding.column),
                ),
            )
        return cls(
            entries=tuple(entries[key] for key in sorted(entries)),
            tool_version=tool_version,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> Baseline:
        if not isinstance(payload, dict):
            raise BaselineError("baseline must be a JSON object")
        unknown = sorted(set(payload) - _TOP_LEVEL)
        if unknown:
            raise BaselineError(f"unsupported baseline keys: {', '.join(unknown)}")
        if payload.get("schema") != BASELINE_SCHEMA:
            raise BaselineError(
                f"baseline schema must be {BASELINE_SCHEMA!r}, "
                f"got {payload.get('schema')!r}"
            )
        version = payload.get("version")
        if type(version) is not int or version != BASELINE_VERSION:
            raise BaselineError(
                f"unsupported baseline version {version!r}; this LeakLens "
                f"supports version {BASELINE_VERSION}"
            )
        schema = payload.get("fingerprint_schema")
        if type(schema) is not int:
            raise BaselineError("fingerprint_schema must be an integer")
        if schema != SCHEMA_VERSION:
            raise BaselineError(
                f"baseline uses fingerprint schema {schema}, but this LeakLens "
                f"computes schema {SCHEMA_VERSION}; regenerate the baseline with "
                "--write-baseline or --update-baseline"
            )
        tool_version = _text(payload.get("tool_version", "unknown"), "tool_version", 40)
        items = payload.get("findings")
        if not isinstance(items, list):
            raise BaselineError("findings must be an array")
        if len(items) > MAX_ENTRIES:
            raise BaselineError(f"baseline has more than {MAX_ENTRIES} entries")
        entries: list[BaselineEntry] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            where = f"findings[{index}]"
            if not isinstance(item, dict):
                raise BaselineError(f"{where} must be an object")
            unknown = sorted(set(item) - _ENTRY_FIELDS)
            if unknown:
                raise BaselineError(
                    f"{where} has unsupported keys: {', '.join(unknown)}"
                )
            missing = sorted(_ENTRY_FIELDS - set(item))
            if missing:
                raise BaselineError(f"{where} is missing: {', '.join(missing)}")
            entry = BaselineEntry(
                fingerprint=item["fingerprint"],
                kind=item["kind"],
                severity=item["severity"],
                column=item["column"],
            )
            if entry.fingerprint in seen:
                raise BaselineError(
                    f"{where} duplicates fingerprint {entry.fingerprint}"
                )
            seen.add(entry.fingerprint)
            entries.append(entry)
        entries.sort(key=lambda entry: entry.fingerprint)
        return cls(
            entries=tuple(entries),
            tool_version=tool_version,
            fingerprint_schema=schema,
            version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise BaselineError(
                f"baseline not found: {source}; create it with --write-baseline"
            ) from None
        except (OSError, UnicodeDecodeError) as exc:
            raise BaselineError(f"cannot read baseline {source}: {exc}") from None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BaselineError(
                f"invalid JSON in baseline {source} at line {exc.lineno} "
                f"column {exc.colno}: {exc.msg}"
            ) from None
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_SCHEMA,
            "version": self.version,
            "fingerprint_schema": self.fingerprint_schema,
            "tool_version": self.tool_version,
            "findings": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        """Deterministic serialization in a fixed field order."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def write(self, path: str | Path, *, update: bool = False) -> Path:
        """Atomically write the baseline; never overwrite without ``update``."""
        target = Path(path)
        if target.is_dir():
            raise BaselineError(f"baseline path is a directory: {target}")
        if target.is_symlink():
            raise BaselineError(
                f"refusing to write a baseline through a symlink: {target}"
            )
        if target.exists() and not update:
            raise BaselineError(
                f"baseline already exists: {target}; pass --update-baseline to replace it"
            )
        parent = target.parent if str(target.parent) else Path(".")
        if not parent.is_dir():
            raise BaselineError(f"baseline directory does not exist: {parent}")
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
            raise BaselineError(f"cannot write baseline {target}: {exc}") from None
        return target


def compare(baseline: Baseline, findings: Any) -> BaselineDiff:
    """Classify current findings against a baseline using stable fingerprints."""
    current: dict[str, Finding] = {}
    for finding in sorted(
        findings, key=lambda f: (f.kind, str(f.column), f.fingerprint)
    ):
        current.setdefault(finding.fingerprint, finding)
    recorded = {entry.fingerprint: entry for entry in baseline.entries}
    new = tuple(f for key, f in current.items() if key not in recorded)
    unchanged = tuple(
        f
        for key, f in current.items()
        if key in recorded and f.severity == recorded[key].severity
    )
    changed = tuple(
        f
        for key, f in current.items()
        if key in recorded and f.severity != recorded[key].severity
    )
    resolved = tuple(entry for key, entry in recorded.items() if key not in current)
    return BaselineDiff(
        new=new, unchanged=unchanged, changed=changed, resolved=resolved
    )
