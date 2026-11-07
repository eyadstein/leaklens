"""Explicit, auditable suppression rules.

Suppression is intentionally narrow: a rule matches one exact stable
fingerprint. There are no wildcards, so a rule cannot hide unrelated future
findings. Every rule needs a human reason and may expire; expired rules are
reported and never applied. Invalid configuration fails closed instead of
silently suppressing anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FIELDS = {"fingerprint", "reason", "expires"}
MAX_REASON = 500


class SuppressionError(ValueError):
    """A suppression rule is malformed, duplicated or cannot be evaluated."""


def _text(value: Any, field: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise SuppressionError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise SuppressionError(f"{field} must not be empty")
    if len(text) > limit:
        raise SuppressionError(f"{field} exceeds {limit} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise SuppressionError(f"{field} contains control characters")
    return text


@dataclass(frozen=True, slots=True)
class SuppressionRule:
    """One exact-fingerprint suppression."""

    fingerprint: str
    reason: str
    expires: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or not _FINGERPRINT.match(
            self.fingerprint
        ):
            raise SuppressionError(
                "fingerprint must be 64 lowercase hexadecimal characters; copy it "
                "from a JSON finding's fingerprint field"
            )
        _text(self.reason, "reason", limit=MAX_REASON)
        if self.expires is not None and not isinstance(self.expires, date):
            raise SuppressionError("expires must be a date or None")

    def to_dict(self) -> dict[str, str]:
        payload = {"fingerprint": self.fingerprint, "reason": self.reason}
        if self.expires is not None:
            payload["expires"] = self.expires.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class SuppressionResult:
    """Rules matched by one audit, plus diagnostics for review."""

    matched: tuple[SuppressionRule, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def fingerprints(self) -> frozenset[str]:
        return frozenset(rule.fingerprint for rule in self.matched)

    def rule_for(self, fingerprint: str) -> SuppressionRule | None:
        for rule in self.matched:
            if rule.fingerprint == fingerprint:
                return rule
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "suppressed": len(self.matched),
            "rules": [rule.to_dict() for rule in self.matched],
            "diagnostics": list(self.diagnostics),
        }


def parse_rules(
    items: Any, *, where: str = "suppressions"
) -> tuple[SuppressionRule, ...]:
    """Validate TOML suppression rules and attach the failing location."""
    if items is None:
        return ()
    if not isinstance(items, list):
        raise SuppressionError(f"{where} must be an array of tables")
    rules: list[SuppressionRule] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(items):
        location = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise SuppressionError(f"{location} must be a table")
        unknown = sorted(set(item) - _FIELDS)
        if unknown:
            raise SuppressionError(
                f"{location} has unsupported keys: {', '.join(unknown)}"
            )
        missing = sorted(_FIELDS - set(item) - {"expires"})
        if missing:
            raise SuppressionError(f"{location} is missing: {', '.join(missing)}")
        raw_expires = item.get("expires")
        expires: date | None = None
        if raw_expires is not None:
            if not isinstance(raw_expires, str) or not _DATE.match(raw_expires):
                raise SuppressionError(
                    f"{location}.expires must be a YYYY-MM-DD date string"
                )
            try:
                expires = date.fromisoformat(raw_expires)
            except ValueError:
                raise SuppressionError(
                    f"{location}.expires is not a valid date: {raw_expires}"
                ) from None
        try:
            rule = SuppressionRule(
                fingerprint=item["fingerprint"],
                reason=item["reason"],
                expires=expires,
            )
        except SuppressionError as exc:
            raise SuppressionError(f"{location}: {exc}") from None
        if rule.fingerprint in seen:
            raise SuppressionError(
                f"{location} duplicates the suppression for fingerprint "
                f"{rule.fingerprint} from {where}[{seen[rule.fingerprint]}]"
            )
        seen[rule.fingerprint] = index
        rules.append(rule)
    return tuple(sorted(rules, key=lambda rule: rule.fingerprint))


def evaluate(
    rules: tuple[SuppressionRule, ...],
    findings: Any,
    *,
    today: date | None = None,
) -> SuppressionResult:
    """Match exact fingerprints; report expired and unmatched rules.

    ``today`` is injectable so tests and reproducible reports do not depend on
    the wall clock. Expired rules are diagnostics only and never suppress.
    """
    current = {finding.fingerprint for finding in findings}
    current_day = today or date.today()
    matched: list[SuppressionRule] = []
    diagnostics: list[str] = []
    for rule in rules:
        if rule.expires is not None and rule.expires < current_day:
            diagnostics.append(
                f"suppression rule {rule.fingerprint} expired on "
                f"{rule.expires.isoformat()} and was not applied"
            )
            continue
        if rule.fingerprint not in current:
            diagnostics.append(
                f"suppression rule {rule.fingerprint} matched no current finding"
            )
            continue
        matched.append(rule)
    return SuppressionResult(matched=tuple(matched), diagnostics=tuple(diagnostics))
