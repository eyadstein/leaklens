"""Exact-fingerprint suppression: validation, expiry, diagnostics, no mutation."""

from datetime import date

import pytest

from leaklens.leakage import Finding
from leaklens.policy import apply_policy
from leaklens.suppression import (
    SuppressionError,
    evaluate,
    parse_rules,
)

FINDING = Finding("high", "target_leakage", "leak", "predicts target")
OTHER = Finding("medium", "id_like", "uid", "identifier")


def _rule(fingerprint=None, **extra):
    payload = {
        "fingerprint": fingerprint or FINDING.fingerprint,
        "reason": "accepted in this dataset",
    }
    payload.update(extra)
    return payload


def test_exact_fingerprint_matches_and_is_reported():
    result = evaluate(parse_rules([_rule()]), [FINDING, OTHER], today=date(2026, 1, 1))
    assert result.fingerprints == {FINDING.fingerprint}
    payload = result.to_dict()
    assert payload["suppressed"] == 1
    assert payload["rules"][0]["reason"] == "accepted in this dataset"


def test_no_wildcard_or_prefix_matching():
    result = evaluate(
        parse_rules([_rule(fingerprint=FINDING.fingerprint[:12] + "0" * 52)]),
        [FINDING],
        today=date(2026, 1, 1),
    )
    assert result.matched == ()
    assert result.diagnostics and "matched no current finding" in result.diagnostics[0]


def test_reason_is_required():
    with pytest.raises(SuppressionError, match="missing: reason"):
        parse_rules([{"fingerprint": FINDING.fingerprint}])


def test_blank_reason_is_rejected():
    with pytest.raises(SuppressionError, match="must not be empty"):
        parse_rules([_rule(reason="   ")])


def test_invalid_fingerprint_is_rejected():
    with pytest.raises(SuppressionError, match="64 lowercase hexadecimal"):
        parse_rules([_rule(fingerprint="not-a-fingerprint")])


def test_unknown_keys_are_rejected():
    with pytest.raises(SuppressionError, match="unsupported keys"):
        parse_rules([_rule(scope="everything")])


def test_duplicate_rules_are_rejected_with_both_locations():
    with pytest.raises(SuppressionError, match=r"suppressions\[1\].*suppressions\[0\]"):
        parse_rules([_rule(), _rule()])


def test_malformed_expiry_is_rejected():
    with pytest.raises(SuppressionError, match="YYYY-MM-DD"):
        parse_rules([_rule(expires="31/12/2026")])
    with pytest.raises(SuppressionError, match="not a valid date"):
        parse_rules([_rule(expires="2026-02-31")])


def test_expired_rule_is_reported_and_not_applied():
    rules = parse_rules([_rule(expires="2025-06-01")])
    result = evaluate(rules, [FINDING], today=date(2026, 1, 1))
    assert result.matched == ()
    assert "expired on 2025-06-01" in result.diagnostics[0]


def test_rule_is_active_through_its_expiry_day():
    rules = parse_rules([_rule(expires="2026-01-01")])
    assert evaluate(rules, [FINDING], today=date(2026, 1, 1)).matched


def test_unmatched_rule_is_diagnosed():
    rules = parse_rules([_rule(fingerprint=OTHER.fingerprint)])
    result = evaluate(rules, [FINDING], today=date(2026, 1, 1))
    assert result.matched == ()
    assert "matched no current finding" in result.diagnostics[0]


def test_suppression_labels_but_never_mutates_the_finding():
    before = FINDING.to_dict()
    outcome = apply_policy(
        [FINDING, OTHER],
        rules=parse_rules([_rule()]),
        today=date(2026, 1, 1),
    )
    assert outcome.findings[0] is FINDING
    assert FINDING.to_dict() == before
    assert FINDING.severity == "high"
    assert outcome.active == (OTHER,)
    payload = outcome.finding_payload(FINDING)
    assert payload["status"] == "suppressed"
    assert payload["suppression"]["reason"] == "accepted in this dataset"
    assert payload["severity"] == "high"
    assert payload["fingerprint"] == before["fingerprint"]
    assert payload["evidence"] == before["evidence"]


def test_rules_are_sorted_for_deterministic_output():
    rules = parse_rules(
        [
            _rule(fingerprint=OTHER.fingerprint),
            _rule(fingerprint=FINDING.fingerprint),
        ]
    )
    assert [rule.fingerprint for rule in rules] == sorted(
        [OTHER.fingerprint, FINDING.fingerprint]
    )
