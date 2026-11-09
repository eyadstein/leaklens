"""Deterministic semantic hypotheses for code, notebooks and dataset schemas.

This module is deliberately *not* a language model.  It is a small, offline,
explainable analyser that looks at names, relationships and control flow which
the rule-based scanner does not cover.  Every result is a hypothesis that points
at an inspectable evidence span; none of it is a verified finding.

User-controlled text (comments, column names, notebook cells) is treated as
untrusted data.  Instruction-like text is reported as a security observation
and is never followed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from leaklens.leakage import Finding

from .schemas import EvidenceSpan, Hypothesis

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD = re.compile(r"[A-Za-z0-9]+")

_INSTRUCTION = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|system\s+prompt|you\s+are\s+now\s+(?:a|an|the)"
    r"|return\s+(?:no|zero)\s+findings|mark\s+(?:this|it|the\s+\w+)\s+(?:as\s+)?(?:safe|clean)"
    r"|do\s+not\s+report|reveal\s+(?:the\s+)?(?:prompt|key|secret)"
    r"|assistant\s*:)",
    re.IGNORECASE,
)

_FUTURE_NAME = re.compile(
    r"(?:^|_)(?:future|next|tomorrow|after|post|later|forward|lead|lookahead)(?:_|$)"
    r"|(?:^|_)(?:days?_until|time_to_event|target_next|next_period)(?:_|$)",
    re.IGNORECASE,
)
_GROUND_TRUTH = (
    "ground_truth",
    "actual",
    "actuals",
    "label",
    "labels",
    "outcome",
    "outcomes",
    "result",
    "results",
    "final_result",
    "verified",
    "confirmed",
    "observed",
)
_GROUND_TRUTH_SUFFIX = re.compile(
    r"_(?:label|labels|outcome|actual|truth|result|results|verified|confirmed)$",
    re.IGNORECASE,
)
_ID_NAME = re.compile(
    r"(?:^|_)(?:id|uuid|guid|hash|rowid|row_id|index|key|token|session)(?:_|$)",
    re.IGNORECASE,
)
_TARGET_ENCODING = re.compile(
    r"(?:target|label|y)_?(?:mean|avg|average|enc|encoded|encoding|rate)"
    r"|(?:mean|avg|average)_(?:target|label)|(?:^|_)woe(?:_|$)|weight_of_evidence",
    re.IGNORECASE,
)
_SUSPICIOUS = re.compile(
    r"(?:^|_)(?:leak|leaky|cheat|hack|synthetic|placeholder|dummy|temp|"
    r"to_drop|drop_me|do_not_use)(?:_|$)",
    re.IGNORECASE,
)
_TARGET_TOKENS = frozenset(
    {
        "target",
        "label",
        "outcome",
        "result",
        "results",
        "response",
        "actual",
        "truth",
        "ground",
        "churn",
        "churned",
        "default",
        "defaulted",
        "fraud",
        "converted",
        "clicked",
        "purchased",
        "approved",
        "accepted",
        "rejected",
        "decision",
        "class",
    }
)
_GENERIC_TOKENS = frozenset(
    {"is", "has", "was", "will", "the", "a", "an", "of", "for", "at", "in"}
)
_DATA_COLUMN = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*(.+)$"
)
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$")

_CORROBORATION = {
    "target_proxy_name": ("target_leakage", "categorical_target_leakage"),
    "label_derived_column": ("target_leakage", "categorical_target_leakage"),
    "target_encoding_artifact": (
        "target_encode_before_split",
        "target_leakage",
        "categorical_target_leakage",
    ),
    "future_information_name": ("temporal_leakage",),
    "shared_entity_key": ("group_leakage_risk", "group_overlap"),
    "identifier_name": ("id_like",),
    "concat_before_split": (
        "split_overlap",
        "feature_overlap",
        "near_duplicate_overlap",
    ),
}

_DEFAULT_LIMITATION = (
    "Name and control-flow heuristics can miss leakage expressed in other ways."
)


@dataclass(frozen=True, slots=True)
class CodeSource:
    """One inspected source unit.  ``cells`` is populated for notebooks."""

    label: str
    source: str
    cells: tuple[str, ...] = ()

    def chunks(self) -> tuple[tuple[str, str], ...]:
        if not self.cells:
            return ((self.label, self.source),)
        return tuple(
            (f"{self.label}[cell {index}]", cell)
            for index, cell in enumerate(self.cells, start=1)
        )


@dataclass(frozen=True, slots=True)
class SchemaInfo:
    """Metadata-only description of a dataset (never row values)."""

    columns: tuple[tuple[str, str], ...] = ()
    target: str | None = None
    group: str | None = None
    time: str | None = None
    test_columns: tuple[str, ...] = ()


def _clean(value: object, limit: int = 220) -> str:
    text = _WHITESPACE.sub(" ", _CONTROL.sub(" ", str(value))).strip()
    return text[:limit]


def _tokens(name: str) -> tuple[str, ...]:
    spaced = _CAMEL.sub("_", _clean(name, 160))
    return tuple(token.lower() for token in _WORD.findall(spaced))


def _meaningful_tokens(name: str) -> frozenset[str]:
    return frozenset(_tokens(name)) - _GENERIC_TOKENS


def _evidence(
    source: str,
    kind: str,
    location: str,
    excerpt: str,
    *,
    supports: bool = True,
) -> EvidenceSpan:
    clean_location = _clean(location, 280)
    clean_excerpt = _clean(excerpt, 320)
    digest = hashlib.sha1(
        f"{source}|{kind}|{clean_location}|{clean_excerpt}".encode("utf-8", "replace"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return EvidenceSpan(
        evidence_id=f"ev-{digest}",
        source=source,
        kind=_clean(kind, 60),
        location=clean_location,
        excerpt=clean_excerpt,
        supports=supports,
    )


class _Builder:
    """Collect hypotheses with stable ids, corroboration and bounded output."""

    def __init__(self, findings: Sequence[Finding] = ()) -> None:
        self._findings = tuple(findings)
        self._items: dict[tuple[str, str], Hypothesis] = {}

    def _corroborating(
        self, concept: str, column: str | None
    ) -> tuple[str | None, bool]:
        kinds = _CORROBORATION.get(concept, ())
        for finding in self._findings:
            if finding.kind not in kinds:
                continue
            if column is None or finding.column in {"*", column}:
                return finding.kind, finding.column == column
        return None, False

    def add(
        self,
        *,
        concept: str,
        title: str,
        explanation: str,
        confidence: float,
        evidence: Iterable[EvidenceSpan],
        location: str,
        severity_hint: str = "medium",
        column: str | None = None,
        limitations: Iterable[str] = (_DEFAULT_LIMITATION,),
    ) -> None:
        spans = tuple(dict.fromkeys(evidence))
        if not spans:
            return
        corroborates, exact_match = self._corroborating(concept, column)
        score = min(0.95, confidence + (0.05 if corroborates else 0.0))
        score = min(0.95, score + (0.05 if exact_match else 0.0))
        clean_location = _clean(location, 280)
        digest = hashlib.sha1(
            f"{concept}|{clean_location}|{spans[0].evidence_id}".encode(
                "utf-8", "replace"
            ),
            usedforsecurity=False,
        ).hexdigest()[:12]
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp-{digest}",
            concept=_clean(concept, 60),
            title=_clean(title, 150),
            explanation=_clean(explanation, 1000),
            confidence=round(score, 4),
            evidence=spans,
            location=clean_location,
            detector="semantics",
            severity_hint=severity_hint,
            confidence_basis="heuristic_weight",
            sufficiency="weak",
            column=_clean(column, 160) if column else None,
            corroborates=corroborates,
            limitations=tuple(_clean(item, 260) for item in limitations)
            or (_DEFAULT_LIMITATION,),
        )
        self._items.setdefault((concept, clean_location), hypothesis)

    def result(self, limit: int = 48) -> tuple[Hypothesis, ...]:
        ordered = sorted(
            self._items.values(),
            key=lambda item: (-item.confidence, item.location, item.hypothesis_id),
        )
        return tuple(ordered[:limit])


def _first_line(lines: Sequence[str], pattern: re.Pattern[str]) -> int | None:
    for index, line in enumerate(lines):
        if pattern.search(line):
            return index
    return None


def _code_hypotheses(builder: _Builder, label: str, source: str) -> None:
    lines = source.splitlines()
    if not lines:
        return
    split_at = _first_line(lines, re.compile(r"train_test_split\s*\("))

    for index, raw in enumerate(lines, start=1):
        line = _clean(raw, 400)
        if not line:
            continue
        location = f"{label}:{index}"

        if _INSTRUCTION.search(line):
            builder.add(
                concept="untrusted_instruction",
                title="Instruction-like text in untrusted input",
                explanation=(
                    "A comment or string in the scanned input resembles instructions to an "
                    "automated reviewer. LeakLens treats it as data, does not follow it, and "
                    "keeps deterministic findings unchanged."
                ),
                confidence=0.6,
                evidence=(
                    _evidence(
                        "code",
                        "prompt_injection",
                        location,
                        line,
                    ),
                ),
                location=location,
                severity_hint="info",
                limitations=(
                    "Detection is heuristic; the text may be ordinary documentation.",
                    "No instruction in scanned code is ever executed or obeyed.",
                ),
            )
            continue

        comment = raw.split("#", 1)[1] if "#" in raw else ""
        if comment and re.search(
            r"\b(leak(?:y|age)?|cheat|hack|shortcut|temporar(?:y|ily)|"
            r"for the demo|not for production|lookahead|tomorrow|ground truth)\b",
            comment,
            re.IGNORECASE,
        ):
            builder.add(
                concept="acknowledged_shortcut",
                title="Comment acknowledges a shortcut",
                explanation=(
                    "A comment describes a shortcut or temporary choice that may bypass a "
                    "clean train/test boundary. Comments are not proof, but they are worth "
                    "checking against the actual data flow."
                ),
                confidence=0.58,
                evidence=(_evidence("code", "comment", location, line),),
                location=location,
                severity_hint="low",
                limitations=(
                    "A comment may be stale, sarcastic, or unrelated to the current code.",
                ),
            )

        if (
            re.search(r"concat\s*\(", raw)
            and re.search(
                r"\b(test|holdout|hold_out|validation|val)\b", raw, re.IGNORECASE
            )
            and (split_at is None or index - 1 <= split_at)
        ):
            builder.add(
                concept="concat_before_split",
                title="Test data may be concatenated before splitting",
                explanation=(
                    "A test or held-out frame appears in a concatenation before the "
                    "train/test split. If those rows enter the training pool, the split "
                    "no longer estimates performance on unseen rows."
                ),
                confidence=0.72,
                evidence=(_evidence("code", "data_flow", location, line),),
                location=location,
                severity_hint="medium",
                limitations=(
                    "The concatenated object may be a temporary frame that is split again.",
                ),
            )

        match = _DATA_COLUMN.match(raw)
        if match:
            _lhs, column, rhs = match.groups()
            rhs_tokens = _meaningful_tokens(rhs)
            target_tokens = _meaningful_tokens(column)
            if rhs_tokens & _TARGET_TOKENS and not target_tokens & _TARGET_TOKENS:
                builder.add(
                    concept="target_derived_feature",
                    title="Feature is derived from the target",
                    explanation=(
                        "An assignment builds a non-target column from an expression that "
                        "references a target-like variable. If that expression is available "
                        "at prediction time it may encode the label; if it is not, it is "
                        "target leakage."
                    ),
                    confidence=0.68,
                    evidence=(_evidence("code", "data_flow", location, line),),
                    location=location,
                    severity_hint="medium",
                    column=column,
                    limitations=(
                        "The token match may refer to an unrelated variable such as a result cache.",
                    ),
                )

        if re.search(
            r"\.bfill\s*\(|fillna\s*\([^)]*method\s*=\s*['\"](?:bfill|backfill)", raw
        ):
            builder.add(
                concept="future_fill",
                title="Backward fill can import future values",
                explanation=(
                    "A backward fill takes the next observed value to fill a gap. In ordered "
                    "data that value may come from the future relative to the row being "
                    "filled, which is unavailable at prediction time."
                ),
                confidence=0.62,
                evidence=(_evidence("code", "data_flow", location, line),),
                location=location,
                severity_hint="medium",
                limitations=(
                    "Backward fill is harmless for unordered cross-sectional data.",
                ),
            )

        if re.search(r"\.rolling\s*\([^)]*center\s*=\s*True", raw):
            builder.add(
                concept="centered_window",
                title="Centered rolling window uses later rows",
                explanation=(
                    "A centered rolling window includes rows after the current one. For time "
                    "series that is future information unless the window is explicitly shifted "
                    "back by the forecast horizon."
                ),
                confidence=0.66,
                evidence=(_evidence("code", "data_flow", location, line),),
                location=location,
                severity_hint="medium",
                limitations=(
                    "Centering is safe for some offline smoothing tasks, not for prediction features.",
                ),
            )

        if re.search(r"\.shift\s*\(\s*-\s*\d", raw):
            builder.add(
                concept="negative_shift",
                title="Negative shift pulls future rows",
                explanation=(
                    "A negative shift moves later observations onto the current row. That "
                    "creates a feature that depends on the future."
                ),
                confidence=0.7,
                evidence=(_evidence("code", "data_flow", location, line),),
                location=location,
                severity_hint="medium",
                limitations=(
                    "A negative shift can be intentional when the label is also shifted by the same horizon.",
                ),
            )

        if (
            re.search(
                r"(?:datetime|pd\.Timestamp|Timestamp)\.(?:now|utcnow)\s*\(|\bdate\.today\s*\(",
                raw,
            )
            and "=" in raw
        ):
            builder.add(
                concept="runtime_clock_feature",
                title="Runtime clock value may leak collection time",
                explanation=(
                    "A wall-clock value is assigned while the pipeline runs. If it becomes a "
                    "feature, it can encode when rows were collected rather than a property "
                    "available at prediction time."
                ),
                confidence=0.5,
                evidence=(_evidence("code", "data_flow", location, line),),
                location=location,
                severity_hint="low",
                limitations=(
                    "The timestamp may be a harmless audit field that is excluded from features.",
                ),
            )

        if re.search(
            r"read_(?:csv|json|parquet|excel)\s*\([^)]*['\"][^'\"]*test",
            raw,
            re.IGNORECASE,
        ):
            builder.add(
                concept="test_file_loaded",
                title="Test file loaded in the same unit",
                explanation=(
                    "The scanned code reads a file whose name contains 'test'. Loading a test "
                    "file is not leakage by itself, but verify it is only used for evaluation "
                    "and never for fitting or feature statistics."
                ),
                confidence=0.4,
                evidence=(_evidence("code", "file_access", location, line),),
                location=location,
                severity_hint="info",
                limitations=(
                    "The filename heuristic can match unrelated files such as contest_test.csv.",
                ),
            )

        if (
            split_at is not None
            and index - 1 < split_at
            and re.search(
                r"\.(?:mean|std|median|describe|quantile|value_counts)\s*\(", raw
            )
        ):
            builder.add(
                concept="global_statistic_before_split",
                title="Global statistics computed before the split",
                explanation=(
                    "Distribution statistics are computed on the full frame before the split. "
                    "Even when the result is not fed to a model directly, statistics that use "
                    "test rows do not describe a deployment-time pipeline."
                ),
                confidence=0.5,
                evidence=(_evidence("code", "data_flow", location, line),),
                location=location,
                severity_hint="low",
                limitations=(
                    "Descriptive statistics used only for reporting do not affect model inputs.",
                ),
            )


def _schema_hypotheses(builder: _Builder, schema: SchemaInfo) -> None:
    target = _clean(schema.target, 160) if schema.target else None
    target_tokens = _meaningful_tokens(target) if target else frozenset()
    test_columns = {str(item) for item in schema.test_columns}
    seen: set[str] = set()

    for raw_name, raw_dtype in schema.columns:
        name = _clean(raw_name, 160)
        dtype = _clean(raw_dtype, 60) or "unknown"
        if not name or name in seen:
            continue
        seen.add(name)
        location = f"column:{name}"
        excerpt = f"{name} ({dtype})"
        tokens = _meaningful_tokens(name)

        if name != target and _INSTRUCTION.search(name):
            builder.add(
                concept="untrusted_instruction",
                title="Instruction-like text in a column name",
                explanation=(
                    "A column name resembles instructions to an automated reviewer. Column "
                    "names are data, not instructions: LeakLens does not follow this text and "
                    "excludes it from optional LLM prompts."
                ),
                confidence=0.62,
                evidence=(_evidence("schema", "prompt_injection", location, excerpt),),
                location=location,
                severity_hint="info",
                column=name,
                limitations=(
                    "The name may be an innocent experiment or a pasted error message.",
                ),
            )

        if name != target and target_tokens and tokens & target_tokens:
            builder.add(
                concept="target_proxy_name",
                title="Column name overlaps the target name",
                explanation=(
                    "A feature name shares meaningful tokens with the target name. Names alone "
                    "cannot prove leakage, but a column such as 'fraud_score' next to target "
                    "'is_fraud' deserves a direct check for target-derived values."
                ),
                confidence=0.56,
                evidence=(_evidence("schema", "name_overlap", location, excerpt),),
                location=location,
                severity_hint="medium",
                column=name,
                limitations=("Shared words are common in legitimate domain features.",),
            )

        lowered = name.lower()
        if name != target and (
            lowered in _GROUND_TRUTH or _GROUND_TRUTH_SUFFIX.search(lowered)
        ):
            builder.add(
                concept="label_derived_column",
                title="Column name suggests a label or outcome",
                explanation=(
                    "The column name looks like an observed outcome, ground truth or final "
                    "decision rather than an input available before prediction. Check when "
                    "the value is recorded."
                ),
                confidence=0.52,
                evidence=(_evidence("schema", "name_semantics", location, excerpt),),
                location=location,
                severity_hint="medium",
                column=name,
                limitations=(
                    "Outcome columns can be legitimate historical features if their timing is known.",
                ),
            )

        if name != target and _FUTURE_NAME.search(lowered):
            builder.add(
                concept="future_information_name",
                title="Column name refers to a future value",
                explanation=(
                    "The name contains a future or look-ahead term (for example 'next', "
                    "'tomorrow' or 'post'). Such a column may not exist at prediction time."
                ),
                confidence=0.6,
                evidence=(_evidence("schema", "name_semantics", location, excerpt),),
                location=location,
                severity_hint="medium",
                column=name,
                limitations=(
                    "Some domains use these words for historical or scheduled inputs.",
                ),
            )

        if name != target and _TARGET_ENCODING.search(lowered):
            builder.add(
                concept="target_encoding_artifact",
                title="Column name suggests target encoding",
                explanation=(
                    "The name looks like a target-mean or weight-of-evidence encoding. These "
                    "encodings must be fitted inside each training fold; a stored column can "
                    "carry held-out label information."
                ),
                confidence=0.62,
                evidence=(_evidence("schema", "name_semantics", location, excerpt),),
                location=location,
                severity_hint="medium",
                column=name,
                limitations=(
                    "A precomputed encoding can be safe when it was fitted on training data only.",
                ),
            )

        if name != target and _SUSPICIOUS.search(lowered):
            builder.add(
                concept="suspicious_column_name",
                title="Column name marks a temporary or synthetic value",
                explanation=(
                    "The column name contains a term such as leak, dummy, placeholder or "
                    "temporary. Verify that it is excluded from model features."
                ),
                confidence=0.55,
                evidence=(_evidence("schema", "name_semantics", location, excerpt),),
                location=location,
                severity_hint="low",
                column=name,
                limitations=("The word may describe a legitimate domain concept.",),
            )

        if (
            name != target
            and name not in {schema.group, schema.time}
            and _ID_NAME.search(lowered)
            and dtype not in {"bool", "boolean"}
        ):
            builder.add(
                concept="identifier_name",
                title="Column name looks like a row or entity identifier",
                explanation=(
                    "The name looks like an identifier. Identifiers can memorise rows or "
                    "entities; the deterministic scanner checks uniqueness, and a random split "
                    "can still share the same entity across train and test."
                ),
                confidence=0.42,
                evidence=(_evidence("schema", "name_semantics", location, excerpt),),
                location=location,
                severity_hint="low",
                column=name,
                limitations=(
                    "Identifier columns are safe when dropped or used only for grouping.",
                ),
            )

        if test_columns and name in test_columns and name == schema.group:
            builder.add(
                concept="shared_entity_key",
                title="Entity key appears in both splits",
                explanation=(
                    "The group/entity column is present in both the training and held-out "
                    "files. A random row split can place rows for the same entity on both "
                    "sides; split by the entity instead."
                ),
                confidence=0.55,
                evidence=(
                    _evidence(
                        "relationship",
                        "shared_key",
                        location,
                        f"{name} appears in train and held-out schemas",
                    ),
                ),
                location=location,
                severity_hint="medium",
                column=name,
                limitations=(
                    "Sharing a key is expected; only shared values in both splits cause leakage.",
                ),
            )

        if test_columns and name in test_columns and name == schema.time:
            builder.add(
                concept="shared_time_key",
                title="Timestamp appears in both splits",
                explanation=(
                    "The time column is present in both files. Check that the split is "
                    "chronological and that training timestamps do not overlap the test "
                    "period."
                ),
                confidence=0.5,
                evidence=(
                    _evidence(
                        "relationship",
                        "shared_key",
                        location,
                        f"{name} appears in train and held-out schemas",
                    ),
                ),
                location=location,
                severity_hint="info",
                column=name,
                limitations=(
                    "Overlapping schemas do not imply overlapping time periods.",
                ),
            )


def analyze_semantics(
    *,
    code: Sequence[CodeSource] = (),
    schema: SchemaInfo | None = None,
    findings: Sequence[Finding] = (),
) -> tuple[Hypothesis, ...]:
    """Return deterministic semantic hypotheses with evidence spans.

    This function performs no network access, writes nothing, and never executes
    scanned code.  Deterministic findings are used only to raise confidence when
    they corroborate the same concept.
    """

    builder = _Builder(findings)
    for source in code:
        for label, text in source.chunks():
            _code_hypotheses(builder, _clean(label, 200), text)
    if schema is not None:
        _schema_hypotheses(builder, schema)
    return builder.result()


__all__ = ["CodeSource", "SchemaInfo", "analyze_semantics"]
