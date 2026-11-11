"""Detector contract manifest and generated documentation.

Every finding kind LeakLens can emit is registered here with its surface,
default severity, advice key, benchmark coverage (or an explicit reason),
documentation anchor and the release that introduced it. The contract test in
``tests/_meta/test_detector_contract.py`` scans the production emitters and
fails when a kind is missing or an entry is stale.

Regenerate ``docs/DETECTORS.md`` with ``python -m leaklens.manifest`` and
verify it with ``python -m leaklens.manifest --check``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from .fixes import FIXES

SURFACES = frozenset({"dataset", "split", "code", "notebook", "image"})
MODALITIES = frozenset({"tabular", "code", "sql", "image", "text", "artifact"})
SEVERITIES = frozenset({"high", "medium", "low"})
EXTRAS = frozenset({"images", "sql", "yaml"})
INTRODUCED = "0.13.0"

# Historical diagnostic kinds score nothing; every other shipped kind is scored.
# New kinds must pass scored=False explicitly until a labelled benchmark graduates
# them (ADR-006).
_DIAGNOSTIC_KINDS = frozenset(
    {
        "distribution_shift",
        "sql_unresolved_column",
        "pipeline_split_overlap",
        "pipeline_unresolved_lineage",
        "pipeline_param_drift",
        "pipeline_missing_digest",
        "pipeline_unpinned_dependency",
    }
)

_SURFACE_MODALITIES = {
    "dataset": ("tabular",),
    "split": ("tabular",),
    "code": ("code",),
    "notebook": ("code",),
    "image": ("image",),
}


@dataclass(frozen=True, slots=True)
class Detector:
    kind: str
    surface: str
    severity: str
    advice: str
    benchmark: str | None
    no_bench_reason: str | None
    docs_anchor: str
    introduced: str
    scored: bool = False
    requires: str | None = None
    modalities: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "surface": self.surface,
            "severity": self.severity,
            "advice": self.advice,
            "benchmark": self.benchmark,
            "no_bench_reason": self.no_bench_reason,
            "docs_anchor": self.docs_anchor,
            "introduced": self.introduced,
            "scored": self.scored,
            "requires": self.requires,
            "modalities": sorted(self.modalities),
        }


def _entry(
    kind: str,
    surface: str,
    severity: str,
    advice: str,
    benchmark: str | None = None,
    no_bench_reason: str | None = None,
    *,
    scored: bool | None = None,
    requires: str | None = None,
    modalities: tuple[str, ...] | None = None,
) -> Detector:
    return Detector(
        kind=kind,
        surface=surface,
        severity=severity,
        advice=advice,
        benchmark=benchmark,
        no_bench_reason=no_bench_reason,
        docs_anchor=f"#{kind}",
        introduced=INTRODUCED,
        scored=kind not in _DIAGNOSTIC_KINDS if scored is None else scored,
        requires=requires,
        modalities=frozenset(modalities or _SURFACE_MODALITIES[surface]),
    )


DETECTORS: tuple[Detector, ...] = (
    # Dataset detectors.
    _entry(
        "target_leakage",
        "dataset",
        "high",
        "target_leakage",
        "target copied into a feature",
    ),
    _entry("id_like", "dataset", "medium", "id_like", "identifier column"),
    _entry(
        "categorical_target_leakage",
        "dataset",
        "high",
        "categorical_target_leakage",
        "high-cardinality categorical target encoding",
    ),
    _entry(
        "group_missing",
        "dataset",
        "low",
        "group_missing",
        no_bench_reason=(
            "Only triggered for a missing --group column; covered by CLI/API option "
            "tests, not the numeric data benchmark."
        ),
    ),
    _entry(
        "group_leakage_risk",
        "dataset",
        "medium",
        "group_leakage_risk",
        "entity repeated across rows",
    ),
    _entry(
        "missingness_target_signal",
        "dataset",
        "medium",
        "missingness_target_signal",
        "binary target association",
    ),
    _entry(
        "missingness_group_signal",
        "dataset",
        "low",
        "missingness_group_signal",
        "group-associated missingness",
    ),
    # Split and duplicate detectors.
    _entry(
        "split_overlap", "split", "high", "split_overlap", "exact rows in both splits"
    ),
    _entry(
        "temporal_leakage",
        "split",
        "high",
        "temporal_leakage",
        "future rows before earlier rows",
    ),
    _entry(
        "missingness_split_shift",
        "split",
        "medium",
        "missingness_split_shift",
        "train and test missingness differ",
    ),
    _entry(
        "distribution_shift",
        "split",
        "low",
        "distribution_shift",
        "numeric shift",
    ),
    _entry("group_overlap", "split", "high", "group_overlap", "entity in both splits"),
    _entry(
        "feature_overlap",
        "split",
        "medium",
        "feature_overlap",
        "identical feature rows with conflicting labels",
    ),
    _entry(
        "near_duplicate_overlap",
        "split",
        "medium",
        "near_duplicate_overlap",
        "case/whitespace near duplicates",
    ),
    # Image and notebook surfaces.
    _entry(
        "image_overlap",
        "image",
        "high",
        "image_overlap",
        requires="images",
        no_bench_reason=(
            "Requires Pillow and image fixtures; covered by tests/test_images.py, not "
            "the numeric data benchmark."
        ),
    ),
    _entry(
        "parse_error",
        "code",
        "low",
        "parse_error",
        no_bench_reason=(
            "Syntax errors are an input error rather than a leakage class; covered by "
            "scanner and notebook tests."
        ),
    ),
    _entry(
        "sql_target_as_feature",
        "code",
        "high",
        "sql_target_as_feature",
        "target expression in projection",
        requires="sql",
        modalities=("sql",),
    ),
    _entry(
        "sql_split_predicate_missing",
        "code",
        "low",
        "sql_split_predicate_missing",
        "split predicate missing",
        requires="sql",
        modalities=("sql",),
    ),
    _entry(
        "sql_unresolved_column",
        "code",
        "low",
        "sql_unresolved_column",
        "ambiguous join is a diagnostic",
        requires="sql",
        modalities=("sql",),
    ),
    _entry(
        "sql_future_window",
        "code",
        "medium",
        "sql_future_window",
        "window reads following rows",
        requires="sql",
        modalities=("sql",),
    ),
    _entry(
        "notebook_out_of_order_execution",
        "notebook",
        "low",
        "notebook_out_of_order_execution",
        no_bench_reason=(
            "Requires an executed .ipynb; covered by notebook tests, not the "
            "source-string benchmark cases."
        ),
    ),
    # Code scanner: ordering and preprocessing.
    _entry(
        "tuning_on_test_data",
        "code",
        "high",
        "tuning_on_test_data",
        "GridSearchCV fitted on test data",
    ),
    _entry(
        "cv_on_test_data",
        "code",
        "medium",
        "cv_on_test_data",
        "cross-validation on the test partition",
    ),
    _entry(
        "tuning_before_split",
        "code",
        "medium",
        "tuning_before_split",
        "search fitted on full data before the split",
    ),
    _entry(
        "tuning_on_full_data",
        "code",
        "medium",
        "tuning_on_full_data",
        no_bench_reason=(
            "No existing benchmark case; adding one would change the frozen benchmark "
            "metrics, so the gap is recorded here instead."
        ),
    ),
    _entry(
        "group_cv_missing_groups",
        "code",
        "medium",
        "group_cv_missing_groups",
        "group CV without groups",
    ),
    _entry(
        "test_set_reuse",
        "code",
        "low",
        "test_set_reuse",
        "repeated scoring on the test partition",
    ),
    _entry(
        "preprocess_outside_cv",
        "code",
        "high",
        "preprocess_outside_cv",
        "scaler fitted on all data then cross_val_score",
    ),
    _entry(
        "resample_outside_cv",
        "code",
        "high",
        "resample_outside_cv",
        "SMOTE on all data then cross_val_score",
    ),
    _entry(
        "fit_on_eval_data",
        "code",
        "high",
        "fit_on_eval_data",
        "scaler fitted on test data",
    ),
    _entry(
        "resample_before_split",
        "code",
        "high",
        "resample_before_split",
        "SMOTE fitted before split",
    ),
    _entry(
        "encoder_before_split",
        "code",
        "low",
        "encoder_before_split",
        "one-hot before split (benign)",
    ),
    _entry(
        "target_encode_before_split",
        "code",
        "high",
        "target_encode_before_split",
        "TargetEncoder fitted before split",
    ),
    _entry(
        "feature_selection_before_split",
        "code",
        "high",
        "feature_selection_before_split",
        "RFE fitted before split",
    ),
    _entry(
        "transformer_before_split",
        "code",
        "medium",
        "transformer_before_split",
        no_bench_reason=(
            "Needs a ColumnTransformer context; no benchmark case covers the ambiguous "
            "branch without changing the frozen metrics."
        ),
    ),
    _entry(
        "preprocess_before_split",
        "code",
        "high",
        "preprocess_before_split",
        "StandardScaler fitted before split",
    ),
    _entry(
        "fit_on_full_data",
        "code",
        "medium",
        "fit_on_full_data",
        "scaler fitted on full X after split",
    ),
    _entry(
        "future_target_feature_before_split",
        "code",
        "high",
        "future_target_feature_before_split",
        "negative target shift before split",
    ),
    _entry(
        "get_dummies_before_split",
        "code",
        "low",
        "get_dummies_before_split",
        "get_dummies before split",
    ),
    _entry(
        "fillna_before_split",
        "code",
        "medium",
        "fillna_before_split",
        "fillna with full-data mean before split",
    ),
    _entry(
        "manual_scale_before_split",
        "code",
        "high",
        "manual_scale_before_split",
        "z-score by hand before split",
    ),
    _entry(
        "group_stat_before_split",
        "code",
        "medium",
        "group_stat_before_split",
        "groupby agg into a column before split",
    ),
    _entry(
        "frequency_encode_before_split",
        "code",
        "low",
        "frequency_encode_before_split",
        "frequency encoding from all rows before split",
    ),
    _entry(
        "temporal_split_shuffle",
        "code",
        "medium",
        "temporal_split_shuffle",
        "time-sorted frame split with default shuffle",
    ),
    _entry(
        "no_random_state",
        "code",
        "low",
        "no_random_state",
        "train_test_split without random_state",
    ),
    _entry(
        "no_stratify",
        "code",
        "low",
        "no_stratify",
        "train_test_split without stratify",
    ),
    # Phase 14 pipeline and MLOps intelligence. Every pipeline kind is a
    # diagnostic: it ships scored=False and never changes the trust score.
    _entry(
        "pipeline_split_overlap",
        "code",
        "high",
        "pipeline_split_overlap",
        "train and eval share one dataset",
    ),
    _entry(
        "pipeline_unresolved_lineage",
        "code",
        "low",
        "pipeline_unresolved_lineage",
        "unresolved stage dependency",
    ),
    _entry(
        "pipeline_param_drift",
        "code",
        "low",
        "pipeline_param_drift",
        "locked parameter drift",
    ),
    _entry(
        "pipeline_missing_digest",
        "code",
        "low",
        "pipeline_missing_digest",
        "locked output without digest",
    ),
    _entry(
        "pipeline_unpinned_dependency",
        "code",
        "low",
        "pipeline_unpinned_dependency",
        "unpinned requirements entry",
    ),
)

BY_KIND: dict[str, Detector] = {entry.kind: entry for entry in DETECTORS}


@dataclass(frozen=True, slots=True)
class Engine:
    """One shipped engine id and the finding kinds it can emit (ADR-005)."""

    id: str
    surface: str
    requires: str | None
    kinds: frozenset[str]


def _engine(
    engine_id: str, surface: str, surfaces: frozenset[str], requires: str | None = None
) -> Engine:
    return Engine(
        engine_id,
        surface,
        requires,
        frozenset(entry.kind for entry in DETECTORS if entry.surface in surfaces),
    )


# Frozen public ids: literals, never derived from __module__ (ADR-005).
ENGINES: dict[str, Engine] = {
    "leaklens.python": _engine(
        "leaklens.python", "dataset", frozenset({"dataset", "split"})
    ),
    "leaklens.scanner.scan_source": _engine(
        "leaklens.scanner.scan_source", "code", frozenset({"code", "notebook"})
    ),
    "leaklens.ai.experimental": Engine(
        "leaklens.ai.experimental", "dataset", None, frozenset()
    ),
    "leaklens.intelligence.deterministic": Engine(
        "leaklens.intelligence.deterministic", "dataset", None, frozenset()
    ),
    "leaklens.pipeline": Engine(
        "leaklens.pipeline",
        "code",
        None,
        frozenset(
            {
                "pipeline_split_overlap",
                "pipeline_unresolved_lineage",
                "pipeline_param_drift",
                "pipeline_missing_digest",
                "pipeline_unpinned_dependency",
                "no_random_state",
            }
        ),
    ),
}


def require_kind(kind: str) -> Detector:
    """Return the manifest entry, or raise ``KeyError`` for an unknown kind."""
    return BY_KIND[kind]


def surface_for(kind: str) -> str:
    entry = BY_KIND.get(kind)
    return entry.surface if entry is not None else "unknown"


def validate_kinds(kinds: set[str] | frozenset[str]) -> None:
    """Raise when any emitted kind is not registered."""
    unknown = sorted(set(kinds) - set(BY_KIND))
    if unknown:
        raise ValueError(f"unregistered finding kinds: {unknown}")


def validate_manifest() -> None:
    """Structurally validate every entry; raises ``ValueError`` on failure."""
    seen: set[str] = set()
    for entry in DETECTORS:
        if entry.kind in seen:
            raise ValueError(f"duplicate manifest entry: {entry.kind}")
        seen.add(entry.kind)
        if entry.surface not in SURFACES:
            raise ValueError(f"{entry.kind}: invalid surface {entry.surface!r}")
        if entry.severity not in SEVERITIES:
            raise ValueError(f"{entry.kind}: invalid severity {entry.severity!r}")
        if entry.advice not in FIXES:
            raise ValueError(
                f"{entry.kind}: advice key {entry.advice!r} is not in FIXES"
            )
        if bool(entry.benchmark) == bool(entry.no_bench_reason):
            raise ValueError(
                f"{entry.kind}: needs exactly one of benchmark or no_bench_reason"
            )
        if entry.docs_anchor != f"#{entry.kind}":
            raise ValueError(f"{entry.kind}: unexpected documentation anchor")
        if not entry.introduced:
            raise ValueError(f"{entry.kind}: missing introduced version")
        if type(entry.scored) is not bool:
            raise ValueError(f"{entry.kind}: scored must be a bool")
        if entry.kind in _DIAGNOSTIC_KINDS and entry.scored:
            raise ValueError(f"{entry.kind}: diagnostic kinds are not scored")
        if not entry.modalities:
            raise ValueError(f"{entry.kind}: needs at least one modality")
        if not entry.modalities <= MODALITIES:
            raise ValueError(
                f"{entry.kind}: unknown modalities {sorted(entry.modalities - MODALITIES)}"
            )
        if entry.requires is not None and entry.requires not in EXTRAS:
            raise ValueError(f"{entry.kind}: unknown extra {entry.requires!r}")
    for engine_id, engine in ENGINES.items():
        if not engine_id or engine.id != engine_id:
            raise ValueError(f"engine registry key/id mismatch: {engine_id!r}")
        unknown = engine.kinds - set(BY_KIND)
        if unknown:
            raise ValueError(f"{engine_id}: unregistered kinds {sorted(unknown)}")
        if engine.requires is not None and engine.requires not in EXTRAS:
            raise ValueError(f"{engine_id}: unknown extra {engine.requires!r}")


def render_docs() -> str:
    """Render docs/DETECTORS.md from the manifest."""
    lines = [
        "# Detector contract",
        "",
        "<!-- Generated by `python -m leaklens.manifest`; do not edit by hand. -->",
        "",
        f"LeakLens emits {len(DETECTORS)} deterministic finding kinds. Each entry "
        "records the detector surface, default severity, advice key, benchmark "
        "coverage (or why no benchmark case exists), documentation anchor, the "
        "release in which the kind was first tracked, whether it participates in the "
        "trust score, the optional extra it needs and the data modalities it reads. "
        "The contract test proves the manifest matches the emitters.",
        "",
        "| Kind | Surface | Default severity | Scored | Requires | Modalities | Advice key | Benchmark coverage | Introduced |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in DETECTORS:
        if entry.benchmark:
            coverage = f"`{entry.benchmark}`"
        else:
            coverage = f"no benchmark case - {entry.no_bench_reason}"
        lines.append(
            f"| [`{entry.kind}`](#{entry.kind}) | {entry.surface} | "
            f"{entry.severity} | {'yes' if entry.scored else 'no'} | "
            f"{entry.requires or '-'} | {', '.join(sorted(entry.modalities))} | "
            f"`{entry.advice}` | {coverage} | {entry.introduced} |"
        )
    lines += ["", "## Fingerprint canonical form", ""]
    lines += [
        "Finding fingerprints are `SHA-256` over a domain-separated, versioned "
        "canonical form (`schema=1`) containing the detector kind, surface, "
        "repository-relative POSIX path, affected feature and an optional "
        "detector-supplied stable key.",
        "",
        "Line numbers, notebook cells, explanatory detail, timestamps and observed "
        "counts are excluded, so moving a finding to another line or rewording its "
        "message keeps the same identity. Moving the file, changing the detector "
        "kind, changing the affected feature or changing the stable key changes the "
        "fingerprint. Dataset and split columns are always treated as features, even "
        "when they contain `/` or `:1`. The bundled code scanners derive the stable "
        "key from the offending construct's structural AST shape and enclosing "
        "function/class (with a deterministic suffix for identical repeats), so "
        "repeated findings of one kind in a file stay distinct and the key contains "
        "no line number, message text or machine path.",
        "",
        "## Finding kinds",
        "",
    ]
    for entry in DETECTORS:
        requires = f"`{entry.requires}`" if entry.requires else "nothing"
        lines += [
            f"### {entry.kind}",
            "",
            f"- Surface: `{entry.surface}`",
            f"- Default severity: `{entry.severity}`",
            f"- Scored: {'yes' if entry.scored else 'no'}",
            f"- Requires: {requires}",
            f"- Modalities: {', '.join(sorted(entry.modalities))}",
            f"- Advice key: `{entry.advice}`",
        ]
        if entry.benchmark:
            lines.append(f"- Benchmark case: `{entry.benchmark}`")
        else:
            lines.append(f"- Benchmark coverage: no case - {entry.no_bench_reason}")
        lines += [
            f"- Introduced: {entry.introduced}",
            f"- Anchor: `{entry.docs_anchor}`",
            "",
            FIXES[entry.advice],
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    target = Path(__file__).resolve().parents[1] / "docs" / "DETECTORS.md"
    content = render_docs()
    if "--check" in args:
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if existing != content:
            print(f"{target} is stale; run python -m leaklens.manifest")
            return 1
        print(f"{target} matches the manifest")
        return 0
    target.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
