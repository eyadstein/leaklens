"""Deterministic orchestration of leakage intelligence analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from leaklens.leakage import Finding

from .confidence import assess_confidence
from .context import AnalysisContext, build_context
from .impact import DatasetImpact
from .provenance import EvidenceGraph, build_evidence_graph
from .schemas import (
    ConfidenceAssessment,
    Evidence,
    EvidenceCluster,
    EvidenceLink,
    Recommendation,
    RootCause,
)


@dataclass(frozen=True, slots=True)
class IntelligenceReport:
    """Structured analysis derived from deterministic scanner findings."""

    evidence: tuple[Evidence, ...]
    confidence: ConfidenceAssessment
    root_causes: tuple[RootCause, ...]
    recommendations: tuple[Recommendation, ...]
    finding_count: int
    impact: DatasetImpact | None = None
    engine: str = "deterministic"
    ai_enriched: bool = False
    clusters: tuple[EvidenceCluster, ...] = ()
    links: tuple[EvidenceLink, ...] = ()
    links_truncated: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        # Completeness detail is exposed on the graph snapshot, not the report.
        data.pop("links_truncated", None)
        return data

    def to_graph(self) -> EvidenceGraph:
        """Return the inspectable graph built from this report."""
        return EvidenceGraph(
            evidence=self.evidence,
            clusters=self.clusters,
            links=self.links,
            links_truncated=self.links_truncated,
        )


_ROOT_CAUSES = {
    "preprocess_before_split": (
        "preprocessing_order",
        "Preprocessing before data splitting",
        "A transformation may have learned information from data that should have remained outside the training set.",
    ),
    "transform_before_split": (
        "preprocessing_order",
        "Transformation before data splitting",
        "A transformation may have used statistics from the complete dataset before the split.",
    ),
    "fit_on_full_data": (
        "preprocessing_order",
        "Model fitting on the complete dataset",
        "A model or transformation appears to have been fitted using data beyond the training partition.",
    ),
    "resample_before_split": (
        "sampling_order",
        "Resampling before data splitting",
        "Resampling before partitioning can allow information from related observations to cross dataset boundaries.",
    ),
    "get_dummies_before_split": (
        "preprocessing_order",
        "Encoding before data splitting",
        "Category encoding was performed before splitting, potentially exposing information about the full dataset.",
    ),
    "target_leakage": (
        "target_contamination",
        "Feature contains target-related information",
        "A feature appears unusually predictive of the target and may contain direct or indirect target information.",
    ),
    "id_like": (
        "identifier_memorization",
        "Identifier-like feature detected",
        "A highly unique feature may allow a model to memorize records instead of learning generalizable patterns.",
    ),
    "split_overlap": (
        "partition_contamination",
        "Training and test rows overlap",
        "Duplicate observations appear in both training and test partitions, potentially inflating evaluation results.",
    ),
    "temporal_leakage": (
        "temporal_order",
        "Temporal ordering violation",
        "The training period appears to overlap or extend beyond the test period.",
    ),
    "group_overlap": (
        "entity_contamination",
        "Groups overlap between partitions",
        "Related entities appear in both training and test data, which may make evaluation overly optimistic.",
    ),
    "group_leakage_risk": (
        "entity_contamination",
        "Random splitting may mix related entities",
        "Multiple observations share the same group, so a random split may place related records in separate partitions.",
    ),
    "distribution_shift": (
        "distribution_shift",
        "Feature distribution differs between train and test",
        "A feature's train and test distributions differ beyond the configured effect floor; this can be an intended temporal or population split, or a sign of mixed partitions.",
    ),
    "missingness_target_signal": (
        "informative_missingness",
        "Missingness is associated with the target",
        "Whether a value is missing tracks the label, so a missing-value indicator can encode the outcome itself.",
    ),
    "missingness_split_shift": (
        "informative_missingness",
        "Missingness differs between train and test",
        "The missing rate changes across partitions, which suggests the split is not exchangeable or a filter treats the partitions differently.",
    ),
    "missingness_group_signal": (
        "informative_missingness",
        "Missingness is associated with group membership",
        "Records missing a value cluster in some groups, so missingness may carry entity information that does not generalize.",
    ),
    "sql_target_as_feature": (
        "sql_lineage",
        "SQL feature derives from the target",
        "Confirmed column lineage links a projected expression to the target column, so the feature cannot be known before the outcome.",
    ),
    "sql_split_predicate_missing": (
        "sql_lineage",
        "SQL query does not filter the split column",
        "A source table that carries the configured split column is read without a predicate on it, so train and test rows may be mixed.",
    ),
    "sql_future_window": (
        "sql_lineage",
        "SQL window frame reads following rows",
        "The window frame ends after the current row, so the feature can use values that are not available at prediction time.",
    ),
    "sql_unresolved_column": (
        "sql_unresolved",
        "SQL lineage could not be resolved",
        "Lineage was ambiguous or the statement was not a SELECT, so the analysis stayed conservative and a target-derived feature may have been missed.",
    ),
}


_RECOMMENDATIONS = {
    "preprocess_before_split": (
        "Move preprocessing after the split",
        "Fit transformations only on training data.",
        (
            "Split the data before fitting transformations.",
            "Use sklearn.pipeline.Pipeline where appropriate.",
        ),
        (
            "Verify that preprocessing is fitted independently inside each training fold.",
        ),
    ),
    "transform_before_split": (
        "Move transformations into the training pipeline",
        "Prevent transformations from learning from validation or test observations.",
        (
            "Split before fitting transformations.",
            "Use a pipeline to fit transformations within each fold.",
        ),
        (
            "Inspect the pipeline and verify that held-out data is only transformed, never fitted.",
        ),
    ),
    "fit_on_full_data": (
        "Fit only on training data",
        "Restrict fitting to the training partition.",
        (
            "Separate training and evaluation data.",
            "Fit the estimator only on training observations.",
        ),
        (
            "Confirm that evaluation observations are never passed to fit or fit_transform.",
        ),
    ),
    "resample_before_split": (
        "Split before resampling",
        "Perform sampling only within training partitions.",
        (
            "Split the original dataset first.",
            "Use an imbalanced-learn pipeline for resampling.",
        ),
        (
            "Verify that validation and test observations are not used during resampling.",
        ),
    ),
    "get_dummies_before_split": (
        "Move encoding into the pipeline",
        "Keep learned preprocessing inside the training workflow.",
        (
            "Split the data before encoding.",
            "Consider sklearn.preprocessing.OneHotEncoder inside a ColumnTransformer.",
        ),
        ("Verify that the encoder is fitted only on training data.",),
    ),
    "target_leakage": (
        "Investigate suspicious predictive features",
        "Check whether the feature contains information derived from the target or from future events.",
        (
            "Trace the feature's data lineage.",
            "Remove features unavailable at prediction time.",
            "Re-evaluate using an independent split.",
        ),
        ("Compare evaluation results with and without the suspicious feature.",),
    ),
    "id_like": (
        "Review identifier-like features",
        "Determine whether the feature represents a record identifier rather than a meaningful predictive signal.",
        (
            "Inspect the feature's semantics.",
            "Remove identifiers when they enable memorization.",
            "Evaluate generalization on unseen entities.",
        ),
        ("Compare model performance with the identifier excluded.",),
    ),
    "split_overlap": (
        "Remove train/test duplication",
        "Ensure evaluation records are independent of training records.",
        (
            "Identify duplicated observations.",
            "Deduplicate before partitioning.",
            "Rebuild the evaluation split.",
        ),
        ("Verify that no exact duplicate rows cross the split boundary.",),
    ),
    "temporal_leakage": (
        "Respect chronological ordering",
        "Ensure training observations precede evaluation observations.",
        (
            "Sort observations by event time.",
            "Use a chronological or time-series split.",
        ),
        (
            "Verify that the latest training timestamp precedes the earliest test timestamp.",
        ),
    ),
    "group_overlap": (
        "Split by entity",
        "Keep related records in the same partition.",
        (
            "Identify the entity or group identifier.",
            "Use GroupShuffleSplit or GroupKFold.",
        ),
        (
            "Verify that no group identifier occurs in both training and test partitions.",
        ),
    ),
    "group_leakage_risk": (
        "Use group-aware validation",
        "Prevent related observations from being distributed across independent partitions.",
        (
            "Identify the correct grouping column.",
            "Use GroupKFold or GroupShuffleSplit.",
        ),
        ("Check group separation across all validation folds.",),
    ),
    "distribution_shift": (
        "Explain or remove the distribution shift",
        "Decide whether the train/test difference is intended before trusting held-out metrics.",
        (
            "Check whether the split should be random, temporal or by group.",
            "Look for a sampling or filter step applied to only one partition.",
        ),
        ("Re-run the drift diagnostics and confirm the shift is explained or gone.",),
    ),
    "missingness_target_signal": (
        "Review informative missingness",
        "Determine why the value is missing before using it as a signal.",
        (
            "Trace whether the value is missing because of the outcome or the labelling process.",
            "Drop missing-value indicators that encode the target.",
        ),
        (
            "Re-run the missingness audit and confirm the association is gone or explained.",
        ),
    ),
    "missingness_split_shift": (
        "Investigate the missingness difference across partitions",
        "Check whether the split is exchangeable or a filter treats partitions differently.",
        (
            "Confirm the split is random or chronological as intended.",
            "Look for a filter or join that populates one partition only.",
        ),
        ("Compare missing rates per partition after rebuilding the split.",),
    ),
    "missingness_group_signal": (
        "Check missingness per entity",
        "Determine whether group membership explains why a value is recorded.",
        (
            "Identify the grouping column and inspect missing rates per group.",
            "Avoid keeping a missing-value indicator when the group is already a feature.",
        ),
        ("Verify that missing rates no longer differ materially between groups.",),
    ),
    "sql_target_as_feature": (
        "Remove target-derived SQL features",
        "Keep every projected feature independent of the label.",
        (
            "Trace the cited projection's lineage to its source column.",
            "Drop the expression or replace it with a pre-outcome source.",
        ),
        ("Re-run the SQL scan and confirm no projection has target lineage.",),
    ),
    "sql_split_predicate_missing": (
        "Filter the split column in SQL",
        "Keep train and test rows separated at the query level.",
        (
            "Add a predicate on the split column to every source table.",
            "Or join the split label explicitly and filter per partition.",
        ),
        (
            "Confirm the query returns training rows only when building training features.",
        ),
    ),
    "sql_future_window": (
        "Bound SQL windows to the current row",
        "Stop features from reading rows that are unavailable at prediction time.",
        (
            "End the frame at CURRENT ROW.",
            "Shift look-ahead windows back by the forecast horizon.",
        ),
        ("Re-run the SQL scan and confirm no frame ends after the current row.",),
    ),
    "sql_unresolved_column": (
        "Resolve the remaining SQL lineage",
        "Make every projection traceable before relying on the analysis.",
        (
            "Provide the table schema so projections resolve to source columns.",
            "Give each projected expression a stable output name.",
        ),
        ("Re-run the SQL scan and confirm unresolved projections are zero.",),
    ),
}


def _recommendation_for(kind: str, severity: str) -> Recommendation:
    template = _RECOMMENDATIONS.get(kind)

    if template is None:
        title = "Investigate the detected leakage risk"
        description = (
            "Review the finding and trace the affected data through the ML pipeline."
        )
        actions = (
            "Inspect the affected code or dataset.",
            "Determine whether information crosses the intended evaluation boundary.",
        )
        validation = (
            "Re-run the leakage scanner after remediation.",
            "Re-evaluate the model using an independent validation strategy.",
        )
    else:
        title, description, actions, validation = template

    priority = (
        severity if severity in {"critical", "high", "medium", "low"} else "medium"
    )

    return Recommendation(
        title=title,
        description=description,
        priority=priority,
        actions=actions,
        validation_steps=validation,
        related_finding_kinds=(kind,),
    )


def analyze_findings(
    findings: Iterable[Finding], *, impact: DatasetImpact | None = None
) -> IntelligenceReport:
    """Produce deterministic intelligence, optionally including measured dataset impact."""

    findings = tuple(findings)
    context: AnalysisContext = build_context(findings)
    confidence = assess_confidence(context.primary_evidence)
    graph = build_evidence_graph(findings, impact=impact, context=context)

    root_causes = []
    recommendations = []

    for kind in context.finding_kinds:
        related = context.evidence_for(kind)
        cause = _ROOT_CAUSES.get(kind)

        if cause is None:
            category = "unclassified_leakage"
            title = f"Unclassified risk: {kind}"
            explanation = (
                "The scanner detected this condition, but the intelligence engine "
                "does not yet have a specialized root-cause template for it."
            )
        else:
            category, title, explanation = cause

        root_causes.append(
            RootCause(
                category=category,
                title=title,
                explanation=explanation,
                related_finding_kinds=(kind,),
                confidence=assess_confidence(related),
            )
        )

        severity = max(
            (item.metadata for item in related),
            key=lambda metadata: {
                "critical": 4,
                "high": 3,
                "medium": 2,
                "low": 1,
            }.get(dict(metadata).get("severity", ""), 0),
            default=(),
        )
        recommendations.append(
            _recommendation_for(kind, dict(severity).get("severity", "medium"))
        )

    return IntelligenceReport(
        evidence=context.evidence,
        confidence=confidence,
        root_causes=tuple(root_causes),
        recommendations=tuple(recommendations),
        finding_count=context.finding_count,
        impact=impact,
        clusters=graph.clusters,
        links=graph.links,
        links_truncated=graph.links_truncated,
    )
