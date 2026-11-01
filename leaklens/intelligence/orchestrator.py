"""Deterministic orchestration of leakage intelligence analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from leaklens.leakage import Finding

from .confidence import assess_confidence
from .context import AnalysisContext, build_context
from .impact import DatasetImpact
from .schemas import ConfidenceAssessment, Evidence, Recommendation, RootCause


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

    def to_dict(self) -> dict:
        return asdict(self)


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

    context: AnalysisContext = build_context(findings)
    confidence = assess_confidence(context.evidence)

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
    )
