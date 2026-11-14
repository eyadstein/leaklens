"""Structured intelligence for data leakage analysis."""

from .confidence import assess_cluster_confidence, assess_confidence
from .context import AnalysisContext, build_context
from .graph_baseline import (
    Completeness,
    GraphBaselineError,
    GraphDiff,
    GraphSnapshot,
    compare_graph_snapshots,
    snapshot_context,
)
from .impact import DatasetImpact, analyze_dataset_impact
from .investigation import (
    InspectEvidence,
    InvestigationPlan,
    MissingEvidence,
    NarrationSlot,
    SuggestedTest,
    build_investigation,
)
from .orchestrator import IntelligenceReport, analyze_findings
from .provenance import EvidenceGraph, anchor_for, build_evidence_graph
from .schemas import (
    ConfidenceAssessment,
    Evidence,
    EvidenceCluster,
    EvidenceLink,
    Recommendation,
    RootCause,
)

__all__ = [
    "AnalysisContext",
    "Completeness",
    "ConfidenceAssessment",
    "DatasetImpact",
    "Evidence",
    "EvidenceCluster",
    "EvidenceGraph",
    "EvidenceLink",
    "GraphBaselineError",
    "GraphDiff",
    "GraphSnapshot",
    "InspectEvidence",
    "IntelligenceReport",
    "InvestigationPlan",
    "MissingEvidence",
    "NarrationSlot",
    "Recommendation",
    "RootCause",
    "SuggestedTest",
    "analyze_dataset_impact",
    "analyze_findings",
    "anchor_for",
    "assess_cluster_confidence",
    "assess_confidence",
    "build_context",
    "build_evidence_graph",
    "build_investigation",
    "compare_graph_snapshots",
    "snapshot_context",
]
