"""Experimental, optional AI intelligence for LeakLens.

Deterministic findings remain authoritative.  Nothing in this package changes
the trust score, severity or the scanner's output.  AI hypotheses are separate,
evidence-bound and clearly marked as unverified.
"""

from .analysis import (
    DatasetInput,
    analyze,
    analyze_code,
    analyze_dataset,
    collect_code_sources,
)
from .corpus import CorpusCase, corpus_cases, corpus_stats
from .learned import (
    LearnedModel,
    ModelEvaluation,
    evaluate_model,
    get_model,
    learned_hypotheses,
    reset_model_cache,
)
from .reasoning import CompletionProvider, build_plan, enrich_plan, note_for
from .schemas import (
    ENGINE_NAME,
    GLOBAL_LIMITATIONS,
    SCHEMA_VERSION,
    AIReport,
    EvidenceSpan,
    Hypothesis,
    LLMEnrichment,
    LLMNote,
    ModelCard,
    ReasoningPlan,
    Remediation,
    RootCauseGroup,
)
from .semantics import CodeSource, SchemaInfo, analyze_semantics

__all__ = [
    "ENGINE_NAME",
    "GLOBAL_LIMITATIONS",
    "SCHEMA_VERSION",
    "AIReport",
    "CodeSource",
    "CompletionProvider",
    "CorpusCase",
    "DatasetInput",
    "EvidenceSpan",
    "Hypothesis",
    "LLMEnrichment",
    "LLMNote",
    "LearnedModel",
    "ModelCard",
    "ModelEvaluation",
    "ReasoningPlan",
    "Remediation",
    "RootCauseGroup",
    "SchemaInfo",
    "analyze",
    "analyze_code",
    "analyze_dataset",
    "analyze_semantics",
    "build_plan",
    "collect_code_sources",
    "corpus_cases",
    "corpus_stats",
    "enrich_plan",
    "evaluate_model",
    "get_model",
    "learned_hypotheses",
    "note_for",
    "reset_model_cache",
]
