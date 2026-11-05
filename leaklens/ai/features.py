"""Small, deterministic feature extraction for the optional learned detector.

The detector is a CPU-only logistic regression over interpretable control-flow
indicators plus hashed identifier tokens.  It never loads a pickle, never calls
the network, and every feature can be named in the report.
"""

from __future__ import annotations

import math
import re

import numpy as np

_INDICATOR_NAMES = (
    "log_lines",
    "log_bytes",
    "has_split",
    "split_count",
    "log_split_position",
    "fit_before_split",
    "fit_transform_before_split",
    "fit_without_split",
    "has_fit",
    "has_fit_transform",
    "has_fit_resample",
    "scaler",
    "imputer",
    "pca",
    "encoder",
    "target_encoder",
    "selector",
    "resampler",
    "vectorizer",
    "pipeline",
    "column_transformer",
    "cross_validation",
    "grid_search",
    "random_search",
    "group_kfold",
    "stratified_kfold",
    "plain_kfold",
    "time_series_split",
    "group_shuffle_split",
    "has_shuffle",
    "has_random_state",
    "sort_values",
    "time_words",
    "shuffle_false",
    "bfill",
    "centered_window",
    "negative_shift",
    "clock_now",
    "concat",
    "test_concat",
    "get_dummies",
    "drop_duplicates",
    "groupby_transform",
    "describe",
    "target_reference",
    "target_in_assignment",
    "read_test",
    "leak_comment",
    "identifier_literal",
    "class_weight",
    "train_only_marker",
    "holdout_eval",
    "unsafe_preprocessing",
    "unsafe_resampling",
    "unsafe_feature_selection",
    "unsafe_target_encoding",
    "test_rows_before_split",
    "temporal_order_risk",
    "group_boundary_ignored",
    "identifier_feature_risk",
    "target_feature_risk",
    "future_feature_risk",
    "train_only_fit",
    "pipeline_boundary",
)

FEATURE_NAMES = _INDICATOR_NAMES

# Size and style signals are deliberately excluded from the learned model:
# they can be shortcuts that do not transfer to new code bases.
EXCLUDED_MODEL_FEATURES = frozenset({"log_lines", "log_bytes", "has_random_state"})
MODEL_FEATURE_NAMES = tuple(
    name for name in FEATURE_NAMES if name not in EXCLUDED_MODEL_FEATURES
)
MODEL_FEATURE_INDICES = tuple(
    index
    for index, name in enumerate(FEATURE_NAMES)
    if name not in EXCLUDED_MODEL_FEATURES
)

_SPLIT = re.compile(r"train_test_split\s*\(")
_FIT = re.compile(r"\.\s*fit\s*\(")
_FIT_TRANSFORM = re.compile(r"\.\s*fit_transform\s*\(")
_FIT_RESAMPLE = re.compile(r"\.\s*fit_resample\s*\(")
_TARGET = re.compile(r"\b(?:y|y_train|label|labels|target|outcome|is_[a-z_]+)\b")
_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\[[^\]]+\])?\s*=\s*(.+)$")
_DATA_COLUMN = re.compile(
    r"^\s*[A-Za-z_]\w*\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*(.+)$"
)


def _has(pattern: str, text: str, flags: int = re.IGNORECASE) -> float:
    return 1.0 if re.search(pattern, text, flags) else 0.0


def _first_line(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    for index, line in enumerate(lines):
        if pattern.search(line):
            return index
    return None


def _fit_before_split(lines: list[str], pattern: re.Pattern[str]) -> float:
    split_at = _first_line(lines, _SPLIT)
    fit_at = _first_line(lines, pattern)
    if fit_at is None:
        return 0.0
    if split_at is None:
        return 1.0
    return 1.0 if fit_at < split_at else 0.0


def extract_features(source: str) -> np.ndarray:
    """Return a fixed-length numeric vector for one Python source unit."""

    text = source or ""
    lines = text.splitlines()
    split_at = _first_line(lines, _SPLIT)
    fit_at = _first_line(lines, _FIT)
    fit_transform_at = _first_line(lines, _FIT_TRANSFORM)
    resample_at = _first_line(lines, _FIT_RESAMPLE)

    def before_split(index: int | None) -> bool:
        if index is None:
            return False
        return split_at is None or index < split_at

    transformer_present = bool(
        re.search(
            r"\b(?:StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|"
            r"QuantileTransformer|PowerTransformer|SimpleImputer|IterativeImputer|"
            r"KNNImputer|PCA|TruncatedSVD|OneHotEncoder|OrdinalEncoder|LabelEncoder|"
            r"SelectKBest|SelectPercentile|SelectFromModel|RFE|RFECV|"
            r"SequentialFeatureSelector|TfidfVectorizer|CountVectorizer)\b",
            text,
        )
    )
    resampler_present = bool(
        re.search(r"\b(?:SMOTE|ADASYN|RandomOverSampler|RandomUnderSampler)\b", text)
    )
    selector_present = bool(
        re.search(
            r"\b(?:SelectKBest|SelectPercentile|SelectFromModel|RFE|RFECV|"
            r"SequentialFeatureSelector)\b",
            text,
        )
    )
    target_encoder_present = bool(
        re.search(
            r"\b(?:TargetEncoder|WOEEncoder|CatBoostEncoder)\b|target.?encod",
            text,
            re.IGNORECASE,
        )
    )
    time_words_present = bool(
        re.search(r"\b(?:date|time|timestamp|datetime|month|year|day)\b", text)
    )
    group_words_present = bool(
        re.search(r"\b(?:customer|account|patient|user|store|device)_?id\b", text)
    )
    identifier_literal = bool(
        re.search(
            r"['\"](?:customer_id|user_id|row_id|uuid|guid|account_id|patient_id|row_hash)['\"]",
            text,
        )
    )
    group_aware = bool(
        re.search(r"\b(?:GroupKFold|GroupShuffleSplit|StratifiedGroupKFold)\b", text)
    )
    unsafe_preprocessing = transformer_present and (
        before_split(fit_at) or before_split(fit_transform_at)
    )
    unsafe_resampling = resampler_present and before_split(resample_at)
    unsafe_selection = selector_present and (
        before_split(fit_at) or before_split(fit_transform_at)
    )
    unsafe_target_encoding = target_encoder_present and (
        split_at is None
        or before_split(fit_at)
        or before_split(fit_transform_at)
        or bool(
            re.search(
                r"(?:groupby\s*\([^)]*\)\s*\[[^\]]*(?:target|label)[^\]]*\].*?"
                r"|\[['\"](?:target|label)['\"]\]\.mean)",
                text,
                re.IGNORECASE,
            )
        )
    )
    test_rows_before_split = bool(
        re.search(r"\.concat\s*\([^)]*(?:test|holdout|validation|val)", text)
    ) and (split_at is None or before_split(_first_line(lines, re.compile(r"concat"))))
    temporal_order_risk = (
        time_words_present
        and bool(
            re.search(r"shuffle\s*=\s*True", text)
            or re.search(r"(?<!Group)(?<!Stratified)\bKFold\s*\(", text)
        )
        and not re.search(r"\bTimeSeriesSplit\b", text)
    )
    group_boundary_ignored = (
        group_words_present
        and not group_aware
        and bool(split_at is not None or re.search(r"\bKFold\s*\(", text))
    )
    identifier_feature_risk = (
        identifier_literal
        and not group_aware
        and not re.search(
            r"drop\s*\([^)]*(?:customer_id|user_id|row_id|uuid|guid|account_id|patient_id)",
            text,
        )
    )
    target_feature_risk = False
    for line in lines:
        column_match = _DATA_COLUMN.match(line)
        if column_match:
            column, rhs = column_match.groups()
            if _TARGET.search(rhs) and not _TARGET.search(column):
                target_feature_risk = True
                break
        assign_match = _ASSIGN.match(line)
        if assign_match:
            lhs, rhs = assign_match.groups()
            if (
                _TARGET.search(rhs)
                and not _TARGET.search(lhs)
                and not re.search(r"drop\s*\([^)]*(?:target|label)", line)
            ):
                target_feature_risk = True
                break
    future_feature_risk = bool(
        re.search(r"\.bfill\s*\(|method\s*=\s*['\"](?:bfill|backfill)", text)
        or re.search(r"\.rolling\s*\([^)]*center\s*=\s*True", text)
        or re.search(r"\.shift\s*\(\s*-\s*\d", text)
    )
    train_only_fit = split_at is not None and fit_at is not None and fit_at > split_at
    pipeline_boundary = bool(
        re.search(r"\bPipeline\s*\(|make_pipeline\s*\(", text)
    ) and bool(split_at is not None and fit_at is not None and fit_at > split_at)

    split_count = len(_SPLIT.findall(text))
    values = [
        math.log1p(len(lines)),
        math.log1p(len(text.encode("utf-8", "replace"))),
        1.0 if split_at is not None else 0.0,
        min(split_count, 8.0),
        math.log1p(split_at) / 5.0 if split_at is not None else 0.0,
        _fit_before_split(lines, _FIT),
        _fit_before_split(lines, _FIT_TRANSFORM),
        1.0 if split_at is None and _FIT.search(text) else 0.0,
        1.0 if _FIT.search(text) else 0.0,
        1.0 if _FIT_TRANSFORM.search(text) else 0.0,
        1.0 if _FIT_RESAMPLE.search(text) else 0.0,
        _has(
            r"\b(?:StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|QuantileTransformer|PowerTransformer)\b",
            text,
        ),
        _has(r"\b(?:SimpleImputer|IterativeImputer|KNNImputer)\b", text),
        _has(r"\b(?:PCA|TruncatedSVD)\b", text),
        _has(r"\b(?:OneHotEncoder|OrdinalEncoder|LabelEncoder)\b", text),
        _has(r"\b(?:TargetEncoder|WOEEncoder|CatBoostEncoder)\b|target.?encod", text),
        _has(
            r"\b(?:SelectKBest|SelectPercentile|SelectFromModel|RFE|RFECV|SequentialFeatureSelector)\b",
            text,
        ),
        _has(r"\b(?:SMOTE|ADASYN|RandomOverSampler|RandomUnderSampler)\b", text),
        _has(r"\b(?:TfidfVectorizer|CountVectorizer|HashingVectorizer)\b", text),
        _has(r"\bPipeline\s*\(", text),
        _has(r"\bColumnTransformer\s*\(", text),
        _has(r"\b(?:cross_val_score|cross_validate|cross_val_predict)\s*\(", text),
        _has(r"\b(?:GridSearchCV|HalvingGridSearchCV)\s*\(", text),
        _has(r"\b(?:RandomizedSearchCV|HalvingRandomSearchCV)\s*\(", text),
        _has(r"\bGroupKFold\s*\(", text),
        _has(r"\bStratifiedKFold\s*\(", text),
        _has(r"(?<!Group)(?<!Stratified)\bKFold\s*\(", text),
        _has(r"\bTimeSeriesSplit\s*\(", text),
        _has(r"\bGroupShuffleSplit\s*\(", text),
        _has(r"shuffle\s*=\s*True", text),
        _has(r"random_state\s*=", text),
        _has(r"\.sort_values\s*\(", text),
        _has(r"\b(?:date|time|timestamp|datetime|month|year|day)\b", text),
        _has(r"shuffle\s*=\s*False", text),
        _has(r"\.bfill\s*\(|method\s*=\s*['\"](?:bfill|backfill)", text),
        _has(r"\.rolling\s*\([^)]*center\s*=\s*True", text),
        _has(r"\.shift\s*\(\s*-\s*\d", text),
        _has(r"\b(?:datetime|Timestamp)\.(?:now|utcnow)\s*\(|\bdate\.today\s*\(", text),
        _has(r"\.concat\s*\(", text),
        _has(r"\.concat\s*\([^)]*(?:test|holdout|validation|val)", text),
        _has(r"get_dummies\s*\(", text),
        _has(r"drop_duplicates\s*\(", text),
        _has(r"groupby\s*\([^)]*\)\s*\.transform\s*\(", text),
        _has(r"\b(?:describe|value_counts)\s*\(", text),
        1.0
        if _TARGET.search(text)
        and re.search(r"\b(?:fit|fit_transform|fit_resample)\s*\(", text)
        else 0.0,
        1.0
        if any(
            _ASSIGN.match(line) and _TARGET.search(_ASSIGN.match(line).group(2))
            for line in lines
        )
        else 0.0,
        _has(r"read_(?:csv|json|parquet|excel)\s*\([^)]*(?:test|holdout)", text),
        _has(r"#.*(?:leak|cheat|hack|shortcut)", text),
        _has(
            r"['\"](?:customer_id|user_id|row_id|uuid|guid|account_id|patient_id)['\"]",
            text,
        ),
        _has(r"class_weight\s*=", text),
        _has(r"\btrain(?:ing)?\b", text),
        _has(r"\b(?:test|holdout|validation)\b.*\b(?:score|evaluate|predict)\b", text),
        1.0 if unsafe_preprocessing else 0.0,
        1.0 if unsafe_resampling else 0.0,
        1.0 if unsafe_selection else 0.0,
        1.0 if unsafe_target_encoding else 0.0,
        1.0 if test_rows_before_split else 0.0,
        1.0 if temporal_order_risk else 0.0,
        1.0 if group_boundary_ignored else 0.0,
        1.0 if identifier_feature_risk else 0.0,
        1.0 if target_feature_risk else 0.0,
        1.0 if future_feature_risk else 0.0,
        1.0 if train_only_fit else 0.0,
        1.0 if pipeline_boundary else 0.0,
    ]

    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(FEATURE_NAMES),):  # pragma: no cover - defensive
        raise RuntimeError("feature vector shape drifted from FEATURE_NAMES")
    return vector


def model_features(source: str) -> np.ndarray:
    """Return only the features the learned model is allowed to use."""

    return extract_features(source)[list(MODEL_FEATURE_INDICES)]


def feature_matrix(sources: list[str], *, model_only: bool = False) -> np.ndarray:
    if not sources:
        width = len(MODEL_FEATURE_NAMES) if model_only else len(FEATURE_NAMES)
        return np.empty((0, width), dtype=float)
    if model_only:
        return np.vstack([model_features(source) for source in sources])
    return np.vstack([extract_features(source) for source in sources])


__all__ = [
    "EXCLUDED_MODEL_FEATURES",
    "FEATURE_NAMES",
    "MODEL_FEATURE_INDICES",
    "MODEL_FEATURE_NAMES",
    "extract_features",
    "feature_matrix",
    "model_features",
]
