"""Small, deterministic feature extraction for the optional learned detector.

The detector is a CPU-only lightweight model (tree ensemble or logistic
regression) over interpretable control-flow indicators.  It never loads a
pickle, never calls the network, and every feature can be named in the report.

Phase 5 adds indicators for *ordering* (was a transformer or selection step
fitted before the split or before cross-validation?), *frame identity* (was that
fit on the training frame, the test frame or the whole frame?), *target-derived
columns*, *duplicate handling*, *grouping*, *temporal ordering* and *EDA-only
summaries*.  The features stay binary/count-based on purpose: they transfer
across projects better than hashed tokens, and every one of them can be pointed
at in a report.
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
    # Phase 5 additions: ordering, frame identity and relationship indicators.
    "fit_on_all_data_after_split",
    "fit_on_test_frame",
    "fit_on_train_frame",
    "transformer_before_split",
    "encoder_before_split",
    "target_encoder_after_split",
    "target_encoder_in_pipeline",
    "target_mean_before_split",
    "target_mean_after_split",
    "manual_full_data_stats",
    "manual_train_stats",
    "cv_on_full_data",
    "cv_after_split",
    "tuning_on_full_data",
    "tuning_after_split",
    "duplicates_after_split",
    "dedupe_before_split",
    "target_relabel_safe",
    "eda_summary_only",
    "summary_used_as_feature",
    "identifier_dropped",
    "group_columns_used",
    "time_ordered_split",
    "target_column_passed",
    "pipeline_cv_full_data",
    "leak_column_name",
    "transform_after_split",
    "fit_count",
    "unsafe_transformer_count_before_split",
    "safe_boundary_ratio",
    "pivot_before_split",
    "group_aggregate_before_split",
    "nested_cv",
    "cv_full_data_safe",
    # Phase 6 additions: functional/manual transforms, fold-local fitting,
    # test-partition scoring and category-only encodings.
    "functional_scaling_before_split",
    "manual_scaling_before_split",
    "full_data_imputation_before_split",
    "get_dummies_before_split",
    "cv_on_test_partition",
    "group_kfold_without_groups",
    "repeated_test_scoring",
    "frequency_encoding_before_split",
    "default_shuffle_after_sort",
    "fold_local_fit",
    "search_over_pipeline_safe",
    "onehot_only_full_data",
    "train_only_transformer_fit",
    "target_lag_safe",
)

FEATURE_NAMES = _INDICATOR_NAMES

# Size and style signals are deliberately excluded from the learned model:
# they can be shortcuts that do not transfer to new code bases.
EXCLUDED_MODEL_FEATURES = frozenset(
    {"log_lines", "log_bytes", "has_random_state", "target_column_passed"}
)
MODEL_FEATURE_NAMES = tuple(
    name for name in FEATURE_NAMES if name not in EXCLUDED_MODEL_FEATURES
)
MODEL_FEATURE_INDICES = tuple(
    index
    for index, name in enumerate(FEATURE_NAMES)
    if name not in EXCLUDED_MODEL_FEATURES
)

_SPLIT = re.compile(r"train_test_split\s*\(")
_FIT_CALL = re.compile(
    r"(?P<receiver>[A-Za-z_]\w*)?\s*\.?\s*"
    r"(?P<kind>fit_transform|fit_resample|fit)\s*\("
)
_TRANSFORM_CALL = re.compile(r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*transform\s*\(")
_CV_CALL = re.compile(r"\b(?:cross_val_score|cross_validate|cross_val_predict)\s*\(")
_TUNING_CALL = re.compile(
    r"\b(?:GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV)\s*\("
)
_TUNING_FIT = re.compile(r"\b(?:search|grid|tuner|gs|rs)\w*\s*\.\s*fit\s*\(")
_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\[[^\]]+\])?\s*=\s*(.+)$")
_DATA_COLUMN = re.compile(
    r"^\s*[A-Za-z_]\w*\s*\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*(.+)$"
)
_IDENT = re.compile(r"[A-Za-z_]\w*")
_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")

_SCALER = re.compile(
    r"\b(?:StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|"
    r"QuantileTransformer|PowerTransformer)\b"
)
_IMPUTER = re.compile(r"\b(?:SimpleImputer|IterativeImputer|KNNImputer)\b")
_PCA = re.compile(r"\b(?:PCA|KernelPCA|IncrementalPCA|SparsePCA|TruncatedSVD)\b")
_ENCODER = re.compile(r"\b(?:OneHotEncoder|OrdinalEncoder|LabelEncoder)\b")
_TARGET_ENCODER = re.compile(r"\b(?:TargetEncoder|WOEEncoder|CatBoostEncoder)\b")
_SELECTOR = re.compile(
    r"\b(?:SelectKBest|SelectPercentile|SelectFromModel|RFE|RFECV|"
    r"SequentialFeatureSelector)\b"
)
_RESAMPLER = re.compile(r"\b(?:SMOTE|ADASYN|RandomOverSampler|RandomUnderSampler)\b")
_VECTORIZER = re.compile(r"\b(?:TfidfVectorizer|CountVectorizer|HashingVectorizer)\b")
_PIPELINE = re.compile(r"\b(?:Pipeline|make_pipeline)\s*\(")
_COLUMN_TRANSFORMER = re.compile(r"\bColumnTransformer\s*\(")
_GROUP_AWARE = re.compile(r"\b(?:GroupKFold|GroupShuffleSplit|StratifiedGroupKFold)\b")
_TIME_WORDS = re.compile(
    r"\b(?:date|time|timestamp|datetime|month|year|day|week|quarter|hour|"
    r"minute|second)\b"
)
# A *column* that carries time, as opposed to a bare word that happens to look
# temporal (``time_start``, ``runtime`` or a comment).  Used to keep temporal
# indicators from firing on ordinary columns.
_TIME_COLUMN = re.compile(
    r"['\"](?:[a-z0-9_]*_)?"
    r"(?:date|time|timestamp|datetime|month|year|week|day|quarter|hour)"
    r"(?:_[a-z0-9_]+)?['\"]|\.dt\.",
    re.IGNORECASE,
)
_GROUP_WORDS = re.compile(
    r"\b(?:customer|account|patient|user|store|device|session|merchant|employee)"
    r"_?id\b"
)
_IDENTIFIER = re.compile(
    r"['\"](?:customer_id|user_id|row_id|uuid|guid|account_id|patient_id|"
    r"session_id|device_id|merchant_id|employee_id|row_hash)['\"]"
)
_LEAK_NAME = re.compile(
    r"(?:ground_truth|final_decision|target_copy|label_copy|answer_key|"
    r"was_fraud|did_convert|did_churn|leak(?:ed)?_|_leak)"
)
_SUMMARY = re.compile(r"\.(?:describe|value_counts|agg|aggregate)\s*\(")
_STATS = re.compile(r"(?:\.|\b)(?:mean|median|std|var|quantile|describe)\s*\(")
_DISPLAY_CALL = re.compile(
    r"^\s*(?:print|display|pprint|logger\.\w+|logging\.\w+)\s*\("
)
_NON_FEATURE_RHS = re.compile(
    r"train_test_split\s*\(|cross_val|cross_validate|GridSearch|RandomizedSearch|"
    r"\.fit\s*\(|\.fit_transform\s*\(|\.fit_resample\s*\(|\.score\s*\(|"
    r"\.predict\s*\(|\.drop\s*\(|\.groupby\s*\(|\.assign\s*\(|"
    r"\.agg\s*\(|\.aggregate\s*\(|\.describe\s*\(|\.value_counts\s*\(|"
    r"\.mean\s*\(|\.median\s*\(|\.std\s*\(|\.sum\s*\("
)
_TARGET = re.compile(r"\b(?:y|label|labels|target|outcome|is_[a-z_]+)\b")
_GENERIC_TARGETS = frozenset(
    {
        "y",
        "y_train",
        "y_test",
        "label",
        "labels",
        "target",
        "targets",
        "outcome",
        "outcomes",
        "is_fraud",
        "is_churn",
        "is_default",
        "converted",
        "churn",
        "fraud",
        "approved",
        "readmitted",
        "defaulted",
        "clicked",
        "purchased",
        "survived",
        "attrition",
    }
)


def _has(pattern: str, text: str, flags: int = re.IGNORECASE) -> float:
    return 1.0 if re.search(pattern, text, flags) else 0.0


def _first_line(lines: list[str], pattern: re.Pattern[str]) -> int | None:
    for index, line in enumerate(lines):
        if pattern.search(line):
            return index
    return None


def _split_arguments(call: str) -> list[str]:
    """Split a call's argument text on top-level commas."""

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in call:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _split_calls(text: str) -> list[tuple[int, list[str]]]:
    calls: list[tuple[int, list[str]]] = []
    for match in re.finditer(r"train_test_split\s*\(([^)]*)\)", text, re.DOTALL):
        calls.append((match.start(), _split_arguments(match.group(1))))
    return calls


def _tokens(expression: str) -> set[str]:
    found = set(_IDENT.findall(expression))
    found.update(_QUOTED.findall(expression))
    return found


def _target_names(text: str, lines: list[str]) -> set[str]:
    names = set(_GENERIC_TARGETS)

    def expression_names(expression: str) -> set[str]:
        quoted = set(_QUOTED.findall(expression))
        identifiers = _IDENT.findall(expression)
        if "[" in expression:
            # ``frame['sales']`` -> ``sales``; the frame is not the target.
            return quoted | set(identifiers[-1:])
        return quoted | set(identifiers)

    for _position, arguments in _split_calls(text):
        if len(arguments) >= 2:
            names.update(expression_names(arguments[1]))
    for match in re.finditer(
        r"\.(?:fit|fit_transform|fit_resample)\s*\(([^)]*)\)", text
    ):
        arguments = _split_arguments(match.group(1))
        if len(arguments) >= 2:
            names.update(expression_names(arguments[1]))
    for line in lines:
        match = _DATA_COLUMN.match(line)
        if match and _QUOTED.search(match.group(2)):
            column = match.group(1)
            if column in names:
                names.add(column)
    for container in (
        "df",
        "X",
        "frame",
        "table",
        "data",
        "train",
        "test",
        "rows",
        "visits",
        "tickets",
        "sales",
        "readings",
        "applications",
        "touches",
        "ledger",
    ):
        names.discard(container)
    return names


def _references_target(expression: str, target_names: set[str]) -> bool:
    return bool(_tokens(expression) & target_names)


def _fit_events(
    lines: list[str],
) -> list[tuple[int, str, str, str, str]]:
    """Return ``(line, receiver, kind, first_argument, class)`` fit events."""

    variable_classes: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(", line)
        if match:
            variable_classes[match.group(1)] = match.group(2)

    def class_for(line: str, receiver: str) -> str:
        known = (
            _SCALER,
            _IMPUTER,
            _PCA,
            _ENCODER,
            _TARGET_ENCODER,
            _SELECTOR,
            _RESAMPLER,
            _VECTORIZER,
            _PIPELINE,
            _COLUMN_TRANSFORMER,
        )
        for pattern in known:
            match = pattern.search(line)
            if match:
                return match.group(0).lstrip("(")
        if receiver in variable_classes:
            return variable_classes[receiver]
        if receiver:
            for _line_index, other_line in enumerate(lines):
                if re.search(
                    rf"\b{re.escape(receiver)}\s*=\s*([A-Za-z_]\w*)\s*\(",
                    other_line,
                ):
                    return variable_classes.get(receiver, receiver)
        return receiver or "unknown"

    events: list[tuple[int, str, str, str, str]] = []
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("import ", "from ")):
            continue
        for match in _FIT_CALL.finditer(line):
            receiver = match.group("receiver") or ""
            kind = match.group("kind")
            after = line[match.end() :]
            depth = 1
            args: list[str] = []
            current: list[str] = []
            quote: str | None = None
            for char in after:
                if quote is not None:
                    current.append(char)
                    if char == quote:
                        quote = None
                    continue
                if char in {"'", '"'}:
                    quote = char
                    current.append(char)
                    continue
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth -= 1
                    if depth == 0:
                        break
                if char == "," and depth == 1:
                    args.append("".join(current).strip())
                    current = []
                    continue
                current.append(char)
            if current:
                args.append("".join(current).strip())
            first = args[0] if args else ""
            events.append((index, receiver, kind, first, class_for(line, receiver)))
    return events


def _is_train_frame(expression: str) -> bool:
    lowered = expression.lower()
    return bool(re.search(r"\btrain\w*\b|\b\w+_train\w*\b|\btr\b|\b\w+_tr\b", lowered))


def _is_test_frame(expression: str) -> bool:
    lowered = expression.lower()
    return bool(
        re.search(
            r"\btest\w*\b|\b\w+_test\w*\b|\bholdout\w*\b|\bvalidation\w*\b|\bval\b",
            lowered,
        )
    )


def _before_split(index: int | None, split_at: int | None) -> bool:
    if index is None:
        return False
    return split_at is None or index < split_at


def extract_features(source: str) -> np.ndarray:
    """Return a fixed-length numeric vector for one Python source unit."""

    text = source or ""
    lines = text.splitlines()
    split_at = _first_line(lines, _SPLIT)
    split_calls = _split_calls(text)
    fit_events = _fit_events(lines)
    target_names = _target_names(text, lines)

    split_frame = ""
    if split_calls:
        arguments = split_calls[0][1]
        if arguments:
            frame_match = _IDENT.search(arguments[0])
            split_frame = frame_match.group(0) if frame_match else ""

    split_outputs = re.findall(
        r"^\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)[^=\n]*=\s*train_test_split\s*\(",
        text,
        re.MULTILINE,
    )
    split_train_name = split_outputs[0][0] if split_outputs else ""
    split_test_name = split_outputs[0][1] if split_outputs else ""

    fit_at = _first_line(lines, _FIT_CALL)
    transform_at = _first_line(lines, _TRANSFORM_CALL)
    cv_at = _first_line(lines, _CV_CALL)
    tuning_at = _first_line(lines, _TUNING_CALL)
    tuning_fit_at = _first_line(lines, _TUNING_FIT)

    # A fit inside a cross-validation loop is fitted per fold.  That is the
    # correct pattern, so it must not be counted as "preprocessing before the
    # split" just because the script has no single ``train_test_split`` call.
    fold_local_rows: set[int] = set()
    loop_indent: int | None = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        if loop_indent is not None and indent <= loop_indent:
            loop_indent = None
        if re.match(r"for\b", stripped) and re.search(
            r"\.split\s*\(|\bKFold\b|\bStratifiedKFold\b|\bGroupKFold\b", stripped
        ):
            loop_indent = indent
            continue
        if (
            loop_indent is not None
            and indent > loop_indent
            and re.search(r"\.(?:fit|fit_transform|fit_resample)\s*\(", stripped)
        ):
            fold_local_rows.add(index)
    fold_local_fit = bool(fold_local_rows)
    boundary_fit_events = [
        event for event in fit_events if event[0] not in fold_local_rows
    ]

    def any_fit_before(classes: tuple[re.Pattern[str], ...]) -> bool:
        for index, _receiver, _kind, _arg, klass in boundary_fit_events:
            if not _before_split(index, split_at):
                continue
            if any(pattern.search(klass) for pattern in classes):
                return True
        return False

    def any_feature_fit_before(classes: tuple[re.Pattern[str], ...]) -> bool:
        """Like ``any_fit_before`` but ignores fits on the label vector only.

        ``LabelEncoder().fit_transform(y)`` encodes the target; it does not
        learn anything from the feature matrix, so it is not a preprocessing
        leak.  A fit that mixes features and the target still counts.
        """

        for index, _receiver, _kind, first, klass in boundary_fit_events:
            if not _before_split(index, split_at):
                continue
            if not any(pattern.search(klass) for pattern in classes):
                continue
            tokens = _tokens(first)
            if tokens and tokens <= target_names:
                continue
            return True
        return False

    transformer_patterns = (_SCALER, _IMPUTER, _PCA, _SELECTOR, _VECTORIZER)
    transformer_before_split = any_fit_before(transformer_patterns)
    encoder_before_split = any_feature_fit_before((_ENCODER,))
    target_encoder_before_split = any_fit_before((_TARGET_ENCODER,))
    selector_before_split = any_fit_before((_SELECTOR,))
    resampler_before_split = any_fit_before((_RESAMPLER,))

    target_encoder_present = bool(_TARGET_ENCODER.search(text))
    target_encoder_in_pipeline = bool(
        re.search(
            r"(?:Pipeline|make_pipeline)\s*\((?:[^()]|\([^)]*\))*"
            r"(?:TargetEncoder|WOEEncoder|CatBoostEncoder)",
            text,
            re.DOTALL,
        )
    )
    target_mean_before_split = False
    target_mean_after_split = False
    for index, line in enumerate(lines):
        if "groupby" not in line or ("transform" not in line and ".mean" not in line):
            continue
        if not (_references_target(line, target_names) or _TARGET.search(line)):
            continue
        if _before_split(index, split_at):
            target_mean_before_split = True
        else:
            target_mean_after_split = True
    # ``df.groupby(column)[target].mean()`` followed by ``map`` is the same leak.
    for index, line in enumerate(lines):
        if ".map(" not in line or not _before_split(index, split_at):
            continue
        if re.search(r"\b(?:means|averages|encoding|map_?values|lookup)\b", line):
            for earlier in lines[:index]:
                if (
                    "groupby" in earlier
                    and (
                        _references_target(earlier, target_names)
                        or _TARGET.search(earlier)
                    )
                    and re.search(r"\.(?:mean|agg)\s*\(", earlier)
                ):
                    target_mean_before_split = True
                    break

    manual_full_data_stats = False
    manual_train_stats = False
    for index, line in enumerate(lines):
        if not _STATS.search(line) and "agg('mean')" not in line:
            continue
        if _DISPLAY_CALL.match(line):
            # ``print(df['x'].mean())`` is inspection, not feature engineering.
            continue
        if re.search(
            r"\b(?:groupby|agg|describe|value_counts|summary|rolling|shift|"
            r"expanding|ewm|bfill|ffill|clip|map|apply)\b",
            line,
        ):
            continue
        if _is_train_frame(line):
            manual_train_stats = True
            continue
        if not _before_split(index, split_at):
            continue
        if (split_frame and re.search(rf"\b{re.escape(split_frame)}\b", line)) or (
            not split_frame and re.search(r"\b(?:df|X|features|data)\b", line)
        ):
            manual_full_data_stats = True

    fit_on_all_data_after_split = False
    fit_on_test_frame = False
    fit_on_train_frame = False
    unsafe_transformer_count = 0
    for index, _receiver, _kind, first, klass in fit_events:
        first_name = first.strip()
        if index in fold_local_rows:
            continue
        if _is_test_frame(first) or (split_test_name and first_name == split_test_name):
            fit_on_test_frame = True
            continue
        if _is_train_frame(first) or (
            split_train_name and first_name == split_train_name
        ):
            if not _before_split(index, split_at):
                fit_on_train_frame = True
            continue
        if not _before_split(index, split_at) and split_at is not None:
            fit_on_all_data_after_split = True
        if _before_split(index, split_at) and any(
            pattern.search(klass) for pattern in transformer_patterns
        ):
            unsafe_transformer_count += 1

    unsafe_preprocessing = transformer_before_split
    unsafe_resampling = resampler_before_split
    unsafe_selection = selector_before_split
    unsafe_target_encoding = bool(
        target_encoder_before_split
        or target_mean_before_split
        or (
            target_encoder_present
            and split_at is None
            and not target_encoder_in_pipeline
        )
    )

    cv_present = cv_at is not None
    tuning_present = tuning_at is not None or tuning_fit_at is not None
    nested_cv = bool(cv_present and tuning_present and tuning_fit_at is None)
    cv_on_full_data = bool(
        cv_present and (split_at is None or _before_split(cv_at, split_at))
    )
    cv_after_split = bool(cv_present and not cv_on_full_data)
    tuning_on_full_data = bool(
        tuning_present
        and tuning_fit_at is not None
        and (split_at is None or _before_split(tuning_fit_at, split_at))
    )
    tuning_after_split = bool(
        tuning_present and tuning_fit_at is not None and not tuning_on_full_data
    )

    pipeline_present = bool(_PIPELINE.search(text))
    pipeline_boundary = pipeline_present and bool(
        split_at is not None and fit_at is not None and fit_at > split_at
    )
    pipeline_cv_full_data = cv_on_full_data and pipeline_present
    cv_full_data_safe = cv_on_full_data and bool(
        pipeline_present or re.search(r"\bTimeSeriesSplit\b", text) or nested_cv
    )

    test_rows_before_split = bool(
        re.search(r"\.concat\s*\([^)]*(?:test|holdout|validation|val)", text)
    ) and (
        split_at is None
        or _before_split(_first_line(lines, re.compile(r"concat")), split_at)
    )
    temporal_order_risk = (
        bool(_TIME_COLUMN.search(text))
        and bool(
            re.search(r"shuffle\s*=\s*True", text)
            or re.search(r"(?<!Group)(?<!Stratified)\bKFold\s*\(", text)
        )
        and not re.search(r"\bTimeSeriesSplit\b", text)
        and not re.search(r"shuffle\s*=\s*False", text)
    )
    group_aware = bool(_GROUP_AWARE.search(text))
    group_columns_used = bool(re.search(r"\bgroups\s*=", text))
    drop_identifier_at = _first_line(
        lines,
        re.compile(
            r"drop\s*\([^)]*(?:customer_id|user_id|row_id|uuid|guid|account_id|"
            r"patient_id|session_id|device_id|merchant_id|row_hash)"
        ),
    )
    identifier_dropped = bool(
        drop_identifier_at is not None and _before_split(drop_identifier_at, split_at)
    )
    # Drop the identifier from the *features*, and the group boundary is still
    # ignored when rows are split randomly: repeated entities straddle the
    # split.  Merely mentioning an id inside ``drop_duplicates`` is not enough
    # evidence that the id is part of the modelling frame.
    group_identifier_in_frame = bool(
        _GROUP_WORDS.search(text)
        and (
            re.search(
                r"\[\s*\[[^\]]*(?:customer|account|patient|user|store|device|"
                r"session|merchant|employee)_?id",
                text,
            )
            or re.search(
                r"\.drop\s*\([^)]*(?:customer|account|patient|user|store|device|"
                r"session|merchant|employee)_?id",
                text,
            )
            or re.search(
                r"groupby\s*\([^)]*(?:customer|account|patient|user|store|"
                r"device|session|merchant|employee)_?id",
                text,
            )
            or re.search(
                r"train_test_split\s*\([^)]*(?:customer|account|patient|user|"
                r"store|device|session|merchant|employee)_?id",
                text,
            )
        )
    )
    group_boundary_ignored = (
        group_identifier_in_frame
        and not group_aware
        and not group_columns_used
        and not identifier_dropped
        and bool(split_at is not None or re.search(r"\bKFold\s*\(", text))
    )
    identifier_feature_risk = (
        bool(_IDENTIFIER.search(text))
        and not identifier_dropped
        and not group_aware
        and not group_columns_used
        and bool(
            re.search(
                r"\[\s*\[[^\]]*(?:customer_id|user_id|row_id|uuid|guid|"
                r"account_id|patient_id|session_id|device_id|merchant_id|row_hash)",
                text,
            )
            or re.search(r"\bX\s*=\s*df\s*\[", text)
        )
    )
    time_ordered_split = bool(
        re.search(r"\.sort_values\s*\(", text)
        and (
            re.search(r"shuffle\s*=\s*False", text)
            or re.search(r"\bTimeSeriesSplit\b", text)
        )
    )

    target_feature_risk = False
    target_relabel_safe = False
    target_in_assignment = False
    summary_vars: dict[str, int] = {}
    for index, line in enumerate(lines):
        column = _DATA_COLUMN.match(line)
        if column is not None:
            lhs, rhs = column.groups()
            if _references_target(rhs, target_names) and not _NON_FEATURE_RHS.search(
                rhs
            ):
                target_in_assignment = True
            is_target_lhs = bool(_tokens(lhs) & target_names)
            if is_target_lhs and _references_target(rhs, target_names):
                target_relabel_safe = True
            if (
                not is_target_lhs
                and _before_split(index, split_at)
                and not _NON_FEATURE_RHS.search(rhs)
                and not re.search(r"\.shift\s*\(\s*[1-9]\d*\s*\)", rhs)
                and (
                    _references_target(rhs, target_names)
                    or bool(_LEAK_NAME.search(rhs))
                )
            ):
                target_feature_risk = True
            if _SUMMARY.search(rhs) and "transform" not in rhs:
                summary_vars[lhs] = index
            continue
        assign = _ASSIGN.match(line)
        if assign:
            lhs, rhs = assign.groups()
            if _references_target(rhs, target_names) and not _NON_FEATURE_RHS.search(
                rhs
            ):
                target_in_assignment = True
            is_target_lhs = bool(_tokens(lhs) & target_names)
            if is_target_lhs and _references_target(rhs, target_names):
                target_relabel_safe = True
            if (
                not is_target_lhs
                and _before_split(index, split_at)
                and not _NON_FEATURE_RHS.search(rhs)
                and not re.search(r"\.shift\s*\(\s*[1-9]\d*\s*\)", rhs)
                and (
                    _references_target(rhs, target_names)
                    or bool(_LEAK_NAME.search(rhs))
                )
            ):
                target_feature_risk = True
            if _SUMMARY.search(rhs) and "transform" not in rhs:
                summary_vars[lhs] = index
    summary_used_as_feature = False
    if summary_vars:
        usage_text = "\n".join(lines)
        for name in summary_vars:
            if re.search(
                rf"train_test_split\s*\([^)]*\b{re.escape(name)}\b", usage_text
            ) or re.search(rf"\bX\s*=[^\n]*\b{re.escape(name)}\b", usage_text):
                summary_used_as_feature = True
    eda_summary_only = bool(summary_vars) and not summary_used_as_feature

    shifted_columns: set[str] = set()
    for line in lines:
        column = _DATA_COLUMN.match(line)
        if column is not None and re.search(r"\.shift\s*\(\s*-\s*\d", column.group(2)):
            shifted_columns.add(column.group(1))
    # A forward shift used only to build the *label* is a forecasting target,
    # not a future feature.  It counts as safe only when the derived column is
    # explicitly dropped from the feature matrix before the split.
    forward_label_excluded = bool(shifted_columns) and all(
        re.search(
            rf"drop\s*\([^)]*['\"]{re.escape(name)}['\"]",
            text,
        )
        for name in shifted_columns
    )
    future_feature_risk = (
        bool(
            re.search(r"\.bfill\s*\(|method\s*=\s*['\"](?:bfill|backfill)", text)
            or re.search(r"\.rolling\s*\([^)]*center\s*=\s*True", text)
            or re.search(r"\.shift\s*\(\s*-\s*\d", text)
            or re.search(
                r"read_(?:csv|json|parquet|excel)\s*\([^)]*\b(?:future|next_)",
                text,
                re.IGNORECASE,
            )
            or re.search(r"\.merge\s*\([^)]*\b(?:future|next_month|next_week)", text)
            or re.search(r"\b(?:future|next)_?\w*\s*=\s*[^\n]*(?:read_|merge\()", text)
        )
        and not forward_label_excluded
    )
    pivot_before_split = bool(
        re.search(r"\.pivot(?:_table)?\s*\(", text)
        and (
            split_at is None
            or _before_split(
                _first_line(lines, re.compile(r"\.pivot(?:_table)?\s*\(")), split_at
            )
        )
    )
    group_aggregate_before_split = False
    group_pattern = re.compile(
        r"\.groupby\s*\([^)]*\).*?"
        r"(?:\.transform\s*\(|\.agg(?:regate)?\s*\(|"
        r"\[[^\]]+\]\s*\.(?:mean|median|sum|std|count|min|max)\s*\()"
    )
    for index, line in enumerate(lines):
        if not _before_split(index, split_at):
            continue
        if not group_pattern.search(line) or "print" in line:
            continue
        if _DATA_COLUMN.match(line):
            group_aggregate_before_split = True
            break
        assign = _ASSIGN.match(line)
        if assign:
            name = assign.group(1)
            if re.search(
                rf"\.map\s*\(\s*{re.escape(name)}\b|"
                rf"merge\s*\([^)]*\b{re.escape(name)}\b|"
                rf"\bX\s*=[^\n]*\b{re.escape(name)}\b",
                text,
            ):
                group_aggregate_before_split = True
                break
    train_only_fit = split_at is not None and fit_at is not None and fit_at > split_at
    fit_without_split = split_at is None and bool(boundary_fit_events)
    leak_column_name = bool(_LEAK_NAME.search(text))
    target_column_passed = any(
        bool(_QUOTED.search(arguments[1]))
        for _position, arguments in split_calls
        if len(arguments) >= 2
    )
    duplicates_after_split = bool(
        re.search(r"drop_duplicates\s*\(", text)
        and _first_line(lines, re.compile(r"drop_duplicates")) is not None
        and _before_split(_first_line(lines, re.compile(r"drop_duplicates")), split_at)
        is False
        and split_at is not None
    )
    dedupe_before_split = bool(
        re.search(r"drop_duplicates\s*\(", text)
        and (
            split_at is None
            or _before_split(
                _first_line(lines, re.compile(r"drop_duplicates")), split_at
            )
        )
    )

    fit_count = min(len(fit_events), 8)
    after_fits = sum(
        1
        for index, _receiver, _kind, _first, _klass in fit_events
        if not _before_split(index, split_at)
    )
    safe_boundary_ratio = round(after_fits / len(fit_events), 4) if fit_events else 0.0

    # Phase 6: local transforms that Phase 5 could not see.  Functional
    # scalers and hand-written normalisation are common in notebooks and in
    # code that predates scikit-learn pipelines.
    def _first_matching(pattern: re.Pattern[str]) -> int | None:
        for index, line in enumerate(lines):
            if pattern.search(line):
                return index
        return None

    functional_scaling = re.compile(
        r"(?<![\w.])(?:scale|normalize|power_transform|quantile_transform|"
        r"robust_scale|minmax_scale|maxabs_scale)\s*\("
    )
    functional_scaling_at = _first_matching(functional_scaling)
    functional_scaling_before_split = (
        functional_scaling_at is not None
        and _before_split(functional_scaling_at, split_at)
    )

    manual_scaling_before_split = False
    for index, line in enumerate(lines):
        if not _before_split(index, split_at):
            continue
        if not (_ASSIGN.match(line) or _DATA_COLUMN.match(line)):
            continue
        if _is_train_frame(line):
            continue
        column = _DATA_COLUMN.match(line)
        assignment = _ASSIGN.match(line)
        if column is not None:
            in_place = bool(
                re.search(rf"\b{re.escape(column.group(1))}\b", column.group(2))
            )
        elif assignment is not None:
            in_place = bool(
                re.search(rf"\b{re.escape(assignment.group(1))}\b", assignment.group(2))
            )
        else:  # pragma: no cover - one of the two patterns always matches above
            in_place = False
        if (
            split_frame
            and not in_place
            and not re.search(rf"\b{re.escape(split_frame)}\b", line)
        ):
            continue
        if not re.search(r"\.(?:max|min|std|mean|var)\s*\(\s*\)", line):
            continue
        if re.search(
            r"\.(?:groupby|agg|aggregate|describe|value_counts|rolling|shift|"
            r"expanding|ewm|fillna|map)\b",
            line,
        ):
            continue
        if re.search(
            r"[/*+-]\s*[A-Za-z_][\w.\[\]'\"]*\.(?:max|min|std|mean)\s*\(", line
        ):
            manual_scaling_before_split = True
            break

    full_data_imputation_before_split = False
    for index, line in enumerate(lines):
        if not _before_split(index, split_at) or _is_train_frame(line):
            continue
        if not re.search(r"\.fillna\s*\(", line):
            continue
        after_fill = line.split(".fillna", 1)[1]
        if re.search(
            r"\.(?:mode|mean|median|value_counts|agg|aggregate|describe)\s*\(",
            after_fill,
        ):
            full_data_imputation_before_split = True
            break

    get_dummies_at = _first_matching(re.compile(r"get_dummies\s*\("))
    get_dummies_before_split = get_dummies_at is not None and _before_split(
        get_dummies_at, split_at
    )

    cv_on_test_partition = bool(
        re.search(
            r"cross_val_(?:score|validate|predict)\s*\((?:[^()]|\([^()]*\))*"
            r"\b(?:X_)?test\w*\b",
            text,
        )
    )

    group_kfold_without_groups = bool(
        re.search(r"\bGroupKFold\s*\(", text) and not re.search(r"\bgroups\s*=", text)
    )

    test_score_calls = len(
        re.findall(
            r"\.score\s*\(\s*(?:X_)?test\w*\s*[,)]|\bscore\s*\(\s*test\w*\s*[,)]",
            text,
        )
    )
    repeated_test_scoring = test_score_calls >= 2

    frequency_encoding_before_split = False
    frequency_vars: set[str] = set()
    for index, line in enumerate(lines):
        if not _before_split(index, split_at):
            continue
        column = _DATA_COLUMN.match(line)
        assign = _ASSIGN.match(line)
        rhs = (
            column.group(2)
            if column is not None
            else (assign.group(2) if assign is not None else "")
        )
        if ".value_counts(" in rhs and ".map(" in rhs:
            frequency_encoding_before_split = True
            break
        if ".map(" in rhs and any(
            re.search(rf"\b{re.escape(name)}\b", rhs) for name in frequency_vars
        ):
            frequency_encoding_before_split = True
            break
        if assign is not None and ".value_counts(" in assign.group(2):
            frequency_vars.add(assign.group(1))

    # Sorting only creates a temporal ordering when the sort key is a time-like
    # column and the sort happens before the split.  Sorting by amount for a
    # report, or sorting after the split, is not a temporal leak.
    sort_at = _first_line(lines, re.compile(r"\.sort_values\s*\("))
    sort_line = lines[sort_at] if sort_at is not None else ""
    default_shuffle_after_sort = bool(
        sort_at is not None
        and split_at is not None
        and sort_at < split_at
        and _TIME_COLUMN.search(sort_line)
        and not re.search(r"shuffle\s*=\s*False", text)
    )

    search_over_pipeline_safe = bool(
        re.search(
            r"\b(?:GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV)\s*\(", text
        )
        and _PIPELINE.search(text)
        and split_at is None
    )

    onehot_only_full_data = False
    non_import_text = "\n".join(
        line for line in lines if not line.lstrip().startswith(("import ", "from "))
    )
    if not re.search(
        r"\b(?:StandardScaler|MinMaxScaler|RobustScaler|MaxAbsScaler|Normalizer|"
        r"PowerTransformer|QuantileTransformer|SimpleImputer|IterativeImputer|"
        r"KNNImputer|PCA|KernelPCA|TruncatedSVD|SelectKBest|SelectFromModel|"
        r"TargetEncoder|WOEEncoder|CatBoostEncoder)\b",
        non_import_text,
    ):
        for index, _receiver, _kind, _first, klass in fit_events:
            if not _before_split(index, split_at):
                continue
            line = lines[index] if index < len(lines) else ""
            if (
                re.search(
                    r"\b(?:OneHotEncoder|OrdinalEncoder|ColumnTransformer)\b",
                    line + klass,
                )
                or "get_dummies" in line
            ):
                onehot_only_full_data = True
                break

    train_only_transformer_fit = any(
        split_at is not None
        and index > split_at
        and (
            _is_train_frame(first)
            or (split_train_name and first.strip() == split_train_name)
        )
        and any(pattern.search(klass) for pattern in transformer_patterns)
        for index, _receiver, _kind, first, klass in fit_events
    )

    target_lag_safe = bool(
        not future_feature_risk
        and re.search(r"\.shift\s*\(\s*[1-9]\d*\s*\)", text)
        and any(
            ".shift(" in line
            and _references_target(line, target_names)
            and _before_split(index, split_at)
            for index, line in enumerate(lines)
        )
    )

    features = {
        "log_lines": math.log1p(len(lines)),
        "log_bytes": math.log1p(len(text.encode("utf-8", "replace"))),
        "has_split": 1.0 if split_at is not None else 0.0,
        "split_count": min(len(split_calls), 8.0),
        "log_split_position": math.log1p(split_at) / 5.0
        if split_at is not None
        else 0.0,
        "fit_before_split": _fitting_before(boundary_fit_events, split_at),
        "fit_transform_before_split": _fitting_before(
            boundary_fit_events, split_at, kind="fit_transform"
        ),
        "fit_without_split": 1.0 if fit_without_split else 0.0,
        "has_fit": 1.0 if fit_events else 0.0,
        "has_fit_transform": _has(r"\.\s*fit_transform\s*\(", text),
        "has_fit_resample": _has(r"\.\s*fit_resample\s*\(", text),
        "scaler": _has(_SCALER.pattern, text),
        "imputer": _has(_IMPUTER.pattern, text),
        "pca": _has(_PCA.pattern, text),
        "encoder": _has(_ENCODER.pattern, text),
        "target_encoder": 1.0 if target_encoder_present else 0.0,
        "selector": _has(_SELECTOR.pattern, text),
        "resampler": _has(_RESAMPLER.pattern, text),
        "vectorizer": _has(_VECTORIZER.pattern, text),
        "pipeline": 1.0 if pipeline_present else 0.0,
        "column_transformer": _has(_COLUMN_TRANSFORMER.pattern, text),
        "cross_validation": 1.0 if cv_present else 0.0,
        "grid_search": _has(r"\b(?:GridSearchCV|HalvingGridSearchCV)\s*\(", text),
        "random_search": _has(
            r"\b(?:RandomizedSearchCV|HalvingRandomSearchCV)\s*\(", text
        ),
        "group_kfold": _has(r"\bGroupKFold\s*\(", text),
        "stratified_kfold": _has(r"\bStratifiedKFold\s*\(", text),
        "plain_kfold": _has(r"(?<!Group)(?<!Stratified)\bKFold\s*\(", text),
        "time_series_split": _has(r"\bTimeSeriesSplit\s*\(", text),
        "group_shuffle_split": _has(r"\bGroupShuffleSplit\s*\(", text),
        "has_shuffle": _has(r"shuffle\s*=\s*True", text),
        "has_random_state": _has(r"random_state\s*=", text),
        "sort_values": _has(r"\.sort_values\s*\(", text),
        "time_words": 1.0 if _TIME_WORDS.search(text) else 0.0,
        "shuffle_false": _has(r"shuffle\s*=\s*False", text),
        "bfill": _has(r"\.bfill\s*\(|method\s*=\s*['\"](?:bfill|backfill)", text),
        "centered_window": _has(r"\.rolling\s*\([^)]*center\s*=\s*True", text),
        "negative_shift": _has(r"\.shift\s*\(\s*-\s*\d", text),
        "clock_now": _has(
            r"\b(?:datetime|Timestamp)\.(?:now|utcnow)\s*\(|\bdate\.today\s*\(", text
        ),
        "concat": _has(r"\.concat\s*\(", text),
        "test_concat": _has(r"\.concat\s*\([^)]*(?:test|holdout|validation|val)", text),
        "get_dummies": _has(r"get_dummies\s*\(", text),
        "drop_duplicates": _has(r"drop_duplicates\s*\(", text),
        "groupby_transform": _has(r"groupby\s*\([^)]*\)\s*\.transform\s*\(", text),
        "describe": _has(r"\b(?:describe|value_counts)\s*\(", text),
        "target_reference": 1.0
        if _references_target(text, target_names) and bool(fit_events)
        else 0.0,
        "target_in_assignment": 1.0 if target_in_assignment else 0.0,
        "read_test": _has(
            r"read_(?:csv|json|parquet|excel)\s*\([^)]*(?:test|holdout)", text
        ),
        "leak_comment": _has(r"#.*(?:leak|cheat|hack|shortcut)", text),
        "identifier_literal": 1.0 if _IDENTIFIER.search(text) else 0.0,
        "class_weight": _has(r"class_weight\s*=", text),
        "train_only_marker": _has(r"\btrain(?:ing)?\b", text),
        "holdout_eval": _has(
            r"\b(?:test|holdout|validation)\b.*\b(?:score|evaluate|predict)\b", text
        ),
        "unsafe_preprocessing": 1.0 if unsafe_preprocessing else 0.0,
        "unsafe_resampling": 1.0 if unsafe_resampling else 0.0,
        "unsafe_feature_selection": 1.0 if unsafe_selection else 0.0,
        "unsafe_target_encoding": 1.0 if unsafe_target_encoding else 0.0,
        "test_rows_before_split": 1.0 if test_rows_before_split else 0.0,
        "temporal_order_risk": 1.0 if temporal_order_risk else 0.0,
        "group_boundary_ignored": 1.0 if group_boundary_ignored else 0.0,
        "identifier_feature_risk": 1.0 if identifier_feature_risk else 0.0,
        "target_feature_risk": 1.0 if target_feature_risk else 0.0,
        "future_feature_risk": 1.0 if future_feature_risk else 0.0,
        "train_only_fit": 1.0 if train_only_fit else 0.0,
        "pipeline_boundary": 1.0 if pipeline_boundary else 0.0,
        "fit_on_all_data_after_split": 1.0 if fit_on_all_data_after_split else 0.0,
        "fit_on_test_frame": 1.0 if fit_on_test_frame else 0.0,
        "fit_on_train_frame": 1.0 if fit_on_train_frame else 0.0,
        "transformer_before_split": 1.0 if transformer_before_split else 0.0,
        "encoder_before_split": 1.0 if encoder_before_split else 0.0,
        "target_encoder_after_split": 1.0
        if target_encoder_present
        and any(
            klass.startswith(("TargetEncoder", "WOEEncoder", "CatBoostEncoder"))
            or _TARGET_ENCODER.search(klass)
            for index, _receiver, _kind, _arg, klass in fit_events
            if split_at is not None and index > split_at
        )
        else 0.0,
        "target_encoder_in_pipeline": 1.0 if target_encoder_in_pipeline else 0.0,
        "target_mean_before_split": 1.0 if target_mean_before_split else 0.0,
        "target_mean_after_split": 1.0 if target_mean_after_split else 0.0,
        "manual_full_data_stats": 1.0 if manual_full_data_stats else 0.0,
        "manual_train_stats": 1.0 if manual_train_stats else 0.0,
        "cv_on_full_data": 1.0 if cv_on_full_data else 0.0,
        "cv_after_split": 1.0 if cv_after_split else 0.0,
        "tuning_on_full_data": 1.0 if tuning_on_full_data else 0.0,
        "tuning_after_split": 1.0 if tuning_after_split else 0.0,
        "duplicates_after_split": 1.0 if duplicates_after_split else 0.0,
        "dedupe_before_split": 1.0 if dedupe_before_split else 0.0,
        "target_relabel_safe": 1.0 if target_relabel_safe else 0.0,
        "eda_summary_only": 1.0 if eda_summary_only else 0.0,
        "summary_used_as_feature": 1.0 if summary_used_as_feature else 0.0,
        "identifier_dropped": 1.0 if identifier_dropped else 0.0,
        "group_columns_used": 1.0 if group_columns_used else 0.0,
        "time_ordered_split": 1.0 if time_ordered_split else 0.0,
        "target_column_passed": 1.0 if target_column_passed else 0.0,
        "pipeline_cv_full_data": 1.0 if pipeline_cv_full_data else 0.0,
        "leak_column_name": 1.0 if leak_column_name else 0.0,
        "transform_after_split": 1.0
        if transform_at is not None and split_at is not None and transform_at > split_at
        else 0.0,
        "fit_count": float(fit_count),
        "unsafe_transformer_count_before_split": float(
            min(unsafe_transformer_count, 8)
        ),
        "safe_boundary_ratio": safe_boundary_ratio,
        "pivot_before_split": 1.0 if pivot_before_split else 0.0,
        "group_aggregate_before_split": 1.0 if group_aggregate_before_split else 0.0,
        "nested_cv": 1.0 if nested_cv else 0.0,
        "cv_full_data_safe": 1.0 if cv_full_data_safe else 0.0,
        "functional_scaling_before_split": 1.0
        if functional_scaling_before_split
        else 0.0,
        "manual_scaling_before_split": 1.0 if manual_scaling_before_split else 0.0,
        "full_data_imputation_before_split": 1.0
        if full_data_imputation_before_split
        else 0.0,
        "get_dummies_before_split": 1.0 if get_dummies_before_split else 0.0,
        "cv_on_test_partition": 1.0 if cv_on_test_partition else 0.0,
        "group_kfold_without_groups": 1.0 if group_kfold_without_groups else 0.0,
        "repeated_test_scoring": 1.0 if repeated_test_scoring else 0.0,
        "frequency_encoding_before_split": 1.0
        if frequency_encoding_before_split
        else 0.0,
        "default_shuffle_after_sort": 1.0 if default_shuffle_after_sort else 0.0,
        "fold_local_fit": 1.0 if fold_local_fit else 0.0,
        "search_over_pipeline_safe": 1.0 if search_over_pipeline_safe else 0.0,
        "onehot_only_full_data": 1.0 if onehot_only_full_data else 0.0,
        "train_only_transformer_fit": 1.0 if train_only_transformer_fit else 0.0,
        "target_lag_safe": 1.0 if target_lag_safe else 0.0,
    }

    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=float)
    if vector.shape != (len(FEATURE_NAMES),):  # pragma: no cover - defensive
        raise RuntimeError("feature vector shape drifted from FEATURE_NAMES")
    return vector


def _fitting_before(
    events: list[tuple[int, str, str, str, str]],
    split_at: int | None,
    *,
    kind: str | None = None,
) -> float:
    for index, _receiver, event_kind, _arg, _klass in events:
        if kind is not None and event_kind != kind:
            continue
        if _before_split(index, split_at):
            return 1.0
    return 0.0


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
