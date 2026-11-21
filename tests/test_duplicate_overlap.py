"""Dataset-level duplicate and near-duplicate overlap checks."""

import pandas as pd

from leaklens.leakage import feature_overlap, near_duplicate_overlap


def _train():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "city": ["Paris", "Lyon", "Nice", "Lille"],
            "y": [0, 1, 0, 1],
        }
    )


def test_feature_overlap_with_conflicting_labels_is_high():
    test = pd.DataFrame({"x": [2.0, 9.0], "city": ["Lyon", "Brest"], "y": [0, 1]})
    findings = feature_overlap(_train(), test, "y")
    assert findings and findings[0].severity == "high"
    assert "different label" in findings[0].detail


def test_feature_overlap_without_conflict_is_medium():
    test = pd.DataFrame({"x": [2.0, 9.0], "city": ["Lyon", "Brest"], "y": [1, 0]})
    findings = feature_overlap(_train(), test, "y")
    assert findings and findings[0].severity == "medium"


def test_disjoint_features_are_clean():
    test = pd.DataFrame({"x": [20.0, 30.0], "city": ["Brest", "Dijon"], "y": [0, 1]})
    assert feature_overlap(_train(), test, "y") == []


def test_target_is_excluded_from_the_feature_hash():
    test = pd.DataFrame({"x": [2.0], "city": ["Lyon"], "y": [0]})
    # The only matching row differs in the target, so the feature vector is
    # what overlaps; this must still be reported.
    assert feature_overlap(_train(), test, "y")


def test_no_feature_columns_is_clean():
    train = pd.DataFrame({"y": [0, 1, 0]})
    test = pd.DataFrame({"y": [0, 1]})
    assert feature_overlap(train, test, "y") == []


def test_near_duplicate_normalisation():
    train = pd.DataFrame({"city": ["Paris", "Lyon"], "x": [1.0, 2.00004], "y": [0, 1]})
    test = pd.DataFrame({"city": ["paris ", "LYON"], "x": [1.0, 2.0], "y": [1, 0]})
    findings = near_duplicate_overlap(train, test, "y")
    assert findings and findings[0].severity == "medium"
    assert findings[0].kind == "near_duplicate_overlap"


def test_near_duplicate_clean_control():
    train = pd.DataFrame({"city": ["Paris", "Lyon"], "x": [1.0, 2.0], "y": [0, 1]})
    test = pd.DataFrame({"city": ["Brest", "Dijon"], "x": [8.0, 9.0], "y": [0, 1]})
    assert near_duplicate_overlap(train, test, "y") == []


def test_unhashable_values_do_not_raise():
    train = pd.DataFrame({"a": [[1, 2], [3, 4]], "y": [0, 1]})
    test = pd.DataFrame({"a": [[1, 2]], "y": [0]})
    assert isinstance(feature_overlap(train, test, "y"), list)
    assert isinstance(near_duplicate_overlap(train, test, "y"), list)


def test_row_limit_zero_is_safe():
    train = _train()
    test = _train()
    assert feature_overlap(train, test, "y", max_rows=0) == []
    assert near_duplicate_overlap(train, test, "y", max_rows=0) == []


# ---------------------------------------------------------------------------
# Sampling disclosure (#622)
# ---------------------------------------------------------------------------

_UNDER_CAP_FEATURE = (
    "1 test row (50.0%) share an identical feature vector with train rows "
    "(the label column is ignored); deduplicate before splitting"
)


def test_feature_overlap_message_is_unchanged_below_the_cap():
    test = pd.DataFrame({"x": [2.0, 9.0], "city": ["Lyon", "Brest"], "y": [1, 0]})
    assert feature_overlap(_train(), test, "y")[0].detail == _UNDER_CAP_FEATURE


def _train_of_two():
    return pd.DataFrame({"x": [1.0, 2.0], "city": ["Paris", "Lyon"], "y": [0, 1]})


def test_feature_overlap_says_how_many_rows_it_examined():
    # Every test row matches, so any deterministic sample still overlaps.
    test = pd.DataFrame(
        {"x": [2.0, 2.0, 2.0, 2.0], "city": ["Lyon"] * 4, "y": [1, 1, 1, 1]}
    )
    detail = feature_overlap(_train_of_two(), test, "y", max_rows=2)[0].detail
    assert "2 of the 2 test rows examined" in detail
    assert "a deterministic sample of 4 test and 2 train rows" in detail
    # The sampled share must not be presented as a share of the split.
    assert "%" not in detail


def test_feature_overlap_conflict_message_says_it_used_a_sample():
    test = pd.DataFrame(
        {"x": [2.0, 2.0, 2.0, 2.0], "city": ["Lyon"] * 4, "y": [0, 0, 0, 0]}
    )
    detail = feature_overlap(_train_of_two(), test, "y", max_rows=2)[0].detail
    assert "in a 2-row sample of 4 rows" in detail


def test_near_duplicate_message_is_unchanged_below_the_cap():
    train = pd.DataFrame({"city": ["Paris"], "x": [1.0], "y": [0]})
    test = pd.DataFrame({"city": ["paris "], "x": [1.0], "y": [0]})
    detail = near_duplicate_overlap(train, test, "y")[0].detail
    assert "1 test rows (100.0%) match a train row after" in detail


def test_near_duplicate_reports_examined_and_declared_counts():
    train = pd.DataFrame({"city": ["Paris", "Lyon"], "x": [1.0, 2.0], "y": [0, 1]})
    test = pd.DataFrame({"city": ["paris "] * 4, "x": [1.0] * 4, "y": [0] * 4})
    finding = near_duplicate_overlap(train, test, "y", max_rows=2)[0]
    assert "2 of the 2 test rows examined match" in finding.detail
    assert "a deterministic sample of 4 test and 2 train rows" in finding.detail
    assert "%" not in finding.detail
    assert (
        finding.evidence[0].observed == "2 of 2 examined test rows matched (4 declared)"
    )


def test_sampling_disclosure_is_deterministic():
    test = pd.DataFrame(
        {"x": [2.0, 2.0, 2.0, 2.0], "city": ["Lyon"] * 4, "y": [1, 1, 1, 1]}
    )
    first = feature_overlap(_train_of_two(), test, "y", max_rows=2)[0]
    second = feature_overlap(_train_of_two(), test, "y", max_rows=2)[0]
    assert (first.kind, first.severity, first.detail) == (
        second.kind,
        second.severity,
        second.detail,
    )
