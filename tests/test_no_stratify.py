from leaklens.fixes import advice
from leaklens.scanner import scan_source

SPLIT = "from sklearn.model_selection import train_test_split\n"


def kinds(src):
    return {f.kind for f in scan_source(src)}


def test_tts_without_stratify_is_low():
    fs = scan_source(SPLIT + "a = train_test_split(X, y, random_state=42)\n")
    f = next(x for x in fs if x.kind == "no_stratify")
    assert f.severity == "low"
    assert "stratify" in f.detail


def test_tts_with_stratify_is_clean():
    assert "no_stratify" not in kinds(
        SPLIT + "a = train_test_split(X, y, random_state=42, stratify=y)\n"
    )


def test_tts_shuffle_false_not_flagged():
    # an ordered split is deliberate, and stratify is incompatible with it
    assert "no_stratify" not in kinds(
        SPLIT + "a = train_test_split(X, y, shuffle=False)\n"
    )


def test_tts_single_array_not_flagged():
    # no separate labels array to stratify on
    assert "no_stratify" not in kinds(
        SPLIT + "a = train_test_split(df, random_state=42)\n"
    )


def test_tts_star_kwargs_not_flagged():
    # a ** expansion can carry stratify; the scanner must not assume absence
    assert "no_stratify" not in kinds(
        SPLIT + "opts = {'stratify': y}\na = train_test_split(X, y, **opts)\n"
    )


def test_tts_multiple_arrays_flagged():
    assert "no_stratify" in kinds(
        SPLIT + "a = train_test_split(X, y, groups, random_state=42)\n"
    )


# --- advice must not claim a classification target without evidence ---

LINEAR = SPLIT + "from sklearn.linear_model import LinearRegression\n"


def _detail(src):
    return next(f for f in scan_source(src) if f.kind == "no_stratify").detail


def test_regression_file_gets_neutral_wording():
    detail = _detail(
        LINEAR
        + "model = LinearRegression()\n"
        + "a = train_test_split(X, y, random_state=42)\n"
        + "model.fit(a[0], a[2])\n"
    )
    assert "imbalanced classification target" not in detail
    assert "if the target is categorical" in detail
    assert "For a regression target, ignore this" in detail


def test_classifier_gets_classification_wording():
    detail = _detail(
        SPLIT
        + "from sklearn.linear_model import LogisticRegression\n"
        + "model = LogisticRegression()\n"
        + "a = train_test_split(X, y, random_state=42)\n"
        + "model.fit(a[0], a[2])\n"
    )
    assert "imbalanced classification target" in detail


def test_classifier_suffix_counts_as_evidence():
    detail = _detail(
        SPLIT
        + "from sklearn.ensemble import RandomForestClassifier\n"
        + "rf = RandomForestClassifier()\n"
        + "a = train_test_split(X, y, random_state=42)\n"
    )
    assert "imbalanced classification target" in detail


def test_classification_metric_counts_as_evidence():
    detail = _detail(
        SPLIT
        + "from sklearn.metrics import accuracy_score\n"
        + "a = train_test_split(X, y, random_state=42)\n"
        + "score = accuracy_score(y, pred)\n"
    )
    assert "imbalanced classification target" in detail


def test_no_model_gets_neutral_wording():
    detail = _detail(SPLIT + "a = train_test_split(X, y, random_state=42)\n")
    assert "imbalanced classification target" not in detail
    assert detail.startswith("train_test_split() without stratify")


def test_no_stratify_has_advice():
    assert "stratify" in advice("no_stratify")
