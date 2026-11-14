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


def test_no_stratify_has_advice():
    assert "stratify" in advice("no_stratify")
