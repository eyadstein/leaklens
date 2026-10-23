from leaklens.fixes import advice
from leaklens.scanner import scan_source

SPLIT = "from sklearn.model_selection import train_test_split\n"


def kinds(src):
    return {f.kind for f in scan_source(src)}


def test_tts_without_random_state_is_low():
    fs = scan_source(SPLIT + "a = train_test_split(X, y)\n")
    f = next(x for x in fs if x.kind == "no_random_state")
    assert f.severity == "low"


def test_tts_with_random_state_is_clean():
    assert not scan_source(SPLIT + "a = train_test_split(X, y, random_state=42)\n")


def test_tts_star_kwargs_not_flagged():
    # a ** expansion can carry random_state; the scanner must not assume absence
    assert "no_random_state" not in kinds(SPLIT + "opts = {'random_state': 0}\na = train_test_split(X, y, **opts)\n")


def test_kfold_shuffle_without_seed():
    fs = scan_source("from sklearn.model_selection import KFold\nkf = KFold(5, shuffle=True)\n")
    assert "no_random_state" in {f.kind for f in fs}


def test_kfold_shuffle_with_seed_clean():
    src = "from sklearn.model_selection import KFold\nkf = KFold(5, shuffle=True, random_state=42)\n"
    assert not scan_source(src)


def test_kfold_without_shuffle_is_clean():
    src = "from sklearn.model_selection import KFold\nkf = KFold(5)\n"
    assert not scan_source(src)


def test_stratified_kfold_shuffle_without_seed():
    src = "from sklearn.model_selection import StratifiedKFold\nskf = StratifiedKFold(n_splits=5, shuffle=True)\n"
    assert "no_random_state" in kinds(src)


def test_repeated_kfold_always_shuffles():
    src = "from sklearn.model_selection import RepeatedKFold\nrkf = RepeatedKFold(n_splits=5, n_repeats=2)\n"
    assert "no_random_state" in kinds(src)
    src2 = ("from sklearn.model_selection import RepeatedKFold\n"
            "rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)\n")
    assert "no_random_state" not in kinds(src2)


def test_module_attribute_call_form():
    src = "import sklearn.model_selection as ms\nkf = ms.KFold(5, shuffle=True)\n"
    assert "no_random_state" in kinds(src)


def test_combined_with_leak_reports_both():
    src = ("from sklearn.preprocessing import StandardScaler\n" + SPLIT +
           "X = StandardScaler().fit_transform(X)\na = train_test_split(X, y)\n")
    assert {"preprocess_before_split", "no_random_state"} <= kinds(src)


def test_fix_hint_registered():
    assert "random_state" in advice("no_random_state")
