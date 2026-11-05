from leaklens.scanner import scan_source

S = "from sklearn.model_selection import train_test_split\n"


def kinds(src):
    return [f.kind for f in scan_source(src)]


def test_shuffle_false_not_flagged():
    assert "no_random_state" not in kinds(
        S + "a = train_test_split(X, y, shuffle=False)\n"
    )


def test_unseeded_split_still_flagged():
    assert "no_random_state" in kinds(S + "a = train_test_split(X, y)\n")


def test_shufflesplit_flagged_and_seeded_clean():
    imp = "from sklearn.model_selection import ShuffleSplit\n"
    assert "no_random_state" in kinds(imp + "ss = ShuffleSplit(3)\n")
    assert "no_random_state" not in kinds(
        imp + "ss = ShuffleSplit(3, random_state=0)\n"
    )
