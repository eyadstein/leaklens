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
    assert "no_random_state" not in kinds(
        SPLIT + "opts = {'random_state': 0}\na = train_test_split(X, y, **opts)\n"
    )


def test_kfold_shuffle_without_seed():
    fs = scan_source(
        "from sklearn.model_selection import KFold\nkf = KFold(5, shuffle=True)\n"
    )
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
    src2 = (
        "from sklearn.model_selection import RepeatedKFold\n"
        "rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=7)\n"
    )
    assert "no_random_state" not in kinds(src2)


def test_module_attribute_call_form():
    src = "import sklearn.model_selection as ms\nkf = ms.KFold(5, shuffle=True)\n"
    assert "no_random_state" in kinds(src)


def test_combined_with_leak_reports_both():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        + SPLIT
        + "X = StandardScaler().fit_transform(X)\na = train_test_split(X, y)\n"
    )
    assert {"preprocess_before_split", "no_random_state"} <= kinds(src)


def test_fix_hint_registered():
    assert "random_state" in advice("no_random_state")


TORCH = "import torch\nfrom torch.utils.data import random_split\n"


def test_random_split_without_generator_is_low():
    fs = scan_source(TORCH + "tr, va = random_split(ds, [0.8, 0.2])\n")
    f = next(x for x in fs if x.kind == "no_random_state")
    assert f.severity == "low"
    assert "random_split" in f.detail


def test_random_split_attribute_call_flagged():
    src = "import torch\ntr, va = torch.utils.data.random_split(ds, [80, 20])\n"
    assert "no_random_state" in kinds(src)


def test_random_split_generator_none_flagged():
    src = TORCH + "tr, va = random_split(ds, [80, 20], generator=None)\n"
    assert "no_random_state" in kinds(src)


def test_random_split_with_generator_keyword_is_clean():
    src = (
        TORCH
        + "g = torch.Generator().manual_seed(42)\n"
        + "tr, va = random_split(ds, [80, 20], generator=g)\n"
    )
    assert "no_random_state" not in kinds(src)


def test_random_split_with_positional_generator_is_clean():
    # generator is the third positional parameter
    src = (
        TORCH
        + "tr, va = random_split(ds, [80, 20], torch.Generator().manual_seed(1))\n"
    )
    assert "no_random_state" not in kinds(src)


def test_random_split_after_global_torch_seed_is_clean():
    # random_split uses the global generator when none is passed, and manual_seed seeds it
    src = TORCH + "torch.manual_seed(0)\ntr, va = random_split(ds, [80, 20])\n"
    assert "no_random_state" not in kinds(src)


def test_random_split_after_seed_everything_is_clean():
    src = (
        TORCH
        + "from lightning import seed_everything\n"
        + "seed_everything(7)\ntr, va = random_split(ds, [80, 20])\n"
    )
    assert "no_random_state" not in kinds(src)


def test_random_split_generator_method_seed_is_not_global_seed():
    # seeding an unrelated generator does not make the global-RNG split reproducible
    src = (
        TORCH
        + "g = torch.Generator()\ng.manual_seed(0)\n"
        + "tr, va = random_split(ds, [80, 20])\n"
    )
    assert "no_random_state" in kinds(src)


def test_random_split_star_expansion_not_flagged():
    # *args / **kwargs can carry the generator; do not assume absence
    assert "no_random_state" not in kinds(TORCH + "a = random_split(*parts)\n")
    assert "no_random_state" not in kinds(
        TORCH + "opts = {'generator': g}\na = random_split(ds, [80, 20], **opts)\n"
    )


def test_random_split_without_torch_import_ignored():
    # an unrelated library's random_split must not be flagged
    assert "no_random_state" not in kinds("a = random_split(ds, [0.8, 0.2])\n")


def test_random_split_finding_advice_mentions_generator():
    assert "generator" in advice("no_random_state")
