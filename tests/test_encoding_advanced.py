"""Target/frequency encoding and feature-selection leakage."""

from leaklens.fixes import advice
from leaklens.scanner import scan_source


def kinds(src):
    return {f.kind for f in scan_source(src)}


def test_target_mean_mapped_into_a_column_is_high():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "stats = df.groupby('city')['target'].mean()\n"
        "df['city_te'] = df['city'].map(stats)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df.drop(columns=['target']), df['target'], random_state=0)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "target_encode_before_split"]
    assert fs and fs[0].severity == "high"
    assert "target-derived" in fs[0].detail


def test_inline_groupby_target_mean_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df['city_te'] = df['city'].map(df.groupby('city')['target'].mean())\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df.drop(columns=['target']), df['target'], random_state=0)\n"
    )
    assert "target_encode_before_split" in kinds(src)


def test_target_encoder_before_split_is_detected():
    src = (
        "from category_encoders import TargetEncoder\n"
        "from sklearn.model_selection import train_test_split\n"
        "enc = TargetEncoder()\n"
        "X = enc.fit_transform(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "target_encode_before_split"]
    assert fs and fs[0].severity == "high"


def test_target_encoder_inside_pipeline_is_clean():
    src = (
        "from category_encoders import TargetEncoder\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "pipe = make_pipeline(TargetEncoder(), model)\n"
        "pipe.fit(X_train, y_train)\n"
    )
    assert "target_encode_before_split" not in kinds(src)


def test_frequency_encoding_before_split_is_low():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "freq = df['city'].value_counts()\n"
        "df['city_freq'] = df['city'].map(freq)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df.drop(columns=['target']), df['target'], random_state=0)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "frequency_encode_before_split"]
    assert fs and fs[0].severity == "low"


def test_frequency_encoding_on_train_only_is_clean():
    src = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "freq = X_train['city'].value_counts()\n"
        "X_train['city_freq'] = X_train['city'].map(freq)\n"
        "X_test['city_freq'] = X_test['city'].map(freq).fillna(0)\n"
    )
    assert "frequency_encode_before_split" not in kinds(src)


def test_target_relabelling_is_not_target_encoding():
    src = (
        "from sklearn.model_selection import train_test_split\n"
        "df['target'] = df['target'].replace({'yes': 1, 'no': 0})\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df.drop(columns=['target']), df['target'], random_state=0)\n"
    )
    assert "target_encode_before_split" not in kinds(src)


def test_frequency_summary_not_used_as_feature_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "summary = df['city'].value_counts()\n"
        "print(summary)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df.drop(columns=['target']), df['target'], random_state=0)\n"
    )
    assert "frequency_encode_before_split" not in kinds(src)


def test_rfe_before_split_is_feature_selection():
    src = (
        "from sklearn.feature_selection import RFE\n"
        "from sklearn.model_selection import train_test_split\n"
        "rfe = RFE(model, n_features_to_select=5)\n"
        "X = rfe.fit_transform(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "feature_selection_before_split"]
    assert fs and fs[0].severity == "high"


def test_select_k_best_before_split_is_feature_selection():
    src = (
        "from sklearn.feature_selection import SelectKBest\n"
        "from sklearn.model_selection import train_test_split\n"
        "X = SelectKBest(k=10).fit_transform(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
    )
    assert "feature_selection_before_split" in kinds(src)


def test_selector_fitted_on_train_only_is_clean():
    src = (
        "from sklearn.feature_selection import SelectKBest\n"
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "sel = SelectKBest(k=10)\n"
        "X_train = sel.fit_transform(X_train, y_train)\n"
        "X_test = sel.transform(X_test)\n"
    )
    assert "feature_selection_before_split" not in kinds(src)


def test_encoding_fix_hints_exist():
    assert advice("feature_selection_before_split")
    assert advice("frequency_encode_before_split")
