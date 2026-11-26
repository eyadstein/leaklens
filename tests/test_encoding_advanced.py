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
    assert advice("target_encode_after_split")


# Post-split target statistics. A statistic read from the frame the split
# divided still holds the held-out labels, so writing it into a feature after
# the split leaks exactly like the pre-split case. Statistics read from a
# train/test partition are legitimate and must stay silent.

POST_SPLIT_FULL_TRANSFORM = (
    "import pandas as pd\n"
    "from sklearn.model_selection import train_test_split\n"
    "df = pd.read_csv('d.csv')\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "df.drop(columns=['target']), df['target'], random_state=0)\n"
    "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
    "X = df.drop(columns=['target'])\n"
)

POST_SPLIT_TRAIN_TRANSFORM = (
    "import pandas as pd\n"
    "from sklearn.model_selection import train_test_split\n"
    "df = pd.read_csv('d.csv')\n"
    "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
    "train = train.copy()\n"
    "train['city_te'] = train.groupby('city')['target'].transform('mean')\n"
)


def test_post_split_transform_from_full_frame_is_high():
    fs = [
        f
        for f in scan_source(POST_SPLIT_FULL_TRANSFORM)
        if f.kind == "target_encode_after_split"
    ]
    assert fs and fs[0].severity == "high"
    assert "after the split" in fs[0].detail


def test_post_split_transform_from_train_only_is_clean():
    assert "target_encode_after_split" not in kinds(POST_SPLIT_TRAIN_TRANSFORM)


def test_post_split_transform_through_a_copy_alias_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "data = df.copy()\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "data['city_te'] = data.groupby('city')['target'].transform('mean')\n"
        "X = data.drop(columns=['target'])\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_post_split_transform_through_x_y_intermediates_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "X = df.drop(columns=['target'])\n"
        "y = df['target']\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_post_split_assign_column_is_detected():
    src = POST_SPLIT_FULL_TRANSFORM.replace(
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n",
        "df = df.assign(city_te=df.groupby('city')['target'].transform('mean'))\n",
    )
    assert "target_encode_after_split" in kinds(src)


def test_post_split_transform_through_a_groupby_object_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "g = df.groupby('city')['target']\n"
        "df['city_te'] = g.transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_aggregate_from_a_groupby_object_mapped_after_split_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "g = df.groupby('city')['target']\n"
        "means = g.mean()\n"
        "alias = means\n"
        "train['city_te'] = train['city'].map(alias)\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_groupby_object_on_train_only_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "g = train.groupby('city')['target']\n"
        "means = g.mean()\n"
        "train['city_te'] = train['city'].map(means)\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_map_result_that_is_not_written_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "means = df.groupby('city')['target'].mean()\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "print(train['city'].map(means))\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_merge_of_train_derived_aggregate_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "means = train.groupby('city')['target'].mean().rename('te').reset_index()\n"
        "train = train.merge(means, on='city', how='left')\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_pre_split_aggregate_mapped_after_split_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "means = df.groupby('city')['target'].mean()\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "train['city_te'] = train['city'].map(means)\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_post_split_full_frame_aggregate_mapped_into_a_partition_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "means = df.groupby('city')['target'].mean()\n"
        "train['city_te'] = train['city'].map(means)\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_full_frame_aggregate_merged_into_a_partition_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "means = df.groupby('city')['target'].mean().rename('te').reset_index()\n"
        "train = train.merge(means, on='city', how='left')\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_inline_full_frame_aggregate_mapped_after_split_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "train['city_te'] = train['city'].map("
        "df.groupby('city')['target'].mean())\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_partition_derived_aggregate_mapped_into_train_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "means = train.groupby('city')['target'].mean()\n"
        "train['city_te'] = train['city'].map(means)\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_post_split_target_summary_not_used_as_a_feature_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "print(df.groupby('city')['target'].mean())\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_post_split_transform_on_a_sampled_subset_is_clean():
    src = (
        "import pandas as pd\n"
        "df = pd.read_csv('d.csv')\n"
        "sub = df.sample(frac=0.8, random_state=0)\n"
        "sub = sub.copy()\n"
        "sub['city_te'] = sub.groupby('city')['target'].transform('mean')\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_post_split_transform_on_an_unrelated_frame_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "other = pd.read_csv('o.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "other['city_te'] = other.groupby('city')['target'].transform('mean')\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_post_split_detection_is_deterministic():
    first = [
        (f.kind, f.severity, f.column, f.detail)
        for f in scan_source(POST_SPLIT_FULL_TRANSFORM)
    ]
    second = [
        (f.kind, f.severity, f.column, f.detail)
        for f in scan_source(POST_SPLIT_FULL_TRANSFORM)
    ]
    assert first == second


# Precision: a full-frame statistic written into the frame the split divided
# only reaches a model when that frame is read again. A dead write or a pure
# display is analysis, not leakage. Partition writes stay flagged because the
# contaminated frame is the one used for training.


def test_dead_full_frame_target_write_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_displaying_the_encoded_frame_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
        "print(df)\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_reassigning_the_frame_is_not_consumption():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
        "df = pd.read_csv('other.csv')\n"
    )
    assert "target_encode_after_split" not in kinds(src)


def test_full_frame_target_write_then_resplit_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
        "train2, test2 = train_test_split(df, test_size=0.2, random_state=1)\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_full_frame_target_write_used_by_an_estimator_is_detected():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
        "model = LogisticRegression()\n"
        "model.fit(df[['city_te']], df['target'])\n"
    )
    assert "target_encode_after_split" in kinds(src)


def test_post_split_full_frame_group_stat_is_medium():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['price_mean'] = df.groupby('city')['price'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "group_stat_after_split"]
    assert fs and fs[0].severity == "medium"
    assert "after the split" in fs[0].detail


def test_post_split_group_stat_on_train_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "train['price_mean'] = train.groupby('city')['price'].transform('mean')\n"
        "X = train.drop(columns=['target'])\n"
    )
    assert "group_stat_after_split" not in kinds(src)


def test_post_split_group_stat_aggregate_then_map_is_medium():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "means = df.groupby('city')['price'].agg('mean')\n"
        "train['price_mean'] = train['city'].map(means)\n"
    )
    assert "group_stat_after_split" in kinds(src)


def test_post_split_group_stat_merge_is_medium():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "counts = df.groupby('city')['price'].count().rename('n').reset_index()\n"
        "train = train.merge(counts, on='city', how='left')\n"
    )
    assert "group_stat_after_split" in kinds(src)


def test_post_split_group_stat_from_train_aggregate_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "means = train.groupby('city')['price'].mean()\n"
        "train['price_mean'] = train['city'].map(means)\n"
    )
    assert "group_stat_after_split" not in kinds(src)


def test_dead_full_frame_group_stat_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['price_mean'] = df.groupby('city')['price'].transform('mean')\n"
    )
    assert "group_stat_after_split" not in kinds(src)


def test_target_stat_after_split_does_not_also_emit_group_stat():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_te'] = df.groupby('city')['target'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    found = kinds(src)
    assert "target_encode_after_split" in found
    assert "group_stat_after_split" not in found


def test_post_split_full_frame_frequency_is_low():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "counts = df['city'].value_counts()\n"
        "train['city_freq'] = train['city'].map(counts)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "frequency_encode_after_split"]
    assert fs and fs[0].severity == "low"
    assert "after the split" in fs[0].detail


def test_inline_post_split_frequency_is_low():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "train['city_freq'] = train['city'].map(df['city'].value_counts())\n"
    )
    assert "frequency_encode_after_split" in kinds(src)


def test_post_split_frequency_through_an_alias_is_low():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "counts = df['city'].value_counts()\n"
        "alias = counts\n"
        "train['city_freq'] = train['city'].map(alias)\n"
    )
    assert "frequency_encode_after_split" in kinds(src)


def test_post_split_frequency_from_train_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "counts = train['city'].value_counts()\n"
        "train['city_freq'] = train['city'].map(counts)\n"
    )
    assert "frequency_encode_after_split" not in kinds(src)


def test_dead_full_frame_frequency_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "df['city_freq'] = df['city'].map(df['city'].value_counts())\n"
    )
    assert "frequency_encode_after_split" not in kinds(src)


def test_group_shuffle_split_loop_reports_a_full_frame_group_stat():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "df = pd.read_csv('d.csv')\n"
        "gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)\n"
        "tr, te = next(gss.split(df, groups=df['g']))\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert kinds(src) == {"group_stat_after_split"}


def test_time_series_split_loop_reports_a_full_frame_target_stat():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import TimeSeriesSplit\n"
        "df = pd.read_csv('d.csv')\n"
        "ts = TimeSeriesSplit(n_splits=5)\n"
        "for tr, te in ts.split(df):\n"
        "    pass\n"
        "df['te'] = df.groupby('c')['target'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert kinds(src) == {"target_encode_after_split"}


def test_kfold_loop_reports_a_full_frame_frequency_encoding():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import KFold\n"
        "df = pd.read_csv('d.csv')\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=0)\n"
        "for tr, te in kf.split(df):\n"
        "    pass\n"
        "df['city_freq'] = df['city'].map(df['city'].value_counts())\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert kinds(src) == {"frequency_encode_after_split"}


def test_split_loop_through_an_alias_still_reports():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import StratifiedKFold\n"
        "df = pd.read_csv('d.csv')\n"
        "skf = StratifiedKFold(n_splits=5)\n"
        "folds = skf.split(df, df['target'])\n"
        "for tr, te in folds:\n"
        "    pass\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert kinds(src) == {"group_stat_after_split"}


def test_nested_split_loop_reports_once():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import GroupKFold\n"
        "df = pd.read_csv('d.csv')\n"
        "gkf = GroupKFold(n_splits=3)\n"
        "for tr, te in gkf.split(df, groups=df['g']):\n"
        "    for tr2, te2 in gkf.split(df, groups=df['g']):\n"
        "        pass\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert kinds(src) == {"group_stat_after_split"}


def test_split_loop_per_fold_statistics_are_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import KFold\n"
        "df = pd.read_csv('d.csv')\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=0)\n"
        "for tr, te in kf.split(df):\n"
        "    X_tr, X_te = df.iloc[tr], df.iloc[te]\n"
        "    X_tr['m'] = X_tr.groupby('c')['x'].transform('mean')\n"
    )
    assert not kinds(src) & {
        "group_stat_after_split",
        "frequency_encode_after_split",
        "target_encode_after_split",
    }


def test_split_loop_dead_full_frame_write_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import KFold\n"
        "df = pd.read_csv('d.csv')\n"
        "kf = KFold(n_splits=5)\n"
        "for tr, te in kf.split(df):\n"
        "    pass\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
    )
    assert "group_stat_after_split" not in kinds(src)


def test_split_loop_on_an_unrelated_frame_is_clean():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import KFold\n"
        "df = pd.read_csv('d.csv')\n"
        "other = pd.read_csv('o.csv')\n"
        "kf = KFold(n_splits=5)\n"
        "for tr, te in kf.split(df):\n"
        "    pass\n"
        "other['m'] = other.groupby('c')['x'].transform('mean')\n"
        "X = other.drop(columns=['target'])\n"
    )
    assert not kinds(src) & {
        "group_stat_after_split",
        "frequency_encode_after_split",
        "target_encode_after_split",
    }


def test_split_call_on_a_frame_alias_reports():
    # `X` is the same rows as `df`, so a fold split of `X` divides `df`.
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import KFold\n"
        "df = pd.read_csv('d.csv')\n"
        "X = df.drop(columns=['target'])\n"
        "kf = KFold(n_splits=5)\n"
        "for tr, te in kf.split(X):\n"
        "    pass\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
        "y = df['target']\n"
    )
    assert kinds(src) == {"group_stat_after_split"}


def test_split_loop_on_a_non_frame_input_is_clean():
    # The splitter divides an array, not the pandas frame, so the frame's
    # provenance is unrelated and the rule deliberately stays silent.
    src = (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import KFold\n"
        "df = pd.read_csv('d.csv')\n"
        "arr = np.load('x.npy')\n"
        "kf = KFold(n_splits=5)\n"
        "for tr, te in kf.split(arr):\n"
        "    pass\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    assert not kinds(src) & {
        "group_stat_after_split",
        "frequency_encode_after_split",
        "target_encode_after_split",
    }


def test_split_loop_feature_detection_is_deterministic():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import GroupKFold\n"
        "df = pd.read_csv('d.csv')\n"
        "gkf = GroupKFold(n_splits=3)\n"
        "for tr, te in gkf.split(df, groups=df['g']):\n"
        "    pass\n"
        "df['m'] = df.groupby('c')['x'].transform('mean')\n"
        "X = df.drop(columns=['target'])\n"
    )
    first = [(f.kind, f.severity, f.detail) for f in scan_source(src)]
    second = [(f.kind, f.severity, f.detail) for f in scan_source(src)]
    assert first == second


def test_post_split_feature_detection_is_deterministic():
    src = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
        "train['price_mean'] = train.groupby('city')['price'].transform('mean')\n"
        "counts = df['city'].value_counts()\n"
        "train['city_freq'] = train['city'].map(counts)\n"
    )
    first = [(f.kind, f.severity, f.detail) for f in scan_source(src)]
    second = [(f.kind, f.severity, f.detail) for f in scan_source(src)]
    assert first == second
