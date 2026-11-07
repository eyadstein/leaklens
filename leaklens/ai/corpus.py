"""Labeled synthetic corpus for the optional learned detector.

Every case is written for this project; no benchmark case is reused for
training.  Cases are grouped by *template family* and grouped cross-validation
never puts two cases from the same family on opposite sides of a fold.  That
reduces, but cannot eliminate, the risk of learning surface syntax instead of
the underlying data flow.

Phase 5 splits the corpus by **scenario source** (a domain/idiom family), not by
random row: a whole source lives in exactly one split.  ``corpus.py`` and
``corpus_validation.py`` form the development corpus; ``corpus_test.py`` is the
locked independent test set and is never imported by the training code.

The corpus is intentionally transparent.  It is a regression corpus, not a
claim that the model understands arbitrary production code.
"""

from __future__ import annotations

from .corpus_phase6 import PHASE6_CASES
from .corpus_phase7 import PHASE7_CASES, PHASE7_HOLDOUT_CASES
from .corpus_types import HOLDOUT, LOCKED_TEST, TRAIN, VALIDATION, CorpusCase
from .corpus_validation import VALIDATION_CASES

_SPLIT = "from sklearn.model_selection import train_test_split\n"


_CASES: list[CorpusCase] = []


def _add(
    group: str,
    category: str,
    leaky: bool,
    name: str,
    code: str,
    *,
    split: str = TRAIN,
    source: str = "phase4-core",
) -> None:
    _CASES.append(CorpusCase(group, name, code, leaky, category, split, source))


# ---------------------------------------------------------------------------
# Leaky families.
# ---------------------------------------------------------------------------

_add(
    "scale_before_split",
    "preprocessing",
    True,
    "standard scaler before split",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "scaler = StandardScaler()\nX = scaler.fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)\n",
)
_add(
    "scale_before_split",
    "preprocessing",
    True,
    "minmax scaler fitted then transformed",
    "from sklearn.preprocessing import MinMaxScaler\n"
    + _SPLIT
    + "scale = MinMaxScaler()\nscale.fit(X)\nX = scale.transform(X)\n"
    "train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=3)\n",
)
_add(
    "scale_before_split",
    "preprocessing",
    True,
    "manual z score before split",
    "import numpy as np\n"
    + _SPLIT
    + "features = (features - np.mean(features)) / np.std(features)\n"
    "x_train, x_test, y_train, y_test = train_test_split(features, labels)\n",
)
_add(
    "scale_before_split",
    "preprocessing",
    True,
    "robust scaler before split",
    "from sklearn.preprocessing import RobustScaler\n"
    + _SPLIT
    + "robust = RobustScaler().fit(features)\nfeatures = robust.transform(features)\n"
    "a, b, c, d = train_test_split(features, labels, random_state=11)\n",
)
_add(
    "impute_before_split",
    "preprocessing",
    True,
    "simple imputer before split",
    "from sklearn.impute import SimpleImputer\n"
    + _SPLIT
    + "imputer = SimpleImputer(strategy='median')\nX = imputer.fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)\n",
)
_add(
    "impute_before_split",
    "preprocessing",
    True,
    "knn imputer before split",
    "from sklearn.impute import KNNImputer\n"
    + _SPLIT
    + "X = KNNImputer(n_neighbors=3).fit_transform(X)\n"
    "tr, te, ytr, yte = train_test_split(X, y, random_state=2)\n",
)
_add(
    "impute_before_split",
    "preprocessing",
    True,
    "column means before split",
    _SPLIT + "for column in df.columns:\n"
    "    df[column] = df[column].fillna(df[column].mean())\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, y)\n",
)
_add(
    "pca_before_split",
    "preprocessing",
    True,
    "pca before split",
    "from sklearn.decomposition import PCA\n"
    + _SPLIT
    + "components = PCA(n_components=10).fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(components, y, random_state=5)\n",
)
_add(
    "pca_before_split",
    "preprocessing",
    True,
    "svd before split",
    "from sklearn.decomposition import TruncatedSVD\n"
    + _SPLIT
    + "svd = TruncatedSVD(n_components=20)\nX = svd.fit_transform(X)\n"
    "train, test, y_train, y_test = train_test_split(X, y)\n",
)
_add(
    "pca_before_split",
    "preprocessing",
    True,
    "select k best before split",
    "from sklearn.feature_selection import SelectKBest, f_classif\n"
    + _SPLIT
    + "selector = SelectKBest(f_classif, k=12)\nX = selector.fit_transform(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)\n",
)
_add(
    "resample_before_split",
    "resampling",
    True,
    "smote before split",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "smote = SMOTE(random_state=4)\nX, y = smote.fit_resample(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)\n",
)
_add(
    "resample_before_split",
    "resampling",
    True,
    "adasyn before split",
    "from imblearn.over_sampling import ADASYN\n"
    + _SPLIT
    + "X, labels = ADASYN().fit_resample(X, labels)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, labels)\n",
)
_add(
    "resample_before_split",
    "resampling",
    True,
    "random oversampler before split",
    "from imblearn.over_sampling import RandomOverSampler\n"
    + _SPLIT
    + "sampler = RandomOverSampler()\nfeatures, target = sampler.fit_resample(features, target)\n"
    "train_x, test_x, train_y, test_y = train_test_split(features, target, random_state=0)\n",
)
_add(
    "target_encode_before_split",
    "target_encoding",
    True,
    "target encoder on all rows",
    "from category_encoders import TargetEncoder\n"
    + _SPLIT
    + "encoder = TargetEncoder()\nX = encoder.fit_transform(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)\n",
)
_add(
    "target_encode_before_split",
    "target_encoding",
    True,
    "group mean target encoding",
    _SPLIT + "means = df.groupby('city')['target'].mean()\n"
    "df['city_encoded'] = df['city'].map(means)\n"
    "X = df[['city_encoded', 'age']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['target'], random_state=6)\n",
)
_add(
    "target_encode_before_split",
    "target_encoding",
    True,
    "weight of evidence before split",
    _SPLIT + "def woe(frame, column, target):\n"
    "    return frame.groupby(column)[target].mean()\n"
    "df['city_woe'] = df['city'].map(woe(df, 'city', 'label'))\n"
    "X_train, X_test, y_train, y_test = train_test_split(df[['city_woe']], df['label'])\n",
)
_add(
    "test_concat",
    "split",
    True,
    "train and test concatenated then split",
    "import pandas as pd\n" + _SPLIT + "all_rows = pd.concat([train, test])\n"
    "X_train, X_test, y_train, y_test = train_test_split(all_rows, y, random_state=8)\n",
)
_add(
    "test_concat",
    "split",
    True,
    "test rows appended before split",
    "import pandas as pd\n"
    + _SPLIT
    + "combined = pd.concat([df_train, df_test], ignore_index=True)\n"
    "train_x, test_x, train_y, test_y = train_test_split(combined, combined['y'])\n",
)
_add(
    "test_concat",
    "split",
    True,
    "near duplicate test row kept in training",
    "import pandas as pd\n" + _SPLIT + "train = pd.concat([train, test.sample(50)])\n"
    "X_train, X_test, y_train, y_test = train_test_split(train, train['label'])\n",
)
_add(
    "target_feature",
    "target_derived",
    True,
    "target copied into feature",
    _SPLIT + "df['score'] = df['target'].map({0: 10, 1: 90})\n"
    "X = df[['age', 'score']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['target'])\n",
)
_add(
    "target_feature",
    "target_derived",
    True,
    "label arithmetic feature",
    _SPLIT + "X['adjusted'] = y * 2 + 1\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)\n",
)
_add(
    "target_feature",
    "target_derived",
    True,
    "outcome encoded as feature",
    _SPLIT + "features['converted_score'] = df['converted'] / df['clicks']\n"
    "X_train, X_test, y_train, y_test = train_test_split(features, df['converted'])\n",
)
_add(
    "future_data",
    "temporal",
    True,
    "backward fill before split",
    "import pandas as pd\n" + _SPLIT + "df = df.sort_values('date')\n"
    "df['revenue'] = df['revenue'].bfill()\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, df['y'], random_state=1)\n",
)
_add(
    "future_data",
    "temporal",
    True,
    "centered rolling average",
    _SPLIT + "feature = df['sales'].rolling(7, center=True).mean()\n"
    "X_train, X_test, y_train, y_test = train_test_split(feature, df['y'])\n",
)
_add(
    "future_data",
    "temporal",
    True,
    "negative shift feature",
    _SPLIT + "df['next_demand'] = df['demand'].shift(-1)\n"
    "X_train, X_test, y_train, y_test = train_test_split(df[['next_demand']], df['y'])\n",
)
_add(
    "future_data",
    "temporal",
    True,
    "future values joined by date",
    _SPLIT + "future = pd.read_csv('future_orders.csv')\n"
    "df = df.merge(future, on='date', how='left')\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, df['y'])\n",
)
_add(
    "group_random_split",
    "grouping",
    True,
    "random rows for repeated customers",
    _SPLIT + "X = df[['customer_id', 'spend', 'visits']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['churn'], random_state=0)\n",
)
_add(
    "group_random_split",
    "grouping",
    True,
    "kfold over accounts",
    "from sklearn.model_selection import KFold, cross_val_score\n"
    "scores = cross_val_score(model, X, df['churn'], cv=KFold(5, shuffle=True))\n",
)
_add(
    "group_random_split",
    "grouping",
    True,
    "patient rows split without group",
    _SPLIT + "train, test = train_test_split(patient_rows, random_state=4)\n"
    "model.fit(train.drop(columns=['patient_id']), train['readmitted'])\n",
)
_add(
    "temporal_random_split",
    "temporal",
    True,
    "shuffled time rows",
    _SPLIT + "X = df[['month', 'orders']]\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "X, y, shuffle=True, random_state=0)\n",
)
_add(
    "temporal_random_split",
    "temporal",
    True,
    "sort then shuffle",
    "import pandas as pd\n" + _SPLIT + "ordered = df.sort_values('timestamp')\n"
    "X_train, X_test, y_train, y_test = train_test_split(ordered, ordered['y'], shuffle=True)\n",
)
_add(
    "temporal_random_split",
    "temporal",
    True,
    "cross validation across time",
    "from sklearn.model_selection import cross_val_score\n"
    "scores = cross_val_score(model, X, y, cv=5)\n",
)
_add(
    "tuning_before_split",
    "tuning",
    True,
    "grid search on full data",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "search = GridSearchCV(model, {'C': [0.1, 1.0]}, cv=5)\n"
    "search.fit(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n",
)
_add(
    "tuning_before_split",
    "tuning",
    True,
    "feature selection cross validated on all rows",
    "from sklearn.feature_selection import RFECV\n"
    + _SPLIT
    + "selector = RFECV(model, cv=5).fit(X, y)\n"
    "X = selector.transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
)
_add(
    "tuning_before_split",
    "tuning",
    True,
    "cross validation used to pick features then test",
    "from sklearn.model_selection import cross_val_score\n"
    + _SPLIT
    + "scores = cross_val_score(model, X, y, cv=10)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
)
_add(
    "identifier_feature",
    "identifier",
    True,
    "customer id kept as feature",
    _SPLIT + "X = df[['customer_id', 'amount']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['fraud'], random_state=0)\n",
)
_add(
    "identifier_feature",
    "identifier",
    True,
    "row hash used as feature",
    _SPLIT + "df['row_hash'] = df.apply(lambda row: hash(tuple(row)), axis=1)\n"
    "X = df[['row_hash', 'amount']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['label'])\n",
)
_add(
    "leakage_column",
    "target_leakage",
    True,
    "leak column included",
    _SPLIT + "X = df[['age', 'leak_score']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['target'])\n",
)
_add(
    "leakage_column",
    "target_leakage",
    True,
    "ground truth copied",
    _SPLIT + "df['ground_truth_copy'] = df['ground_truth']\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, df['label'])\n",
)
_add(
    "leakage_column",
    "target_leakage",
    True,
    "final decision as feature",
    _SPLIT + "X = df[['income', 'final_decision']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['approved'])\n",
)
_add(
    "aggregate_before_split",
    "preprocessing",
    True,
    "group transform before split",
    _SPLIT
    + "df['customer_mean'] = df.groupby('customer_id')['spend'].transform('mean')\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, df['y'])\n",
)
_add(
    "aggregate_before_split",
    "preprocessing",
    True,
    "full frame pivot before split",
    "import pandas as pd\n"
    + _SPLIT
    + "pivot = df.pivot_table(index='user', columns='day', values='clicks')\n"
    "X_train, X_test, y_train, y_test = train_test_split(pivot, df['converted'])\n",
)
_add(
    "duplicate_overlap",
    "split",
    True,
    "duplicates removed after split",
    _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(df, df['y'], random_state=0)\n"
    "X_train = X_train.drop_duplicates()\n",
)


# ---------------------------------------------------------------------------
# Legitimate families.
# ---------------------------------------------------------------------------

_add(
    "safe_scale_after_split",
    "preprocessing",
    False,
    "standard scaler fitted on train",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)\n"
    "scaler = StandardScaler().fit(X_train)\n"
    "X_train = scaler.transform(X_train)\nX_test = scaler.transform(X_test)\n",
)
_add(
    "safe_scale_after_split",
    "preprocessing",
    False,
    "minmax in a pipeline",
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.preprocessing import MinMaxScaler\n"
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)\n"
    "model = make_pipeline(MinMaxScaler(), LogisticRegression())\n"
    "model.fit(X_train, y_train)\nprint(model.score(X_test, y_test))\n",
)
_add(
    "safe_scale_after_split",
    "preprocessing",
    False,
    "manual train statistics",
    "import numpy as np\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)\n"
    "mean = X_train.mean()\nstd = X_train.std()\n"
    "X_train = (X_train - mean) / std\nX_test = (X_test - mean) / std\n",
)
_add(
    "safe_impute_after_split",
    "preprocessing",
    False,
    "imputer fitted on train",
    "from sklearn.impute import SimpleImputer\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)\n"
    "imputer = SimpleImputer(strategy='median').fit(X_train)\n"
    "X_train = imputer.transform(X_train)\nX_test = imputer.transform(X_test)\n",
)
_add(
    "safe_impute_after_split",
    "preprocessing",
    False,
    "train column means only",
    _SPLIT + "train, test = train_test_split(df, random_state=1)\n"
    "for column in train.columns:\n"
    "    fill = train[column].mean()\n"
    "    train[column] = train[column].fillna(fill)\n"
    "    test[column] = test[column].fillna(fill)\n",
)
_add(
    "safe_pca_after_split",
    "preprocessing",
    False,
    "pca fitted on train",
    "from sklearn.decomposition import PCA\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)\n"
    "pca = PCA(n_components=5).fit(X_train)\n"
    "X_train = pca.transform(X_train)\nX_test = pca.transform(X_test)\n",
)
_add(
    "safe_pca_after_split",
    "preprocessing",
    False,
    "feature selection inside pipeline",
    "from sklearn.feature_selection import SelectKBest\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)\n"
    "pipe = Pipeline([('select', SelectKBest(k=10)), ('model', LogisticRegression())])\n"
    "pipe.fit(X_train, y_train)\nprint(pipe.score(X_test, y_test))\n",
)
_add(
    "safe_resample_train_only",
    "resampling",
    False,
    "smote after split on train",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)\n"
    "X_train, y_train = SMOTE(random_state=4).fit_resample(X_train, y_train)\n"
    "model.fit(X_train, y_train)\n",
)
_add(
    "safe_resample_train_only",
    "resampling",
    False,
    "sampler only in training branch",
    "from imblearn.pipeline import Pipeline\n"
    "from imblearn.over_sampling import RandomOverSampler\n"
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)\n"
    "pipe = Pipeline([('sample', RandomOverSampler()), ('model', LogisticRegression())])\n"
    "pipe.fit(X_train, y_train)\n",
)
_add(
    "safe_target_encoding",
    "target_encoding",
    False,
    "target encoder inside cross validation",
    "from sklearn.model_selection import cross_val_score\n"
    "from category_encoders import TargetEncoder\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "pipe = Pipeline([('encode', TargetEncoder()), ('model', LogisticRegression())])\n"
    "scores = cross_val_score(pipe, X, y, cv=5)\n",
)
_add(
    "safe_target_encoding",
    "target_encoding",
    False,
    "target means computed on train",
    _SPLIT + "train, test = train_test_split(df, random_state=6)\n"
    "means = train.groupby('city')['target'].mean()\n"
    "train['city_encoded'] = train['city'].map(means)\n"
    "test['city_encoded'] = test['city'].map(means)\n",
)
_add(
    "safe_group_split",
    "grouping",
    False,
    "group shuffle split",
    "from sklearn.model_selection import GroupShuffleSplit\n"
    "splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)\n"
    "train_idx, test_idx = next(splitter.split(X, y, groups=df['customer_id']))\n"
    "model.fit(X.iloc[train_idx], y.iloc[train_idx])\n",
)
_add(
    "safe_group_split",
    "grouping",
    False,
    "group k fold",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "scores = cross_val_score(model, X, y, cv=GroupKFold(5), groups=df['account_id'])\n",
)
_add(
    "safe_group_split",
    "grouping",
    False,
    "stratified group k fold",
    "from sklearn.model_selection import StratifiedGroupKFold, cross_val_score\n"
    "cv = StratifiedGroupKFold(n_splits=5)\n"
    "scores = cross_val_score(model, X, y, cv=cv, groups=df['patient_id'])\n",
)
_add(
    "safe_temporal_split",
    "temporal",
    False,
    "sorted chronological split",
    "import pandas as pd\n" + _SPLIT + "ordered = df.sort_values('timestamp')\n"
    "X_train, X_test, y_train, y_test = train_test_split(ordered, ordered['y'], shuffle=False)\n",
)
_add(
    "safe_temporal_split",
    "temporal",
    False,
    "time series split",
    "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\n"
    "scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))\n",
)
_add(
    "safe_temporal_split",
    "temporal",
    False,
    "trailing rolling feature",
    _SPLIT + "df['rolling_mean'] = df['sales'].rolling(7).mean()\n"
    "X_train, X_test, y_train, y_test = train_test_split(df[['rolling_mean']], df['y'])\n",
)
_add(
    "safe_pipeline_cv",
    "tuning",
    False,
    "grid search inside training split",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "search = GridSearchCV(model, {'C': [0.1, 1.0]}, cv=5)\n"
    "search.fit(X_train, y_train)\nprint(search.score(X_test, y_test))\n",
)
_add(
    "safe_pipeline_cv",
    "tuning",
    False,
    "cross validation on training rows",
    "from sklearn.model_selection import cross_val_score\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
    "model.fit(X_train, y_train)\n",
)
_add(
    "safe_pipeline_cv",
    "tuning",
    False,
    "column transformer inside pipeline",
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "pre = ColumnTransformer([('num', StandardScaler(), ['age']), "
    "('cat', OneHotEncoder(), ['city'])])\n"
    "pipe = Pipeline([('pre', pre), ('model', model)])\n"
    "pipe.fit(X_train, y_train)\n",
)
_add(
    "safe_features",
    "identifier",
    False,
    "identifier dropped",
    _SPLIT + "X = df.drop(columns=['customer_id', 'row_hash'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, df['fraud'], random_state=0)\n",
)
_add(
    "safe_features",
    "identifier",
    False,
    "identifier used only for grouping",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "features = df.drop(columns=['customer_id'])\n"
    "scores = cross_val_score(model, features, df['churn'], cv=GroupKFold(5), "
    "groups=df['customer_id'])\n",
)
_add(
    "safe_duplicates",
    "split",
    False,
    "deduplicate before split",
    _SPLIT + "df = df.drop_duplicates()\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, df['y'], random_state=0)\n",
)
_add(
    "safe_duplicates",
    "split",
    False,
    "explicit separate frames",
    _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)\n"
    "model.fit(X_train, y_train)\nprint(model.score(X_test, y_test))\n",
)
_add(
    "safe_misc",
    "split",
    False,
    "target dropped explicitly",
    _SPLIT + "features = df.drop(columns=['target'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(features, df['target'], random_state=0)\n",
)
_add(
    "safe_misc",
    "split",
    False,
    "train test split with stratify",
    "from sklearn.model_selection import train_test_split\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, "
    "stratify=y, random_state=42)\n",
)
_add(
    "safe_misc",
    "temporal",
    False,
    "shifted target horizon",
    _SPLIT + "df['next_target'] = df['target'].shift(-1)\n"
    "df = df.dropna()\n"
    "X_train, X_test, y_train, y_test = train_test_split(df, df['next_target'], shuffle=False)\n",
)
_add(
    "safe_misc",
    "preprocessing",
    False,
    "fill with training median",
    "from sklearn.impute import SimpleImputer\n"
    + _SPLIT
    + "train, test = train_test_split(df, random_state=1)\n"
    "imputer = SimpleImputer(strategy='median').fit(train)\n"
    "train = imputer.transform(train)\ntest = imputer.transform(test)\n",
)
_add(
    "safe_misc",
    "preprocessing",
    False,
    "documented no leakage",
    _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "# all transformers are fitted on X_train only, never on the test split\n"
    "train_stats = X_train.describe()\n",
)


# ---------------------------------------------------------------------------
# Phase 5 training sources.  Each source is one authored scenario family and
# lives entirely in the training split.
# ---------------------------------------------------------------------------

_add(
    "t5_quantile_full",
    "preprocessing",
    True,
    "quantile transform before partition",
    "from sklearn.preprocessing import QuantileTransformer\n"
    + _SPLIT
    + "qt = QuantileTransformer(n_quantiles=100)\n"
    "train_x = qt.fit_transform(train_x)\n"
    "a, b, c, d = train_test_split(train_x, train_y, random_state=12)\n",
    source="train:sklearn",
)
_add(
    "t5_power_full",
    "preprocessing",
    True,
    "power transform on the full matrix",
    "from sklearn.preprocessing import PowerTransformer\n"
    + _SPLIT
    + "pt = PowerTransformer()\n"
    "matrix = pt.fit_transform(matrix)\n"
    "x_tr, x_te, y_tr, y_te = train_test_split(matrix, labels)\n",
    source="train:sklearn",
)
_add(
    "t5_iterative_full",
    "preprocessing",
    True,
    "iterative imputer on the full table",
    "from sklearn.experimental import enable_iterative_imputer\n"
    "from sklearn.impute import IterativeImputer\n"
    + _SPLIT
    + "filled = IterativeImputer(max_iter=10).fit_transform(table)\n"
    "x_tr, x_te, y_tr, y_te = train_test_split(filled, table['outcome'])\n",
    source="train:sklearn",
)
_add(
    "t5_kernel_pca_full",
    "preprocessing",
    True,
    "kernel pca before the holdout",
    "from sklearn.decomposition import KernelPCA\n"
    + _SPLIT
    + "embedding = KernelPCA(n_components=8, kernel='rbf').fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(embedding, y)\n",
    source="train:sklearn",
)
_add(
    "t5_mutual_info_full",
    "preprocessing",
    True,
    "mutual information selection before the holdout",
    "from sklearn.feature_selection import SelectKBest, mutual_info_classif\n"
    + _SPLIT
    + "keep = SelectKBest(mutual_info_classif, k=15).fit_transform(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(keep, y)\n",
    source="train:sklearn",
)
_add(
    "t5_full_copy_fit",
    "preprocessing",
    True,
    "scaler fitted on a full-frame copy after the split",
    "from sklearn.preprocessing import StandardScaler\n" + _SPLIT + "raw = X.copy()\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "sc = StandardScaler()\nsc.fit(raw)\n",
    source="train:sklearn",
)
_add(
    "t5_scale_test_partition",
    "preprocessing",
    True,
    "scaler fitted on the test partition",
    "from sklearn.preprocessing import MinMaxScaler\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "scale = MinMaxScaler()\nX_test = scale.fit_transform(X_test)\n",
    source="train:sklearn",
)
_add(
    "t5_quantile_train",
    "preprocessing",
    False,
    "quantile transform fitted on the train partition",
    "from sklearn.preprocessing import QuantileTransformer\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)\n"
    "qt = QuantileTransformer(n_quantiles=100).fit(X_train)\n"
    "X_train = qt.transform(X_train)\nX_test = qt.transform(X_test)\n",
    source="train:sklearn",
)
_add(
    "t5_iterative_train",
    "preprocessing",
    False,
    "iterative imputer fitted on the train partition",
    "from sklearn.experimental import enable_iterative_imputer\n"
    "from sklearn.impute import IterativeImputer\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "imp = IterativeImputer(max_iter=10).fit(X_train)\n"
    "X_train = imp.transform(X_train)\nX_test = imp.transform(X_test)\n",
    source="train:sklearn",
)
_add(
    "t5_kernel_pca_train",
    "preprocessing",
    False,
    "kernel pca fitted on the train partition",
    "from sklearn.decomposition import KernelPCA\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "kp = KernelPCA(n_components=6, kernel='rbf').fit(X_train)\n"
    "X_train = kp.transform(X_train)\nX_test = kp.transform(X_test)\n",
    source="train:sklearn",
)
_add(
    "t5_mutual_info_pipeline",
    "preprocessing",
    False,
    "mutual information selection inside a pipeline",
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.feature_selection import SelectKBest, mutual_info_classif\n"
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "pipe = Pipeline([('select', SelectKBest(mutual_info_classif, k=10)), "
    "('model', LogisticRegression())])\n"
    "pipe.fit(X_train, y_train)\n",
    source="train:sklearn",
)
_add(
    "t5_column_transformer_train",
    "preprocessing",
    False,
    "column transformer fitted after the split",
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "pre = ColumnTransformer([('num', StandardScaler(), ['age']), "
    "('cat', OneHotEncoder(), ['city'])])\n"
    "pipe = Pipeline([('pre', pre), ('model', model)])\n"
    "pipe.fit(X_train, y_train)\n",
    source="train:sklearn",
)
_add(
    "t5_ohe_before_split",
    "preprocessing",
    False,
    "one hot vocabulary before the split is harmless",
    "from sklearn.preprocessing import OneHotEncoder\n"
    + _SPLIT
    + "encoder = OneHotEncoder(handle_unknown='ignore')\n"
    "encoded = encoder.fit_transform(categories)\n"
    "X_train, X_test, y_train, y_test = train_test_split(encoded, y)\n",
    source="train:sklearn",
)
_add(
    "t5_label_target",
    "preprocessing",
    False,
    "label encoder on the target before the split",
    "from sklearn.preprocessing import LabelEncoder\n"
    + _SPLIT
    + "le = LabelEncoder()\n"
    "y = le.fit_transform(y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)\n",
    source="train:sklearn",
)

_add(
    "t5_zip_target_mean",
    "target_encoding",
    True,
    "target mean by zip before the partition",
    _SPLIT + "zip_means = frame.groupby('zip')['churn'].mean()\n"
    "frame['zip_score'] = frame['zip'].map(zip_means)\n"
    "X = frame[['zip_score', 'tenure']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['churn'])\n",
    source="train:pandas",
)
_add(
    "t5_region_target_mean",
    "target_encoding",
    True,
    "smoothed target mean mapped on the full frame",
    _SPLIT + "smooth = frame.groupby('region')['defaulted'].agg('mean')\n"
    "frame['region_rate'] = frame['region'].map(smooth)\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['defaulted'])\n",
    source="train:pandas",
)
_add(
    "t5_account_median_full",
    "preprocessing",
    True,
    "median spend by account before the partition",
    _SPLIT
    + "frame['account_median'] = frame.groupby('account_id')['spend'].transform('median')\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['churn'])\n",
    source="train:pandas",
)
_add(
    "t5_pivot_full",
    "preprocessing",
    True,
    "pivot table built before the partition",
    "import pandas as pd\n"
    + _SPLIT
    + "daily = frame.pivot_table(index='session', columns='day', values='clicks')\n"
    "X_train, X_test, y_train, y_test = train_test_split(daily, frame['purchased'])\n",
    source="train:pandas",
)
_add(
    "t5_refund_feature",
    "target_derived",
    True,
    "refund feature derived from the outcome",
    _SPLIT + "frame['refund_total'] = frame['refunded'] * frame['refund_fee']\n"
    "X = frame[['refund_total', 'age']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['refunded'])\n",
    source="train:pandas",
)
_add(
    "t5_final_status_feature",
    "target_leakage",
    True,
    "final status used as a predictor",
    _SPLIT
    + "frame['final_status'] = frame['approved'].map({0: 'declined', 1: 'accepted'})\n"
    "X = frame[['income', 'final_status']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['approved'])\n",
    source="train:pandas",
)
_add(
    "t5_duplicates_after",
    "split",
    True,
    "duplicate rows pruned after the partition",
    _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(frame, frame['label'])\n"
    "X_train = X_train.drop_duplicates()\n",
    source="train:pandas",
)
_add(
    "t5_zip_target_mean_train",
    "target_encoding",
    False,
    "target mean fitted on the train partition",
    _SPLIT + "train, test = train_test_split(frame, random_state=6)\n"
    "means = train.groupby('zip')['churn'].mean()\n"
    "train['zip_score'] = train['zip'].map(means)\n"
    "test['zip_score'] = test['zip'].map(means)\n",
    source="train:pandas",
)
_add(
    "t5_account_median_train",
    "preprocessing",
    False,
    "account median learned on the train partition",
    "from sklearn.model_selection import GroupShuffleSplit\n"
    + _SPLIT
    + "splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=3)\n"
    "train_idx, test_idx = next(splitter.split(frame, groups=frame['account_id']))\n"
    "train = frame.iloc[train_idx]\ntest = frame.iloc[test_idx]\n"
    "medians = train.groupby('account_id')['spend'].median()\n"
    "train['account_median'] = train['account_id'].map(medians)\n"
    "test['account_median'] = test['account_id'].map(medians)\n",
    source="train:pandas",
)
_add(
    "t5_next_target",
    "temporal",
    False,
    "shifted target horizon",
    _SPLIT + "frame['next_churn'] = frame['churn'].shift(-1)\n"
    "frame = frame.dropna()\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "frame, frame['next_churn'], shuffle=False)\n",
    source="train:pandas",
)
_add(
    "t5_group_summary_only",
    "split",
    False,
    "group summary printed but never used",
    "summary = frame.groupby('region')['revenue'].agg('mean')\n"
    "print(summary.head())\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split("
    "frame.drop(columns=['churn']), frame['churn'])\n",
    source="train:pandas",
)
_add(
    "t5_dedupe_before",
    "split",
    False,
    "duplicates removed before the partition",
    _SPLIT + "frame = frame.drop_duplicates(subset=['customer_id', 'month'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['churn'])\n",
    source="train:pandas",
)
_add(
    "t5_target_recode",
    "split",
    False,
    "target recoded before the partition",
    _SPLIT + "frame['churn'] = frame['churn'].map({'yes': 1, 'no': 0})\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "frame.drop(columns=['churn']), frame['churn'])\n",
    source="train:pandas",
)

_add(
    "t5_shuffled_months",
    "temporal",
    True,
    "shuffled month rows",
    _SPLIT + "X = frame[['month', 'demand']]\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "X, y, shuffle=True, random_state=1)\n",
    source="train:timeseries",
)
_add(
    "t5_kfold_daily",
    "temporal",
    True,
    "kfold across daily demand",
    "from sklearn.model_selection import KFold, cross_val_score\n"
    "X = frame[['day', 'temperature']]\n"
    "scores = cross_val_score(model, X, frame['sales'], cv=KFold(4))\n",
    source="train:timeseries",
)
_add(
    "t5_backfill_sensor",
    "temporal",
    True,
    "backfill sensor values before the partition",
    _SPLIT + "frame = frame.sort_values('date')\n"
    "frame['inventory'] = frame['inventory'].bfill()\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['sales'])\n",
    source="train:timeseries",
)
_add(
    "t5_centered_rolling",
    "temporal",
    True,
    "centered rolling average over the full series",
    _SPLIT + "frame['smooth'] = frame['sales'].rolling(14, center=True).mean()\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['sales'])\n",
    source="train:timeseries",
)
_add(
    "t5_negative_shift",
    "temporal",
    True,
    "negative lag of future orders",
    _SPLIT + "frame['next_orders'] = frame['orders'].shift(-2)\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['sales'])\n",
    source="train:timeseries",
)
_add(
    "t5_future_merge",
    "temporal",
    True,
    "future invoices joined by month",
    "import pandas as pd\n"
    + _SPLIT
    + "invoices = pd.read_csv('next_month_invoices.csv')\n"
    "frame = frame.merge(invoices, on='month', how='left')\n"
    "X_train, X_test, y_train, y_test = train_test_split(frame, frame['sales'])\n",
    source="train:timeseries",
)
_add(
    "t5_chronological",
    "temporal",
    False,
    "chronological split without shuffling",
    _SPLIT + "ordered = frame.sort_values('date')\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "ordered, ordered['sales'], shuffle=False)\n",
    source="train:timeseries",
)
_add(
    "safe_temporal_split",
    "temporal",
    False,
    "time series split on dates",
    "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\n"
    "scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=4))\n",
    source="train:timeseries",
)
_add(
    "t5_trailing_window",
    "temporal",
    False,
    "trailing rolling window",
    _SPLIT + "frame['trailing_7'] = frame['sales'].shift(1).rolling(7).mean()\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "frame[['trailing_7']], frame['sales'], shuffle=False)\n",
    source="train:timeseries",
)
_add(
    "t5_sorted_holdout",
    "temporal",
    False,
    "sorted holdout on event time",
    "ordered = frame.sort_values('event_time')\n"
    "cut = int(len(ordered) * 0.8)\n"
    "train, test = ordered.iloc[:cut], ordered.iloc[cut:]\n",
    source="train:timeseries",
)

_add(
    "t5_store_random_split",
    "grouping",
    True,
    "random split over repeated stores",
    _SPLIT + "X = frame[['store_id', 'spend', 'visits']]\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "X, frame['churn'], random_state=3)\n",
    source="train:groups",
)
_add(
    "t5_device_kfold",
    "grouping",
    True,
    "kfold over device events",
    "from sklearn.model_selection import KFold, cross_val_score\n"
    "X = frame.drop(columns=['device_id'])\n"
    "scores = cross_val_score(model, X, frame['failed'], cv=KFold(5))\n",
    source="train:groups",
)
_add(
    "t5_session_rows",
    "grouping",
    True,
    "session rows split without a group key",
    _SPLIT + "rows = events[['session_id', 'clicks']]\n"
    "train, test = train_test_split(rows, random_state=2)\n"
    "model.fit(train.drop(columns=['session_id']), train['converted'])\n",
    source="train:groups",
)
_add(
    "t5_group_shuffle",
    "grouping",
    False,
    "group shuffle split on store id",
    "from sklearn.model_selection import GroupShuffleSplit\n"
    "splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)\n"
    "train_idx, test_idx = next(splitter.split(X, y, groups=frame['store_id']))\n",
    source="train:groups",
)
_add(
    "t5_group_kfold",
    "grouping",
    False,
    "group kfold on device id",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "scores = cross_val_score(model, X, y, cv=GroupKFold(5), groups=frame['device_id'])\n",
    source="train:groups",
)
_add(
    "t5_group_loop",
    "grouping",
    False,
    "group kfold loop over accounts",
    "from sklearn.model_selection import GroupKFold\n"
    "for train_idx, test_idx in GroupKFold(n_splits=4).split("
    "X, y, groups=frame['account_id']):\n"
    "    model.fit(X.iloc[train_idx], y.iloc[train_idx])\n",
    source="train:groups",
)
_add(
    "t5_identifier_dropped",
    "identifier",
    False,
    "identifier columns dropped before modeling",
    _SPLIT + "features = frame.drop(columns=['customer_id', 'session_id'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(features, frame['churn'])\n",
    source="train:groups",
)
_add(
    "t5_identifier_grouping_only",
    "identifier",
    False,
    "merchant id used only for grouping",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "features = frame.drop(columns=['merchant_id'])\n"
    "scores = cross_val_score(model, features, frame['fraud'], "
    "cv=GroupKFold(5), groups=frame['merchant_id'])\n",
    source="train:groups",
)

_add(
    "t5_nb_scale_order",
    "preprocessing",
    True,
    "scale cell runs before the split cell",
    "# In[3]:\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "scaled = StandardScaler().fit_transform(features)\n"
    "# In[4]:\n"
    "from sklearn.model_selection import train_test_split\n"
    "X_train, X_test, y_train, y_test = train_test_split(scaled, target)\n",
    source="train:notebook",
)
_add(
    "t5_nb_impute_order",
    "preprocessing",
    True,
    "impute cell runs on the full table",
    "# In[5]:\n"
    "from sklearn.impute import KNNImputer\n"
    "filled = KNNImputer(n_neighbors=5).fit_transform(table)\n"
    "# In[6]:\n"
    "X_train, X_test, y_train, y_test = train_test_split(filled, table['label'])\n",
    source="train:notebook",
)
_add(
    "t5_nb_target_order",
    "target_encoding",
    True,
    "target mean cell runs before the split cell",
    "# In[7]:\n"
    "means = table.groupby('country')['target'].mean()\n"
    "table['country_rate'] = table['country'].map(means)\n"
    "# In[8]:\n"
    "X_train, X_test, y_train, y_test = train_test_split(table, table['target'])\n",
    source="train:notebook",
)
_add(
    "t5_nb_value_counts",
    "split",
    False,
    "value counts cell is EDA only",
    "# In[2]:\n"
    "counts = table['city'].value_counts()\n"
    "counts.head(10)\n"
    "# In[3]:\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "table.drop(columns=['target']), table['target'])\n",
    source="train:notebook",
)
_add(
    "t5_nb_describe",
    "split",
    False,
    "describe cell is EDA only",
    "# In[2]:\n"
    "table.describe()\n"
    "# In[3]:\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "table.drop(columns=['target']), table['target'])\n",
    source="train:notebook",
)
_add(
    "t5_nb_split_first",
    "preprocessing",
    False,
    "split cell runs before the scale cell",
    "# In[3]:\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "# In[4]:\n"
    "scaler = StandardScaler().fit(X_train)\n"
    "X_train = scaler.transform(X_train)\nX_test = scaler.transform(X_test)\n",
    source="train:notebook",
)
_add(
    "t5_nb_target_recode",
    "split",
    False,
    "target recode cell before the split",
    "table['target'] = table['target'].map({'yes': 1, 'no': 0})\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "table.drop(columns=['target']), table['target'])\n",
    source="train:notebook",
)
_add(
    "t5_nb_pipeline",
    "preprocessing",
    False,
    "pipeline cell fitted on the train split",
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "pipe = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())\n"
    "pipe.fit(X_train, y_train)\n",
    source="train:notebook",
)

_add(
    "t5_grid_full",
    "tuning",
    True,
    "grid search before the partition",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "search = GridSearchCV(model, {'max_depth': [3, 5]}, cv=4)\n"
    "search.fit(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
    source="train:tuning",
)
_add(
    "t5_random_full",
    "tuning",
    True,
    "random search on all rows",
    "from sklearn.model_selection import RandomizedSearchCV\n"
    "search = RandomizedSearchCV(model, params, n_iter=20, cv=5)\n"
    "search.fit(frame[features], frame['target'])\n"
    "train, test = train_test_split(frame, random_state=4)\n",
    source="train:tuning",
)
_add(
    "tuning_before_split",
    "tuning",
    True,
    "recursive feature elimination before the partition",
    "from sklearn.feature_selection import RFECV\n"
    + _SPLIT
    + "selector = RFECV(model, cv=5).fit(X, y)\n"
    "X = selector.transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
    source="train:tuning",
)
_add(
    "t5_grid_train",
    "tuning",
    False,
    "grid search on the train partition",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "search = GridSearchCV(model, {'max_depth': [3, 5]}, cv=4)\n"
    "search.fit(X_train, y_train)\nprint(search.score(X_test, y_test))\n",
    source="train:tuning",
)
_add(
    "t5_random_train",
    "tuning",
    False,
    "random search after the partition",
    "from sklearn.model_selection import RandomizedSearchCV\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "search = RandomizedSearchCV(model, params, n_iter=20, cv=5)\n"
    "search.fit(X_train, y_train)\n",
    source="train:tuning",
)
_add(
    "t5_cv_train",
    "tuning",
    False,
    "cross validation on the training rows",
    "from sklearn.model_selection import cross_val_score\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "scores = cross_val_score(model, X_train, y_train, cv=5)\n",
    source="train:tuning",
)
_add(
    "t5_pipeline_cv_all",
    "tuning",
    False,
    "cross validation through a full pipeline",
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.feature_selection import SelectKBest\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.model_selection import cross_val_score\n"
    "pipe = Pipeline([('scale', StandardScaler()), ('select', SelectKBest(k=8)), "
    "('model', LogisticRegression())])\n"
    "scores = cross_val_score(pipe, X, y, cv=5)\n",
    source="train:tuning",
)
_add(
    "t5_nested_cv",
    "tuning",
    False,
    "nested cross validation",
    "from sklearn.model_selection import GridSearchCV, cross_val_score\n"
    "inner = GridSearchCV(model, params, cv=3)\n"
    "scores = cross_val_score(inner, X, y, cv=5)\n",
    source="train:tuning",
)

_add(
    "t5_smote_full",
    "resampling",
    True,
    "smote before the partition",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "X, y = SMOTE(random_state=1).fit_resample(X, y)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
    source="train:resampling",
)
_add(
    "t5_undersample_full",
    "resampling",
    True,
    "random undersampler before the partition",
    "from imblearn.under_sampling import RandomUnderSampler\n"
    + _SPLIT
    + "sampler = RandomUnderSampler()\n"
    "X, y = sampler.fit_resample(X, y)\n"
    "train_x, test_x, train_y, test_y = train_test_split(X, y)\n",
    source="train:resampling",
)
_add(
    "t5_borderline_cv",
    "resampling",
    True,
    "borderline smote before cross validation",
    "from imblearn.over_sampling import BorderlineSMOTE\n"
    "from sklearn.model_selection import cross_val_score\n"
    "X, y = BorderlineSMOTE().fit_resample(X, y)\n"
    "scores = cross_val_score(model, X, y, cv=5)\n",
    source="train:resampling",
)
_add(
    "t5_smote_train",
    "resampling",
    False,
    "smote on the train partition only",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "X_train, y_train = SMOTE().fit_resample(X_train, y_train)\n"
    "model.fit(X_train, y_train)\n",
    source="train:resampling",
)
_add(
    "t5_imblearn_pipeline",
    "resampling",
    False,
    "imblearn pipeline fitted on train",
    "from imblearn.pipeline import Pipeline\n"
    "from imblearn.over_sampling import SMOTE\n"
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "pipe = Pipeline([('sample', SMOTE()), ('model', LogisticRegression())])\n"
    "pipe.fit(X_train, y_train)\n",
    source="train:resampling",
)
_add(
    "t5_class_weight",
    "resampling",
    False,
    "class weights instead of resampling",
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    "model = LogisticRegression(class_weight='balanced')\n"
    "model.fit(X_train, y_train)\n",
    source="train:resampling",
)

_add(
    "t5_uuid_feature",
    "identifier",
    True,
    "uuid kept as a predictor",
    _SPLIT + "X = frame[['uuid', 'amount']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['fraud'])\n",
    source="train:identifiers",
)
_add(
    "t5_row_hash_feature",
    "identifier",
    True,
    "row hash used as a predictor",
    _SPLIT + "frame['row_hash'] = frame.apply(lambda row: hash(tuple(row)), axis=1)\n"
    "X = frame[['row_hash', 'amount']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['fraud'])\n",
    source="train:identifiers",
)
_add(
    "t5_merchant_feature",
    "identifier",
    True,
    "merchant id included in the matrix",
    _SPLIT + "X = frame[['merchant_id', 'amount', 'hour']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['fraud'])\n",
    source="train:identifiers",
)
_add(
    "t5_uuid_dropped",
    "identifier",
    False,
    "uuid dropped before the split",
    _SPLIT + "X = frame.drop(columns=['uuid', 'row_hash'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['fraud'])\n",
    source="train:identifiers",
)
_add(
    "t5_merchant_groups",
    "identifier",
    False,
    "merchant id used for grouping only",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "X = frame.drop(columns=['merchant_id'])\n"
    "scores = cross_val_score(model, X, frame['fraud'], cv=GroupKFold(5), "
    "groups=frame['merchant_id'])\n",
    source="train:identifiers",
)
_add(
    "t5_hash_removed",
    "identifier",
    False,
    "row hash removed before the split",
    _SPLIT + "X = frame.drop(columns=['row_hash', 'session_id'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, frame['fraud'])\n",
    source="train:identifiers",
)


_CASES.extend(PHASE6_CASES)
_CASES.extend(PHASE7_CASES)


def phase7_holdout_cases() -> tuple[CorpusCase, ...]:
    """Return the frozen Phase 7 holdout.  Never used for fitting."""

    return PHASE7_HOLDOUT_CASES


def corpus_cases(split: str | None = None) -> tuple[CorpusCase, ...]:
    """Return development cases, or only one split.

    ``None`` means train + validation.  The locked test set lives in
    :mod:`leaklens.ai.corpus_test` and the frozen Phase 7 holdout lives in
    :mod:`leaklens.ai.corpus_phase7`; neither is part of the development corpus.
    """

    development = tuple(_CASES) + VALIDATION_CASES
    if split is None or split == "development":
        return development
    if split == TRAIN:
        return tuple(_CASES)
    if split == VALIDATION:
        return VALIDATION_CASES
    if split == HOLDOUT:
        return PHASE7_HOLDOUT_CASES
    raise ValueError(f"unknown corpus split: {split}")


def training_cases() -> tuple[CorpusCase, ...]:
    return tuple(_CASES)


def validation_cases() -> tuple[CorpusCase, ...]:
    return VALIDATION_CASES


def development_cases() -> tuple[CorpusCase, ...]:
    return tuple(_CASES) + VALIDATION_CASES


def _stats(cases: tuple[CorpusCase, ...]) -> dict[str, int]:
    groups = {case.group for case in cases}
    return {
        "cases": len(cases),
        "leaky": sum(case.leaky for case in cases),
        "legitimate": sum(not case.leaky for case in cases),
        "groups": len(groups),
        "leaky_groups": len({case.group for case in cases if case.leaky}),
        "legitimate_groups": len({case.group for case in cases if not case.leaky}),
        "sources": len({case.source for case in cases}),
    }


def corpus_stats() -> dict[str, object]:
    train = _stats(tuple(_CASES))
    validation = _stats(VALIDATION_CASES)
    development = _stats(development_cases())
    holdout = _stats(PHASE7_HOLDOUT_CASES)
    return {
        **development,
        "train_cases": train["cases"],
        "train_leaky": train["leaky"],
        "train_legitimate": train["legitimate"],
        "train_groups": train["groups"],
        "validation_cases": validation["cases"],
        "validation_leaky": validation["leaky"],
        "validation_legitimate": validation["legitimate"],
        "validation_groups": validation["groups"],
        "train_sources": train["sources"],
        "validation_sources": validation["sources"],
        "holdout_cases": holdout["cases"],
        "holdout_leaky": holdout["leaky"],
        "holdout_legitimate": holdout["legitimate"],
        "holdout_groups": holdout["groups"],
        "holdout_sources": holdout["sources"],
    }


__all__ = [
    "HOLDOUT",
    "LOCKED_TEST",
    "TRAIN",
    "VALIDATION",
    "CorpusCase",
    "corpus_cases",
    "corpus_stats",
    "development_cases",
    "phase7_holdout_cases",
    "training_cases",
    "validation_cases",
]
