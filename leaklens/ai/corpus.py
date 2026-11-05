"""Labeled synthetic corpus for the optional learned detector.

Every case is written for this project; no benchmark case is reused for
training.  Cases are grouped by *template family* and grouped cross-validation
never puts two cases from the same family on opposite sides of a fold.  That
reduces, but cannot eliminate, the risk of learning surface syntax instead of
the underlying data flow.

The corpus is intentionally small and transparent.  It is a regression corpus,
not a claim that the model understands arbitrary production code.
"""

from __future__ import annotations

from dataclasses import dataclass

_SPLIT = "from sklearn.model_selection import train_test_split\n"


@dataclass(frozen=True, slots=True)
class CorpusCase:
    group: str
    name: str
    code: str
    leaky: bool
    category: str


_CASES: list[CorpusCase] = []


def _add(
    group: str,
    category: str,
    leaky: bool,
    name: str,
    code: str,
) -> None:
    _CASES.append(CorpusCase(group, name, code, leaky, category))


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
    _SPLIT + "X = df[['spend', 'visits']]\n"
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
    _SPLIT
    + "X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)\n",
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
    _SPLIT + "X = df.drop(columns=['target'])\n"
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


def corpus_cases() -> tuple[CorpusCase, ...]:
    return tuple(_CASES)


def corpus_stats() -> dict[str, int]:
    groups = {case.group for case in _CASES}
    return {
        "cases": len(_CASES),
        "leaky": sum(case.leaky for case in _CASES),
        "legitimate": sum(not case.leaky for case in _CASES),
        "groups": len(groups),
        "leaky_groups": len({case.group for case in _CASES if case.leaky}),
        "legitimate_groups": len({case.group for case in _CASES if not case.leaky}),
    }


__all__ = ["CorpusCase", "corpus_cases", "corpus_stats"]
