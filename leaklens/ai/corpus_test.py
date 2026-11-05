"""Locked independent test split.

This module is imported only by the benchmark and its tests.  The training code
(``leaklens.ai.learned``) imports :mod:`leaklens.ai.corpus`, never this module.
Cases were authored before model selection and are never used for feature
selection, calibration or threshold tuning.

Provenance: every case is hand-written for this project from a scenario source
that does not appear in the development corpus.  No case is a copy of a
training, validation or scanner-benchmark case.
"""

from __future__ import annotations

from .corpus_types import LOCKED_TEST, CorpusCase

_SPLIT = "from sklearn.model_selection import train_test_split\n"

_CASES: list[CorpusCase] = []


def _add(
    group: str,
    category: str,
    leaky: bool,
    name: str,
    code: str,
    *,
    source: str,
) -> None:
    _CASES.append(CorpusCase(group, name, code, leaky, category, LOCKED_TEST, source))


# --- locked: credit underwriting ------------------------------------------

_add(
    "l5_credit_scale_full",
    "preprocessing",
    True,
    "applicant income scaled across all rows",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "applications['income_z'] = StandardScaler().fit_transform("
    "applications[['income']])\n"
    "train, test = train_test_split(applications, random_state=11)\n",
    source="test:credit",
)
_add(
    "l5_credit_employer_mean",
    "target_encoding",
    True,
    "default rate by employer before the holdout",
    _SPLIT + "employer_rate = applications.groupby('employer')['defaulted'].mean()\n"
    "applications['employer_rate'] = applications['employer'].map(employer_rate)\n"
    "train, test = train_test_split(applications, random_state=11)\n",
    source="test:credit",
)
_add(
    "l5_credit_collection_feature",
    "target_derived",
    True,
    "collection amount copied from the default flag",
    _SPLIT + "applications['collection_amount'] = applications['defaulted'] * 500\n"
    "features = applications[['collection_amount', 'income']]\n"
    "train, test = train_test_split(features, applications['defaulted'])\n",
    source="test:credit",
)
_add(
    "l5_credit_random_applicant",
    "grouping",
    True,
    "random split over repeated applicants",
    _SPLIT + "rows = applications[['applicant_id', 'income']]\n"
    "train, test = train_test_split(rows, random_state=11)\n"
    "model.fit(train.drop(columns=['applicant_id']), train['defaulted'])\n",
    source="test:credit",
)
_add(
    "l5_credit_bfill_bureau",
    "temporal",
    True,
    "bureau score backfilled from later months",
    _SPLIT + "applications = applications.sort_values('statement_month')\n"
    "applications['bureau_score'] = applications['bureau_score'].bfill()\n"
    "train, test = train_test_split(applications, random_state=11)\n",
    source="test:credit",
)
_add(
    "l5_credit_test_quantile",
    "preprocessing",
    True,
    "quantile transform fitted on the test partition",
    "from sklearn.preprocessing import QuantileTransformer\n"
    + _SPLIT
    + "train, test = train_test_split(applications, random_state=11)\n"
    "qt = QuantileTransformer()\n"
    "test[['income']] = qt.fit_transform(test[['income']])\n",
    source="test:credit",
)
_add(
    "l5_credit_dupes_after",
    "split",
    True,
    "duplicate applications pruned after the holdout",
    _SPLIT + "train, test = train_test_split(applications, random_state=11)\n"
    "train = train.drop_duplicates(subset=['applicant_id', 'statement_month'])\n",
    source="test:credit",
)
_add(
    "l5_credit_scale_train",
    "preprocessing",
    False,
    "scaler fitted on the approved train rows",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "train, test = train_test_split(applications, random_state=11)\n"
    "scaler = StandardScaler().fit(train[['income', 'debt']])\n"
    "train[['income', 'debt']] = scaler.transform(train[['income', 'debt']])\n"
    "test[['income', 'debt']] = scaler.transform(test[['income', 'debt']])\n",
    source="test:credit",
)
_add(
    "l5_credit_employer_train",
    "target_encoding",
    False,
    "employer default rate learned on train",
    _SPLIT + "train, test = train_test_split(applications, random_state=11)\n"
    "employer_rate = train.groupby('employer')['defaulted'].mean()\n"
    "train['employer_rate'] = train['employer'].map(employer_rate)\n"
    "test['employer_rate'] = test['employer'].map(employer_rate)\n",
    source="test:credit",
)
_add(
    "l5_credit_group_shuffle",
    "grouping",
    False,
    "group shuffle split over applicants",
    "from sklearn.model_selection import GroupShuffleSplit\n"
    "splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=11)\n"
    "train_idx, test_idx = next(splitter.split("
    "features, target, groups=applications['applicant_id']))\n",
    source="test:credit",
)
_add(
    "l5_credit_collection_dropped",
    "target_leakage",
    False,
    "collection fields dropped explicitly",
    _SPLIT
    + "features = applications.drop(columns=['defaulted', 'collection_amount'])\n"
    "train, test = train_test_split(features, applications['defaulted'])\n",
    source="test:credit",
)
_add(
    "l5_credit_chronological",
    "temporal",
    False,
    "chronological application holdout",
    _SPLIT + "ordered = applications.sort_values('statement_month')\n"
    "train, test = train_test_split(ordered, ordered['defaulted'], shuffle=False)\n",
    source="test:credit",
)


# --- locked: campaign attribution -----------------------------------------

_add(
    "l5_marketing_campaign_mean",
    "target_encoding",
    True,
    "click rate by campaign before the holdout",
    _SPLIT + "campaign_rate = touches.groupby('campaign')['clicked'].mean()\n"
    "touches['campaign_rate'] = touches['campaign'].map(campaign_rate)\n"
    "train, test = train_test_split(touches, random_state=9)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_smote_full",
    "resampling",
    True,
    "smote before the campaign holdout",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "features, clicked = SMOTE().fit_resample(touches[predictors], touches['clicked'])\n"
    "train, test = train_test_split(features, clicked)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_conversion_feature",
    "target_derived",
    True,
    "conversion flag turned into a feature",
    _SPLIT
    + "touches['conversion_value'] = touches['converted'] * touches['order_value']\n"
    "features = touches[['conversion_value', 'channel']]\n"
    "train, test = train_test_split(features, touches['converted'])\n",
    source="test:marketing",
)
_add(
    "l5_marketing_rfecv_full",
    "tuning",
    True,
    "recursive elimination on every touch",
    "from sklearn.feature_selection import RFECV\n"
    + _SPLIT
    + "selector = RFECV(model, cv=5).fit(touches[predictors], touches['converted'])\n"
    "chosen = selector.transform(touches[predictors])\n"
    "train, test = train_test_split(chosen, touches['converted'])\n",
    source="test:marketing",
)
_add(
    "l5_marketing_grid_full",
    "tuning",
    True,
    "grid search before the holdout",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "search = GridSearchCV(model, grid, cv=5)\n"
    "search.fit(touches[predictors], touches['converted'])\n"
    "train, test = train_test_split(touches, random_state=9)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_channel_mean_map",
    "target_encoding",
    True,
    "channel mean mapped onto the full frame",
    _SPLIT + "lookup = touches.groupby('channel')['converted'].agg('mean')\n"
    "touches['channel_rate'] = touches['channel'].map(lookup)\n"
    "train, test = train_test_split(touches, random_state=9)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_pipeline_cv",
    "target_encoding",
    False,
    "target encoder inside a cross-validated pipeline",
    "from sklearn.model_selection import cross_val_score\n"
    "from sklearn.pipeline import Pipeline\n"
    "from category_encoders import TargetEncoder\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "pipe = Pipeline([('encode', TargetEncoder()), ('model', LogisticRegression())])\n"
    "scores = cross_val_score(pipe, touches[predictors], touches['converted'], cv=5)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_smote_train",
    "resampling",
    False,
    "smote on the train campaign rows",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "train, test = train_test_split(touches, random_state=9)\n"
    "x_train, y_train = SMOTE().fit_resample(train[predictors], train['converted'])\n"
    "model.fit(x_train, y_train)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_grid_train",
    "tuning",
    False,
    "grid search on the campaign train split",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "train, test = train_test_split(touches, random_state=9)\n"
    "search = GridSearchCV(model, grid, cv=5)\n"
    "search.fit(train[predictors], train['converted'])\n",
    source="test:marketing",
)
_add(
    "l5_marketing_trailing_window",
    "temporal",
    False,
    "trailing email window",
    _SPLIT + "touches['trailing_opens'] = touches['opens'].shift(1).rolling(7).mean()\n"
    "train, test = train_test_split(touches, shuffle=False)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_summary_only",
    "split",
    False,
    "campaign summary is EDA only",
    "summary = touches.groupby('campaign')['spend'].agg('sum')\n"
    "summary.head()\n"
    + _SPLIT
    + "train, test = train_test_split(touches.drop(columns=['converted']), random_state=9)\n",
    source="test:marketing",
)
_add(
    "l5_marketing_dedupe_before",
    "split",
    False,
    "duplicate touches removed before the holdout",
    _SPLIT
    + "touches = touches.drop_duplicates(subset=['user_id', 'campaign', 'day'])\n"
    "train, test = train_test_split(touches, random_state=9)\n",
    source="test:marketing",
)


# --- locked: sensor anomaly ------------------------------------------------

_add(
    "l5_sensor_centered",
    "temporal",
    True,
    "centered rolling window over every reading",
    _SPLIT
    + "readings['smooth'] = readings['pressure'].rolling(9, center=True).mean()\n"
    "train, test = train_test_split(readings, random_state=13)\n",
    source="test:sensor",
)
_add(
    "l5_sensor_bfill",
    "temporal",
    True,
    "future sensor reading backfilled",
    _SPLIT + "readings = readings.sort_values('observed_at')\n"
    "readings['pressure'] = readings['pressure'].bfill()\n"
    "train, test = train_test_split(readings, random_state=13)\n",
    source="test:sensor",
)
_add(
    "l5_sensor_pca_full",
    "preprocessing",
    True,
    "pca over all sensor readings",
    "from sklearn.decomposition import PCA\n"
    + _SPLIT
    + "embedding = PCA(n_components=6).fit_transform(readings[channels])\n"
    "train, test = train_test_split(embedding, readings['fault'])\n",
    source="test:sensor",
)
_add(
    "l5_sensor_random_device",
    "grouping",
    True,
    "random split over repeated devices",
    _SPLIT + "rows = readings[['device_id', 'pressure']]\n"
    "train, test = train_test_split(rows, random_state=13)\n"
    "model.fit(train.drop(columns=['device_id']), train['fault'])\n",
    source="test:sensor",
)
_add(
    "l5_sensor_dupes_after",
    "split",
    True,
    "duplicate readings pruned after the holdout",
    _SPLIT + "train, test = train_test_split(readings, random_state=13)\n"
    "train = train.drop_duplicates(subset=['device_id', 'observed_at'])\n",
    source="test:sensor",
)
_add(
    "l5_sensor_test_scale",
    "preprocessing",
    True,
    "scaler fitted on the test readings",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "train, test = train_test_split(readings, random_state=13)\n"
    "scaler = StandardScaler()\n"
    "test[channels] = scaler.fit_transform(test[channels])\n",
    source="test:sensor",
)
_add(
    "l5_sensor_trailing",
    "temporal",
    False,
    "trailing sensor window",
    _SPLIT + "readings['trailing'] = readings['pressure'].shift(1).rolling(5).mean()\n"
    "train, test = train_test_split(readings, shuffle=False)\n",
    source="test:sensor",
)
_add(
    "l5_sensor_pca_train",
    "preprocessing",
    False,
    "pca fitted on the train readings",
    "from sklearn.decomposition import PCA\n"
    + _SPLIT
    + "train, test = train_test_split(readings, random_state=13)\n"
    "pca = PCA(n_components=6).fit(train[channels])\n"
    "train[channels] = pca.transform(train[channels])\n"
    "test[channels] = pca.transform(test[channels])\n",
    source="test:sensor",
)
_add(
    "l5_sensor_group_kfold",
    "grouping",
    False,
    "group kfold over devices",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "features = readings.drop(columns=['device_id'])\n"
    "scores = cross_val_score(model, features, readings['fault'], "
    "cv=GroupKFold(5), groups=readings['device_id'])\n",
    source="test:sensor",
)
_add(
    "l5_sensor_chronological",
    "temporal",
    False,
    "chronological reading holdout",
    _SPLIT + "ordered = readings.sort_values('observed_at')\n"
    "train, test = train_test_split(ordered, ordered['fault'], shuffle=False)\n",
    source="test:sensor",
)
_add(
    "l5_sensor_dedupe_before",
    "split",
    False,
    "duplicate readings removed before the holdout",
    _SPLIT
    + "readings = readings.drop_duplicates(subset=['device_id', 'observed_at'])\n"
    "train, test = train_test_split(readings, random_state=13)\n",
    source="test:sensor",
)


# --- locked: support ticket triage ----------------------------------------

_add(
    "l5_support_vectorizer_full",
    "preprocessing",
    True,
    "ticket vectorizer fitted on every row",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    + _SPLIT
    + "vectorizer = TfidfVectorizer(max_features=500)\n"
    "text = vectorizer.fit_transform(tickets['body'])\n"
    "train, test = train_test_split(text, tickets['escalated'])\n",
    source="test:support",
)
_add(
    "l5_support_resolution_feature",
    "target_derived",
    True,
    "resolution code copied from the outcome",
    _SPLIT + "tickets['resolution_score'] = tickets['escalated'].map({0: 1, 1: 9})\n"
    "features = tickets[['resolution_score', 'priority']]\n"
    "train, test = train_test_split(features, tickets['escalated'])\n",
    source="test:support",
)
_add(
    "l5_support_queue_mean",
    "target_encoding",
    True,
    "escalation rate by queue before the holdout",
    _SPLIT + "queue_rate = tickets.groupby('queue')['escalated'].mean()\n"
    "tickets['queue_rate'] = tickets['queue'].map(queue_rate)\n"
    "train, test = train_test_split(tickets, random_state=17)\n",
    source="test:support",
)
_add(
    "l5_support_agent_kfold",
    "grouping",
    True,
    "kfold leaves agents on both sides",
    "from sklearn.model_selection import KFold, cross_val_score\n"
    "X = tickets[['priority', 'agent_id']]\n"
    "scores = cross_val_score(model, X, tickets['escalated'], cv=KFold(5))\n",
    source="test:support",
)
_add(
    "l5_support_impute_full",
    "preprocessing",
    True,
    "satisfaction imputed over the full table",
    "from sklearn.impute import KNNImputer\n"
    + _SPLIT
    + "filled = KNNImputer().fit_transform(tickets[numeric])\n"
    "train, test = train_test_split(filled, tickets['escalated'])\n",
    source="test:support",
)
_add(
    "l5_support_satisfaction_feature",
    "target_derived",
    True,
    "satisfaction score derived from escalation",
    _SPLIT + "tickets['satisfaction_delta'] = 5 - tickets['escalated'] * 3\n"
    "X = tickets[['satisfaction_delta', 'priority']]\n"
    "train, test = train_test_split(X, tickets['escalated'])\n",
    source="test:support",
)
_add(
    "l5_support_vectorizer_pipeline",
    "preprocessing",
    False,
    "vectorizer inside a fitted pipeline",
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
    "from sklearn.linear_model import LogisticRegression\n"
    + _SPLIT
    + "train, test = train_test_split(tickets, random_state=17)\n"
    "pipe = Pipeline([('text', TfidfVectorizer()), ('model', LogisticRegression())])\n"
    "pipe.fit(train['body'], train['escalated'])\n",
    source="test:support",
)
_add(
    "l5_support_queue_train",
    "target_encoding",
    False,
    "queue rate learned on the train split",
    _SPLIT + "train, test = train_test_split(tickets, random_state=17)\n"
    "queue_rate = train.groupby('queue')['escalated'].mean()\n"
    "train['queue_rate'] = train['queue'].map(queue_rate)\n"
    "test['queue_rate'] = test['queue'].map(queue_rate)\n",
    source="test:support",
)
_add(
    "l5_support_group_agents",
    "grouping",
    False,
    "group kfold over agents",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "features = tickets.drop(columns=['agent_id'])\n"
    "scores = cross_val_score(model, features, tickets['escalated'], "
    "cv=GroupKFold(5), groups=tickets['agent_id'])\n",
    source="test:support",
)
_add(
    "l5_support_impute_train",
    "preprocessing",
    False,
    "imputer fitted on the train tickets",
    "from sklearn.impute import KNNImputer\n"
    + _SPLIT
    + "train, test = train_test_split(tickets, random_state=17)\n"
    "imp = KNNImputer().fit(train[numeric])\n"
    "train[numeric] = imp.transform(train[numeric])\n"
    "test[numeric] = imp.transform(test[numeric])\n",
    source="test:support",
)
_add(
    "l5_support_resolution_dropped",
    "target_leakage",
    False,
    "resolution fields dropped explicitly",
    _SPLIT + "features = tickets.drop(columns=['escalated', 'resolution_code'])\n"
    "train, test = train_test_split(features, tickets['escalated'])\n",
    source="test:support",
)
_add(
    "l5_support_eda_only",
    "split",
    False,
    "ticket length summary is EDA only",
    "summary = tickets.groupby('queue')['body'].agg('count')\n"
    "print(summary)\n"
    + _SPLIT
    + "train, test = train_test_split(tickets.drop(columns=['escalated']), random_state=17)\n",
    source="test:support",
)


LOCKED_TEST_CASES: tuple[CorpusCase, ...] = tuple(_CASES)

__all__ = ["LOCKED_TEST_CASES"]
