"""Validation split: authored scenario sources never used for final fitting.

These cases are used for threshold selection, calibration checks and model
selection only.  They are grouped by scenario source and template family, and
every source is disjoint from the training sources and from the locked test.
"""

from __future__ import annotations

from .corpus_types import VALIDATION, CorpusCase

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
    _CASES.append(CorpusCase(group, name, code, leaky, category, VALIDATION, source))


# --- validation: subscription churn ---------------------------------------

_add(
    "v5_churn_scaler_full",
    "preprocessing",
    True,
    "tenure scaled across every customer",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "scaled = StandardScaler().fit_transform(customer_features)\n"
    "x_tr, x_te, y_tr, y_te = train_test_split(scaled, churn, random_state=1)\n",
    source="validation:churn",
)
_add(
    "v5_churn_impute_full",
    "preprocessing",
    True,
    "support tickets imputed over the full table",
    "from sklearn.impute import KNNImputer\n"
    + _SPLIT
    + "filled = KNNImputer(n_neighbors=4).fit_transform(tickets)\n"
    "x_tr, x_te, y_tr, y_te = train_test_split(filled, tickets['churn'])\n",
    source="validation:churn",
)
_add(
    "v5_churn_target_mean",
    "target_encoding",
    True,
    "plan target rate before the holdout",
    _SPLIT + "plan_rate = accounts.groupby('plan')['churn'].mean()\n"
    "accounts['plan_rate'] = accounts['plan'].map(plan_rate)\n"
    "train, test = train_test_split(accounts, random_state=2)\n",
    source="validation:churn",
)
_add(
    "v5_churn_reason_feature",
    "target_derived",
    True,
    "cancellation reason copied into a feature",
    _SPLIT + "accounts['reason_score'] = accounts['cancel_reason'].factorize()[0]\n"
    "X = accounts[['reason_score', 'tenure']]\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "X, accounts['cancel_reason'])\n",
    source="validation:churn",
)
_add(
    "v5_churn_grid_full",
    "tuning",
    True,
    "grid search before the holdout",
    "from sklearn.model_selection import GridSearchCV\n"
    + _SPLIT
    + "search = GridSearchCV(estimator, space, cv=4)\n"
    "search.fit(customer_features, churn)\n"
    "train, test = train_test_split(accounts, random_state=2)\n",
    source="validation:churn",
)
_add(
    "v5_churn_dupes_after",
    "split",
    True,
    "duplicate customers pruned after the holdout",
    _SPLIT + "train, test = train_test_split(accounts, random_state=2)\n"
    "train = train.drop_duplicates(subset=['account_id'])\n",
    source="validation:churn",
)
_add(
    "v5_churn_scaler_train",
    "preprocessing",
    False,
    "scaler trained on the customer train split",
    "from sklearn.preprocessing import StandardScaler\n"
    + _SPLIT
    + "train, test = train_test_split(accounts, random_state=2)\n"
    "scaler = StandardScaler().fit(train[['tenure', 'spend']])\n"
    "train[['tenure', 'spend']] = scaler.transform(train[['tenure', 'spend']])\n"
    "test[['tenure', 'spend']] = scaler.transform(test[['tenure', 'spend']])\n",
    source="validation:churn",
)
_add(
    "v5_churn_target_train",
    "target_encoding",
    False,
    "plan rate learned on the train split",
    _SPLIT + "train, test = train_test_split(accounts, random_state=2)\n"
    "plan_rate = train.groupby('plan')['churn'].mean()\n"
    "train['plan_rate'] = train['plan'].map(plan_rate)\n"
    "test['plan_rate'] = test['plan'].map(plan_rate)\n",
    source="validation:churn",
)
_add(
    "v5_churn_dedupe_before",
    "split",
    False,
    "duplicate customers removed before the holdout",
    _SPLIT + "accounts = accounts.drop_duplicates(subset=['account_id', 'month'])\n"
    "train, test = train_test_split(accounts, random_state=2)\n",
    source="validation:churn",
)
_add(
    "v5_churn_summary_only",
    "split",
    False,
    "revenue summary is EDA only",
    "summary = accounts.groupby('plan')['revenue'].agg('mean')\n"
    "print(summary)\n"
    + _SPLIT
    + "train, test = train_test_split(accounts.drop(columns=['churn']), random_state=2)\n",
    source="validation:churn",
)


# --- validation: card fraud ------------------------------------------------

_add(
    "v5_fraud_winsor_full",
    "preprocessing",
    True,
    "amount winsorized on the full ledger",
    _SPLIT + "low = ledger['amount'].quantile(0.01)\n"
    "high = ledger['amount'].quantile(0.99)\n"
    "ledger['amount'] = ledger['amount'].clip(low, high)\n"
    "train, test = train_test_split(ledger, random_state=0)\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_chargeback_feature",
    "target_leakage",
    True,
    "chargeback flag used as a feature",
    _SPLIT + "ledger['chargeback_score'] = ledger['chargeback'] * 10\n"
    "X = ledger[['chargeback_score', 'amount']]\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "X, ledger['chargeback'])\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_merchant_mean",
    "target_encoding",
    True,
    "merchant target mean before the holdout",
    _SPLIT + "merchant_rate = ledger.groupby('merchant')['fraud'].mean()\n"
    "ledger['merchant_rate'] = ledger['merchant'].map(merchant_rate)\n"
    "train, test = train_test_split(ledger, random_state=0)\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_kfold_merchant",
    "grouping",
    True,
    "kfold leaves merchants on both sides",
    "from sklearn.model_selection import KFold, cross_val_score\n"
    "X = ledger[['amount', 'merchant_id']]\n"
    "scores = cross_val_score(estimator, X, ledger['fraud'], cv=KFold(5))\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_smote_full",
    "resampling",
    True,
    "smote applied before the holdout",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "X, y = SMOTE().fit_resample(ledger[features], ledger['fraud'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_winsor_train",
    "preprocessing",
    False,
    "winsorization uses train quantiles",
    _SPLIT + "train, test = train_test_split(ledger, random_state=0)\n"
    "low = train['amount'].quantile(0.01)\n"
    "high = train['amount'].quantile(0.99)\n"
    "train['amount'] = train['amount'].clip(low, high)\n"
    "test['amount'] = test['amount'].clip(low, high)\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_group_merchant",
    "grouping",
    False,
    "group kfold over merchants",
    "from sklearn.model_selection import GroupKFold, cross_val_score\n"
    "X = ledger.drop(columns=['merchant_id'])\n"
    "scores = cross_val_score(estimator, X, ledger['fraud'], "
    "cv=GroupKFold(5), groups=ledger['merchant_id'])\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_chargeback_dropped",
    "target_leakage",
    False,
    "chargeback fields dropped explicitly",
    _SPLIT + "X = ledger.drop(columns=['chargeback', 'chargeback_date'])\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, ledger['fraud'])\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_smote_train",
    "resampling",
    False,
    "smote only on the train ledger",
    "from imblearn.over_sampling import SMOTE\n"
    + _SPLIT
    + "train, test = train_test_split(ledger, random_state=0)\n"
    "x_train, y_train = SMOTE().fit_resample(train[features], train['fraud'])\n"
    "estimator.fit(x_train, y_train)\n",
    source="validation:fraud",
)
_add(
    "v5_fraud_dedupe_stratify",
    "split",
    False,
    "deduplicate then stratified holdout",
    _SPLIT + "ledger = ledger.drop_duplicates(subset=['transaction_id'])\n"
    "train, test = train_test_split(ledger, stratify=ledger['fraud'], random_state=0)\n",
    source="validation:fraud",
)


# --- validation: hospital readmission -------------------------------------

_add(
    "v5_health_pca_full",
    "preprocessing",
    True,
    "lab pca before the patient holdout",
    "from sklearn.decomposition import PCA\n"
    + _SPLIT
    + "components = PCA(n_components=12).fit_transform(labs)\n"
    "train, test = train_test_split(components, readmitted)\n",
    source="validation:healthcare",
)
_add(
    "v5_health_patient_mean",
    "preprocessing",
    True,
    "patient mean labs before the holdout",
    _SPLIT
    + "visits['patient_mean'] = visits.groupby('patient_id')['lab'].transform('mean')\n"
    "train, test = train_test_split(visits, random_state=5)\n",
    source="validation:healthcare",
)
_add(
    "v5_health_score_feature",
    "target_derived",
    True,
    "readmission score copied from the outcome",
    _SPLIT + "visits['score_copy'] = visits['readmitted'] * 5\n"
    "X = visits[['score_copy', 'age']]\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, visits['readmitted'])\n",
    source="validation:healthcare",
)
_add(
    "v5_health_random_patient",
    "grouping",
    True,
    "random split over repeated patients",
    _SPLIT + "rows = visits[['patient_id', 'lab']]\n"
    "train, test = train_test_split(rows, random_state=5)\n"
    "model.fit(train.drop(columns=['patient_id']), train['readmitted'])\n",
    source="validation:healthcare",
)
_add(
    "v5_health_bfill",
    "temporal",
    True,
    "future lab value backfilled",
    _SPLIT + "visits = visits.sort_values('admitted_at')\n"
    "visits['lab'] = visits['lab'].bfill()\n"
    "train, test = train_test_split(visits, random_state=5)\n",
    source="validation:healthcare",
)
_add(
    "v5_health_pca_train",
    "preprocessing",
    False,
    "pca fitted on the patient train split",
    "from sklearn.decomposition import PCA\n"
    + _SPLIT
    + "train, test = train_test_split(visits, random_state=5)\n"
    "pca = PCA(n_components=10).fit(train[labs])\n"
    "train[labs] = pca.transform(train[labs])\n"
    "test[labs] = pca.transform(test[labs])\n",
    source="validation:healthcare",
)
_add(
    "v5_health_patient_train",
    "preprocessing",
    False,
    "patient mean learned on the train split",
    "from sklearn.model_selection import GroupShuffleSplit\n"
    + _SPLIT
    + "splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=5)\n"
    "train_idx, test_idx = next(splitter.split(visits, groups=visits['patient_id']))\n"
    "train = visits.iloc[train_idx]\ntest = visits.iloc[test_idx]\n"
    "means = train.groupby('patient_id')['lab'].mean()\n"
    "train['patient_mean'] = train['patient_id'].map(means)\n"
    "test['patient_mean'] = test['patient_id'].map(means)\n",
    source="validation:healthcare",
)
_add(
    "v5_health_group_shuffle",
    "grouping",
    False,
    "group shuffle split over patients",
    "from sklearn.model_selection import GroupShuffleSplit\n"
    "splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=5)\n"
    "train_idx, test_idx = next(splitter.split(X, y, groups=visits['patient_id']))\n",
    source="validation:healthcare",
)
_add(
    "v5_health_trailing_lab",
    "temporal",
    False,
    "trailing lab window",
    _SPLIT + "visits['lab_lag'] = visits['lab'].shift(1).rolling(3).mean()\n"
    "train, test = train_test_split(visits, shuffle=False)\n",
    source="validation:healthcare",
)
_add(
    "v5_health_identifier_drop",
    "identifier",
    False,
    "patient identifier dropped",
    _SPLIT + "X = visits.drop(columns=['patient_id', 'row_hash'])\n"
    "train, test = train_test_split(X, visits['readmitted'])\n",
    source="validation:healthcare",
)


# --- validation: store demand ---------------------------------------------

_add(
    "v5_demand_centered",
    "temporal",
    True,
    "centered rolling window over all stores",
    _SPLIT + "sales['smooth'] = sales['units'].rolling(21, center=True).mean()\n"
    "train, test = train_test_split(sales, random_state=7)\n",
    source="validation:demand",
)
_add(
    "v5_demand_shuffle_time",
    "temporal",
    True,
    "time rows shuffled into the holdout",
    _SPLIT + "X = sales[['week', 'price']]\n"
    "train, test = train_test_split(X, sales['units'], shuffle=True, random_state=7)\n",
    source="validation:demand",
)
_add(
    "v5_demand_store_mean",
    "target_encoding",
    True,
    "store target mean before the holdout",
    _SPLIT + "store_rate = sales.groupby('store')['units'].mean()\n"
    "sales['store_rate'] = sales['store'].map(store_rate)\n"
    "train, test = train_test_split(sales, random_state=7)\n",
    source="validation:demand",
)
_add(
    "v5_demand_rfecv",
    "tuning",
    True,
    "recursive elimination on all rows",
    "from sklearn.feature_selection import RFECV\n"
    + _SPLIT
    + "selector = RFECV(estimator, cv=4).fit(sales[predictors], sales['units'])\n"
    "chosen = selector.transform(sales[predictors])\n"
    "train, test = train_test_split(chosen, sales['units'])\n",
    source="validation:demand",
)
_add(
    "v5_demand_future_merge",
    "temporal",
    True,
    "future shipments merged by week",
    _SPLIT + "shipments = pd.read_csv('future_shipments.csv')\n"
    "sales = sales.merge(shipments, on='week', how='left')\n"
    "train, test = train_test_split(sales, random_state=7)\n",
    source="validation:demand",
)
_add(
    "v5_demand_trailing",
    "temporal",
    False,
    "trailing demand window",
    _SPLIT + "sales['trailing'] = sales['units'].shift(1).rolling(14).mean()\n"
    "train, test = train_test_split(sales, shuffle=False)\n",
    source="validation:demand",
)
_add(
    "v5_demand_timeseries_cv",
    "temporal",
    False,
    "time series cross validation",
    "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\n"
    "scores = cross_val_score(estimator, X, sales['units'], "
    "cv=TimeSeriesSplit(n_splits=5))\n",
    source="validation:demand",
)
_add(
    "v5_demand_store_train",
    "target_encoding",
    False,
    "store rate learned from train weeks",
    _SPLIT + "train, test = train_test_split(sales, shuffle=False)\n"
    "store_rate = train.groupby('store')['units'].mean()\n"
    "train['store_rate'] = train['store'].map(store_rate)\n"
    "test['store_rate'] = test['store'].map(store_rate)\n",
    source="validation:demand",
)
_add(
    "v5_demand_selection_pipeline",
    "tuning",
    False,
    "selection inside a fitted pipeline",
    "from sklearn.pipeline import Pipeline\n"
    "from sklearn.feature_selection import SelectFromModel\n"
    + _SPLIT
    + "train, test = train_test_split(sales, shuffle=False)\n"
    "pipe = Pipeline([('select', SelectFromModel(estimator)), ('model', estimator)])\n"
    "pipe.fit(train[predictors], train['units'])\n",
    source="validation:demand",
)
_add(
    "v5_demand_chronological",
    "temporal",
    False,
    "chronological demand holdout",
    _SPLIT + "ordered = sales.sort_values('week')\n"
    "train, test = train_test_split(ordered, ordered['units'], shuffle=False)\n",
    source="validation:demand",
)


VALIDATION_CASES: tuple[CorpusCase, ...] = tuple(_CASES)

__all__ = ["VALIDATION_CASES"]
