"""M6 false-positive guards: safe CV, temporal splits, label carriers and EDA.

Each guard has a safe case, a nearby legitimate lookalike and a genuine leak that
must keep firing.  The tests assert emitted features, not model probabilities, so
they stay valid when the frozen weights are regenerated.
"""

from __future__ import annotations

import numpy as np

from leaklens.ai.features import FEATURE_NAMES, extract_features


def _active(code: str) -> set[str]:
    values = extract_features(code)
    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(values).all()
    return {FEATURE_NAMES[index] for index, value in enumerate(values) if value}


def test_label_arguments_of_gradient_boosting_containers_are_not_features():
    code = (
        "import xgboost as xgb\n"
        "ordered = df.sort_values('date')\n"
        "cutoff = int(len(ordered) * 0.8)\n"
        "train, test = ordered.iloc[:cutoff], ordered.iloc[cutoff:]\n"
        "dtrain = xgb.DMatrix(train[features], label=train['sales'])\n"
        "dtest = xgb.DMatrix(test[features], label=test['sales'])\n"
        "booster = xgb.train(params, dtrain, num_boost_round=100)\n"
    )
    active = _active(code)

    assert "target_feature_risk" not in active
    assert "target_in_assignment" not in active
    assert "time_ordered_split" in active


def test_target_inside_the_feature_matrix_still_fires():
    code = (
        "import xgboost as xgb\n"
        "dtrain = xgb.DMatrix(train[['x', 'target']], label=train['target'])\n"
    )
    assert "target_feature_risk" in _active(code)


def test_lightgbm_dataset_label_is_not_a_target_feature():
    code = (
        "import lightgbm as lgb\n"
        "lgb_train = lgb.Dataset(X_train, y_train)\n"
        "lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)\n"
        "gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid])\n"
    )
    active = _active(code)

    assert "target_feature_risk" not in active
    assert "target_in_assignment" not in active


def test_positional_time_split_is_recognised_as_a_split_boundary():
    code = (
        "import pandas as pd\n"
        "from catboost import CatBoostRegressor\n"
        "ordered = df.sort_values('timestamp').reset_index(drop=True)\n"
        "cutoff = int(len(ordered) * 0.8)\n"
        "train = ordered.iloc[:cutoff]\n"
        "test = ordered.iloc[cutoff:]\n"
        "model = CatBoostRegressor(iterations=300, random_seed=0)\n"
        "model.fit(train[features], train['demand'])\n"
        "score = model.score(test[features], test['demand'])\n"
    )
    active = _active(code)

    assert "time_ordered_split" in active
    assert "fit_before_split" not in active
    assert "fit_transform_before_split" not in active
    assert "fit_without_split" not in active


def test_positional_cut_on_a_non_time_column_is_not_a_time_split():
    code = (
        "ordered = df.sort_values('amount')\n"
        "cutoff = int(len(ordered) * 0.8)\n"
        "train, test = ordered.iloc[:cutoff], ordered.iloc[cutoff:]\n"
        "model.fit(train[features], train['demand'])\n"
    )
    assert "time_ordered_split" not in _active(code)


def test_preprocessing_before_a_positional_cut_still_fires():
    code = (
        "ordered = df.sort_values('date')\n"
        "scaled = scaler.fit_transform(ordered[features])\n"
        "cutoff = int(len(ordered) * 0.8)\n"
        "train, test = ordered.iloc[:cutoff], ordered.iloc[cutoff:]\n"
        "model.fit(train[features], train['demand'])\n"
    )
    assert "fit_before_split" in _active(code)


def test_plain_fold_local_cv_without_preprocessing_is_safe():
    code = (
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "cv = KFold(n_splits=5, shuffle=True, random_state=0)\n"
        "scores = cross_val_score(model, X, y, cv=cv)\n"
    )
    active = _active(code)

    assert "cv_full_data_safe" in active
    assert "cv_on_full_data" not in active
    assert "plain_kfold" not in active
    assert "has_shuffle" not in active


def test_cross_validation_reused_as_a_heldout_estimate_is_not_safe():
    code = (
        "from sklearn.model_selection import cross_val_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "scores = cross_val_score(model, X, y, cv=5)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    )
    active = _active(code)

    assert "cv_full_data_safe" not in active
    assert "cv_on_full_data" in active


def test_cross_validation_with_full_frame_preprocessing_is_not_safe():
    code = (
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "scaled = StandardScaler().fit_transform(X)\n"
        "cv = KFold(n_splits=5, shuffle=True, random_state=0)\n"
        "scores = cross_val_score(model, scaled, y, cv=cv)\n"
    )
    active = _active(code)

    assert "cv_full_data_safe" not in active
    assert "transformer_before_split" in active


def test_plot_only_full_frame_scaling_is_eda_not_leakage():
    code = (
        "X_plot = X[['proline', 'hue']]\n"
        "X_plot_scaled = scaler.fit_transform(X_plot)\n"
        "clf = KNeighborsClassifier(n_neighbors=20)\n"
        "fit_and_plot_model(X_plot, y, clf, ax1)\n"
        "fit_and_plot_model(X_plot_scaled, y, clf, ax2)\n"
    )
    active = _active(code)

    assert "eda_summary_only" in active
    assert "fit_before_split" not in active
    assert "fit_transform_before_split" not in active
    assert "fit_without_split" not in active


def test_full_frame_scaling_followed_by_a_split_is_still_leakage():
    code = (
        "from sklearn.model_selection import train_test_split\n"
        "scaled = scaler.fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(scaled, y)\n"
    )
    active = _active(code)

    assert "fit_before_split" in active
    assert "eda_summary_only" not in active


def test_appending_test_targets_is_not_a_safe_target_relabel():
    code = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "X_train = pd.concat([X_train, X_test.head(50)])\n"
        "y_train = pd.concat([y_train, y_test.head(50)])\n"
        "model.fit(X_train, y_train)\n"
    )
    active = _active(code)

    assert "target_relabel_safe" not in active
    assert "test_concat" in active


def test_mapping_the_target_is_still_a_safe_relabel():
    code = (
        "frame['target'] = frame['target'].map({'yes': 1, 'no': 0})\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "frame.drop(columns=['target']), frame['target'])\n"
    )
    assert "target_relabel_safe" in _active(code)


def test_prefixed_resampler_before_cross_validation_is_not_safe():
    code = (
        "from imblearn.over_sampling import BorderlineSMOTE\n"
        "from sklearn.model_selection import cross_val_score\n"
        "X, y = BorderlineSMOTE().fit_resample(X, y)\n"
        "scores = cross_val_score(model, X, y, cv=5)\n"
    )
    active = _active(code)

    assert "resampler" in active
    assert "unsafe_resampling" in active
    assert "cv_full_data_safe" not in active


def test_identifier_used_as_a_cv_feature_is_not_safe():
    code = (
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "X = tickets[['priority', 'agent_id']]\n"
        "scores = cross_val_score(model, X, tickets['escalated'], cv=KFold(5))\n"
    )
    active = _active(code)

    assert "cv_full_data_safe" not in active
    assert "cv_on_full_data" in active


def test_identifier_dropped_before_cross_validation_is_safe():
    code = (
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "X = tickets.drop(columns=['agent_id'])\n"
        "scores = cross_val_score(model, X, tickets['escalated'], cv=KFold(5))\n"
    )
    active = _active(code)

    assert "cv_full_data_safe" in active
    assert "cv_on_full_data" not in active


def test_label_carrier_built_before_the_split_is_still_a_target_feature():
    code = (
        "import xgboost as xgb\n"
        "from sklearn.model_selection import train_test_split\n"
        "dtrain_all = xgb.DMatrix(X, label=y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "dtrain = xgb.DMatrix(X_train, label=y_train)\n"
    )
    assert "target_feature_risk" in _active(code)


def test_guards_are_deterministic():
    code = (
        "ordered = df.sort_values('date')\n"
        "cutoff = int(len(ordered) * 0.8)\n"
        "train, test = ordered.iloc[:cutoff], ordered.iloc[cutoff:]\n"
        "model.fit(train[features], train['demand'])\n"
    )
    first = extract_features(code)
    second = extract_features(code)
    np.testing.assert_array_equal(first, second)
