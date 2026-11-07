"""Frozen real-world holdout corpus for the Phase 6 external evaluation.

Why this module exists
----------------------
Phase 5 measured the learned detector on an *authored* locked test and on the
scanner benchmark.  Both are useful regression guards, but neither says whether
the detector transfers to code written by other people.  This module collects a
second, separate corpus that is **only** used to evaluate a model that has
already been trained, calibrated and thresholded.  Nothing here is imported by
``learned.py`` at training time.

Provenance discipline
---------------------
Every case records its repository, licence, commit-pinned URL, path, retrieval
date and a manual label rationale.  Two provenance classes are used and must not
be conflated:

``verbatim_excerpt``
    Lines copied from the cited public file at the cited commit.  The upstream
    licence permits redistribution; attribution is kept in
    ``docs/REALWORLD_CORPUS.md``.

``reconstructed_excerpt``
    A short, hand-written variant of the idiom that the cited file demonstrates.
    It is **not** a copy of upstream code and must never be described as one.  It
    exists because most public examples are clean: genuinely leaky production
    code is rarely published under a permissive licence with a citable commit.

The split is by **repository and pipeline family**.  Every family lives in one
repository, and no repository in this module appears in the authored training,
validation or Phase 5 locked-test corpora.  That makes the repository-level
holdout genuinely unseen, but it does not make it a random sample of all
production code.

Freeze protocol
---------------
The corpus was written and labelled before Phase 6 changed feature extraction or
model selection.  ``manifest_digest()`` hashes every labelled field so the
benchmark report can prove that the holdout was not edited after it was scored.
The Phase 6 report records the digest that was frozen at the first commit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

HOLDOUT_ID = "phase6-realworld-holdout-v1"
HOLDOUT_ROLE = "single_pass_evaluation"
RETRIEVED = "2026-11-05"


@dataclass(frozen=True, slots=True)
class PublicRepository:
    """A public repository used as a provenance anchor."""

    slug: str
    title: str
    url: str
    license_id: str
    license_url: str
    commit: str
    branch: str


@dataclass(frozen=True, slots=True)
class RealWorldCase:
    """One reviewed pipeline unit with complete provenance."""

    name: str
    repository: str
    family: str
    code: str
    leaky: bool
    category: str
    label_rationale: str
    provenance: str
    path: str
    source_url: str
    review: str = "manual-review-2026-11-05"
    notes: str = ""


REPOSITORIES: tuple[PublicRepository, ...] = (
    PublicRepository(
        slug="scikit-learn/scikit-learn",
        title="scikit-learn examples and documentation",
        url="https://github.com/scikit-learn/scikit-learn",
        license_id="BSD-3-Clause",
        license_url="https://github.com/scikit-learn/scikit-learn/blob/main/COPYING",
        commit="3c5e5effc5e11309209234507cb768c5fb38971b",
        branch="main",
    ),
    PublicRepository(
        slug="mlflow/mlflow",
        title="MLflow tracking and model examples",
        url="https://github.com/mlflow/mlflow",
        license_id="Apache-2.0",
        license_url="https://github.com/mlflow/mlflow/blob/master/LICENSE.txt",
        commit="5c6af89333648b7e14ae726cc37cb2465f1917e1",
        branch="master",
    ),
    PublicRepository(
        slug="optuna/optuna-examples",
        title="Optuna integration examples",
        url="https://github.com/optuna/optuna-examples",
        license_id="MIT",
        license_url="https://github.com/optuna/optuna-examples/blob/main/LICENSE",
        commit="9450de9b8b042e396bb816fdbe7fb4b5227b3323",
        branch="main",
    ),
    PublicRepository(
        slug="ageron/handson-ml3",
        title="Hands-On Machine Learning notebooks (3rd edition)",
        url="https://github.com/ageron/handson-ml3",
        license_id="Apache-2.0",
        license_url="https://github.com/ageron/handson-ml3/blob/main/LICENSE",
        commit="ef1c9f5ab11083584635b9b3a7846f49c58a6084",
        branch="main",
    ),
    PublicRepository(
        slug="feature-engine/feature_engine",
        title="Feature-engine preprocessing transformers",
        url="https://github.com/feature-engine/feature_engine",
        license_id="BSD-3-Clause",
        license_url=(
            "https://github.com/feature-engine/feature_engine/blob/main/LICENSE.md"
        ),
        commit="d866312874a66364c6dee3c247e69476b478395c",
        branch="main",
    ),
    PublicRepository(
        slug="microsoft/FLAML",
        title="FLAML automated machine learning examples",
        url="https://github.com/microsoft/FLAML",
        license_id="MIT",
        license_url="https://github.com/microsoft/FLAML/blob/main/LICENSE",
        commit="798cacbd638c92fef35b8ba3113eec5564418952",
        branch="main",
    ),
    PublicRepository(
        slug="aladdinpersson/Machine-Learning-Collection",
        title="Machine Learning Collection tutorial notebooks",
        url="https://github.com/aladdinpersson/Machine-Learning-Collection",
        license_id="MIT",
        license_url=(
            "https://github.com/aladdinpersson/Machine-Learning-Collection/"
            "blob/master/LICENSE.txt"
        ),
        commit="558557c7989f0b10fee6e8d8f953d7269ae43d4f",
        branch="master",
    ),
    PublicRepository(
        slug="scikit-learn-contrib/imbalanced-learn",
        title="imbalanced-learn resampling examples",
        url="https://github.com/scikit-learn-contrib/imbalanced-learn",
        license_id="MIT",
        license_url=(
            "https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/LICENSE"
        ),
        commit="8504e95f0160f61d1b617ca66f779646d2ee609e",
        branch="master",
    ),
    PublicRepository(
        slug="automl/auto-sklearn",
        title="auto-sklearn examples",
        url="https://github.com/automl/auto-sklearn",
        license_id="BSD-3-Clause",
        license_url=(
            "https://github.com/automl/auto-sklearn/blob/development/LICENSE.txt"
        ),
        commit="673211252ca508b6f5bb92cf5fa87c6455bbad99",
        branch="development",
    ),
    PublicRepository(
        slug="dmlc/xgboost",
        title="XGBoost Python examples",
        url="https://github.com/dmlc/xgboost",
        license_id="Apache-2.0",
        license_url="https://github.com/dmlc/xgboost/blob/master/LICENSE",
        commit="f109cc0503aef1bcb21e7e0dc0b4e93a1ef9740a",
        branch="master",
    ),
    PublicRepository(
        slug="microsoft/LightGBM",
        title="LightGBM Python examples",
        url="https://github.com/microsoft/LightGBM",
        license_id="MIT",
        license_url="https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE",
        commit="3ee03c26ba8c9e8ddc05fa8687717cae8a7d61d9",
        branch="main",
    ),
    PublicRepository(
        slug="catboost/catboost",
        title="CatBoost tutorials and examples",
        url="https://github.com/catboost/catboost",
        license_id="Apache-2.0",
        license_url="https://github.com/catboost/catboost/blob/master/LICENSE",
        commit="ecf5e12040890db408ad73fda7b99da3dd394354",
        branch="master",
    ),
)

_BY_SLUG = {repository.slug: repository for repository in REPOSITORIES}


def _case(
    repository: str,
    family: str,
    name: str,
    code: str,
    *,
    leaky: bool,
    category: str,
    rationale: str,
    provenance: str,
    path: str,
    notes: str = "",
) -> RealWorldCase:
    repo = _BY_SLUG[repository]
    if provenance not in {"verbatim_excerpt", "reconstructed_excerpt"}:
        raise ValueError(f"unknown provenance class: {provenance}")
    return RealWorldCase(
        name=name,
        repository=repository,
        family=family,
        code=code,
        leaky=leaky,
        category=category,
        label_rationale=rationale,
        provenance=provenance,
        path=path,
        source_url=(f"https://github.com/{repository}/blob/{repo.commit}/{path}"),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# scikit-learn/scikit-learn — examples/preprocessing/plot_scaling_importance.py
# ---------------------------------------------------------------------------
_SKLEARN = "scikit-learn/scikit-learn"
_SKLEARN_SCALING = "examples/preprocessing/plot_scaling_importance.py"
_SKLEARN_CV_PREDICT = "examples/model_selection/plot_cv_predict.py"


_CASES: list[RealWorldCase] = [
    _case(
        _SKLEARN,
        "sklearn-scaling",
        "sklearn_scaling_fit_after_split",
        (
            "from sklearn.datasets import load_wine\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "X, y = load_wine(return_X_y=True, as_frame=True)\n"
            'scaler = StandardScaler().set_output(transform="pandas")\n'
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.30, random_state=42\n"
            ")\n"
            "scaled_X_train = scaler.fit_transform(X_train)\n"
        ),
        leaky=False,
        category="preprocessing_scaling",
        rationale=(
            "The scaler is fitted on X_train only, after the split. Test rows "
            "never influence the scaling statistics."
        ),
        provenance="verbatim_excerpt",
        path=_SKLEARN_SCALING,
    ),
    _case(
        _SKLEARN,
        "sklearn-scaling",
        "sklearn_scaling_for_plot_only",
        (
            'X_plot = X[["proline", "hue"]]\n'
            "X_plot_scaled = scaler.fit_transform(X_plot)\n"
            "clf = KNeighborsClassifier(n_neighbors=20)\n"
            "fit_and_plot_model(X_plot, y, clf, ax1)\n"
            "fit_and_plot_model(X_plot_scaled, y, clf, ax2)\n"
        ),
        leaky=False,
        category="eda_only",
        rationale=(
            "The full-frame fit is used only to draw a decision-boundary figure; "
            "no held-out score is computed from it, so there is no evaluation leak."
        ),
        provenance="verbatim_excerpt",
        path=_SKLEARN_SCALING,
        notes="Illustrates why full-frame transforms are not automatically leaks.",
    ),
    _case(
        _SKLEARN,
        "sklearn-pca",
        "sklearn_pca_fit_before_split",
        (
            "from sklearn.decomposition import PCA\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.pipeline import make_pipeline\n"
            "\n"
            "components = PCA(n_components=5).fit_transform(X)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    components, y, random_state=0\n"
            ")\n"
            "model = make_pipeline(LogisticRegression()).fit(X_train, y_train)\n"
            "print(model.score(X_test, y_test))\n"
        ),
        leaky=True,
        category="preprocessing_scaling",
        rationale=(
            "PCA learns its components from every row, including rows that later "
            "become the test split. The reported score is therefore optimistic."
        ),
        provenance="reconstructed_excerpt",
        path=_SKLEARN_SCALING,
        notes=(
            "Condensed variant of the example's PCA section; upstream fits PCA on "
            "X_train, so this is a leaky mutation of a clean idiom."
        ),
    ),
    _case(
        _SKLEARN,
        "sklearn-cv",
        "sklearn_pipeline_inside_cross_val",
        (
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.model_selection import KFold, cross_val_predict\n"
            "from sklearn.pipeline import make_pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "pipeline = make_pipeline(StandardScaler(), Ridge())\n"
            "cv = KFold(n_splits=5, shuffle=True, random_state=0)\n"
            "predicted = cross_val_predict(pipeline, X, y, cv=cv)\n"
        ),
        leaky=False,
        category="cv_misuse",
        rationale=(
            "The scaler lives inside the pipeline, so cross_val_predict refits it "
            "on each training fold and transforms the held-out fold with training "
            "statistics only."
        ),
        provenance="reconstructed_excerpt",
        path=_SKLEARN_CV_PREDICT,
    ),
    # -----------------------------------------------------------------------
    # mlflow/mlflow — examples/sklearn_elasticnet_wine/train.py
    # -----------------------------------------------------------------------
    _case(
        "mlflow/mlflow",
        "mlflow-wine",
        "mlflow_wine_split_before_fit",
        (
            "import pandas as pd\n"
            "from sklearn.linear_model import ElasticNet\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'data = pd.read_csv("winequality-red.csv", sep=";")\n'
            "train, test = train_test_split(data)\n"
            'train_x = train.drop(["quality"], axis=1)\n'
            'test_x = test.drop(["quality"], axis=1)\n'
            'train_y = train[["quality"]]\n'
            'test_y = test[["quality"]]\n'
            "lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)\n"
            "lr.fit(train_x, train_y)\n"
            "rmse = mean_squared_error(test_y, lr.predict(test_x)) ** 0.5\n"
        ),
        leaky=False,
        category="preprocessing_order",
        rationale=(
            "Rows are split before any fitting; the model only sees training rows "
            "and the metric is computed on the untouched test partition."
        ),
        provenance="verbatim_excerpt",
        path="examples/sklearn_elasticnet_wine/train.py",
    ),
    _case(
        "mlflow/mlflow",
        "mlflow-wine",
        "mlflow_wine_scaler_on_full_frame",
        (
            "import pandas as pd\n"
            "from sklearn.linear_model import ElasticNet\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            'data = pd.read_csv("winequality-red.csv", sep=";")\n'
            'features = data.drop(columns=["quality"])\n'
            "scaled = StandardScaler().fit_transform(features)\n"
            "train_x, test_x, train_y, test_y = train_test_split(\n"
            '    scaled, data["quality"], random_state=42\n'
            ")\n"
            "lr = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(train_x, train_y)\n"
        ),
        leaky=True,
        category="preprocessing_scaling",
        rationale=(
            "StandardScaler is fitted on all rows before the split, so test rows "
            "contribute their mean and variance to the training features."
        ),
        provenance="reconstructed_excerpt",
        path="examples/sklearn_elasticnet_wine/train.py",
        notes="Leaky variant of the shipped MLflow example.",
    ),
    _case(
        "mlflow/mlflow",
        "mlflow-pipeline",
        "mlflow_pipeline_logged",
        (
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, stratify=y, random_state=0\n"
            ")\n"
            "model = Pipeline([\n"
            '    ("impute", SimpleImputer(strategy="median")),\n'
            '    ("scale", StandardScaler()),\n'
            '    ("clf", LogisticRegression(max_iter=1000)),\n'
            "])\n"
            "model.fit(X_train, y_train)\n"
            'mlflow.sklearn.log_model(model, name="model")\n'
        ),
        leaky=False,
        category="complex_transform",
        rationale=(
            "Imputation and scaling are pipeline steps fitted on the training "
            "partition; the serialized model carries the same fitted state."
        ),
        provenance="reconstructed_excerpt",
        path="examples/sklearn_logistic_regression/train.py",
    ),
    _case(
        "mlflow/mlflow",
        "mlflow-tuning",
        "mlflow_grid_search_on_test",
        (
            "from sklearn.model_selection import GridSearchCV, train_test_split\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "search = GridSearchCV(\n"
            '    RandomForestClassifier(), {"max_depth": [2, 4, 8]}, cv=5\n'
            ")\n"
            "search.fit(X_test, y_test)\n"
            "print(search.best_score_)\n"
        ),
        leaky=True,
        category="tuning",
        rationale=(
            "Hyperparameters are selected by fitting the search on the test "
            "partition, so the reported best score is not held out at all."
        ),
        provenance="reconstructed_excerpt",
        path="examples/sklearn_elasticnet_wine/train.py",
    ),
    # -----------------------------------------------------------------------
    # optuna/optuna-examples — sklearn/sklearn_simple.py
    # -----------------------------------------------------------------------
    _case(
        "optuna/optuna-examples",
        "optuna-cv",
        "optuna_cross_val_objective",
        (
            "import optuna\n"
            "import sklearn.datasets\n"
            "import sklearn.ensemble\n"
            "import sklearn.model_selection\n"
            "\n"
            "def objective(trial):\n"
            "    iris = sklearn.datasets.load_iris()\n"
            "    x, y = iris.data, iris.target\n"
            '    max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)\n'
            "    classifier_obj = sklearn.ensemble.RandomForestClassifier(\n"
            "        max_depth=max_depth, n_estimators=10\n"
            "    )\n"
            "    score = sklearn.model_selection.cross_val_score(\n"
            "        classifier_obj, x, y, n_jobs=-1, cv=3\n"
            "    )\n"
            "    return score.mean()\n"
        ),
        leaky=False,
        category="tuning",
        rationale=(
            "There is no separate test partition in this example, so nothing is "
            "held out and nothing is contaminated. Tuning on the same CV folds "
            "does bias the reported score, but that is selection bias, not leakage."
        ),
        provenance="reconstructed_excerpt",
        path="sklearn/sklearn_simple.py",
        notes="Structure preserved; the suggest call is named for readability.",
    ),
    _case(
        "optuna/optuna-examples",
        "optuna-target",
        "optuna_target_encoding_full_data",
        (
            "import optuna\n"
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "def objective(trial):\n"
            '    global_means = df.groupby("city")["target"].mean()\n'
            '    df["city_encoded"] = df["city"].map(global_means)\n'
            "    X_train, X_test, y_train, y_test = train_test_split(\n"
            '        df[["city_encoded", "age"]], df["target"], random_state=0\n'
            "    )\n"
            '    params = {"max_depth": trial.suggest_int("max_depth", 2, 12)}\n'
            "    model = DecisionTreeClassifier(**params).fit(X_train, y_train)\n"
            "    return model.score(X_test, y_test)\n"
        ),
        leaky=True,
        category="target_encoding",
        rationale=(
            "The target mean per city is computed on every row, then mapped onto "
            "training and test rows alike. Each test row's encoding contains its "
            "own label contribution."
        ),
        provenance="reconstructed_excerpt",
        path="sklearn/sklearn_simple.py",
    ),
    _case(
        "optuna/optuna-examples",
        "optuna-pipeline",
        "optuna_pipeline_cv_objective",
        (
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.pipeline import make_pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.svm import SVC\n"
            "\n"
            "def objective(trial):\n"
            '    c = trial.suggest_float("svc_c", 1e-3, 1e3, log=True)\n'
            "    model = make_pipeline(StandardScaler(), SVC(C=c))\n"
            "    scores = cross_val_score(model, X, y, cv=5)\n"
            "    return scores.mean()\n"
        ),
        leaky=False,
        category="cv_misuse",
        rationale=(
            "StandardScaler is part of the pipeline passed to cross_val_score, so "
            "every fold fits its own scaler on training rows only."
        ),
        provenance="reconstructed_excerpt",
        path="sklearn/sklearn_simple.py",
    ),
    _case(
        "optuna/optuna-examples",
        "optuna-group",
        "optuna_group_kfold_without_groups",
        (
            "from sklearn.model_selection import GroupKFold, cross_val_score\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "kf = GroupKFold(n_splits=5)\n"
            "model = RandomForestClassifier(n_estimators=50, random_state=0)\n"
            "scores = cross_val_score(model, X, y, cv=kf)\n"
        ),
        leaky=True,
        category="grouping",
        rationale=(
            "GroupKFold needs a groups array. Without groups= it raises or falls "
            "back to positional behaviour, so repeated entities can still cross "
            "the fold boundary."
        ),
        provenance="reconstructed_excerpt",
        path="sklearn/sklearn_simple.py",
    ),
    # -----------------------------------------------------------------------
    # ageron/handson-ml3 — 02_end_to_end_machine_learning_project.ipynb
    # -----------------------------------------------------------------------
    _case(
        "ageron/handson-ml3",
        "homl-stratified",
        "homl_stratified_holdout",
        (
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'housing["income_cat"] = pd.cut(\n'
            '    housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf],\n'
            "    labels=[1, 2, 3, 4, 5],\n"
            ")\n"
            "strat_train_set, strat_test_set = train_test_split(\n"
            '    housing, test_size=0.2, stratify=housing["income_cat"],\n'
            "    random_state=42,\n"
            ")\n"
            "for dataset in (strat_train_set, strat_test_set):\n"
            '    dataset.drop("income_cat", axis=1, inplace=True)\n'
        ),
        leaky=False,
        category="grouping",
        rationale=(
            "Binning is target-independent and used only as a stratification key, "
            "then dropped. No target statistic is shared across the boundary."
        ),
        provenance="reconstructed_excerpt",
        path="02_end_to_end_machine_learning_project.ipynb",
        notes="Notebook cell sequence condensed into one unit.",
    ),
    _case(
        "ageron/handson-ml3",
        "homl-pipeline",
        "homl_pipeline_fit_on_train",
        (
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.pipeline import make_pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "\n"
            "num_pipeline = make_pipeline(\n"
            '    SimpleImputer(strategy="median"), StandardScaler()\n'
            ")\n"
            "preprocess = ColumnTransformer([\n"
            '    ("num", num_pipeline, num_attribs),\n'
            '    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),\n'
            "])\n"
            "housing_prepared = preprocess.fit_transform(housing)\n"
        ),
        leaky=False,
        category="preprocessing_order",
        rationale=(
            "In this notebook housing is already the training partition, so fitting "
            "the preprocessing pipeline on it does not touch the test set."
        ),
        provenance="reconstructed_excerpt",
        path="02_end_to_end_machine_learning_project.ipynb",
    ),
    _case(
        "ageron/handson-ml3",
        "homl-pipeline",
        "homl_pipeline_rebuilt_each_cv_fold",
        (
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "scores = cross_val_score(\n"
            "    make_pipeline(preprocess, RandomForestRegressor(random_state=42)),\n"
            '    housing, housing_labels, scoring="neg_mean_squared_error", cv=10,\n'
            ")\n"
            "rmse_scores = np.sqrt(-scores)\n"
        ),
        leaky=False,
        category="cv_misuse",
        rationale=(
            "The fitted preprocessing object is cloned inside the pipeline for "
            "each fold; cross_val_score refits it on the fold's training rows."
        ),
        provenance="reconstructed_excerpt",
        path="02_end_to_end_machine_learning_project.ipynb",
    ),
    _case(
        "ageron/handson-ml3",
        "homl-scaling",
        "homl_scaler_before_split",
        (
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "scaled = StandardScaler().fit_transform(housing_num)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    scaled, housing_labels, test_size=0.2, random_state=42\n"
            ")\n"
            "model = LinearRegression().fit(X_train, y_train)\n"
        ),
        leaky=True,
        category="preprocessing_scaling",
        rationale=(
            "The scaler sees the held-out rows before the split, leaking their "
            "distribution into the training features."
        ),
        provenance="reconstructed_excerpt",
        path="02_end_to_end_machine_learning_project.ipynb",
    ),
    # -----------------------------------------------------------------------
    # feature-engine/feature_engine — docs examples
    # -----------------------------------------------------------------------
    _case(
        "feature-engine/feature_engine",
        "feature-engine-target",
        "feature_engine_target_encoder_in_pipeline",
        (
            "from feature_engine.encoding import MeanEncoder\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import cross_val_score\n"
            "\n"
            "pipe = Pipeline([\n"
            '    ("encoder", MeanEncoder(variables=["city", "channel"])),\n'
            '    ("model", LogisticRegression(max_iter=500)),\n'
            "])\n"
            "scores = cross_val_score(pipe, X, y, cv=5)\n"
        ),
        leaky=False,
        category="target_encoding",
        rationale=(
            "MeanEncoder is fitted inside the pipeline, so each CV fold derives its "
            "encoding from training rows only."
        ),
        provenance="reconstructed_excerpt",
        path="docs/user_guide/encoding/MeanEncoder.rst",
    ),
    _case(
        "feature-engine/feature_engine",
        "feature-engine-target",
        "feature_engine_target_encoder_full_fit",
        (
            "from feature_engine.encoding import MeanEncoder\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'encoder = MeanEncoder(variables=["city", "channel"])\n'
            "X_encoded = encoder.fit_transform(X, y)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_encoded, y, random_state=0\n"
            ")\n"
            "model.fit(X_train, y_train)\n"
            "print(model.score(X_test, y_test))\n"
        ),
        leaky=True,
        category="target_encoding",
        rationale=(
            "The encoder is fitted with the full target vector before the split, so "
            "test-row labels shape the category means used as features."
        ),
        provenance="reconstructed_excerpt",
        path="docs/user_guide/encoding/MeanEncoder.rst",
    ),
    _case(
        "feature-engine/feature_engine",
        "feature-engine-selection",
        "feature_engine_drop_constant_before_split",
        (
            "from feature_engine.selection import DropConstantFeatures\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "selector = DropConstantFeatures(tol=0.98)\n"
            "X_reduced = selector.fit_transform(X)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_reduced, y, random_state=0\n"
            ")\n"
        ),
        leaky=True,
        category="feature_selection",
        rationale=(
            "Feature filtering uses the full frame, so held-out rows decide which "
            "columns survive. The effect is small here but the ordering is wrong."
        ),
        provenance="reconstructed_excerpt",
        path="docs/user_guide/selection/DropConstantFeatures.rst",
    ),
    _case(
        "feature-engine/feature_engine",
        "feature-engine-selection",
        "feature_engine_selector_after_split",
        (
            "from feature_engine.selection import DropCorrelatedFeatures\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, random_state=0\n"
            ")\n"
            "selector = DropCorrelatedFeatures(threshold=0.9)\n"
            "X_train_reduced = selector.fit_transform(X_train)\n"
            "X_test_reduced = selector.transform(X_test)\n"
        ),
        leaky=False,
        category="feature_selection",
        rationale=(
            "The correlation structure is learned on training rows only and then "
            "applied to the test partition."
        ),
        provenance="reconstructed_excerpt",
        path="docs/user_guide/selection/DropCorrelatedFeatures.rst",
    ),
    # -----------------------------------------------------------------------
    # microsoft/FLAML — examples
    # -----------------------------------------------------------------------
    _case(
        "microsoft/FLAML",
        "flaml-automl",
        "flaml_split_before_automl",
        (
            "from flaml import AutoML\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=0\n"
            ")\n"
            "automl = AutoML()\n"
            'automl.fit(X_train, y_train, task="classification", time_budget=60)\n'
            "print(automl.score(X_test, y_test))\n"
        ),
        leaky=False,
        category="tuning",
        rationale=(
            "AutoML searches and fits on the training partition only; the test "
            "partition is scored once at the end."
        ),
        provenance="reconstructed_excerpt",
        path="test/automl/test_classification.py",
    ),
    _case(
        "microsoft/FLAML",
        "flaml-timeseries",
        "flaml_timeseries_random_split",
        (
            "import pandas as pd\n"
            "from flaml import AutoML\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'df = df.sort_values("timestamp")\n'
            "train, test = train_test_split(df, test_size=0.2, random_state=0)\n"
            "automl = AutoML()\n"
            'automl.fit(train.drop(columns=["demand"]), train["demand"],\n'
            '           task="regression", time_budget=60)\n'
        ),
        leaky=True,
        category="temporal",
        rationale=(
            "Sorting does nothing when train_test_split shuffles by default, so "
            "later timestamps can land in training while earlier ones are tested."
        ),
        provenance="reconstructed_excerpt",
        path="test/automl/test_regression.py",
    ),
    _case(
        "microsoft/FLAML",
        "flaml-group",
        "flaml_random_split_with_customer_id",
        (
            "from flaml import AutoML\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'features = df[["customer_id", "tenure", "monthly_charge"]]\n'
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            '    features, df["churn"], test_size=0.2, random_state=0\n'
            ")\n"
            "automl = AutoML()\n"
            'automl.fit(X_train, y_train, task="classification", time_budget=60)\n'
        ),
        leaky=True,
        category="grouping",
        rationale=(
            "Rows are split randomly and customer_id is kept as a feature, so the "
            "same customer can appear in both partitions and be memorised by id."
        ),
        provenance="reconstructed_excerpt",
        path="test/automl/test_classification.py",
    ),
    _case(
        "microsoft/FLAML",
        "flaml-impute",
        "flaml_imputer_full_data",
        (
            "from sklearn.impute import KNNImputer\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "imputed = KNNImputer(n_neighbors=5).fit_transform(numeric_features)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    imputed, y, test_size=0.2, random_state=0\n"
            ")\n"
        ),
        leaky=True,
        category="imputation",
        rationale=(
            "KNNImputer borrows values from every row, so test rows help fill "
            "training gaps and vice versa before the split."
        ),
        provenance="reconstructed_excerpt",
        path="test/automl/test_classification.py",
    ),
    # -----------------------------------------------------------------------
    # aladdinpersson/Machine-Learning-Collection — tutorial notebooks
    # -----------------------------------------------------------------------
    _case(
        "aladdinpersson/Machine-Learning-Collection",
        "aladdin-notebook",
        "aladdin_notebook_scale_cell_before_split",
        (
            "# In[1]:\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "X_scaled = StandardScaler().fit_transform(X)\n"
            "# In[2]:\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_scaled, y, test_size=0.2, random_state=0\n"
            ")\n"
        ),
        leaky=True,
        category="notebook_order",
        rationale=(
            "Cell 1 fits the scaler on the whole matrix and cell 2 splits the "
            "already-scaled result, so execution order leaks test rows."
        ),
        provenance="reconstructed_excerpt",
        path="ML/Scikit_learn/KNN/knn.ipynb",
    ),
    _case(
        "aladdinpersson/Machine-Learning-Collection",
        "aladdin-notebook",
        "aladdin_notebook_split_cell_before_scale",
        (
            "# In[1]:\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=0, stratify=y\n"
            ")\n"
            "# In[2]:\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler().fit(X_train)\n"
            "X_train = scaler.transform(X_train)\n"
            "X_test = scaler.transform(X_test)\n"
            "# In[3]:\n"
            "from sklearn.neighbors import KNeighborsClassifier\n"
            "knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)\n"
            'print("accuracy", knn.score(X_test, y_test))\n'
        ),
        leaky=False,
        category="notebook_order",
        rationale=(
            "The split cell runs before the scaler cell, and the scaler is fitted "
            "on X_train only."
        ),
        provenance="reconstructed_excerpt",
        path="ML/Scikit_learn/KNN/knn.ipynb",
    ),
    _case(
        "aladdinpersson/Machine-Learning-Collection",
        "aladdin-svm",
        "aladdin_svm_full_scale",
        (
            "from sklearn import svm\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import scale\n"
            "\n"
            "X = scale(X)\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            'clf = svm.SVC(kernel="rbf", C=10)\n'
            "clf.fit(X_train, y_train)\n"
            "print(clf.score(X_test, y_test))\n"
        ),
        leaky=True,
        category="preprocessing_scaling",
        rationale=(
            "sklearn.preprocessing.scale is a function, but it still computes the "
            "mean and variance of every row before the split."
        ),
        provenance="reconstructed_excerpt",
        path="ML/Scikit_learn/SVM/svm.ipynb",
    ),
    _case(
        "aladdinpersson/Machine-Learning-Collection",
        "aladdin-impute",
        "aladdin_knn_imputer_train_only",
        (
            "from sklearn.impute import KNNImputer\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=0\n"
            ")\n"
            "imputer = KNNImputer(n_neighbors=5).fit(X_train)\n"
            "X_train = imputer.transform(X_train)\n"
            "X_test = imputer.transform(X_test)\n"
        ),
        leaky=False,
        category="imputation",
        rationale=(
            "The imputer's neighbour structure is learned on training rows and "
            "applied unchanged to the test partition."
        ),
        provenance="reconstructed_excerpt",
        path="ML/Scikit_learn/KNN/knn.ipynb",
    ),
    # -----------------------------------------------------------------------
    # scikit-learn-contrib/imbalanced-learn — examples
    # -----------------------------------------------------------------------
    _case(
        "scikit-learn-contrib/imbalanced-learn",
        "imblearn-smote",
        "imblearn_smote_before_split",
        (
            "from imblearn.over_sampling import SMOTE\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import classification_report\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_res, y_res = SMOTE(random_state=0, k_neighbors=3).fit_resample(X, y)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_res, y_res, test_size=0.2, random_state=0\n"
            ")\n"
            "model = LogisticRegression(max_iter=1000).fit(X_train, y_train)\n"
            "predictions = model.predict(X_test)\n"
            "print(classification_report(y_test, predictions, digits=3))\n"
        ),
        leaky=True,
        category="resampling",
        rationale=(
            "SMOTE interpolates between rows before the split, so synthetic copies "
            "of what should be test rows appear in training."
        ),
        provenance="reconstructed_excerpt",
        path="examples/over_sampling/plot_smote.py",
    ),
    _case(
        "scikit-learn-contrib/imbalanced-learn",
        "imblearn-smote",
        "imblearn_smote_pipeline_cv",
        (
            "from imblearn.over_sampling import SMOTE\n"
            "from imblearn.pipeline import Pipeline\n"
            "from sklearn.model_selection import cross_val_score\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "\n"
            "pipeline = Pipeline([\n"
            '    ("smote", SMOTE(random_state=0)),\n'
            '    ("model", LogisticRegression(max_iter=1000)),\n'
            "])\n"
            'scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")\n'
        ),
        leaky=False,
        category="resampling",
        rationale=(
            "imblearn.pipeline applies SMOTE inside each training fold, so no "
            "synthetic test row exists."
        ),
        provenance="reconstructed_excerpt",
        path="examples/over_sampling/plot_smote.py",
    ),
    _case(
        "scikit-learn-contrib/imbalanced-learn",
        "imblearn-tomek",
        "imblearn_tomek_links_full_data",
        (
            "from imblearn.under_sampling import TomekLinks\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_clean, y_clean = TomekLinks().fit_resample(X, y)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_clean, y_clean, test_size=0.2, random_state=0\n"
            ")\n"
        ),
        leaky=True,
        category="duplicates",
        rationale=(
            "Under-sampling decisions are made with test rows present, so the "
            "surviving training set depends on held-out neighbours."
        ),
        provenance="reconstructed_excerpt",
        path="examples/under_sampling/plot_tomek_links.py",
    ),
    _case(
        "scikit-learn-contrib/imbalanced-learn",
        "imblearn-cv",
        "imblearn_resample_inside_fold",
        (
            "from imblearn.over_sampling import RandomOverSampler\n"
            "from sklearn.model_selection import StratifiedKFold\n"
            "\n"
            "kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)\n"
            "for train_idx, valid_idx in kf.split(X, y):\n"
            "    X_tr, y_tr = RandomOverSampler().fit_resample(\n"
            "        X[train_idx], y[train_idx]\n"
            "    )\n"
            "    model.fit(X_tr, y_tr)\n"
            "    score = model.score(X[valid_idx], y[valid_idx])\n"
        ),
        leaky=False,
        category="resampling",
        rationale=(
            "Oversampling runs inside the loop on training indices only; the "
            "validation fold is left untouched."
        ),
        provenance="reconstructed_excerpt",
        path="examples/over_sampling/plot_random_over_sampling.py",
    ),
    # -----------------------------------------------------------------------
    # automl/auto-sklearn — examples
    # -----------------------------------------------------------------------
    _case(
        "automl/auto-sklearn",
        "autosklearn-holdout",
        "autosklearn_holdout_before_fit",
        (
            "import autosklearn.classification\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=1\n"
            ")\n"
            "automl = autosklearn.classification.AutoSklearnClassifier(\n"
            "    time_left_for_this_task=60\n"
            ")\n"
            "automl.fit(X_train, y_train)\n"
            "print(automl.score(X_test, y_test))\n"
        ),
        leaky=False,
        category="tuning",
        rationale=(
            "The holdout is created before the search starts and is used once for "
            "the final score."
        ),
        provenance="reconstructed_excerpt",
        path="examples/40_advanced/example_methods_to_load_and_export_models.py",
    ),
    _case(
        "automl/auto-sklearn",
        "autosklearn-preprocess",
        "autosklearn_preprocess_full_data",
        (
            "import autosklearn.classification\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_reduced = PCA(n_components=20).fit_transform(X)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X_reduced, y, test_size=0.2, random_state=1\n"
            ")\n"
            "automl = autosklearn.classification.AutoSklearnClassifier()\n"
            "automl.fit(X_train, y_train)\n"
        ),
        leaky=True,
        category="preprocessing_scaling",
        rationale=(
            "PCA is fitted on the full matrix before the holdout exists, so the "
            "projection is informed by test rows."
        ),
        provenance="reconstructed_excerpt",
        path="examples/40_advanced/example_methods_to_load_and_export_models.py",
    ),
    _case(
        "automl/auto-sklearn",
        "autosklearn-cv",
        "autosklearn_cv_on_test_partition",
        (
            "from sklearn.model_selection import cross_val_score, train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=1\n"
            ")\n"
            "scores = cross_val_score(model, X_test, y_test, cv=5)\n"
            "print(scores.mean(), scores.std())\n"
        ),
        leaky=True,
        category="cv_misuse",
        rationale=(
            "Cross-validation is run on the held-out partition, so every fold "
            "reuses the same test rows and the reported mean is not an out-of-sample "
            "estimate."
        ),
        provenance="reconstructed_excerpt",
        path="examples/40_advanced/example_methods_to_load_and_export_models.py",
    ),
    _case(
        "automl/auto-sklearn",
        "autosklearn-duplicates",
        "autosklearn_duplicates_across_split",
        (
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=1\n"
            ")\n"
            "X_train = pd.concat([X_train, X_test.head(50)])\n"
            "y_train = pd.concat([y_train, y_test.head(50)])\n"
            "model.fit(X_train, y_train)\n"
        ),
        leaky=True,
        category="duplicates",
        rationale=(
            "Fifty test rows are concatenated back into training, so those rows are "
            "both memorised and scored."
        ),
        provenance="reconstructed_excerpt",
        path="examples/40_advanced/example_methods_to_load_and_export_models.py",
    ),
    # -----------------------------------------------------------------------
    # dmlc/xgboost — python examples
    # -----------------------------------------------------------------------
    _case(
        "dmlc/xgboost",
        "xgboost-cv",
        "xgboost_kfold_cv",
        (
            "import xgboost as xgb\n"
            "from sklearn.model_selection import KFold, cross_val_score\n"
            "\n"
            "model = xgb.XGBClassifier(n_estimators=200, max_depth=4)\n"
            "cv = KFold(n_splits=5, shuffle=True, random_state=0)\n"
            "scores = cross_val_score(model, X, y, cv=cv)\n"
        ),
        leaky=False,
        category="cv_misuse",
        rationale=(
            "No preprocessing is fitted outside the folds and no row appears in "
            "both training and validation within a fold."
        ),
        provenance="reconstructed_excerpt",
        path="demo/guide-python/cross_validation.py",
    ),
    _case(
        "dmlc/xgboost",
        "xgboost-earlystop",
        "xgboost_early_stopping_on_test",
        (
            "import xgboost as xgb\n"
            "\n"
            "dtrain = xgb.DMatrix(X_train, label=y_train)\n"
            "dtest = xgb.DMatrix(X_test, label=y_test)\n"
            "booster = xgb.train(\n"
            "    params, dtrain, num_boost_round=500,\n"
            '    evals=[(dtest, "test")], early_stopping_rounds=20,\n'
            ")\n"
            "print(booster.best_score, booster.best_iteration)\n"
        ),
        leaky=True,
        category="tuning",
        rationale=(
            "The number of boosting rounds is selected by watching the test set, so "
            "the test set influences model selection and the final score is "
            "optimistic."
        ),
        provenance="reconstructed_excerpt",
        path="demo/guide-python/basic_walkthrough.py",
    ),
    _case(
        "dmlc/xgboost",
        "xgboost-matrix",
        "xgboost_dmatrix_before_split",
        (
            "import xgboost as xgb\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "dtrain_all = xgb.DMatrix(X, label=y)\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=0\n"
            ")\n"
            "dtrain = xgb.DMatrix(X_train, label=y_train)\n"
            "dtest = xgb.DMatrix(X_test, label=y_test)\n"
        ),
        leaky=True,
        category="complex_transform",
        rationale=(
            "The full DMatrix is built before the split. Building a matrix is "
            "usually harmless, but any binning or categorical vocabulary learned "
            "from it crosses the boundary; the dead full-frame object is a smell."
        ),
        provenance="reconstructed_excerpt",
        path="demo/guide-python/basic_walkthrough.py",
    ),
    _case(
        "dmlc/xgboost",
        "xgboost-timeseries",
        "xgboost_time_ordered_split",
        (
            "import xgboost as xgb\n"
            "import pandas as pd\n"
            "\n"
            'ordered = df.sort_values("date")\n'
            "cutoff = int(len(ordered) * 0.8)\n"
            "train, test = ordered.iloc[:cutoff], ordered.iloc[cutoff:]\n"
            'dtrain = xgb.DMatrix(train[features], label=train["sales"])\n'
            'dtest = xgb.DMatrix(test[features], label=test["sales"])\n'
            "booster = xgb.train(params, dtrain, num_boost_round=100)\n"
        ),
        leaky=False,
        category="temporal",
        rationale=(
            "The split is positional on a time-sorted frame, so every test "
            "timestamp is later than every training timestamp."
        ),
        provenance="reconstructed_excerpt",
        path="demo/guide-python/cross_validation.py",
    ),
    # -----------------------------------------------------------------------
    # microsoft/LightGBM — examples/python-guide/simple_example.py
    # -----------------------------------------------------------------------
    _case(
        "microsoft/LightGBM",
        "lightgbm-simple",
        "lightgbm_early_stopping_on_test",
        (
            "import lightgbm as lgb\n"
            "\n"
            "lgb_train = lgb.Dataset(X_train, y_train)\n"
            "lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)\n"
            "gbm = lgb.train(\n"
            "    params, lgb_train, num_boost_round=20, valid_sets=lgb_eval,\n"
            "    callbacks=[lgb.early_stopping(stopping_rounds=5)],\n"
            ")\n"
            "y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)\n"
            "rmse_test = mean_squared_error(y_test, y_pred) ** 0.5\n"
        ),
        leaky=True,
        category="tuning",
        rationale=(
            "This is the shipped simple example: early stopping watches the test "
            "set, so the reported test RMSE is selected using the test set itself."
        ),
        provenance="verbatim_excerpt",
        path="examples/python-guide/simple_example.py",
        notes=(
            "Labelled leaky on review even though it is an upstream example; the "
            "pattern is test-set model selection, not a broken file."
        ),
    ),
    _case(
        "microsoft/LightGBM",
        "lightgbm-simple",
        "lightgbm_train_valid_files",
        (
            "import lightgbm as lgb\n"
            "\n"
            'df_train = pd.read_csv("regression.train", header=None, sep="\\t")\n'
            'df_valid = pd.read_csv("regression.valid", header=None, sep="\\t")\n'
            "X_train, y_train = df_train.drop(0, axis=1), df_train[0]\n"
            "X_valid, y_valid = df_valid.drop(0, axis=1), df_valid[0]\n"
            "lgb_train = lgb.Dataset(X_train, y_train)\n"
            "lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)\n"
            "gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid])\n"
        ),
        leaky=False,
        category="preprocessing_order",
        rationale=(
            "Training and validation come from separate files and early stopping "
            "uses the validation file, keeping the final test file untouched."
        ),
        provenance="reconstructed_excerpt",
        path="examples/python-guide/simple_example.py",
    ),
    _case(
        "microsoft/LightGBM",
        "lightgbm-categorical",
        "lightgbm_categorical_full_encode",
        (
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'df["city_code"] = pd.factorize(df["city"])[0]\n'
            'df["city_freq"] = df["city"].map(df["city"].value_counts())\n'
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            '    df[["city_code", "city_freq", "age"]], df["churn"],\n'
            "    test_size=0.2, random_state=0,\n"
            ")\n"
        ),
        leaky=True,
        category="target_encoding",
        rationale=(
            "Categorical codes and frequencies are computed on every row, so the "
            "vocabulary and counts used for training include test rows."
        ),
        provenance="reconstructed_excerpt",
        path="examples/python-guide/advanced_example.py",
    ),
    _case(
        "microsoft/LightGBM",
        "lightgbm-group",
        "lightgbm_group_split",
        (
            "import lightgbm as lgb\n"
            "from sklearn.model_selection import GroupShuffleSplit\n"
            "\n"
            "splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)\n"
            'train_idx, test_idx = next(splitter.split(X, y, groups=df["store_id"]))\n'
            "dtrain = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])\n"
            "dtest = lgb.Dataset(X.iloc[test_idx], y.iloc[test_idx])\n"
        ),
        leaky=False,
        category="grouping",
        rationale=(
            "The split is taken by store_id, so a store never appears on both "
            "sides of the evaluation."
        ),
        provenance="reconstructed_excerpt",
        path="examples/python-guide/advanced_example.py",
    ),
    # -----------------------------------------------------------------------
    # catboost/catboost — tutorials
    # -----------------------------------------------------------------------
    _case(
        "catboost/catboost",
        "catboost-target",
        "catboost_target_statistics_full_frame",
        (
            "from catboost import CatBoostClassifier, Pool\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            'pool = Pool(df[features], df["target"], cat_features=["city"])\n'
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            '    df[features], df["target"], test_size=0.2, random_state=0\n'
            ")\n"
            "model = CatBoostClassifier(iterations=200, random_seed=0)\n"
            'model.fit(X_train, y_train, cat_features=["city"])\n'
        ),
        leaky=True,
        category="target_encoding",
        rationale=(
            "The full Pool is built before the split. CatBoost computes ordered "
            "target statistics internally on the frame it is given, so the unused "
            "pool still represents statistics taken from held-out rows."
        ),
        provenance="reconstructed_excerpt",
        path="catboost/tutorials/classification/classification_tutorial.ipynb",
    ),
    _case(
        "catboost/catboost",
        "catboost-target",
        "catboost_pool_after_split",
        (
            "from catboost import CatBoostClassifier, Pool\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            '    df[features], df["target"], test_size=0.2, random_state=0\n'
            ")\n"
            'train_pool = Pool(X_train, y_train, cat_features=["city"])\n'
            'test_pool = Pool(X_test, y_test, cat_features=["city"])\n'
            "model = CatBoostClassifier(iterations=200, random_seed=0)\n"
            "model.fit(train_pool)\n"
        ),
        leaky=False,
        category="target_encoding",
        rationale=(
            "Target statistics are computed inside fit from the training pool only; "
            "the test pool is built after the split."
        ),
        provenance="reconstructed_excerpt",
        path="catboost/tutorials/classification/classification_tutorial.ipynb",
    ),
    _case(
        "catboost/catboost",
        "catboost-duplicates",
        "catboost_duplicate_rows_before_split",
        (
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "\n"
            "df = pd.concat([df, df.sample(frac=0.2, random_state=0)])\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            '    df[features], df["target"], test_size=0.2, random_state=0\n'
            ")\n"
        ),
        leaky=True,
        category="duplicates",
        rationale=(
            "Twenty percent of the rows are duplicated before splitting, so near "
            "copies of test rows are almost guaranteed to land in training."
        ),
        provenance="reconstructed_excerpt",
        path="catboost/tutorials/classification/classification_tutorial.ipynb",
    ),
    _case(
        "catboost/catboost",
        "catboost-timeseries",
        "catboost_ordered_time_split",
        (
            "import pandas as pd\n"
            "from catboost import CatBoostRegressor, Pool\n"
            "\n"
            'ordered = df.sort_values("timestamp").reset_index(drop=True)\n'
            "cutoff = int(len(ordered) * 0.8)\n"
            "train = ordered.iloc[:cutoff]\n"
            "test = ordered.iloc[cutoff:]\n"
            "model = CatBoostRegressor(iterations=300, random_seed=0)\n"
            'model.fit(train[features], train["demand"])\n'
            'score = model.score(test[features], test["demand"])\n'
        ),
        leaky=False,
        category="temporal",
        rationale=(
            "The frame is ordered by timestamp and cut positionally, so training "
            "only contains earlier observations."
        ),
        provenance="reconstructed_excerpt",
        path="catboost/tutorials/regression/regression_tutorial.ipynb",
    ),
]


REALWORLD_CASES: tuple[RealWorldCase, ...] = tuple(_CASES)


def holdout_cases() -> tuple[RealWorldCase, ...]:
    """Return the frozen holdout.  Never merge this into training."""

    return REALWORLD_CASES


def repositories() -> tuple[PublicRepository, ...]:
    return tuple(
        sorted(
            {_BY_SLUG[case.repository] for case in REALWORLD_CASES},
            key=lambda r: r.slug,
        )
    )


def corpus_stats() -> dict[str, object]:
    cases = REALWORLD_CASES
    categories = sorted({case.category for case in cases})
    return {
        "holdout_id": HOLDOUT_ID,
        "role": HOLDOUT_ROLE,
        "cases": len(cases),
        "leaky": sum(1 for case in cases if case.leaky),
        "legitimate": sum(1 for case in cases if not case.leaky),
        "repositories": len({case.repository for case in cases}),
        "families": len({case.family for case in cases}),
        "categories": len(categories),
        "verbatim_excerpts": sum(
            1 for case in cases if case.provenance == "verbatim_excerpt"
        ),
        "reconstructed_excerpts": sum(
            1 for case in cases if case.provenance == "reconstructed_excerpt"
        ),
        "per_category": {
            category: {
                "cases": sum(1 for case in cases if case.category == category),
                "leaky": sum(
                    1 for case in cases if case.category == category and case.leaky
                ),
            }
            for category in categories
        },
        "per_repository": {
            repository.slug: {
                "cases": sum(1 for case in cases if case.repository == repository.slug),
                "families": len(
                    {
                        case.family
                        for case in cases
                        if case.repository == repository.slug
                    }
                ),
                "license": repository.license_id,
            }
            for repository in repositories()
        },
    }


def _manifest() -> list[dict[str, object]]:
    return [
        {
            "name": case.name,
            "repository": case.repository,
            "family": case.family,
            "code": case.code,
            "leaky": case.leaky,
            "category": case.category,
            "label_rationale": case.label_rationale,
            "provenance": case.provenance,
            "path": case.path,
            "source_url": case.source_url,
            "review": case.review,
            "notes": case.notes,
        }
        for case in sorted(REALWORLD_CASES, key=lambda case: case.name)
    ]


def manifest_digest() -> str:
    """SHA-256 of every labelled field, used to prove the holdout was frozen."""

    payload = json.dumps(_manifest(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest() -> dict[str, object]:
    """Machine-readable freeze record for ``docs/REALWORLD_CORPUS.md``."""

    return {
        "holdout_id": HOLDOUT_ID,
        "role": HOLDOUT_ROLE,
        "retrieved": RETRIEVED,
        "digest": manifest_digest(),
        "stats": corpus_stats(),
        "repositories": [
            {
                "slug": repository.slug,
                "title": repository.title,
                "url": repository.url,
                "license": repository.license_id,
                "license_url": repository.license_url,
                "commit": repository.commit,
                "branch": repository.branch,
            }
            for repository in repositories()
        ],
        "cases": [
            {
                "name": case.name,
                "repository": case.repository,
                "family": case.family,
                "leaky": case.leaky,
                "category": case.category,
                "provenance": case.provenance,
                "path": case.path,
                "source_url": case.source_url,
                "label_rationale": case.label_rationale,
                "notes": case.notes,
            }
            for case in sorted(REALWORLD_CASES, key=lambda case: case.name)
        ],
    }


__all__ = [
    "HOLDOUT_ID",
    "HOLDOUT_ROLE",
    "REALWORLD_CASES",
    "REPOSITORIES",
    "RETRIEVED",
    "PublicRepository",
    "RealWorldCase",
    "corpus_stats",
    "holdout_cases",
    "manifest",
    "manifest_digest",
    "repositories",
]
