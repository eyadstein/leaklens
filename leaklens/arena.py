from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .leakage import scan


def _models(clf, seed):
    if clf:
        return {
            "linear": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "tree_ensemble": HistGradientBoostingClassifier(random_state=seed),
            "neural_net": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                early_stopping=True,
                random_state=seed,
            ),
        }
    return {
        "linear": Ridge(),
        "tree_ensemble": HistGradientBoostingRegressor(random_state=seed),
        "neural_net": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    }


def _prep(num, cat):
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=20,
                    sparse_output=False,
                ),
                cat,
            ),
        ]
    )


def _metrics(clf, y, pipe, X):
    out = {}
    pred = pipe.predict(X)
    if clf:
        proba = pipe.predict_proba(X)
        if proba.shape[1] == 2:
            s = proba[:, 1]
            try:
                out["roc_auc"] = roc_auc_score(y, s)
                out["pr_auc"] = average_precision_score(y, s)
                p, r, _ = precision_recall_curve(y, s)
                out["best_f1"] = float((2 * p * r / np.maximum(p + r, 1e-12)).max())
            except ValueError:
                pass
            out["f1"] = f1_score(y, pred, zero_division=0)
            out["precision"] = precision_score(y, pred, zero_division=0)
            out["recall"] = recall_score(y, pred, zero_division=0)
        else:
            out["f1_macro"] = f1_score(y, pred, average="macro", zero_division=0)
            out["accuracy"] = accuracy_score(y, pred)
    else:
        out["mae"] = mean_absolute_error(y, pred)
        out["rmse"] = float(np.sqrt(mean_squared_error(y, pred)))
        out["r2"] = r2_score(y, pred)
    return {k: float(v) for k, v in out.items()}


def run(
    df,
    target,
    time_col=None,
    out_dir="models",
    tracking_uri="sqlite:///mlflow.db",
    experiment="leaklens-arena",
    drop_leaks=True,
    seed=0,
):
    d = df[df[target].notna()].copy()
    if len(d) < 50:
        raise ValueError("need at least 50 labeled rows")
    dropped = []
    if drop_leaks:
        dropped = [
            f.column
            for f in scan(d, target)
            if f.kind == "id_like"
            or (f.kind == "target_leakage" and f.severity == "high")
        ]
    y_raw = d[target]
    clf = y_raw.dtype.kind not in "iuf" or y_raw.nunique() <= 20
    if clf:
        cats = y_raw.astype("category")
        y = cats.cat.codes.to_numpy()
        classes = [str(c) for c in cats.cat.categories]
    else:
        y = y_raw.astype(float).to_numpy()
        classes = []
    drop = {target, *dropped}
    if time_col:
        drop.add(time_col)
    X = d.drop(columns=[c for c in drop if c in d.columns])
    X = X.drop(columns=X.select_dtypes(include=["datetime", "datetimetz"]).columns)
    for c in X.select_dtypes("bool").columns:
        X[c] = X[c].astype(int)
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if num:
        X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    for c in cat:
        X[c] = X[c].astype(str)
    n = len(d)
    if time_col:
        t = pd.to_datetime(d[time_col], errors="coerce").to_numpy()
        order = np.argsort(t, kind="stable")
        cut = int(n * 0.8)
        tr_i, te_i = order[:cut], order[cut:]
    else:
        strat = y if clf and np.bincount(y).min() >= 2 else None
        tr_i, te_i = train_test_split(
            np.arange(n), test_size=0.2, stratify=strat, random_state=seed
        )
    if clf and len(np.unique(y[tr_i])) < 2:
        raise ValueError("training split has a single class")
    Xtr, Xte, ytr, yte = X.iloc[tr_i], X.iloc[te_i], y[tr_i], y[te_i]
    primary = ("pr_auc" if len(classes) == 2 else "f1_macro") if clf else "rmse"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    rows, pipes = [], {}
    for name, model in _models(clf, seed).items():
        pipe = Pipeline([("prep", _prep(num, cat)), ("model", model)])
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model": name,
                    "task": "classification" if clf else "regression",
                    "split": "temporal" if time_col else "random",
                    "n_train": len(tr_i),
                    "n_test": len(te_i),
                    "dropped": ",".join(map(str, dropped)) or "none",
                }
            )
            pipe.fit(Xtr, ytr)
            m = _metrics(clf, yte, pipe, Xte)
            mlflow.log_metrics({k: v for k, v in m.items() if np.isfinite(v)})
        pipes[name] = pipe
        rows.append({"model": name, **m})
    res = pd.DataFrame(rows)
    if primary not in res:
        res[primary] = np.nan
    res = res.sort_values(
        primary, ascending=(primary == "rmse"), na_position="last"
    ).reset_index(drop=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipes[res.loc[0, "model"]],
            "features": list(X.columns),
            "target": target,
            "task": "classification" if clf else "regression",
            "classes": classes,
            "dropped": dropped,
            "model": res.loc[0, "model"],
        },
        Path(out_dir) / "best.joblib",
    )
    res.attrs["dropped"] = dropped
    res.attrs["primary"] = primary
    return res
