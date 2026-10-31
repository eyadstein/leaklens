FIXES = {
    "preprocess_outside_cv": "Put the preprocessing inside a sklearn Pipeline (or make_pipeline) and pass the pipeline to cross_val_score / GridSearchCV so every fold fits it on its own training part.",
    "resample_outside_cv": "Use imblearn.pipeline.Pipeline with the sampler as a step so resampling happens inside each training fold only.",
    "preprocess_before_split": "Split first, then fit on the training part only. Wrap the steps in sklearn.pipeline.Pipeline and fit/cross-validate the pipeline.",
    "transformer_before_split": "Split first, then fit the transformer on the training part only (put it inside a Pipeline).",
    "resample_before_split": "Resample only the training fold: imblearn.pipeline.Pipeline([('smote', SMOTE()), ('clf', model)]).",
    "fit_on_full_data": "Fit on the training split only (X_train). If this is the final refit after evaluation it is fine; otherwise put the steps in a Pipeline.",
    "fit_on_eval_data": "Fit on X_train, then call .transform() (not .fit_transform()) on validation/test data.",
    "encoder_before_split": "Usually harmless, but fit encoders on training data and use handle_unknown='ignore' to be safe.",
    "target_leakage": "Drop the column or confirm it is known at prediction time; on its own it predicts the target.",
    "id_like": "Drop identifier columns; they let models memorize rows instead of learning patterns.",
    "split_overlap": "Deduplicate before splitting, or split by group so the same row cannot land on both sides.",
    "temporal_leakage": "Split by time: train on earlier data and test on later data.",
    "group_overlap": "Split by entity with GroupShuffleSplit or GroupKFold so no entity appears in both splits.",
    "group_leakage_risk": "Split by entity with GroupShuffleSplit or GroupKFold instead of a random split.",
    "no_random_state": "Pass random_state=<int> to train_test_split() and to KFold-family constructors (KFold(..., shuffle=True, random_state=...)); RepeatedKFold and RepeatedStratifiedKFold shuffle too and need it.",
    "target_encode_before_split": "Compute the target encoding from the training fold only (sklearn.preprocessing.TargetEncoder in a Pipeline, or out-of-fold encoding) so held-out rows never help build it.",
    "manual_scale_before_split": "Compute the statistics (mean, std, min, max, ...) on the training split only and apply them to the test split, or use a scaler inside a Pipeline so held-out rows never help set the scale.",
    "fillna_before_split": "Impute using training data only: split first, then compute the mean/median on the training split and apply it to both splits, or use SimpleImputer inside a Pipeline.",
    "group_stat_before_split": "Compute group statistics on the training split only, or wrap the encoding in a Pipeline and evaluate with cross-validation.",
}


def advice(kind: str) -> str:
    return FIXES.get(kind, "")