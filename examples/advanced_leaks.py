"""Deliberately leaky pipeline covering the 0.14.0 detectors.

Run ``leaklens --scan examples/advanced_leaks.py --explain`` to see each
finding and its fix.  Nothing here touches the network or runs real training;
the scanner only parses the source.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split

# --- target encoding built from every row, then feature selection on the label.
df = pd.read_csv("claims.csv")
city_target_mean = df.groupby("city")["fraud"].mean()
df["city_te"] = df["city"].map(city_target_mean)
selector = SelectKBest(k=12)
X = selector.fit_transform(df.drop(columns=["fraud"]), df["fraud"])

# --- hyperparameters chosen on the full frame before the held-out split exists.
search = GridSearchCV(RandomForestClassifier(), {"max_depth": [3, 6, 9]}, cv=5)
search.fit(X, df["fraud"])

# --- shuffled split of time-ordered rows, then tuning on the test partition.
ordered = df.sort_values("event_date")
X_train, X_test, y_train, y_test = train_test_split(
    ordered.drop(columns=["fraud"]), ordered["fraud"], random_state=0
)
search.fit(X_test, y_test)

# --- group-aware CV without the entity column.
kf = GroupKFold(n_splits=5)
for train_idx, valid_idx in kf.split(X_train, y_train):
    pass
