from __future__ import annotations

import pandas as pd


def audit(df: pd.DataFrame, target: str | None = None) -> dict:
    out = {
        "rows": len(df),
        "cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_rate": {
            c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0
        },
        "cardinality": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    if target is not None and target in df.columns:
        vc = df[target].value_counts(normalize=True)
        out["target_distribution"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        out["minority_rate"] = round(float(vc.min()), 6)
    return out
