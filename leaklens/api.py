from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import __version__
from .audit import audit
from .leakage import scan, score
from .readers import read_dataset
from .scanner import scan_source

app = FastAPI(title="LeakLens", version=__version__)


class ScanReq(BaseModel):
    source: str
    name: str = "snippet.py"


class PredictReq(BaseModel):
    rows: list[dict]
    model_path: str = "models/best.joblib"


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/audit")
async def audit_csv(file: UploadFile = File(...), target: str = Form(...), threshold: float = Form(0.95)):
    try:
        data = io.BytesIO(await file.read())
        df = read_dataset(data, file.filename or "")
    except Exception as e:
        raise HTTPException(400, f"bad input: {e}")
    if target not in df.columns:
        raise HTTPException(422, f"target '{target}' not in columns")
    fs = scan(df, target, threshold)
    return {"trust_score": score(fs), "audit": audit(df, target), "findings": [f.to_dict() for f in fs]}


@app.post("/scan")
def scan_code(req: ScanReq):
    fs = scan_source(req.source, req.name)
    return {"trust_score": score(fs), "findings": [f.to_dict() for f in fs]}


@app.post("/predict")
def predict(req: PredictReq):
    p = Path(req.model_path)
    if not p.exists():
        raise HTTPException(404, "model not found; run: leaklens data.csv --target T --arena")
    b = joblib.load(p)
    df = pd.DataFrame(req.rows)
    miss = [c for c in b["features"] if c not in df.columns]
    if miss:
        raise HTTPException(422, f"missing features: {miss}")
    X = df[b["features"]].copy()
    pipe = b["pipeline"]
    tf = pipe.named_steps["prep"].transformers_
    for c in tf[0][2]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in tf[1][2]:
        X[c] = X[c].astype(str)
    if b["task"] == "classification":
        out = []
        for pr in pipe.predict_proba(X):
            i = int(np.argmax(pr))
            out.append({"label": b["classes"][i], "score": float(pr[1] if len(pr) == 2 else pr[i])})
    else:
        out = [{"value": float(v)} for v in pipe.predict(X)]
    return {"model": b["model"], "task": b["task"], "predictions": out}
