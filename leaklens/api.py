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
from .readers import DatasetLimitError, read_dataset
from .scanner import scan_source

app = FastAPI(title="LeakLens", version=__version__)


class ScanReq(BaseModel):
    source: str
    name: str = "snippet.py"


class PredictReq(BaseModel):
    rows: list[dict]
    model_path: str = "models/best.joblib"  # relative path; resolved against CWD below


class UploadLimits:
    """DoS guard for uploaded datasets in POST /audit (size cap, no secrets in logs)."""

    MAX_BYTES = 8 * 1024 * 1024  # 8 MiB
    MAX_ROWS = 50_000


def validate_upload_size(name: str, byte_count: int) -> None:
    if byte_count > UploadLimits.MAX_BYTES:
        raise HTTPException(413, f"upload too large (>8 MiB): {name}")


def _finding_payload(finding) -> dict:
    """Additive payload: existing fields plus privacy-safe evidence/fingerprint."""
    return finding.to_dict()


def _check_upload(name: str, data: bytes) -> None:
    validate_upload_size(name, len(data))
    # Text formats are row-counted here. Binary containers (xlsx, parquet,
    # feather, json) are deliberately not newline-counted -- their row count
    # is only known after decompression, which is enforced in read_dataset().
    if name.lower().endswith((".xlsx", ".parquet", ".feather", ".json")):
        return
    rows = data.count(b"\n")
    if rows > UploadLimits.MAX_ROWS:
        raise HTTPException(413, f"too many rows (>50k): {name}")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/audit")
async def audit_csv(
    file: UploadFile = File(...),  # noqa: B008
    target: str = Form(...),
    threshold: float = Form(0.95),
):
    try:
        data = io.BytesIO(await file.read())
        _check_upload(file.filename or "upload", data.getvalue())
        df = read_dataset(data, file.filename or "", max_rows=UploadLimits.MAX_ROWS)
    except DatasetLimitError as e:
        raise HTTPException(413, str(e)) from None
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - re-wrap as HTTP 400 (fail closed)
        raise HTTPException(400, f"bad input: {e}")
    if target not in df.columns:
        raise HTTPException(422, f"target '{target}' not in columns")
    fs = scan(df, target, threshold)
    return {
        "trust_score": score(fs),
        "audit": audit(df, target),
        "findings": [_finding_payload(f) for f in fs],
    }


@app.post("/scan")
def scan_code(req: ScanReq):
    if len(req.source) > UploadLimits.MAX_BYTES:
        raise HTTPException(413, f"scan source too large (>8 MiB): {req.name}")
    fs = scan_source(req.source, req.name)
    return {"trust_score": score(fs), "findings": [_finding_payload(f) for f in fs]}


@app.post("/predict")
def predict(req: PredictReq):
    p = Path(req.model_path)
    # Reject absolute or traversal paths; the model must live under the deployer's
    # models/ tree so callers cannot load arbitrary (e.g. attacker-supplied) jobs.
    # WARNING: joblib.load() unpickles the file; it is pickle-based deserialization
    # and can execute arbitrary code if the model file is malicious. Only place
    # model artifacts you trust in ./models/. Path validation alone does not make
    # an untrusted joblib file safe.
    resolved = (Path.cwd() / p).resolve()
    models_root = (Path.cwd() / "models").resolve()
    if models_root not in resolved.parents:
        raise HTTPException(
            422, f"model_path must be under ./models/ (got: {req.model_path})"
        )
    if not resolved.exists():
        raise HTTPException(
            404, "model not found; run: leaklens data.csv --target T --arena"
        )
    try:
        b = joblib.load(resolved)
    except Exception:  # noqa: BLE001 - re-wrap as HTTP 400 (fail closed)
        # Corrupt/malicious pickle surfaces as EOFError, UnpicklingError
        # (KeyError/ValueError inside) or OSError; treat any load failure
        # as invalid model rather than leaking internal traces. pickle
        # can execute arbitrary code, but the trusted ./models/ gate
        # above already ensures only operator-placed artifacts reach here.
        raise HTTPException(400, "invalid model file") from None
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
            out.append(
                {
                    "label": b["classes"][i],
                    "score": float(pr[1] if len(pr) == 2 else pr[i]),
                }
            )
    else:
        out = [{"value": float(v)} for v in pipe.predict(X)]
    return {"model": b["model"], "task": b["task"], "predictions": out}
