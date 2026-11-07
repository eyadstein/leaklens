from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import __version__
from .audit import audit
from .baseline import BaselineError
from .drift import DEFAULT_SETTINGS as DEFAULT_DRIFT_SETTINGS
from .drift import audit_drift
from .intelligence import analyze_findings
from .leakage import scan, score
from .missingness import audit_missingness
from .policy import apply_policy, parse_baseline, parse_suppressions
from .readers import DatasetLimitError, read_dataset
from .scanner import scan_source
from .sqlscan import SqlDialectError, SqlUnavailable, scan_sql
from .suppression import SuppressionError

app = FastAPI(title="LeakLens", version=__version__)


class ScanReq(BaseModel):
    source: str
    name: str = "snippet.py"
    baseline: dict | None = None
    suppressions: list[dict] | None = None
    dialect: str = "ansi"
    target: str | None = None
    split_column: str | None = None
    schema_map: dict[str, dict[str, str]] | None = None
    intelligence: bool = False


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


def _finding_payloads(outcome) -> list[dict]:
    """Existing fields plus optional status/suppression; never mutates findings."""
    return [outcome.finding_payload(finding) for finding in outcome.findings]


def _policy_payload(outcome) -> dict:
    if outcome.diff is None and outcome.suppression is None:
        return {}
    payload = {"policy": outcome.summary()}
    if outcome.diff is not None and outcome.diff.resolved:
        payload["resolved"] = [entry.to_dict() for entry in outcome.diff.resolved]
    return payload


def _apply_policy(findings, baseline=None, suppressions=None):
    try:
        baseline_doc = parse_baseline(baseline)
        rules = parse_suppressions(suppressions)
    except (BaselineError, SuppressionError) as exc:
        raise HTTPException(422, f"invalid policy: {exc}") from None
    return apply_policy(findings, baseline=baseline_doc, rules=rules)


def _json_form(raw: str | None, name: str):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"{name} is not valid JSON: {exc.msg}") from None


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
    baseline: str | None = Form(None),
    suppressions: str | None = Form(None),
    test: UploadFile | None = File(None),  # noqa: B008
    drift: bool = Form(False),
    intelligence: bool = Form(False),
):
    try:
        data = io.BytesIO(await file.read())
        _check_upload(file.filename or "upload", data.getvalue())
        df = read_dataset(data, file.filename or "", max_rows=UploadLimits.MAX_ROWS)
        test_df = None
        if test is not None:
            raw = await test.read()
            _check_upload(test.filename or "test", raw)
            test_df = read_dataset(
                io.BytesIO(raw),
                test.filename or "",
                max_rows=UploadLimits.MAX_ROWS,
            )
    except DatasetLimitError as e:
        raise HTTPException(413, str(e)) from None
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - re-wrap as HTTP 400 (fail closed)
        raise HTTPException(400, f"bad input: {e}")
    if target not in df.columns:
        raise HTTPException(422, f"target '{target}' not in columns")
    fs = scan(df, target, threshold)
    missing = audit_missingness(df, target)
    fs += missing.findings
    drift_report = None
    if drift:
        drift_report = audit_drift(
            df,
            test_df,
            target=target,
            settings=replace(DEFAULT_DRIFT_SETTINGS, enabled=True),
        )
        fs += drift_report.findings
    outcome = _apply_policy(
        fs,
        baseline=_json_form(baseline, "baseline"),
        suppressions=_json_form(suppressions, "suppressions"),
    )
    payload = {
        "trust_score": score(outcome.active),
        "audit": audit(df, target),
        "missingness": missing.to_dict(),
        "findings": _finding_payloads(outcome),
        **_policy_payload(outcome),
    }
    if drift_report is not None:
        payload["drift"] = drift_report.to_dict()
    if intelligence:
        payload["intelligence"] = analyze_findings(fs).to_dict()
    return payload


@app.post("/scan")
def scan_code(req: ScanReq):
    if len(req.source) > UploadLimits.MAX_BYTES:
        raise HTTPException(413, f"scan source too large (>8 MiB): {req.name}")
    if req.name.lower().endswith(".sql"):
        try:
            fs = scan_sql(
                req.source,
                req.name,
                dialect=req.dialect,
                target=req.target,
                schema=req.schema_map,
                split_column=req.split_column,
            )
        except SqlUnavailable as exc:
            raise HTTPException(422, str(exc)) from None
        except (SqlDialectError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"invalid SQL scan options: {exc}") from None
    else:
        fs = scan_source(req.source, req.name)
    outcome = _apply_policy(fs, baseline=req.baseline, suppressions=req.suppressions)
    payload = {
        "trust_score": score(outcome.active),
        "findings": _finding_payloads(outcome),
        **_policy_payload(outcome),
    }
    if req.intelligence:
        payload["intelligence"] = analyze_findings(fs).to_dict()
    return payload


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
