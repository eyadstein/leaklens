from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from . import __version__, assemble
from .audit import audit
from .baseline import BaselineError
from .drift import DEFAULT_SETTINGS as DEFAULT_DRIFT_SETTINGS
from .drift import audit_drift
from .intelligence import (
    GraphBaselineError,
    GraphSnapshot,
    analyze_findings,
    compare_graph_snapshots,
    snapshot_context,
)
from .leakage import scan
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
    graph_baseline: dict | None = None
    include_graph_snapshot: bool = False


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
    return assemble.from_outcome(outcome).findings


def _policy_payload(outcome) -> dict:
    sections = assemble.from_outcome(outcome)
    payload = {"policy": sections.policy} if sections.policy else {}
    if sections.resolved:
        payload["resolved"] = sections.resolved
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


MAX_GRAPH_SNAPSHOT_BYTES = 2 * 1024 * 1024


def _graph_baseline(payload) -> GraphSnapshot | None:
    """Validate an API-supplied snapshot without touching the filesystem."""
    if payload is None:
        return None
    try:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):
        raise HTTPException(422, "graph baseline must be a JSON object") from None
    if len(encoded) > MAX_GRAPH_SNAPSHOT_BYTES:
        raise HTTPException(
            413,
            f"graph baseline exceeds {MAX_GRAPH_SNAPSHOT_BYTES} bytes",
        )
    try:
        return GraphSnapshot.from_dict(payload)
    except GraphBaselineError as exc:
        raise HTTPException(422, f"invalid graph baseline: {exc}") from None


def _graph_payload(
    findings,
    *,
    baseline: GraphSnapshot | None,
    include_snapshot: bool,
    include_intelligence: bool,
    context: dict[str, str],
) -> dict:
    """Build one graph and derive every optional response field from it."""
    report = None
    snapshot = None
    payload: dict = {}
    if baseline is not None or include_snapshot or include_intelligence:
        report = analyze_findings(findings)
        snapshot = GraphSnapshot.from_findings(
            findings,
            graph=report.to_graph(),
            complete=not any(f.kind == "parse_error" for f in findings),
            incomplete_reasons=(
                ("a source file could not be parsed completely",)
                if any(f.kind == "parse_error" for f in findings)
                else ()
            ),
            context=context,
        )
    if include_intelligence and report is not None:
        payload["intelligence"] = report.to_dict()
    if baseline is not None and snapshot is not None:
        try:
            payload["graph_baseline"] = compare_graph_snapshots(
                baseline, snapshot
            ).to_dict()
        except GraphBaselineError as exc:
            raise HTTPException(422, f"cannot compare graph baseline: {exc}") from None
    if include_snapshot and snapshot is not None:
        payload["graph_snapshot"] = snapshot.to_dict()
    return payload


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
    graph_baseline: str | None = Form(None),
    include_graph_snapshot: bool = Form(False),
):
    graph_doc = _graph_baseline(_json_form(graph_baseline, "graph baseline"))
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
        "trust_score": assemble.from_outcome(outcome).trust_score,
        "audit": audit(df, target),
        "missingness": missing.to_dict(),
        "findings": _finding_payloads(outcome),
        **_policy_payload(outcome),
    }
    if drift_report is not None:
        payload["drift"] = drift_report.to_dict()
    payload.update(
        _graph_payload(
            fs,
            baseline=graph_doc,
            include_snapshot=include_graph_snapshot,
            include_intelligence=intelligence,
            context=snapshot_context(
                mode="dataset",
                target=target,
                test=test_df is not None,
                group="",
                time="",
                threshold=threshold,
                drift=bool(drift),
            ),
        )
    )
    return payload


@app.post("/scan")
def scan_code(req: ScanReq):
    if len(req.source) > UploadLimits.MAX_BYTES:
        raise HTTPException(413, f"scan source too large (>8 MiB): {req.name}")
    graph_doc = _graph_baseline(req.graph_baseline)
    is_sql = req.name.lower().endswith(".sql")
    if is_sql:
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
        "trust_score": assemble.from_outcome(outcome).trust_score,
        "findings": _finding_payloads(outcome),
        **_policy_payload(outcome),
    }
    payload.update(
        _graph_payload(
            fs,
            baseline=graph_doc,
            include_snapshot=req.include_graph_snapshot,
            include_intelligence=req.intelligence,
            context=snapshot_context(
                mode="code",
                target=req.target or "",
                split=req.split_column or "",
                dialect=req.dialect if is_sql else "",
                threshold=0.95,
            ),
        )
    )
    return payload


@lru_cache(maxsize=8)
def _load_model(path: Path):
    """Load one operator-fixed model per resolved path; the cache survives requests."""
    return joblib.load(path)


@app.post("/predict")
def predict(req: PredictReq):
    # The model is fixed by the server, never selected by a request (G1 / M-S0).
    # joblib.load() unpickles the file, so a request-selected path would let anyone
    # who can write under ./models/ execute code in this process. Keep the fixed
    # path operator-controlled, and only load artifacts the operator trusts.
    configured = Path(os.environ.get("LEAKLENS_MODEL_PATH", "models/best.joblib"))
    if Path(req.model_path) != configured:
        raise HTTPException(
            422, "model_path is fixed by the server (LEAKLENS_MODEL_PATH)"
        )
    resolved = configured.resolve()
    if not resolved.is_file():
        raise HTTPException(
            404, "model not found; run: leaklens data.csv --target T --arena"
        )
    try:
        b = _load_model(resolved)
    except Exception:  # noqa: BLE001 - re-wrap as HTTP 400 (fail closed)
        # Corrupt/malicious pickle surfaces as EOFError, UnpicklingError or OSError;
        # treat any load failure as an invalid model rather than leaking internals.
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
