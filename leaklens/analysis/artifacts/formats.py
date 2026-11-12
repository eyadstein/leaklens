"""One typed artifact intake: magic first, budgets, a reason on every refusal.

This is the artifact half of M-S1 in ``ARTIFACT_SECURITY_STRATEGY.md``: the same
rules as ``upload_guard`` (never trust a name, refuse before parse, cap size and
fields) applied to model files. It never unpickles, never extracts an archive and
never writes anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...evidence import Evidence
from ...leakage import Finding
from ...paths import PathOutsideRoot, resolve_within

# Local module name only; this file never imports the pickle package.
from . import ArtifactMeta, ArtifactParseError, onnx, pickle_guard, safetensors  # nosec B403

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_KEYS = 10_000

SAFE_FORMATS = {".safetensors": "safetensors", ".onnx": "onnx", ".json": "json"}


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ArtifactParseError(f"duplicate JSON key {key[:300]!r}")
        out[key] = value
    return out


def _depth(value: object, limit: int = MAX_JSON_DEPTH) -> int:
    stack = [(value, 1)]
    deepest = 0
    while stack:
        item, level = stack.pop()
        deepest = max(deepest, level)
        if deepest > limit:
            return deepest
        if isinstance(item, dict):
            stack.extend((value, level + 1) for value in item.values())
        elif isinstance(item, list):
            stack.extend((value, level + 1) for value in item)
    return deepest


def inspect_json(data: bytes, *, max_bytes: int = MAX_JSON_BYTES) -> ArtifactMeta:
    """Parse artifact JSON with size/depth caps and record its digest."""
    if len(data) > max_bytes:
        raise ArtifactParseError("the JSON artifact exceeds the byte cap")
    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ArtifactParseError("the JSON artifact is not valid JSON") from error
    if _depth(document) > MAX_JSON_DEPTH:
        raise ArtifactParseError("the JSON artifact exceeds the depth cap")
    summary: dict[str, object] = {"top_level": type(document).__name__}
    if isinstance(document, dict):
        summary["keys"] = min(len(document), MAX_JSON_KEYS)
    elif isinstance(document, list):
        summary["items"] = min(len(document), MAX_JSON_KEYS)
    return ArtifactMeta("json", hashlib.sha256(data).hexdigest(), summary)


def _unsafe_finding(rel: str, reason: str) -> Finding:
    detail = (
        f"serialized checkpoint refused without loading it: {reason} format. "
        "LeakLens never unpickles a model file; re-export metadata as "
        "safetensors, ONNX or JSON to audit it."
    )
    return Finding(
        "high",
        "artifact_unsafe_format",
        rel,
        detail,
        evidence=(
            Evidence(
                source="code",
                kind="parse",
                description="a pickle-based or archive checkpoint was refused",
                location=rel,
                observed=f"{reason} magic or extension",
                expected="safetensors, ONNX or JSON metadata",
            ),
        ),
        stable_key=f"artifact:{rel}:{reason}",
        diagnostic=True,
    )


def _summary(meta: object) -> dict[str, object]:
    if isinstance(meta, safetensors.SafetensorsMeta):
        return {
            "tensor_count": meta.tensor_count,
            "tensors": list(meta.tensors),
            "dtypes": meta.dtypes,
            "metadata_keys": sorted(meta.metadata),
            "metadata_truncated": meta.metadata_truncated,
            "data_bytes": meta.data_bytes,
        }
    if isinstance(meta, onnx.OnnxMeta):
        return {
            "producer": meta.producer,
            "graph": meta.graph_name,
            "opset": list(meta.opset),
            "inputs": list(meta.inputs),
            "outputs": list(meta.outputs),
            "initializer_count": meta.initializer_count,
        }
    return {}


def inspect_artifact(
    root: str | Path,
    path: str | Path,
    *,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[ArtifactMeta | None, Finding | None, str | None]:
    """Inspect one artifact; return ``(meta, refusal finding, skip reason)``."""
    base = Path(root).resolve()
    try:
        target = resolve_within(base, str(path))
    except PathOutsideRoot:
        return None, None, "artifact_outside_root"
    except OSError:
        return None, None, "artifact_unreadable"
    try:
        if target.stat().st_size > max_bytes:
            return None, None, "artifact_too_large"
        with target.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return None, None, "artifact_unreadable"
    rel = target.relative_to(base).as_posix()
    # Refuse on the first bytes so a refused 64 MiB checkpoint is never read.
    reason = pickle_guard.unsafe_reason(target.name, head)
    if reason == "pickle":
        return None, _unsafe_finding(rel, reason), None
    if reason == "archive":
        return None, None, "archive_refused"

    suffix = target.suffix.lower()
    if suffix not in SAFE_FORMATS:
        return None, None, "artifact_unknown_format"
    try:
        data = target.read_bytes()
    except OSError:
        return None, None, "artifact_unreadable"
    if suffix == ".json":
        try:
            meta = inspect_json(data, max_bytes=min(max_bytes, MAX_JSON_BYTES))
        except ArtifactParseError:
            return None, None, "artifact_unreadable"
        return meta, None, None
    try:
        if suffix == ".safetensors":
            parsed = safetensors.parse_safetensors(data)
            digest = parsed.header_digest
        else:
            parsed = onnx.parse_onnx(data, max_bytes=max_bytes)
            digest = parsed.digest
    except ArtifactParseError:
        return None, None, "artifact_unreadable"
    return ArtifactMeta(SAFE_FORMATS[suffix], digest, _summary(parsed)), None, None
