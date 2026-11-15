"""Collect the real-world *expansion* corpus (candidates, not labels).

Three independent channels, all public and licence-recorded:

  github_search   GitHub Code Search across ML workflow queries (``gh`` auth)
  github_repos    whole-tree enumeration of curated public ML repositories
  kaggle          DistilKaggle (Zenodo 10317389, CC-BY-4.0) streamed by byte range

Nothing collected here is executed. Notebooks are scanned statically by
``leaklens.scanner`` and only their provenance, hashes and findings are
committed; the files themselves stay in a content-addressed cache outside Git.

Usage::

    python scripts/collect_expansion.py collect --channels github_search
    python scripts/collect_expansion.py collect            # all channels, resumable
    python scripts/collect_expansion.py finalize           # dedupe, queue, stats
    python scripts/collect_expansion.py review --id rwx_.. --state legitimate ...
    python scripts/collect_expansion.py validate

The cache defaults to ``%LOCALAPPDATA%\\leaklens-corpus-cache`` (or
``$XDG_CACHE_HOME``/``~/.cache``) and can be moved with ``LEAKLENS_CORPUS_CACHE``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaklens import corpus_expansion as corpus  # noqa: E402
from leaklens.scanner import scan_file  # noqa: E402

OWNER = "eyadstein/leaklens"
EXPANSION = ROOT / "realworld_expansion"
MANIFEST = EXPANSION / "manifest.jsonl"
QUEUE = EXPANSION / "review_queue.jsonl"
REVIEWS = EXPANSION / "reviews.jsonl"
STATE = EXPANSION / "collection_state.json"
STATS = EXPANSION / "stats.json"

MAX_BYTES = 2_000_000
CODE_SEARCH_PAUSE = 7.0  # 10 requests/minute authenticated
CORE_PAUSE = 0.2
RETRIES = 3
REPO_FILES = 8
_REPO_CACHE: dict[str, dict] = {}

# Translated copies and checkpoint debris are the same notebook, not new samples.
SKIP_PATH = re.compile(
    r"(^|/)(translations?|i18n|\.ipynb_checkpoints)(/|$)"
    r"|\.nbconvert\.ipynb$|-translations?/",
    re.IGNORECASE,
)

# Workflow coverage, not leakage hints: a diverse slice of real ML pipelines.
SEARCH_QUERIES = [
    "train_test_split extension:ipynb",
    "cross_val_score extension:ipynb",
    "GridSearchCV extension:ipynb",
    "StratifiedKFold extension:ipynb",
    "TimeSeriesSplit extension:ipynb",
    "StandardScaler extension:ipynb",
    "MinMaxScaler extension:ipynb",
    "RobustScaler extension:ipynb",
    "fit_transform extension:ipynb",
    "SimpleImputer extension:ipynb",
    "OneHotEncoder extension:ipynb",
    "LabelEncoder extension:ipynb",
    "ColumnTransformer extension:ipynb",
    "pipeline fit extension:ipynb",
    "TfidfVectorizer extension:ipynb",
    "CountVectorizer extension:ipynb",
    "SMOTE extension:ipynb",
    "RandomOverSampler extension:ipynb",
    "SelectKBest extension:ipynb",
    "RFE extension:ipynb",
    "PCA extension:ipynb",
    "xgboost extension:ipynb",
    "lightgbm extension:ipynb",
    "catboost extension:ipynb",
    "RandomForestClassifier extension:ipynb",
    "LogisticRegression extension:ipynb",
    "LinearRegression extension:ipynb",
    "KMeans extension:ipynb",
    "groupby transform extension:ipynb",
    "groupby agg extension:ipynb",
    "rolling mean extension:ipynb",
    "shift -1 extension:ipynb",
    "target encoding extension:ipynb",
    "feature_importances_ extension:ipynb",
    "classification_report extension:ipynb",
    "roc_auc_score extension:ipynb",
    "mean_squared_error extension:ipynb",
    "balanced_accuracy_score extension:ipynb",
    "ydata-profiling extension:ipynb",
    "read_csv train.csv extension:ipynb",
    "submission.csv extension:ipynb",
    "data leakage extension:ipynb",
]

# Years of public ML teaching and example code, chosen for workflow diversity.
CURATED_REPOS = [
    "microsoft/Data-Science-For-Beginners",
    "microsoft/ML-For-Beginners",
    "microsoft/AI-For-Beginners",
    "microsoft/generative-ai-for-beginners",
    "jakevdp/PythonDataScienceHandbook",
    "rasbt/python-machine-learning-book-3rd-edition",
    "rasbt/machine-learning-book",
    "amueller/introduction_to_ml_with_python",
    "mrdbourke/zero-to-mastery-ml",
    "mrdbourke/pytorch-deep-learning",
    "nyandwi/machine_learning_complete",
    "huggingface/notebooks",
    "d2l-ai/d2l-en",
    "udacity/machine-learning",
    "Kaggle/python",
    "fchollet/deep-learning-with-python-notebooks",
    "hunkim/DeepLearningZeroToAll",
    "GoogleCloudPlatform/training-data-analyst",
    "aws/amazon-sagemaker-examples",
    "Azure/MachineLearningNotebooks",
    "tensorflow/docs",
    "pytorch/tutorials",
    "openai/openai-cookbook",
    "anthropics/anthropic-cookbook",
    "iterative/dvc",
    "evidentlyai/evidently",
    "wandb/examples",
    "alirezadir/Machine-Learning-Interviews",
    "codebasics/py",
    "joanby/machinelearning-az",
    "Pierian-Data/Complete-Python-3-Bootcamp",
    "ageron/tf2_course",
    "khanhnamle1994/cracking-the-data-science-interview",
    "NVIDIA/DeepLearningExamples",
    "h2oai/h2o-tutorials",
    "tensorflow/examples",
    "dmlc/dgl",
    "onnx/tutorials",
    "mlflow/recipes-examples",
]

KAGGLE_CODE = "https://zenodo.org/api/records/10317389/files/code.csv/content"
KAGGLE_RECORD = "https://zenodo.org/record/10317389"
KAGGLE_LICENSE = "CC-BY-4.0"
KAGGLE_BLOCK_BYTES = 1_200_000
KAGGLE_BLOCKS = 36
KAGGLE_PER_BLOCK = 7

MARKERS = re.compile(
    r"\b(sklearn|scikit_learn|xgboost|lightgbm|catboost|tensorflow|torch|keras|"
    r"statsmodels|imblearn|yellowbrick)\b"
)
WORKFLOW = {
    "split": re.compile(
        r"train_test_split|KFold|StratifiedKFold|GroupKFold|TimeSeriesSplit"
    ),
    "fit": re.compile(r"\.fit\(|fit_transform|fit_resample"),
    "cv": re.compile(
        r"cross_val_score|cross_validate|GridSearchCV|RandomizedSearchCV|optuna|hyperopt"
    ),
    "scale": re.compile(r"StandardScaler|MinMaxScaler|RobustScaler|Normalizer"),
    "impute": re.compile(r"SimpleImputer|KNNImputer|fillna\("),
    "encode": re.compile(r"OneHotEncoder|LabelEncoder|get_dummies|TargetEncoder"),
    "resample": re.compile(r"SMOTE|RandomOverSampler|NearMiss|TomekLinks"),
}


def cache_root() -> pathlib.Path:
    env = os.environ.get("LEAKLENS_CORPUS_CACHE")
    if env:
        return pathlib.Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return pathlib.Path(base) / "leaklens-corpus-cache"
    return pathlib.Path.home() / ".cache" / "leaklens-corpus-cache"


def object_path(sha: str, file_type: str) -> pathlib.Path:
    return cache_root() / "objects" / sha[:2] / f"{sha}.{file_type}"


def log(message: str) -> None:
    # repository paths are Unicode; a cp1252 console must not abort a collection
    safe = message.encode(sys.stdout.encoding or "utf-8", "replace").decode(
        sys.stdout.encoding or "utf-8", "replace"
    )
    print(safe, flush=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- transport --------------------------------------------------------------


def gh_api(path: str, fields: dict | None = None, *, pause: float = CORE_PAUSE):
    """One ``gh api`` call with retries. Never prints or stores the token."""
    args = ["gh", "api", "-X", "GET", path]
    for key, value in (fields or {}).items():
        args += ["-f", f"{key}={value}"]
    for attempt in range(RETRIES):
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            time.sleep(pause)
            return json.loads(result.stdout or "{}")
        log(
            f"    gh api {path} failed (attempt {attempt + 1}): "
            f"{result.stderr.strip()[:120]}"
        )
        time.sleep(10 * (attempt + 1))
    return None


def fetch_bytes(url: str, *, limit: int = MAX_BYTES, headers: dict | None = None):
    """Download with retries; returns None when unavailable or too large."""
    request_headers = {"User-Agent": f"leaklens-corpus/{OWNER}"}
    request_headers.update(headers or {})
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > limit:
                    return None
                data = response.read(limit + 1)
                if len(data) > limit:
                    return None
                return data
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            log(f"    fetch failed (attempt {attempt + 1}): {str(error)[:100]}")
            time.sleep(5 * (attempt + 1))
    return None


def content_length(url: str) -> int:
    for _ in range(RETRIES):
        try:
            request = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": f"leaklens-corpus/{OWNER}"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return int(response.headers.get("Content-Length") or 0)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(5)
    return 0


def repo_info(repo: str) -> dict:
    """Licence and immutable commit for one repository, cached per run."""
    if repo in _REPO_CACHE:
        return _REPO_CACHE[repo]
    meta = gh_api(f"repos/{repo}") or {}
    branch = meta.get("default_branch", "")
    commit = gh_api(f"repos/{repo}/commits/{branch}") if branch else None
    info = {
        "license_spdx": (meta.get("license") or {}).get("spdx_id") or "",
        "commit_sha": (commit or {}).get("sha", ""),
        "default_branch": branch,
        "private": bool(meta.get("private")),
        "available": bool(meta.get("full_name")),
    }
    _REPO_CACHE[repo] = info
    return info


# --- record assembly --------------------------------------------------------


def workflow_flags(code: str) -> dict:
    return {name: bool(pattern.search(code)) for name, pattern in WORKFLOW.items()}


def ml_relevant(code: str, flags: dict) -> bool:
    return bool(MARKERS.search(code)) or (flags["split"] and flags["fit"])


def build_record(
    *,
    source_id: str,
    source_kind: str,
    repo: str,
    path: str,
    file_url: str,
    data: bytes,
    file_type: str,
    license_spdx: str,
    collected_at: str,
    commit_sha: str = "",
    repo_url: str = "",
    extra: dict | None = None,
) -> dict:
    sha = corpus.content_sha256(data)
    code = corpus.code_text(data, file_type)
    flags = workflow_flags(code)
    stored = object_path(sha, file_type)
    stored.parent.mkdir(parents=True, exist_ok=True)
    if not stored.exists():
        stored.write_bytes(data)
    try:
        findings = [
            {
                "severity": finding.severity,
                "kind": finding.kind,
                "column": finding.column,
                "detail": finding.detail,
            }
            for finding in scan_file(stored)
        ]
    except Exception as error:  # noqa: BLE001 - one bad file must not stop a run
        findings = [
            {
                "severity": "low",
                "kind": "scan_error",
                "column": path,
                "detail": str(error)[:200],
            }
        ]
    record = {
        "id": corpus.candidate_id(source_id, repo, path, sha),
        "source_id": source_id,
        "source_kind": source_kind,
        "repo": repo,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "path": path,
        "file_url": file_url,
        "collected_at": collected_at,
        "license_spdx": license_spdx,
        "file_type": file_type,
        "sha256": sha,
        "code_sha256": corpus.code_sha256(code),
        "bytes": len(data),
        "sketch": list(corpus.sketch(code)),
        "workflow": flags,
        "ml_relevant": ml_relevant(code, flags),
        "eligible_for_review": bool(code.strip()) and ml_relevant(code, flags),
        "cache_key": f"{sha}.{file_type}",
        "overlaps_frozen": False,
        "duplicate_of": None,
        "code_duplicate_of": None,
        "near_of": None,
        "findings": findings,
        "review": {"state": "not_reviewed"},
    }
    record.update(extra or {})
    return record


def frozen_hashes() -> set[str]:
    """Hashes of the frozen 60-notebook corpus, so overlap is visible not silent."""
    folder = ROOT / "realworld" / "notebooks"
    return {corpus.content_sha256(path.read_bytes()) for path in folder.glob("*.ipynb")}


# --- channels ---------------------------------------------------------------


def github_search_channel(limit: int, per_query: int, state: dict) -> int:
    collected: list[dict] = []
    seen = {(rec["repo"], rec["path"]) for rec in load_records()}
    for query in SEARCH_QUERIES:
        if len(collected) >= limit:
            break
        cursor = state.setdefault("queries", {}).setdefault(query, {"page": 1})
        page = cursor["page"]
        payload = gh_api(
            "search/code",
            {"q": query, "per_page": 100, "page": page},
            pause=CODE_SEARCH_PAUSE,
        )
        if not payload:
            continue
        items = payload.get("items", [])
        log(f"  {query!r} page {page}: {len(items)} hits")
        for item in items[:per_query]:
            if len(collected) >= limit:
                break
            repo = item["repository"]["full_name"]
            path = item["path"]
            if (repo, path) in seen or SKIP_PATH.search(path):
                continue
            seen.add((repo, path))
            record = collect_github_file(
                repo, path, source_id="github_search", query=query
            )
            if record:
                collected.append(record)
                log(f"    {record['id']} {repo}/{path} ({len(record['findings'])})")
        # persist before advancing the cursor: a crash must never skip work
        save(collected)
        cursor["page"] = page + 1
        save_state(state)
    return len(collected)


def collect_github_file(
    repo: str, path: str, *, source_id: str, query: str = ""
) -> dict | None:
    info = repo_info(repo)
    if info["private"] or not info["available"]:
        return None
    url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(path)}"
    data = fetch_bytes(url)
    if not data:
        return None
    return build_record(
        source_id=source_id,
        source_kind="github_search" if source_id == "github_search" else "github_repos",
        repo=repo,
        repo_url=f"https://github.com/{repo}",
        path=path,
        file_url=f"https://github.com/{repo}/blob/"
        f"{info['commit_sha'] or 'HEAD'}/{path}",
        data=data,
        file_type=path.rsplit(".", 1)[-1].lower(),
        license_spdx=info["license_spdx"],
        commit_sha=info["commit_sha"],
        collected_at=now(),
        extra={"query": query} if query else {},
    )


def github_repos_channel(limit: int) -> list[dict]:
    collected = []
    seen = {(rec["repo"], rec["path"]) for rec in load_records()}
    for repo in CURATED_REPOS:
        if len(collected) >= limit:
            break
        info = repo_info(repo)
        if not info["available"] or info["private"]:
            log(f"  skip {repo} (unavailable or private)")
            continue
        branch = info["default_branch"] or "HEAD"
        tree = gh_api(f"repos/{repo}/git/trees/{branch}?recursive=1")
        if not tree:
            continue
        blobs = [
            item
            for item in tree.get("tree", [])
            if item.get("type") == "blob"
            and item.get("path", "").lower().endswith((".ipynb", ".py"))
            and (item.get("size") or 0) <= MAX_BYTES
            and not SKIP_PATH.search(item.get("path", ""))
        ]
        # notebooks first, then scripts; spread across the tree by path order
        blobs.sort(key=lambda item: (not item["path"].endswith(".ipynb"), item["path"]))
        taken = 0
        for item in blobs:
            if len(collected) >= limit or taken >= REPO_FILES:
                break
            path = item["path"]
            if path.startswith((".github/", "docs/")) or (repo, path) in seen:
                continue
            url = (
                f"https://raw.githubusercontent.com/{repo}/{branch}/"
                f"{urllib.parse.quote(path)}"
            )
            data = fetch_bytes(url)
            if not data:
                continue
            seen.add((repo, path))
            record = build_record(
                source_id="github_repos",
                source_kind="github_repos",
                repo=repo,
                repo_url=f"https://github.com/{repo}",
                path=path,
                file_url=f"https://github.com/{repo}/blob/"
                f"{tree.get('sha', branch)}/{path}",
                data=data,
                file_type=path.rsplit(".", 1)[-1].lower(),
                license_spdx=info["license_spdx"],
                commit_sha=tree.get("sha", ""),
                collected_at=now(),
            )
            if record["ml_relevant"]:
                collected.append(record)
                taken += 1
                log(f"    {record['id']} {repo}/{path} ({len(record['findings'])})")
        log(f"  {repo}: {taken} kept (total {len(collected)})")
        save(collected)
    return len(collected)


def kaggle_channel(state: dict) -> int:
    """Stream DistilKaggle ``code.csv`` by byte range; rows are kernel-contiguous.

    The file is 3.9 GB and row groups are contiguous per kernel_id, so evenly
    spaced ranges give a spread across the whole id space for ~45 MB in total.
    """
    total = content_length(KAGGLE_CODE)
    if not total:
        log("  kaggle: code.csv unavailable, channel skipped")
        return 0
    collected: list[dict] = []
    done = set(state.setdefault("kaggle_blocks", []))
    for index in range(KAGGLE_BLOCKS):
        if index in done:
            continue
        start = index * total // KAGGLE_BLOCKS
        block = fetch_bytes(
            KAGGLE_CODE,
            limit=KAGGLE_BLOCK_BYTES,
            headers={"Range": f"bytes={start}-{start + KAGGLE_BLOCK_BYTES - 1}"},
        )
        if not block:
            continue
        notebooks = notebooks_from_block(block, per_block=KAGGLE_PER_BLOCK)
        log(f"  kaggle block {index}: {len(notebooks)} notebooks")
        for kernel_id, cells in notebooks:
            record = kaggle_record(kernel_id, cells)
            if record and record["ml_relevant"]:
                collected.append(record)
                log(f"    {record['id']} kaggle/{kernel_id}")
        save(collected)
        state["kaggle_blocks"].append(index)
        save_state(state)
    return len(collected)


def notebooks_from_block(block: bytes, *, per_block: int):
    """Parse a mid-file byte range into complete kernels with their cells."""
    csv.field_size_limit(4 << 20)  # Kaggle cells exceed the 128 KB default
    text = block.decode("utf-8", "replace")
    start = text.find("\n")
    if start < 0:
        return []
    rows = list(csv.reader(io.StringIO(text[start + 1 :])))
    runs: dict[str, list[tuple[int, str]]] = collections.OrderedDict()
    for row in rows:
        if len(row) < 3 or not row[0].isdigit() or not row[1].isdigit():
            continue
        runs.setdefault(row[0], []).append((int(row[1]), row[2]))
    # the last kernel in a range is probably truncated; keep the complete ones
    complete = list(runs.items())[:-1]
    return complete[:per_block]


def kaggle_record(kernel_id: str, cells) -> dict | None:
    blocks = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
        for _index, source in sorted(cells)
        if source.strip()
    ]
    if not blocks:
        return None
    notebook = {
        "cells": blocks,
        "metadata": {"kernel_id": kernel_id, "origin": "DistilKaggle"},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return build_record(
        source_id="kaggle_distilkaggle",
        source_kind="kaggle",
        repo=f"kaggle/kernel-{kernel_id}",
        repo_url=KAGGLE_RECORD,
        path=f"code.csv#kernel={kernel_id}",
        file_url=f"https://www.kaggle.com/kernels/{kernel_id}",
        data=json.dumps(notebook, indent=1).encode("utf-8"),
        file_type="ipynb",
        license_spdx=KAGGLE_LICENSE,
        collected_at=now(),
        extra={"dataset_doi": "10.5281/zenodo.10317389"},
    )


# --- persistence ------------------------------------------------------------


def load_records() -> list[dict]:
    return corpus.load_manifest(MANIFEST) if MANIFEST.exists() else []


def save_state(state: dict) -> None:
    EXPANSION.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps(state, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save(records: list[dict]) -> int:
    """Merge ``records`` into the manifest by stable id; returns the new total."""
    merged = {record["id"]: record for record in load_records()}
    for record in records:
        merged.setdefault(record["id"], record)
    ordered = [merged[key] for key in sorted(merged)]
    corpus.deduplicate(ordered)
    corpus.write_manifest(MANIFEST, ordered)
    return len(ordered)


def finalize() -> dict:
    records = load_records()
    frozen = frozen_hashes()
    for record in records:
        record["overlaps_frozen"] = record["sha256"] in frozen
    corpus.deduplicate(records)
    corpus.write_manifest(MANIFEST, records)
    queue_ids = corpus.queue(records)
    by_id = {record["id"]: record for record in records}
    EXPANSION.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8", newline="\n") as handle:
        for ident in queue_ids:
            handle.write(json.dumps(by_id[ident], sort_keys=True) + "\n")
    stats = corpus.summarize(records, queue_ids)
    STATS.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


# --- commands ---------------------------------------------------------------


def command_collect(args) -> int:
    cache_root().mkdir(parents=True, exist_ok=True)
    state = load_state()
    before = len(load_records())
    channels = [name.strip() for name in args.channels.split(",") if name.strip()]
    if "github_search" in channels:
        log("[github_search]")
        github_search_channel(args.limit, args.per_query, state)
    if "github_repos" in channels:
        log("[github_repos]")
        github_repos_channel(args.limit)
    if "kaggle" in channels:
        log("[kaggle]")
        kaggle_channel(state)
    stats = finalize()
    total = len(load_records())
    log(
        f"collected {total - before} new candidates "
        f"({total} total, {stats['unique_samples']} unique) -> {MANIFEST}"
    )
    return 0


def command_finalize(_args) -> int:
    print(json.dumps(finalize(), indent=2, sort_keys=True))
    return 0


def command_review(args) -> int:
    records = load_records()
    by_id = {record["id"]: record for record in records}
    if args.id not in by_id:
        log(f"unknown sample id: {args.id}")
        return 2
    updated = corpus.review(
        by_id[args.id],
        state=args.state,
        evidence=args.evidence,
        explanation=args.explanation,
        mechanism=args.mechanism,
        confidence=args.confidence,
        reviewer=args.reviewer,
    )
    records = [updated if record["id"] == args.id else record for record in records]
    corpus.write_manifest(MANIFEST, records)
    EXPANSION.mkdir(parents=True, exist_ok=True)
    with REVIEWS.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(updated["review"], sort_keys=True) + "\n")
    finalize()
    log(f"recorded {args.state} for {args.id}")
    return 0


def command_validate(_args) -> int:
    records = load_records()
    errors = corpus.validate(records)
    for error in errors[:40]:
        log(error)
    log(f"{len(records)} records, {len(errors)} provenance problems")
    return 1 if errors else 0


def command_stats(_args) -> int:
    print(json.dumps(finalize(), indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="acquire candidates (resumable)")
    collect.add_argument("--channels", default="github_search,github_repos,kaggle")
    collect.add_argument("--limit", type=int, default=400, help="per channel")
    collect.add_argument("--per-query", type=int, default=12, help="hits per search")
    collect.set_defaults(func=command_collect)

    for name, func, help_text in (
        ("finalize", command_finalize, "dedupe, rebuild queue and stats"),
        ("stats", command_stats, "print corpus statistics"),
        ("validate", command_validate, "check provenance completeness"),
    ):
        part = sub.add_parser(name, help=help_text)
        part.set_defaults(func=func)

    review = sub.add_parser("review", help="record one human decision")
    review.add_argument("--id", required=True)
    review.add_argument("--state", required=True, choices=corpus.REVIEW_STATES)
    review.add_argument("--evidence", default="")
    review.add_argument("--explanation", default="")
    review.add_argument("--mechanism", default=None)
    review.add_argument("--confidence", type=float, default=None)
    review.add_argument("--reviewer", default="unassigned")
    review.set_defaults(func=command_review)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
