"""Build the real-world notebook validation corpus (Priority 1).

Searches public GitHub notebooks with GitHub Code Search (``gh`` CLI, needs
auth), fetches each notebook via raw.githubusercontent.com, **statically**
scans it with LeakLens (no notebook code is executed), and writes a
reproducible corpus under ``realworld/``:

  realworld/manifest.jsonl    one line per notebook: source, hash, findings
  realworld/labels.json       hand-adjustable expected verdicts per notebook
  realworld/notebooks/nb_*.ipynb   fetched fixtures (fetched, not copied)

Usage:
  .venv\\Scripts\\python.exe scripts/realworld.py [pages_per_query] [--per-query N]
  Defaults: 1 page per query, per-query cap 12, global cap ~150 notebooks.

Notebooks are fetched from their public GitHub URLs (MIT/attribution
preserved by reference in the manifest; the .ipynb files stay in the repo as
test fixtures). Query strings mirror scripts/study.py and add coverage for
target encoding, feature selection, time-series and NLP workflows.

The labels file is the ground truth for the real-world benchmark:
  {"notebook_id": {"verdict": "confirmed_leakage"|"potential_leakage"|"legitimate"|"uncertain",
                   "reason": "...", "expected_kinds": [...]}}

``leaklens/realworld_eval.py`` scores the scanner against these labels.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from leaklens.scanner import scan_file

QUERIES = [
    # study.py core queries (scaling / imputation / resampling / encoding)
    "StandardScaler fit_transform",
    "MinMaxScaler fit_transform",
    "RobustScaler fit_transform",
    "SMOTE fit_resample",
    "RandomOverSampler fit_resample",
    "SimpleImputer fit_transform",
    "OneHotEncoder fit_transform",
    "LabelEncoder fit_transform",
    "PCA fit_transform",
    "TfidfVectorizer fit_transform",
    "CountVectorizer fit_transform",
    "SelectKBest fit_transform",
    # priority-1 diversity: target encoding, group stats, feature selection
    "groupby transform extension:ipynb",
    "train_test_split extension:ipynb",
    # time series & NLP signals
    "rolling center True extension:ipynb",
    "shift -1 extension:ipynb",
    "TfidfVectorizer train_test_split extension:ipynb",
]


def _gh_search(query, pages, per_page=100):
    out = []
    for page in range(1, pages + 1):
        r = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "GET",
                "search/code",
                "-f",
                f"q={query}",
                "-f",
                f"per_page={per_page}",
                "-f",
                f"page={page}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        time.sleep(2)
        if r.returncode != 0:
            print(
                f"search failed for {query!r}: {r.stderr.strip()[:160]}",
                file=sys.stderr,
            )
            break
        items = json.loads(r.stdout or "{}").get("items", [])
        for it in items:
            out.append(
                {
                    "path": it["path"],
                    "repository": it["repository"]["full_name"],
                    "query": query,
                }
            )
        if len(items) < per_page:
            break
    return out


def _fetch(item):
    repo = item["repository"]
    url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(item['path'])}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return item, r.read(3_000_000), None
    except Exception as e:  # noqa: BLE001 - network flakes are not fatal
        return item, None, str(e)[:120]


def _is_notebook(item):
    return item["path"].lower().endswith(".ipynb")


def build(pages=1, per_query=12, global_cap=150):
    root = pathlib.Path("realworld")
    nb_dir = root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    seen = {}
    for q in QUERIES:
        for it in _gh_search(q, pages):
            key = (it["repository"], it["path"])
            if key not in seen:
                seen[key] = it
    print(f"{len(seen)} unique notebooks found")

    # stable order so the corpus is reproducible across runs
    items = sorted(seen.values(), key=lambda it: (it["repository"], it["path"]))
    per_query_count = collections.Counter()
    chosen = []
    for it in items:
        # notebooks only: scan_file expects real Jupyter notebooks
        if not _is_notebook(it):
            continue
        if per_query_count[it["query"]] >= per_query:
            continue
        if len(chosen) >= global_cap:
            break
        per_query_count[it["query"]] += 1
        chosen.append(it)
    print(f"{len(chosen)} selected (cap {global_cap})")

    records = []
    with cf.ThreadPoolExecutor(8) as ex:
        for item, data, err in ex.map(_fetch, chosen):
            if data is None:
                print(
                    f"  skip (fetch error): {item['repository']}/{item['path']}: {err}"
                )
                continue
            nb_id = f"nb_{len(records):04d}"
            path = nb_dir / f"{nb_id}.ipynb"
            path.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()[:16]
            src = f"https://github.com/{item['repository']}/blob/HEAD/{item['path']}"
            findings = []
            try:
                findings = [
                    {
                        "severity": f.severity,
                        "kind": f.kind,
                        "column": f.column,
                        "detail": f.detail,
                    }
                    for f in scan_file(path)
                ]
            except Exception as e:  # noqa: BLE001
                findings = [
                    {
                        "severity": "low",
                        "kind": "scan_error",
                        "column": nb_id,
                        "detail": str(e)[:200],
                    }
                ]
            records.append(
                {
                    "id": nb_id,
                    "source_repo": item["repository"],
                    "source_path": item["path"],
                    "source_url": src,
                    "query": item["query"],
                    "sha256": sha,
                    "bytes": len(data),
                    "findings": findings,
                }
            )
            print(
                f"  {nb_id} {item['repository']}/{pathlib.Path(item['path']).name}: "
                f"{len(findings)} findings"
            )

    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    labels_path = root / "labels.json"
    if not labels_path.exists():
        labels = {
            r["id"]: {
                "verdict": (
                    "confirmed_leakage"
                    if any(f["severity"] == "high" for f in r["findings"])
                    else "uncertain"
                ),
                "reason": "auto-seeded from scanner high-severity findings; review by hand",
                "expected_kinds": sorted({f["kind"] for f in r["findings"]}),
            }
            for r in records
        }
        labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {labels_path} (auto-seeded; hand-review recommended)")

    print(f"done: {len(records)} notebooks -> {root}/")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pages = int(args[0]) if args else 1
    per_query = 12
    for a in sys.argv[1:]:
        if a.startswith("--per-query"):
            per_query = int(a.split("=")[-1])
    raise SystemExit(build(pages=pages, per_query=per_query))
