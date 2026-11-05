"""Scan public GitHub notebooks and report how many trip LeakLens's pipeline-leak rules.

Usage: python scripts/study.py [pages_per_query]   (needs the gh CLI, logged in)
"""

import collections
import concurrent.futures as cf
import datetime
import json
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

from leaklens.scanner import scan_file

QUERIES = [
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
]


def search(q, pages):
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
                f"q={q} extension:ipynb",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        time.sleep(7)
        if r.returncode != 0:
            print("search failed:", r.stderr.strip()[:160])
            break
        items = json.loads(r.stdout or "{}").get("items", [])
        out += [
            {
                "path": i["path"],
                "repository": {"nameWithOwner": i["repository"]["full_name"]},
            }
            for i in items
        ]
        if len(items) < 100:
            break
    return out


def fetch(item):
    repo = item["repository"]["nameWithOwner"]
    url = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(item['path'])}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return item, r.read(3_000_000)
    except (OSError, ValueError):  # network/parse failures are not fatal here; skip
        return item, None


def main(pages=3):
    seen, items = set(), []
    for q in QUERIES:
        res = search(q, pages)
        print(q, len(res))
        for it in res:
            k = (it["repository"]["nameWithOwner"], it["path"])
            if k not in seen:
                seen.add(k)
                items.append(it)
    print(len(items), "unique notebooks found")
    kinds = collections.Counter()
    flagged_items = []
    scanned = flagged = 0
    with tempfile.TemporaryDirectory() as td, cf.ThreadPoolExecutor(8) as ex:
        for i, (it, data) in enumerate(ex.map(fetch, items)):
            if not data:
                continue
            if b"train_test_split" not in data and b"KFold" not in data:
                continue
            p = pathlib.Path(td) / f"{i}.ipynb"
            p.write_bytes(data)
            fs = scan_file(p)
            if any(f.kind == "parse_error" for f in fs):
                continue
            scanned += 1
            hi = {f.kind for f in fs if f.severity == "high"}
            if hi:
                flagged += 1
                flagged_items.append(
                    (
                        it,
                        [f.column + " " + f.detail for f in fs if f.severity == "high"],
                    )
                )
                kinds.update(hi)
    if not scanned:
        print("nothing scanned (check gh auth / network)")
        return 1
    import random

    random.seed(0)
    with open("study_sample.txt", "w", encoding="utf-8") as fh:
        for it, d in random.sample(flagged_items, min(25, len(flagged_items))):
            fh.write(
                "https://github.com/"
                + it["repository"]["nameWithOwner"]
                + "/blob/HEAD/"
                + it["path"]
                + "\n  "
                + "\n  ".join(d)
                + "\n"
            )
    pct = 100 * flagged / scanned
    lines = [
        "# Leakage study",
        "",
        f"Run on {datetime.date.today()} with `python scripts/study.py`.",
        "",
        f"- Public notebooks with a train/test split scanned: **{scanned}**",
        f"- Flagged with at least one high-severity pipeline leak: **{flagged} ({pct:.0f}%)**",
    ]
    if scanned < 100:
        lines.append("- Small sample: treat the percentage as anecdotal.")
    lines += ["", "| Pattern | Notebooks |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in kinds.most_common()]
    lines += [
        "",
        "## Method and caveats",
        "Notebooks come from GitHub code search for common scikit-learn preprocessing calls, "
        "so the sample is not random. The scanner is a static heuristic: it can flag code that is fine "
        "(for example a fit on a different, already-split frame) and miss leaks it cannot see. "
        "Treat the numbers as a rough prevalence signal, not a verdict on any notebook. "
        "Notebooks with syntax errors were skipped. No repositories are named.",
    ]
    out = pathlib.Path("docs") / "STUDY.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
