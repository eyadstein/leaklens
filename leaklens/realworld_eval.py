"""Real-world benchmark evaluation: score the scanner against hand-reviewed labels.

Usage:
    python -m leaklens.realworld_eval [--write docs/REALWORLD.md]

Reads ``realworld/labels.json`` (verdict per notebook) and
``realworld/manifest.jsonl`` (scanner findings), then computes
precision/recall/F1 overall and per verdict class. High-severity findings are
the counted prediction; low findings (no_random_state, parse_error,
get_dummies_before_split) are reported separately as noise.

Verdict classes (see docs/REALWORLD.md):
  confirmed_leakage  - hand-confirmed real leak (high finding expected)
  potential_leakage  - plausible but uncertain (e.g. commented-out code,
                       separate train/test files, unused transforms)
  legitimate         - notebook is fine; any finding is a false positive
  uncertain          - cannot tell from static analysis alone

Only the 8 notebooks with verdicts `confirmed_leakage` (5) or `legitimate`
(3) have definitive reviewed labels. The other 52 (`potential_leakage` 2 +
`uncertain` 50) are **excluded** from precision/recall/F1: they are reported
in the "by verdict class" table for transparency but never scored as
positive or negative ground truth.
"""

from __future__ import annotations

import json
import pathlib


def _load():
    root = pathlib.Path("realworld")
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    records = {}
    for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["id"]] = r
    return labels, records


def _high_findings(record):
    return [f for f in record["findings"] if f["severity"] in ("high", "medium")]


def evaluate():
    labels, records = _load()
    per_nb = []
    for nb_id, label in labels.items():
        rec = records.get(nb_id, {"findings": []})
        high = _high_findings(rec)
        predicted = len(high) > 0
        expected = label["verdict"] == "confirmed_leakage"
        per_nb.append(
            {
                "id": nb_id,
                "verdict": label["verdict"],
                "predicted": predicted,
                "high_findings": len(high),
                "kinds": sorted({f["kind"] for f in high}),
                "tp": predicted and expected,
                "fp": predicted and not expected and label["verdict"] == "legitimate",
                "fn": (not predicted) and expected,
                "tn": (not predicted) and label["verdict"] == "legitimate",
            }
        )

    tp = sum(1 for x in per_nb if x["tp"])
    fp = sum(1 for x in per_nb if x["fp"])
    fn = sum(1 for x in per_nb if x["fn"])
    tn = sum(1 for x in per_nb if x["tn"])
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    by_verdict = {}
    for v in ("confirmed_leakage", "potential_leakage", "legitimate", "uncertain"):
        group = [x for x in per_nb if x["verdict"] == v]
        by_verdict[v] = {
            "n": len(group),
            "flagged": sum(1 for x in group if x["predicted"]),
        }

    return {
        "n": len(per_nb),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "by_verdict": by_verdict,
        "per_nb": per_nb,
    }


def markdown(m):
    lines = [
        "# Real-world notebook validation",
        "",
        f"Evaluated {m['n']} public notebooks from `realworld/` against hand-reviewed "
        "labels (`realworld/labels.json`). Only 8 labels are scored (see below); "
        "the other 52 are reported but excluded from precision/recall/F1.",
        "",
        "High/medium findings are the prediction; low findings (no_random_state, "
        "parse_error, get_dummies_before_split) are noise and not counted.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Notebooks evaluated | {m['n']} |",
        "| Definitive labels scored | 8 (5 confirmed_leakage + 3 legitimate) |",
        "| Notebooks evaluated but excluded from scoring | 52 (2 potential_leakage + 50 uncertain) |",
        f"| True positives | {m['tp']} |",
        f"| False positives | {m['fp']} |",
        f"| False negatives | {m['fn']} |",
        f"| True negatives | {m['tn']} |",
        f"| Precision (on the 8 scored labels) | {m['precision']:.2f} |",
        f"| Recall (on the 8 scored labels) | {m['recall']:.2f} |",
        f"| F1 (on the 8 scored labels) | {m['f1']:.2f} |",
        "",
        "Only 8 notebooks have definitive reviewed labels: 5 `confirmed_leakage` "
        "and 3 `legitimate`. The remaining 52 (2 `potential_leakage` + 50 "
        "`uncertain`) are excluded from precision/recall/F1 — they are not "
        "ground truth, and counting them would inflate or deflate the score. "
        "The 8 reviewed labels are a very small sample; treat the resulting "
        "metrics as a smoke check, not a population accuracy estimate.",
        "",
        "## Reviewed-label breakdown (scored)",
        "",
        "| Verdict | Notebooks | Flagged with high/medium | Role in scoring |",
        "|---|---:|---:|---|",
    ]
    for v, d in m["by_verdict"].items():
        role = (
            "positive ground truth"
            if v == "confirmed_leakage"
            else (
                "negative ground truth"
                if v == "legitimate"
                else "excluded (not scored)"
            )
        )
        lines.append(f"| {v} | {d['n']} | {d['flagged']} | {role} |")
    lines += [
        "",
        "## Excluded notebooks (reported, not scored)",
        "",
        "2 `potential_leakage` and 50 `uncertain` notebooks are listed in the",
        "table above so the full 60-notebook corpus stays transparent. One",
        "`uncertain` notebook does carry a high/medium finding, but it is **not**",
        "counted as a false positive — without a hand-confirmed verdict it",
        "cannot be scored. `potential_leakage` notebooks are plausible but",
        "unproven (commented-out code, separate train/test files, unused",
        "transforms) and are likewise excluded.",
        "",
        "## By verdict class (full corpus, for context)",
        "",
        "| Verdict | Notebooks | Flagged with high/medium |",
        "|---|---:|---:|",
    ]
    for v, d in m["by_verdict"].items():
        lines.append(f"| {v} | {d['n']} | {d['flagged']} |")
    lines += [
        "",
        "## Methodology and caveats",
        "",
        "- Notebooks come from GitHub Code Search with the same queries as "
        "`scripts/study.py` plus groupby/shift/rolling queries. The sample is "
        "not random, so results are not a population prevalence estimate.",
        "- Verdicts were hand-reviewed from static inspection of notebook code; "
        "no notebook was executed. Uncertain cases are kept separate and are not "
        "counted against the scanner.",
        "- This real-world corpus is **kept separate** from the synthetic "
        "`leaklens/bench.py` benchmark (different directory, different scoring).",
        "- Scanner findings are high/medium only; low-severity noise (unseeded "
        "splits, parse errors, get_dummies) is excluded from precision/recall.",
        "- Regenerate this file with `python -m leaklens.realworld_eval --write docs/REALWORLD.md`",
        "  (do not edit the numbers by hand — edit `realworld/labels.json` instead).",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    m = evaluate()
    text = markdown(m)
    print(text)
    if len(argv) >= 2 and argv[0] == "--write":
        pathlib.Path(argv[1]).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
