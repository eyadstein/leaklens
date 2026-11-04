# Real-world validation corpus

60 public GitHub notebooks fetched via Code Search (`gh` CLI) using the
same queries as `scripts/study.py` plus groupby/shift/rolling/TFIDF queries.
Each notebook is **only statically analyzed — never executed**.

## Contents

- `manifest.jsonl` — one JSON record per notebook: source repo/path/URL,
  SHA-256 of the fixture, bytes, query, and scanner findings.
- `labels.json` — hand-reviewed verdict per notebook:
  `confirmed_leakage` / `potential_leakage` / `legitimate` / `uncertain`,
  with a reason per verdict.
- `notebooks/` — the fetched `.ipynb` fixtures (`nb_0000.ipynb` … `nb_0059.ipynb`,
  attributed by URL in the manifest; do not redistribute beyond this test corpus).

## Labels

| Verdict | Count | Scored? |
|---|---:|---|
| `confirmed_leakage` | 5 | Yes — positive ground truth |
| `legitimate` | 3 | Yes — negative ground truth |
| `potential_leakage` | 2 | No — excluded |
| `uncertain` | 50 | No — excluded |
| **Total** | **60** | **8 scored, 52 excluded** |

Only the **8** definitive labels (5 `confirmed_leakage` + 3 `legitimate`) contribute to precision, recall and F1. The other 52 stay visible in the output for transparency but are never scored as positive or negative ground truth.

### Excluded verdicts

- **`potential_leakage` (2)** — plausible but unproven: commented-out code, separate train/test files, or an unused transform.
- **`uncertain` (50)** — cannot tell from static analysis alone. One carries a high/medium finding; it is not counted as a false positive.

## Methodology and caveats

- Notebooks come from GitHub Code Search with a fixed query set (see `scripts/realworld.py`). The sample is **not random**, so metrics are not a population prevalence estimate.
- Verdicts were hand-reviewed from static inspection of notebook code; **no notebook was executed**.
- The corpus is **kept separate** from the synthetic `leaklens/bench.py` benchmark (different directory, different scoring, `docs/BENCHMARK.md`).
- Scanner predictions are high/medium findings only. Low-severity noise (`no_random_state`, `parse_error`, `get_dummies_before_split`) is excluded from precision/recall.
- The sample is tiny (8 scored) and non-random — treat the reported metrics as a smoke check, not a population accuracy estimate.

## Reproduce

Scoring (read-only; do not regenerate the corpus unless you need to extend it):

```sh
.venv\Scripts\python.exe -m leaklens.realworld_eval
.venv\Scripts\python.exe -m pytest tests/test_realworld.py -q
```

Regenerate the corpus (requires `gh` CLI; overwrites `notebooks/`, `manifest.jsonl`):

```sh
.venv\Scripts\python.exe scripts/realworld.py
```

Write the eval report:

```sh
.venv\Scripts\python.exe -m leaklens.realworld_eval --write docs/REALWORLD.md
```

The synthetic benchmark (`leaklens/bench.py`, `docs/BENCHMARK.md`) stays
separate from this corpus. Full report: [../docs/REALWORLD.md](../docs/REALWORLD.md).
