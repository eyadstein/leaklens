# Review guide — adjudicating an expansion-corpus dossier

A dossier is **evidence plus an unverified hypothesis**. It is not a label.
Your decision is the label, and it only exists once you record it.

## What you are given

`realworld_expansion/review_batch_01.jsonl` (machine-readable) and
`REVIEW_BATCH_01.md` (the same content as a document). Each dossier has:

| Field | What to do with it |
|---|---|
| `provenance` | Check the repo, commit, path and SHA-256. The commit is pinned, so the code will not drift under you. |
| `license` | `redistributable: true` means code text is quoted in the dossier. Otherwise only references are shipped — read the source yourself. |
| `scanner` | What the scanner claimed. A `high` signal is a **suspicion**, not a finding you must confirm. |
| `workflow` | A one-paragraph summary of what the notebook does. Use it to orient, not to decide. |
| `mechanism` | One value from a closed vocabulary. Treat it as a question ("is this mechanism present?"), not a conclusion. |
| `why_it_may_apply` | The AI's argument. Verify it against the cited lines. |
| `evidence_against` | The strongest counter-argument already found. If it survives, say so in your rationale. |
| `questions` | The specific points you must resolve. Answering them in the rationale is what makes the decision auditable. |
| `code_refs` | `cell N:L` = zero-based notebook cell index, one-based line within that cell. Script files use `line N`. |
| `hypothesis` | Always `author: "ai_unverified"`, `is_ground_truth: false`. Ignore its confidence when deciding. |
| `decision` | Empty. It stays empty until you record one. |

`python scripts/collect_expansion.py show --id <candidate>` prints the dossier
and the cached source lines each reference points at (`--context 5` widens the
window). If the cache is gone, use the pinned `file_url`.

## The five states

| State | Meaning | Scored? |
|---|---|---|
| `not_reviewed` | Nobody has looked. The default. | no |
| `confirmed_leakage` | You found a mechanism where held-out information reaches the fit or the reported score. | **yes** |
| `legitimate` | You checked the mechanism and it does not hold — correct order, unused block, row-wise operation, nominal encoding only. | **yes** |
| `potential` | The shape is suspicious but the evidence cannot settle it (e.g. out-of-order execution hides what fed what). | no |
| `uncertain` | You cannot decide from static evidence: missing file, ambiguous provenance, non-executable code. | no |

Only `confirmed_leakage` and `legitimate` become ground truth. `potential` and
`uncertain` are recorded so the next reviewer does not repeat the work — they
are never scored as positive or negative.

## Recording a decision

```sh
python scripts/collect_expansion.py review \
  --id rwx_11f261a63388 \
  --state confirmed_leakage \
  --rationale "cell 21 line 3 refits StandardScaler on the test file, so the holdout defines its own transform; cells 33/43/50 then score that same block repeatedly" \
  --ref "cell 21:1-3" --ref "cell 33:1" \
  --mechanism preprocess_before_split \
  --confidence 0.9 \
  --reviewer <your name>
```

Rules the command enforces, so you cannot accidentally fabricate evidence:

- `--reviewer` is required and `ai`, `unassigned`, `anonymous`, `bot` and
  similar placeholders are rejected.
- `--rationale` is required for a scored verdict, and at least one `--ref` is
  always required. `--ref` must look like `cell 128:4`, `cell 128:3-6` or
  `line 42`.
- `--state`, `--mechanism` and `--confidence` are validated against closed
  vocabularies; `confidence` is `[0, 1]` and is your uncertainty, not the AI's.
- The decision is **appended** to `reviews.jsonl`. `manifest.jsonl`,
  `review_queue.jsonl` and the batch files are never rewritten.

If you record the wrong state, do not edit the log by hand: record again. The
last decision for a candidate wins, and the whole history stays visible.

## Judging the common shapes

Ask what the held-out rows could have contributed to, not whether the call
looks scary.

- **Split first, then `fit_transform(X_train)` / `transform(X_test)`** —
  legitimate. This is the pattern the scanner often cannot distinguish from
  leakage because it only sees the call.
- **`fit_transform` on a training partition whose name the scanner cannot
  resolve** (e.g. `X_t` after a `train_test_split`) — legitimate, with the
  split cited.
- **Row-wise reductions** (`axis=1` softmax stabilisation, per-row
  normalisation) — cannot carry information between rows.
- **Nominal one-hot or label encoding over the full frame** — no target
  statistic; usually legitimate, note it in the rationale.
- **Imputation or scaling statistics computed over the full frame, then split**
  — the held-out rows contribute to the transform every training row sees.
  `confirmed_leakage` unless the affected column never reaches the fit or the
  score.
- **Tuning or early stopping on the block later reported as test** — the score
  is selection-biased. Confirm if that score is what the notebook presents;
  `potential` if it is an exploratory side branch.
- **Time-ordered rows split randomly** — neighbouring observations land on both
  sides. Confirm when the notebook reports a score from that split.
- **Grouped data split by row** (segments of one recording, multiple images of
  one patient, several rows per customer) — the group identity is the leak.
- **Code that cannot run** (Python 2 syntax, undefined variables, template with
  answers removed) — `uncertain`, not `legitimate`: absence of evidence is not
  evidence of absence.

## What this guide does not license

Do not change a detector, threshold or score to match a decision. A dossier
that reveals a scanner false positive is a recorded observation for a later
mission, not an edit to the scanner. Do not add a label to `realworld/labels.json`
or to the frozen holdouts — those corpora are closed.
