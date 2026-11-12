# Pipeline and MLOps intelligence (Phase 14)

`--pipeline` audits the metadata a training pipeline leaves in the repository. It answers one
question the data and code detectors cannot: **is this evaluation circular?** A model card can say
"trained on v3, evaluated on v4" while v4 is a re-split of v3; the pipeline graph shows whether the
two sides derive from disjoint artifacts.

Everything here is read-only. LeakLens never runs a DVC stage, imports DVC or MLflow, starts an
orchestrator or contacts a service.

```sh
leaklens --scan . --pipeline                  # audit metadata next to the scanned code
leaklens --scan pipelines/dvc.yaml --pipeline # audit one DVC file
pip install "leaklens-ml[yaml]"               # optional, for DVC YAML files
```

## What is parsed

| Artifact | Format | Parsed into |
|---|---|---|
| `dvc.yaml` | YAML (`safe_load`) | stages: `cmd`, `deps`, `outs`, `params` references |
| `dvc.lock` | YAML (`safe_load`) | stages with locked parameter values and output digests |
| `*.dvc` | YAML (`safe_load`) | one stage with its `deps`/`outs` |
| `dag.json`, `*.dag.json` | JSON | `{"stages": [{"name", "deps", "outs"}]}` |
| `*.ipynb` | JSON + `ast` | one stage per code cell that reads or writes a file, ordered by `execution_count` |
| `requirements*.txt`, `requirements/*.txt` | text | dependency pins |
| `params.yaml` | YAML (`safe_load`) | current parameter values, compared with `dvc.lock` |

A directory is scanned once; file discovery is sorted, bounded (`MAX_FILES`) and skips `.git`,
virtualenvs, caches and `graphify-out/`. Any file that does not parse is recorded in `skipped` and
never raised or guessed around.

## Findings

All six are opt-in, `diagnostic: true`, and ship `scored: false`: they never change the trust score,
the exit code or any existing finding.

| Kind | Severity | What it means | Evidence |
|---|---|---|---|
| `pipeline_split_overlap` | high | a training stage and an evaluation stage read the same data artifact, or evaluation reads a data artifact the training stage produced | artifact path, both stage ids, source file |
| `pipeline_unresolved_lineage` | low | a declared dependency has no producing stage and no file on disk | dependency path, source file |
| `pipeline_param_drift` | low | `dvc.lock` records a parameter value that the current parameter file no longer has | parameter key, locked value, current value |
| `pipeline_missing_digest` | low | a locked output has no `md5`/`hash`, so its identity cannot be verified | stage id, artifact path |
| `pipeline_unpinned_dependency` | low | a requirements entry has no exact pin (`package==version`) | package, requirements file |
| `no_random_state` | low | a training stage declares model parameters but no seed (the shipped kind is reused, not duplicated) | stage id, declared parameter keys |

Only a confirmed pipeline lineage path can produce `high`. Model artifacts are deliberately excluded
from the overlap check: an evaluation stage consuming `model.pkl` is normal.

## Facts, deterministic findings and AI hypotheses

- **Facts** are the parsed stages, dependencies, outputs and locked values, exposed as
  `report["pipeline"]` (`status`, `stages`, `edges`, `unresolved`, `skipped`).
- **Deterministic findings** are the table above. Each carries `Evidence(source="code", …)` with a
  location, what was observed and what was expected, plus a stable key.
- **AI hypotheses** are not produced by this engine. The existing experimental `--ai` layer may
  interpret these findings, but it never changes their severity, evidence or fingerprint, and its
  output is not proof.

## Determinism and limits

- Stages and `LineageEdge`s are sorted; running the audit twice produces the same payload.
- `pipeline_split_overlap` only compares stages whose names identify a train side (`train`, `fit`,
  `fitting`) and an eval side (`eval`, `evaluate`, `test`, `score`, `predict`, `validate`), and only
  data-like suffixes (`.csv`, `.parquet`, `.json`, …). A pipeline with other stage names is not
  checked; no guess is made.
- `pipeline_unpinned_dependency` treats any non-`==` specifier as unpinned, including `>=`.
- Without PyYAML, directory audits skip DVC files and record `yaml_not_installed`; naming a DVC file
  explicitly raises `PipelineUnavailable` naming `pip install "leaklens-ml[yaml]"`.
- A lock that references a parameter file outside the audited root is skipped
  (`params_outside_root`) instead of being read, because lock metadata is untrusted input.
- Not implemented in this phase: DVC/MLflow integration modules (M5), the workspace lineage view
  (M6) and OpenLineage/Feast readers (M8). No orchestration, experiment tracking, DAG editor, graph
  database or network access is added.

## Graduation decision

The new kinds ship `scored=False` because there is no real-world labelled corpus yet. The synthetic
benchmark ([`PIPELINE_BENCHMARK.md`](PIPELINE_BENCHMARK.md)) reports precision/recall per family but
is a regression guard, not an accuracy claim. Graduation to the trust score requires a labelled
corpus of real repositories with recorded precision/recall; that decision is not part of Phase 14.
