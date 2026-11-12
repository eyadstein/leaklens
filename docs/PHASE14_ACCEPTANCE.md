# Phase 14 acceptance record (2026-11-11)

Scope is [AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) §3. This record states what
shipped, what is deliberately deferred, and why. Phase 13 is accepted in
[PHASE13_ACCEPTANCE.md](PHASE13_ACCEPTANCE.md).

## Milestones

| Id | Milestone | Status | Evidence |
|---|---|---|---|
| M1 | Stage graph from repo artifacts (notebook order, `.dvc`, `dvc.yaml`, serialized DAG JSON); unresolved edges labelled | **Shipped** | `leaklens/pipeline.py` (`Stage`, `PipelineReport` carrying stages + edges), `tests/test_pipeline.py::test_three_stage_graph_yields_the_expected_edges`, `::test_notebook_stages_follow_execution_order` |
| M2 | `LineageEdge` contract + `sqlscan` column-map export | **Shipped** | `leaklens/lineage.py` (frozen, validated, sorted `(stage, column, transform, locator, confidence)`); `sqlscan.analyze_sql` returns `SqlAnalysis.column_map` once; `scan_sql` unchanged; `tests/test_sqlscan.py::test_analyze_sql_exposes_one_resolved_column_map` |
| M3 | Train/eval split-disjointness check across stages; same input is `high` with artifact and stage ids as evidence | **Shipped** | `_split_findings`, `tests/test_pipeline.py::test_shared_train_eval_input_is_high_with_evidence` |
| M4 | Reproducibility checks: unpinned deps, missing seeds, non-deterministic ops, missing digests; reuse `no_random_state`; framework-neutral advice | **Partial (3 of 4 checks)** | `pipeline_unpinned_dependency`, `pipeline_missing_digest`, `pipeline_param_drift` (config drift) and `no_random_state` reuse. Non-deterministic operations are **deferred**: metadata alone does not identify them without guessing, and the roadmap's own false-positive mitigation forbids guessing. Recorded below. |
| M5 | `integrations/dvc.py`, `integrations/mlflow.py` | **Deferred** | Local DVC metadata is read directly by M1; an integration package is interface work without new capability. MLflow has no fixture in this repository. |
| M6 | Lineage view payload + workspace view with a DOM-level unresolved label/legend test | **Deferred** | Needs the workspace surface and a front-end test harness; no new intake is introduced by deferring it. |
| M7 | Pipeline benchmark corpus ≥20 labelled pipelines with precision/recall published, misses listed | **Shipped** | `leaklens/bench_pipeline.py` (22 cases, 8 formats), [PIPELINE_BENCHMARK.md](PIPELINE_BENCHMARK.md); misses/false alarms printed and exit 1 |
| M8 | `integrations/openlineage.py`, then `feast.py` | **Deferred** | No fixtures in this repository; each milestone is its own review. |

## Acceptance criteria

| # | Criterion | Result | Evidence |
|---:|---|---|---|
| 1 | A confirmed lineage path is the only way to a `high` severity; unresolved stays low/diagnostic | Pass | Only `pipeline_split_overlap` (high) is emitted, and it requires two parsed stages plus a confirmed shared/produced data artifact. Everything else is low and `diagnostic=True`. |
| 2 | Lineage edges are deterministic and stable across runs (sorted) | Pass | `lineage.sort_edges`; `tests/test_pipeline.py::test_the_same_metadata_always_produces_the_same_graph` |
| 3 | No orchestration system is executed or contacted | Pass | `pipeline.py` imports only `ast`, `json`, stdlib, `availability`, `evidence`, `leakage`, `lineage` and the shared `paths` containment helper; no subprocess, no network, no DVC/MLflow import. A lock that points a parameter file outside the root is skipped (`params_outside_root`), never read. |
| 4 | All new findings ship `scored=False` initially, with the graduation decision recorded | Pass | Manifest entries + `_DIAGNOSTIC_KINDS`; findings carry `diagnostic=True`; [PIPELINE.md](PIPELINE.md#graduation-decision) records the decision. `no_random_state` is an existing scored kind; its pipeline finding is still `diagnostic=True`. |
| 5 | Reading a malformed external metadata file degrades to a skip | Pass | `audit_pipeline` records `malformed_yaml` / `malformed_json` / `malformed_notebook` / `unreadable_file`; `tests/test_pipeline.py::test_malformed_metadata_degrades_to_a_skip` and three benchmark cases. |

## Integration contract

- **Engine id**: `leaklens.pipeline` added to `manifest.ENGINES`; the frozen-id test was updated
  additively (no existing id changed).
- **Kinds**: five new kinds, all `surface="code"`, `requires=None`, `scored=False`; `no_random_state`
  is reused, not duplicated.
- **Assembly**: findings are ordinary `Finding` objects appended to the same list and serialized by
  `assemble.build`; there is no second scoring path. `--pipeline` is opt-in and the CLI emits
  `report["pipeline"]` alongside the existing sections.
- **Availability**: `yaml = ["PyYAML>=6"]` extra, `safe_load` only, `PipelineUnavailable`
  (an `EngineUnavailable`) naming `pip install "leaklens-ml[yaml]"`.
- **No new dependencies** beyond the optional extra; no version bump.

## Deferred work and why the roadmap permits it

The roadmap states each milestone "is its own review" (M8) and each phase is independently
shippable. M5/M6/M8 add surfaces (integrations, a workspace view) rather than correctness, and M4's
non-deterministic-operation check cannot be deterministic from metadata alone. Deferring them keeps
"unresolved-by-default" and the false-positive mitigation intact. Phase 15 eligibility is unaffected:
the shared evidence, provenance, availability and assembly paths are unchanged and reusable.
