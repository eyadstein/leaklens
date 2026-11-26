# The leakage-detection landscape

**Retrieved:** 2026-11-26. **Method:** primary sources only — repository metadata
and source files read through the GitHub API, release feeds, and arXiv abstracts.
Stars, release dates and file lists are observations on that date, not claims
about quality. Where a capability could not be verified in source or docs it is
marked *unverified* and is not used to rank anything.

This page exists to keep LeakLens honest about what it is: one tool in a crowded
space, strongest on the code that builds a split and weakest on everything a
production monitoring platform does.

## 1. What each project actually checks

| Project | Verified artefact | What it reads | Leakage relevance | Licence | Latest release |
|---|---|---|---|---|---|
| **LeakLens** | `docs/DETECTORS.md`, 65 kinds | Python/notebook AST, dataframes, SQL text, image manifests, DL configs | Pre-split/pre-CV ordering, target and group encodings, CV misuse, notebook order, split overlap, temporal/group leakage | MIT (this repo) | 0.14.0 |
| **Deepchecks** | `deepchecks/tabular/checks/train_test_validation/` | A train frame and a test frame the caller already split | `IndexTrainTestLeakage`, `TrainTestSamplesMix`, `DateTrainTestLeakageDuplicates`, `DateTrainTestLeakageOverlap`, `FeatureLabelCorrelationChange`, drift | AGPL-3.0 | 0.19.1, 2024-12-15 |
| **Evidently** | `src/evidently/presets/` (`drift`, `classification`, `regression`, `recsys`, `dataset_stats`, `special`) | Reference and current datasets | Drift and quality; no train/test leakage check in the preset list | Apache-2.0 | v0.7.23, 2026-09-11 |
| **Great Expectations** | Expectation suites + checkpoints | Batches from a data context | Schema/distribution expectations; no ML or split vocabulary | Apache-2.0 | 1.25.0, 2026-11-12 |
| **Pandera** | DataFrame schemas and hypothesis tests | In-memory frames (pandas/polars/pyspark) | Runtime type/range/statistical contracts; no split awareness | MIT | v0.36.0, 2026-11-10 |
| **Alibi Detect** | Drift, outlier and adversarial detectors | Reference and test arrays | Distribution shift, not evaluation validity | NOASSERTION (custom) | v0.13.0, 2025-12-11 |
| **whylogs** | Mergeable profiles + constraints | Streams of rows | Profiling and monitoring; project last pushed 2025-01-10 | Apache-2.0 | v1.6.4, 2024-12-03 |
| **`malusamayo/leakage-analysis`** | ASE'22 tool: AST → transformed IR → Datalog (Souffle) + pyright | Python files converted from notebooks | Static detection of test-data leakage; the closest prior art to LeakLens' code scanner | MIT | no releases; last push 2023-05-02 |
| **SE4AIResearch Leakage Detector** | VS Code extension | Notebooks | Multi-test, overlap and preprocessing leakage in notebooks | MIT | no releases; last push 2025-05-07 |

The last two rows matter most. LeakLens did not invent static leakage detection
in notebooks: Yang et al., *Data Leakage in Notebooks: Static Detection and
Better Processes* (ASE'22) published a Datalog-based analyser with an empirical
evaluation, and a course-built VS Code extension ships the same three-family
taxonomy. Any claim that LeakLens is uniquely capable here would be false.

## 2. What LeakLens does well

- **It reads the code that produces the split.** Deepchecks, Evidently, GX,
  Pandera, Alibi Detect and whylogs all require the caller to hand over data that
  is already split (or already streamed). None can see that
  `StandardScaler().fit(df)` ran two lines before `train_test_split`, because
  that fact is not in the frames. LeakLens' `preprocess_before_split`,
  `target_encode_before_split`, `cv_on_test_data`, `tuning_on_test_data` and
  `notebook_out_of_order_execution` only exist at that layer.
- **One dependency-light path.** `pip install leaklens` plus the standard
  library covers the default scan. The ASE'22 tool needs Souffle and a custom
  pyright build; the VS Code extension needs Node plus prebuilt binaries or
  Docker. That difference is deployment, not analysis depth.
- **Evidence and identity are first-class.** Fingerprints, baselines,
  suppressions, SARIF with evidence, and a manifest of detector kinds and
  severities are part of the contract rather than an afterthought.
- **Breadth beyond code.** SQL lineage, declared-split audits, image
  near-duplicates, text/LLM contamination and DL artifact inspection share one
  report and one exit gate.

## 3. Where competitors are stronger

- **Statistical testing on real data.** Deepchecks' checks return measured
  statistics with conditions; LeakLens' data detectors are heuristics with
  severities. A team that wants "is this drift significant at n=10,000" should
  use Deepchecks or Alibi Detect.
- **Production monitoring.** Evidently and whylogs are built for ongoing
  windows, dashboards and service integration. LeakLens is a one-shot audit.
- **Framework breadth and integrations.** GX has mature orchestration
  integrations (Airflow, dbt); Pandera has deep DataFrame-typing support across
  pandas, polars and pyspark. LeakLens reads files and a few frame shapes.
- **Research evaluation.** The ASE'22 work evaluated a static analyser on real
  notebook corpora and published the result. LeakLens has authored matrices and
  a candidate corpus with single-reviewer labels — not an equivalent evaluation.
  This is the largest credibility gap and no amount of feature work closes it.

## 4. Gaps that no project fills

1. **No tool connects code-level causality to data-level statistics.** One side
   sees the ordering bug but not its measured effect; the other sees the effect
   but not the cause. A report that pairs "this scaler was fitted on the full
   frame" with "the resulting AUC inflation is X points on this dataset" does not
   exist in any of the projects above.
2. **Notebook execution order is largely unmodelled.** Only LeakLens (in this
   list) reports out-of-order execution, and only syntactically.
3. **No shared leakage benchmark.** Each tool reports its own cases; there is no
   agreed corpus with independent labels. The ASE'22 materials and the papers in
   §5 are the closest thing, and none is a maintained benchmark.
4. **Licence friction is real.** Deepchecks is AGPL-3.0, so embedding it in a
   product is a legal decision, not an import.

## 5. Recent research (verified abstracts)

- **Sharma, *Not All Data Leakage Is Equal: A Controlled Benchmark of Evaluation
  Inflation in Tabular Classification*** (arXiv 2611.18598, 2026-08-31). 600
  paired clean-versus-leaked comparisons across six datasets, three model
  families and five repeated splits. Global target encoding inflated held-out
  ROC-AUC by 2.42 points (95% CI 0.62–4.89), global supervised feature selection
  by 1.93 (0.63–3.39), duplicate contamination by 1.52 (0.40–2.80); global
  scaling and global median imputation had near-zero average effects under i.i.d.
  settings. **Implication for LeakLens:** severity should stay severity-aware —
  label-dependent transforms and record reuse above distribution-only transforms.
  Our `target_encode_before_split` is `high` while `transformer_before_split` is
  `medium`; this paper is evidence for that ordering, and evidence against
  claiming that every pre-split scaler is equally harmful.
- **Essus, Vatsavai, Rachunok, *Data Leakage Inflates Generalizability of Power
  Outage Prediction Models*** (arXiv 2608.24665, 2026-08-25). Random splits
  inflate performance through spatial and temporal autocorrelation; under
  leave-one-state-out and leave-one-event-out the models often fail to beat a
  null baseline. **Implication:** the group and temporal families
  (`group_overlap`, `temporal_leakage`, `group_cv_missing_groups`) are not
  cosmetic, and a holdout must be split by entity and time, not by row.
- **Roy, Minu, *Unmasking Shortcut Learning in IoT Intrusion Detection*** (arXiv
  2609.28725, 2026-09-23). On 3.6M flows, conventional random-flow splitting
  raised attack recall by up to 14 points relative to PCAP-disjoint splits;
  raw timestamps let tree models reach 99.28% Macro-F1 against 90.62% for a
  linear model. **Implication:** identifier and timestamp leakage
  (`id_like`, temporal features) inflates results through model capacity, and the
  unit of separation must be the session/file, not the row.

These three are consistent with the taxonomy LeakLens implements. None of them
validates LeakLens itself; they validate that the problem is real and
measurable.

## 6. Integrate or build

| Capability | Decision | Reason |
|---|---|---|
| Drift/outlier statistics | **Integrate, do not build** | Alibi Detect and Evidently are stronger and maintained; LeakLens' `--drift` is a deliberately small opt-in diagnostic. |
| Expectation/schema contracts | **Integrate (read their JSON) when asked** | GX and Pandera own this; duplicating it adds dependencies without a leakage story. |
| Code-level ordering analysis | **Build** | No maintained tool with a pip-only install covers the families LeakLens covers; the ASE'22 tool is dormant and needs Souffle. |
| Measured effect of a detected leak | **Build, narrowly** | This is the ecosystem gap in §4.1. A single deterministic "re-run without the leaked column/step" measurement is more valuable than another detector. |
| Dashboards and monitoring | **Do not build** | Evidently/whylogs own it; LeakLens is an audit. |
| Shared leakage benchmark | **Build with others, not alone** | Only independent labels make it useful; a self-authored corpus is what we already have. |

## 7. Capability selection for this phase (Phase 20P)

The candidate gaps were weighed against what the code and tests actually show,
not against how many detectors exist. Current state: 65 deterministic finding
kinds (`docs/DETECTORS.md`); the authored 22-case matrix still has 0 failures;
the scanner floods least on single-call preprocessing order and most on
multi-stage flow, which is why the ordering detectors keep growing.

| Candidate gap | Evidence checked | Decision |
|---|---|---|
| Structured evidence for data findings | Six `Finding()` calls in `leakage.py` — `id_like`, `categorical_target_leakage`, `group_missing`, `group_leakage_risk` and both `feature_overlap` branches — serialised `"evidence": []`, so they could not join the evidence graph or SARIF properties (#621). | **Build.** Done: every data finding now carries a bounded `Evidence` record (source, kind, observed, expected). |
| Cross-file and notebook lineage | `flow.py`/scanner already follow assignments within a file and across executed cells; a cross-file claim would need import resolution the scanner does not do. | **Defer.** The blind spot is real but a sound version needs a module graph, which is a project, not a detector. |
| Multi-stage pipeline information flow | `pipeline_unfitted_step`, `pipeline_unpinned_dependency` and the post-split rules cover the visible order; no failing realistic case was found beyond them. | **Defer.** Adding a rule here costs false positives on legitimate `Pipeline` use. |
| Evaluation-contamination detection | Text engine is hash-only against caller-supplied eval fingerprints; it already reports contamination when given the evidence. Detecting it from a notebook alone needs the eval corpus. | **Do not build.** It would require shipping or fetching data, which the offline default forbids. |
| Split-aware transformations | `group_stat_after_split` and `frequency_encode_after_split` closed this (#656). | **Done.** |
| Measured effect of a detected leak | Already selected in §6 as the ecosystem gap; still unbuilt and still the largest single value item. | **Next phase.** |
| Indirect leakage through a feature-generation function | The scanner resolves calls inside a file but not through a helper module; same blocker as cross-file lineage. | **Defer.** |

Chosen work this phase: **evidence completeness**, because it is testable, needs
no new dependency, and turns six previously opaque findings into machine-readable
ones. The next capability with a clear contract is the measured
"re-run without the leaked column/step" effect in §6, not another syntactic rule.

## 8. Open questions

- Does LeakLens reproduce the ASE'22 tool's findings on the same notebooks, and
  where do the two disagree? A cross-tool comparison on a shared, licensed set is
  the cheapest credible external validation available.
- How large is the measured inflation for each LeakLens kind, in the style of
  arXiv 2611.18598? Severity is currently reasoned, not measured.
- Can a notebook's executed-cell order be recovered from `.ipynb` metadata
  reliably enough to raise `notebook_out_of_order_execution` above `low`?
- Is there a licensing-compatible way to redistribute a labelled leakage corpus?
  Every candidate so far has at least one non-permissive member.

## Sources

Repository metadata and source files read 2026-11-26 via
`api.github.com/repos/{deepchecks/deepchecks, evidentlyai/evidently,
great-expectations/great_expectations, unionai-oss/pandera,
SeldonIO/alibi-detect, whylabs/whylogs, malusamayo/leakage-analysis,
SE4AIResearch/DataLeakage_JupyterNotebook_Fall2024}`; arXiv abstracts
2611.18598, 2608.24665, 2609.28725; `docs/DETECTORS.md` in this repo.
Unverified in this pass: Deepchecks' exact statistical thresholds, the ASE'22
paper's reported precision/recall, and whether the SE4AIResearch extension is
maintained beyond its last push.
