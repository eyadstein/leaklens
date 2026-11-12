# AI/ML ecosystem research

**Research date:** 2026-11-10. **Method:** direct fetches of primary sources — GitHub REST
repository and release endpoints, raw `LICENSE` files, PyPI JSON metadata, and official
documentation pages. `websearch` was unavailable in this environment (HTTP 402), so discovery was
deliberate rather than exhaustive: this is a purposeful sample, not a systematic review. Nothing
was installed, executed, vendored or copied. Version numbers and licenses are point-in-time and
must be re-checked before quoting in a later release.

**Purpose of this document:** decide which projects LeakLens should integrate with, which it
should compete with, and which it should ignore — and record the licence facts that constrain
embedding any of them.

---

## 1. How to read this

Not every project listed is a competitor. Three relationships matter:

- **Complement** — produces metadata or evidence LeakLens can consume read-only (trackers,
  orchestrators, lineage emitters).
- **Adjacent** — overlaps on one capability (duplicates, drift) but answers a different question.
- **Substitute** — a user could pick it instead of LeakLens for the same job.

The recurring finding across all categories is in §7: **every one of these systems records what a
user asserted or what a statistic measured; none verifies that an evaluation was valid.** That gap
is the product.

---

## 2. Data validation and quality suites

| Project | Purpose / user | Core abstraction | License (verified) | Latest observed | Relationship |
|---|---|---|---|---|---|
| **Deepchecks** | Continuous validation of ML models and data; ML engineers | `Check` → `CheckResult` with conditions; `Suite` | **AGPL-3.0** ([LICENSE](https://github.com/deepchecks/deepchecks/blob/main/LICENSE)) | 0.19.1, 2024-12-15 ([release](https://github.com/deepchecks/deepchecks/releases/tag/0.19.1)); no release in ~23 months | Adjacent (closest conceptual overlap: train/test validation suites) |
| **Evidently** | Data/model/LLM drift and quality reports; ML and platform teams | `Dataset` + `DataDefinition`, `Report([metrics])`, `Tests` | Apache-2.0 ([LICENSE](https://github.com/evidentlyai/evidently/blob/main/LICENSE)) | 0.7.23, 2026-09-11 ([release](https://github.com/evidentlyai/evidently/releases/tag/v0.7.23)) | Complement (drift), adjacent (reports) |
| **Great Expectations** | Declarative data-quality expectations; analytics and data engineering | `DataContext` → Asset/Batch → Expectation Suite → Checkpoint | Apache-2.0 ([LICENSE](https://github.com/fivetran/great_expectations/blob/develop/LICENSE)); repository now resolves to the `fivetran` org | 1.24.1, 2026-10-22 ([release](https://github.com/great_expectations/great_expectations/releases/tag/1.24.1)) | Complement (expectations as evidence) |
| **TensorFlow Data Validation** | Schema/stats validation inside TFX pipelines; TFX owners | `StatisticsProto` → `Schema` → `Anomalies` | Apache-2.0 ([LICENSE](https://github.com/tensorflow/data-validation/blob/master/LICENSE)) | 1.21.0, 2026-06-11 ([release](https://github.com/tensorflow/data-validation/releases/tag/v1.21.0)) | Complement (schema diff), ignore for deps |
| **whylogs** | Lightweight statistical profiling / data logging; monitoring teams | Mergeable `DatasetProfile` from sketches, constraints evaluated separately | Apache-2.0 ([LICENSE](https://github.com/whylabs/whylogs/blob/mainline/LICENSE)) | 1.6.4, 2024-12-03 ([release](https://github.com/whylabs/whylogs/releases/tag/v1.6.4)); repo pushed 2025-01-10 — dormant | Ignore |
| **Soda Core** | YAML data contracts verified by CLI; analytics engineers | SodaCL checks + `soda scan` over a SQL source | **Elastic License 2.0** ([LICENSE](https://github.com/sodadata/soda-core/blob/main/LICENSE)); PyPI metadata says "Proprietary" | 4.30.0, 2026-11-05 ([release](https://github.com/sodadata/soda-core/releases/tag/v4.30.0)) | Ignore (licence + SQL-only + telemetry default) |
| **Amazon Deequ** | "Unit tests for data" on Spark; JVM data engineers | `AnalysisRunner` metrics, `VerificationSuite` constraints | Apache-2.0 ([LICENSE](https://github.com/awslabs/deequ/blob/master/LICENSE)) | 2.1.0, 2026-09-16 ([release](https://github.com/awslabs/deequ/releases/tag/2.1.0)) | Ignore (JVM/Spark) |
| **Cleanlab** | Label-noise and data-issue detection; data-centric ML engineers | `Datalab`, `find_label_issues(labels, pred_probs)` | Apache-2.0 ([LICENSE](https://github.com/cleanlab/cleanlab/blob/master/LICENSE)) | 2.9.0, 2026-01-13 ([release](https://github.com/cleanlab/cleanlab/releases/tag/v2.9.0)) | Adjacent (label/dedup evidence), needs model probabilities — LeakLens must not |
| **DataProfiler** | Profile and label data content; governance and data engineering | `Profiler` variants producing a serializable profile | Apache-2.0 ([LICENSE](https://github.com/capitalone/DataProfiler/blob/main/LICENSE)) | 0.14.0, 2026-09-10 ([release](https://github.com/capitalone/DataProfiler/releases/tag/0.14.0)) | Complement at most; `[ml]` extra pulls TensorFlow/Keras |

**What they do well:** expectation vocabularies, report renderers, schema diffing, and the
"assert on data" habit. **What none does:** decide whether an evaluation is valid. Deepchecks is
the only one whose suites approach the question (train/test validation), and it is AGPL, stalled,
and plotting-heavy. Evidently's checks are distributional against a reference window. GX has no
ML vocabulary. whylogs is lossy and dormant. Soda is SQL-only and licence-constrained. Deequ needs
a JVM. Cleanlab needs a trained model's probabilities. DataProfiler profiles but never judges.

**Reusable formats for LeakLens:** GX expectation suites and Evidently report JSON are both
stable enough to read as *evidence*, not as truth, behind an optional adapter. Nothing here should
be a dependency.

---

## 3. Experiment tracking, orchestration and feature stores

| Project | Purpose / user | Metadata it defines | Offline without a server? | License (verified) | Latest observed | Relationship |
|---|---|---|---|---|---|---|
| **MLflow** | Tracking + registry (+ GenAI tracing in 3.x); ML engineers, platform teams | `run_id` params/metrics/tags/artifacts; `MLmodel` YAML with `flavors`, `run_id`, `signature`; registry protos (`RegisteredModel`, `ModelVersion`) | Yes: local `mlruns/` + filesystem artifacts | Apache-2.0 ([LICENSE](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)) | 3.18.0, 2026-11-10 ([PyPI](https://pypi.org/project/mlflow/)) | Complement (already an `arena` extra) |
| **DVC** | Data and experiment versioning beside git; data scientists | `.dvc` (content hashes), `dvc.yaml` stages, `dvc.lock` | Yes, no server, no DB | Apache-2.0 ([LICENSE](https://github.com/iterative/dvc/blob/main/LICENSE)) | 3.67.1, 2026-03-31 ([release](https://github.com/treeverse/dvc/releases/tag/3.67.1)) | Complement — best-in-class for pipeline truth |
| **Kubeflow Pipelines** | K8s-native pipeline orchestration with MLMD lineage; platform teams | v2 `PipelineSpec` IR YAML; MLMD `Context`/`Execution`/`Artifact` | No (cluster + object store + metadata DB) | Apache-2.0 ([LICENSE](https://github.com/kubeflow/pipelines/blob/master/LICENSE)) | 2.18.0, 2026-10-28 ([release](https://github.com/kubeflow/pipelines/releases/tag/2.18.0)) | Complement on compiled YAML only |
| **Feast** | Feature store with point-in-time joins; ML platform teams | protobuf `Registry` (`Entity`, `FeatureView`, `FeatureService`), `feature_store.yaml` | Yes: file offline store + SQLite | Apache-2.0 ([LICENSE](https://github.com/feast-dev/feast/blob/master/LICENSE)) | 0.67.0, 2026-10-13 ([release](https://github.com/feast-dev/feast/releases/tag/v0.67.0)) | Complement — strongest high-value audit target |
| **Metaflow** | Python-native flow authoring with automatic versioning; data scientists | `.metaflow/` metadata tree, flow/run/step/task ids, content-addressed artifacts (pickles) | Yes, local datastore | Apache-2.0 ([LICENSE](https://github.com/Netflix/metaflow/blob/master/LICENSE)) | 2.19.41, 2026-11-06 ([release](https://github.com/Netflix/metaflow/releases/tag/2.19.41)) | Complement, metadata only |
| **Apache Airflow** | General DAG scheduling; data engineers | Serialized DAG JSON (JSON-Schema), XCom rows, task-instance state | No true no-server mode (`standalone` is a trial mode) | Apache-2.0 ([LICENSE](https://github.com/apache/airflow/blob/main/LICENSE)) | 3.4.0, 2026-11-06 ([release](https://github.com/apache/airflow/releases/tag/3.4.0)) | Complement on serialized DAG JSON |
| **Prefect** | Pythonic orchestration; data/ML engineers | Flow/task run records, `prefect.yaml`, Blocks, persisted results | Local API server is the normal mode; "no server" **unverified** | Apache-2.0 ([LICENSE](https://github.com/PrefectHQ/prefect/blob/main/LICENSE)) | 3.9.2, 2026-11-06 ([PyPI](https://pypi.org/project/prefect/)) | Low priority |
| **Hugging Face Hub + `datasets`** | Model/dataset hosting and loading; researchers, app developers | Repo revisions, model-card and dataset-card frontmatter, `config.json`, `safetensors` weights, Arrow caches | `huggingface_hub` works offline against the cache; `datasets` is CPU-only | Apache-2.0 ([hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE), [datasets](https://github.com/huggingface/datasets/blob/main/LICENSE)) | hub 2.4.0, 2026-10-29; datasets 5.2.0, 2026-11-09 | Complement — highest-volume real-world leakage surface |

**The pattern:** all of these model *flow* — which task consumed which artifact — and none models
*correctness*. A pipeline that trains on `dataset@v3` and evaluates on `dataset@v3` is a valid
lineage graph in MLMD, DVC, OpenLineage, Airflow and Metaflow alike. MLflow records the metric
your code logged; Feast performs the as-of join your timestamp field declared; DVC hashes the file
your stage wrote. Metadata is self-asserted everywhere.

**Best integration targets, ranked by value per unit of effort:**

1. **DVC** — `.dvc`/`dvc.yaml`/`dvc.lock` are plain text, offline, and give a real stage DAG. Auditing
   the artifacts behind the hashes is exactly the missing semantic layer.
2. **MLflow** — already an extra; `mlruns/` and `MLmodel` YAML are readable without a server, and
   logged dataset tags can be cross-checked for train/eval overlap.
3. **Feast** — the registry plus the produced training frame is the only place where "label-derived
   feature" and "fanout join" are detectable after the fact.
4. **OpenLineage JSON** — an event spec, not a service. Consume JSONL events directly; do not
   require Marquez.
5. **Hugging Face snapshot** — local parquet/Arrow shards; exact and near-duplicate overlap across
   declared splits. Highest real-world hit rate.

**Deliberately not integrated:** Kubeflow (cluster), Airflow (server + DB), Prefect (server),
Marquez (JVM + Postgres). Read their exported metadata formats instead of running them.

---

## 4. Lineage

**OpenLineage** — Apache-2.0 ([LICENSE](https://github.com/OpenLineage/OpenLineage/blob/main/LICENSE)),
1.55.0 on 2026-10-30 ([release](https://github.com/OpenLineage/OpenLineage/releases/tag/1.55.0)).
JSON-Schema run events (`RunEvent`, `JobEvent`, `DatasetEvent`) with **facets** as the extension
point: `schema`, `columnLineage`, `dataQuality`, `parent`, `nominalTime`. The Python client can
emit to a file/stdout transport, so an audit can consume events with no server. Column-level
lineage exists only when an integration emits it; row-level, time-window and split lineage are
absent by design. **Marquez** (reference server) is Apache-2.0 but last released 0.50.0 on
2024-10-24 with continued pushes — release cadence looks stale, so do not depend on it.

**Why this matters for LeakLens:** OpenLineage is the right *substrate* for pipeline lineage —
adopt its vocabulary (`Dataset`/`Job`/`Run`/facet) rather than inventing a competing one, and read
events as an optional input. Do not implement an emitter until a user asks; do not run Marquez.

---

## 5. LLM, RAG and NLP

| Project | Purpose | Cost | License (verified) | Latest observed | Relationship |
|---|---|---|---|---|---|
| **transformers** | Model loading/fine-tuning | 12.4 MB wheel; torch ≈2 GB for real use | Apache-2.0 ([LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE)) | 5.22.2, 2026-11-06; now requires Python ≥3.11 | Optional far-future embedding backend; never a default |
| **datasets** | Uniform dataset access (Arrow) | `pyarrow` + `fsspec` + `dill` | Apache-2.0 ([LICENSE](https://github.com/huggingface/datasets/blob/main/LICENSE)) | 5.2.0, 2026-11-09 | Complement; read parquet directly with pyarrow instead |
| **sentence-transformers** | Embeddings and reranking | torch + weights | Apache-2.0 ([LICENSE](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)) | 6.2.0, 2026-10-21 | Optional only; embeddings are non-reproducible evidence |
| **LangChain** | LLM app orchestration | Small core, 100+ transitive packages typical | MIT ([LICENSE](https://github.com/langchain-ai/langchain/blob/master/LICENSE)) | `langchain` 1.4.10 / `langchain-core` 1.7.7, 2026-11-10 | Audit target (static wiring), never a dependency |
| **LlamaIndex** | RAG/data framework | Per-integration + vector store + model | MIT ([LICENSE](https://github.com/run-llama/llama_index/blob/main/LICENSE)) | 0.14.25, 2026-09-21 | Audit target |
| **Ragas** | Reference-free RAG evaluation | LLM calls per sample | Apache-2.0 ([LICENSE](https://github.com/explodinggradients/ragas/blob/main/LICENSE)) | 0.4.3, 2026-01-13 | Adjacent; LLM-judge metrics are not reproducible |

**The verified gap:** Ragas' complete metric index (Faithfulness, Context Precision/Recall, Context
Entities Recall, Noise Sensitivity, Response Relevancy, factual/semantic/string, SQL, testset
generation) contains **no train/test or corpus-overlap metric**. LlamaIndex's
`RetrieverEvaluator` measures hit rate and MRR — retrieval *quality* — so a retriever that returns
the eval item's own source document scores perfectly. LangChain has no provenance layer: nothing
stops the train corpus and the eval corpus from sharing one index.

**Design consequences for LeakLens:**

- Start with **no dependency at all**: exact hash of normalized text, then token-shingle MinHash,
  both reproducible and CPU-cheap. Embeddings must be opt-in, model-version-pinned and labelled as
  non-reproducible evidence.
- Audit **wiring statically** (AST): which loader feeds which index, which index serves which
  retriever, whether a train and an eval path reach the same store or the same source files.
- Refuse to evaluate answers with an LLM judge in the deterministic path.

---

## 6. Code analysis and computer vision

| Project | Purpose | License (verified) | Latest observed | Relationship |
|---|---|---|---|---|
| **Semgrep OSS** | Pattern-based static analysis over YAML rules | **LGPL-2.1** ([LICENSE](https://github.com/semgrep/semgrep/blob/develop/LICENSE)) | 1.185.0, 2026-11-10 | Optional subprocess integration only |
| **CodeQL** | Query code as a relational database | Queries **MIT** ([LICENSE](https://github.com/github/codeql/blob/main/LICENSE)); **CLI under GitHub CodeQL Terms** ([terms](https://github.com/github/codeql-cli-binaries/blob/main/LICENSE.md)) — OSS analysis, demos and academic use only without GHAS | rolling | Not integrable at zero cost; GitHub's own `codeql-action` already runs on this repo |
| **FiftyOne** | Dataset curation, visualization and model evaluation | Apache-2.0 ([LICENSE](https://github.com/voxel51/fiftyone/blob/main/LICENSE)) | 1.22.1, 2026-09-29 | Adjacent; requires MongoDB — never a dependency |
| **CleanVision** | Image dataset defect scan (duplicates, blur, size, …) | Apache-2.0 ([LICENSE](https://github.com/cleanlab/cleanvision/blob/main/LICENSE)) | 0.3.7, 2026-01-05; low activity | Adjacent; its exact-duplicate hashing is the primitive LeakLens needs |

**Semgrep licensing, worked through:** LGPL §5 puts a work that *uses* the library without
containing it outside the licence; §6 obligations attach when distributing the library together
with your work. An MIT CLI that neither vendors nor imports Semgrep source, and shells out to a
user-installed binary (optional extra or `shutil.which` probe), stays MIT-clean — with two
conditions: pass `--metrics off` (Semgrep phones home by default, see
[semgrep.dev/docs/metrics](https://semgrep.dev/docs/metrics)), and **do not ship rules from
`semgrep/semgrep-rules`**, which is under the separate
[Semgrep Rules License](https://semgrep.dev/legal/rules-license) and forbids redistribution. Write
our own rules or use the user's `--config`. This is an engineering reading of the licence text,
not legal advice; counsel should confirm before shipping the path.

**Annotation format authorities (primary):** COCO — https://cocodataset.org/#format-data (single
JSON with `images`/`annotations`/`categories`, bbox `[x,y,w,h]`, RLE masks). YOLO —
https://docs.ultralytics.com/datasets/detect/ (one `.txt` per image, `class cx cy w h` normalised;
note this is a de-facto convention documented by Ultralytics — whose *code* is AGPL-3.0, separate
from the format). Pascal VOC —
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html (per-image XML plus
`ImageSets/Main/*_{train,val}.txt` split files; the site answered only over plain HTTP on
2026-11-10).

The important detail: **VOC `ImageSets` and COCO's `train2017`/`val2017` directories are the
declared split authority.** Reading the declared split and hashing image bytes plus annotation
content is a fully deterministic train/test contamination check — and none of the tools above does
it across splits.

### 6.1 Model-artifact and near-duplicate prior art

Verified 2026-11-11 via the GitHub repository and release APIs:

| Project | Purpose | License (verified) | Latest observed | Relationship |
|---|---|---|---|---|
| **picklescan** (`mmaitre314/picklescan`) | Scans pickle opcodes for dangerous globals before loading | MIT | v1.0.5, 2026-07-01 | Adjacent. It scans serialized payloads; LeakLens refuses pickle instead of inspecting it (ADR-016). A refusal is simpler and cannot execute, so no dependency |
| **fickling** (`trailofbits/fickling`) | Pickle analysis, decompilation and safe-ish rewriting | LGPL-3.0 | v0.1.13, 2026-11-03 | Reference for what pickle analysis can see; LGPL and an execution-adjacent surface keep it out of the base install |
| **Netron** (`lutzroeder/netron`) | Interactive viewer for ONNX/TF/PyTorch graphs | MIT | v9.3.5, 2026-11-06 | Ignore as a dependency: it is a viewer, not a headless audit API. Confirms ONNX graph metadata is readable without a runtime |
| **datasketch** (`ekzhu/datasketch`) | MinHash/LSH sketches | MIT | v2.0.0, 2026-07-05 | Reference implementation for `near_duplicate.py`; the deterministic primitive is small enough to own, so no dependency |
| **text-dedup** (`ChenghaoMou/text-dedup`) | Pipeline-scale text deduplication (MinHash, suffix arrays) | Apache-2.0 | tag 2023.09.20; repo pushed 2026-03-09 | Reference for corpus-scale dedup; it is a pipeline framework, heavier than one primitive |

**What none does:** decide whether an evaluation was valid. Pickle scanners answer "is this file
dangerous?", viewers answer "what is in this graph?", dedup libraries answer "which texts are
similar?" — all useful inputs, none an evaluation-integrity verdict.

---

## 7. Cross-cutting findings

1. **Nobody verifies claims; everybody records them.** Trackers record asserted metrics,
   orchestrators record flow, data suites record statistics, HF cards record prose. The verification
   layer is empty.
2. **Licences constrain embedding more than capability does.** AGPL-3.0 (Deepchecks) and ELv2
   (Soda Core) are viral/restrictive for a distributed MIT tool; CodeQL's CLI terms exclude
   non-OSS use. This narrows "integrate" to *read their data* or *shell out optionally*.
3. **Telemetry is on by default in several tools** (Soda via OpenTelemetry, Semgrep unless
   `--metrics off`). Any LeakLens integration must be offline by default; this is a differentiator,
   not a checkbox.
4. **Approximate or churning outputs defeat CI baselines.** whylogs profiles are lossy; Evidently's
   report schema changed repeatedly across 0.4→0.7; TFDV pins TensorFlow and protobuf. A committed
   baseline needs byte-stable, exact output — which LeakLens already produces.
5. **Metadata formats are the real integration surface.** DVC lock files, MLflow `MLmodel`, Feast
   registry protos, OpenLineage JSON, serialized Airflow DAGs, HF cards. Reading them is cheap;
   running the systems is not.
6. **Heavy stacks are normal in this space** (TensorFlow for TFDV/DataProfiler, Spark for
   Deequ, MongoDB for FiftyOne, torch for anything embedding-based). Keeping those behind optional
   extras with per-engine gating is a defensible architectural choice, not a compromise.
7. **Contamination detection is the most under-served high-value capability observed.** Benchmarks
   and published datasets are routinely re-used in training; no general-purpose tool in this sample
   checks across declared splits.

---

## 8. What users actually need (as opposed to what tools advertise)

Derived from the documented limitations above, not from marketing:

- A **pass/fail gate that runs in CI without a server, a cluster, a GPU or a network**.
- A result they can **attach to a review**: stable identity per finding, evidence, and a diff
  against last week's baseline.
- **Determinism** so a gate does not flake when a dependency upgrades or a distribution shifts
  slightly.
- **Honest scope**: what was not checked, what was skipped for lack of a dependency, what is
  uncertain.
- **No model-training prerequisite** to audit data (Cleanlab's requirement is a fine research tool
  and a bad CI gate).
- **Reviewable remediation**, not just alerts.

LeakLens meets the gate, review-attachment and determinism needs today, and most of honest scope:
missingness and SQL report structured skips, but a single "what was skipped and why" list for every
engine only arrives with Phase 13's engine-status. The no-training-prerequisite need is met.
**Reviewable remediation** is the Phase 19 target, not a shipped capability. Extending into new
modalities is credible because the hard part is the contract, not the detector.

---

## 9. Integration decisions for LeakLens

| Project | Decision | Form | Phase |
|---|---|---|---|
| MLflow | Keep (already `arena`) | Read local `mlruns/` + `MLmodel` YAML; no server | 14 |
| DVC | Adopt | Parse `.dvc`, `dvc.yaml`, `dvc.lock`; audit content behind hashes | 14 |
| OpenLineage | Adopt the vocabulary | Read run-event JSONL; no emitter, no Marquez | 14 |
| Feast | Adopt | Read registry protobuf export + produced frame | 14 M8 |
| Airflow | Read only | Serialized DAG JSON in `analysis/pipeline/dag_files.py` (no adapter module) | 14 |
| Hugging Face | Adopt | Local snapshot parquet/Arrow; split overlap | 15 M1 |
| Semgrep | Optional subprocess | User-installed binary, `--metrics off`, own rules | 19 |
| Great Expectations / Evidently | Not scheduled | Read their report JSON as evidence **if** a user needs it; no adapter is planned in Phases 13–19 | deferred |
| CleanVision / FiftyOne | Do not integrate | Reimplement exact-hash duplicate detection natively | 17 |
| Deepchecks / Soda / Deequ / TFDV / whylogs / CodeQL / Ragas | Do not integrate | Licence, weight or irreproducibility | — |

Rule for every integration: **read-only, optional, offline, and never a mandatory dependency.** An
integration that fails to parse must degrade to a recorded skip, never to a failed audit.

---

## 10. Sources

All URLs fetched 2026-11-10. Licence claims cite the repository licence file; version claims cite
the release or PyPI page named above.

Data quality: `github.com/deepchecks/deepchecks`, `github.com/evidentlyai/evidently`,
`github.com/great_expectations/great_expectations` (→ `fivetran/`),
`github.com/tensorflow/data-validation`, `github.com/whylabs/whylogs`,
`github.com/sodadata/soda-core`, `github.com/awslabs/deequ`, `github.com/cleanlab/cleanlab`,
`github.com/capitalone/DataProfiler`.

MLOps: `github.com/mlflow/mlflow`, `github.com/iterative/dvc` (→ `treeverse/dvc`),
`github.com/kubeflow/pipelines`, `github.com/feast-dev/feast`, `github.com/Netflix/metaflow`,
`github.com/apache/airflow`, `github.com/PrefectHQ/prefect`, `github.com/OpenLineage/OpenLineage`,
`github.com/MarquezProject/marquez`, `github.com/huggingface/huggingface_hub`,
`github.com/huggingface/datasets`.

LLM/CV/SAST: `github.com/huggingface/transformers`, `github.com/UKPLab/sentence-transformers`,
`github.com/langchain-ai/langchain`, `github.com/run-llama/llama_index`,
`github.com/explodinggradients/ragas`, `github.com/semgrep/semgrep`,
`github.com/github/codeql`, `github.com/github/codeql-cli-binaries`,
`github.com/voxel51/fiftyone`, `github.com/cleanlab/cleanvision`, `cocodataset.org`,
`docs.ultralytics.com`, `host.robots.ox.ac.uk/pascal/VOC/voc2012`.

Artifact and dedup prior art (verified 2026-11-11):
`github.com/mmaitre314/picklescan`, `github.com/trailofbits/fickling`,
`github.com/lutzroeder/netron`, `github.com/ekzhu/datasketch`,
`github.com/ChenghaoMou/text-dedup`.

**Unverified in this pass:** Evidently's polars/Spark support; TFDV's custom-validator API;
WhyLabs SaaS status; the GX→Fivetran transfer date; DataProfiler CLI existence; Prefect 3's exact
"no server" ephemeral mode; Kubeflow Docs reachability during research; CodeQL CLI size and exact
tag. Treat all of these as open if they become load-bearing.
