# Pipeline benchmark

Synthetic, labelled pipeline repositories for the Phase 14 pipeline/MLOps engine. Truth is the exact `(kind, column)` set per case. These are hand-written controls, not a sample of real repositories, so the numbers are a regression guard and do not estimate real-world accuracy. Times and memory are measured on the machine that generated this file.

## Metrics

| Metric | Value |
|---|---:|
| Cases | 22 |
| Expected findings | 12 |
| Reported findings | 12 |
| True positives | 12 |
| False positives | 0 |
| False negatives | 0 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| Findings with complete evidence | 12 (1.00) |
| Formats covered | dag_json, dvc, dvc_file, dvc_lock, empty, malformed, notebook, requirements |
| Total runtime | 75 ms |
| Median case | 2 ms |
| Slowest case | 37 ms |
| Peak memory | 968 KiB |

## By family

| Family | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| no_random_state | 1 | 0 | 0 | 1.00 | 1.00 |
| pipeline_missing_digest | 1 | 0 | 0 | 1.00 | 1.00 |
| pipeline_param_drift | 2 | 0 | 0 | 1.00 | 1.00 |
| pipeline_split_overlap | 2 | 0 | 0 | 1.00 | 1.00 |
| pipeline_unpinned_dependency | 3 | 0 | 0 | 1.00 | 1.00 |
| pipeline_unresolved_lineage | 3 | 0 | 0 | 1.00 | 1.00 |

## Cases

| Case | Family | Formats | Status | Findings | Skipped | Evidence | ms | Peak KiB |
|---|---|---|---|---|---|---:|---:|---:|
| clean three stage dvc pipeline | clean | dvc | ok | - | - | - | 37.3 | 968 |
| train and eval share one dataset | pipeline_split_overlap | dvc | ok | pipeline_split_overlap:data/all.csv | - | 1/1 | 2.8 | 137 |
| evaluation reads the training artifact | pipeline_split_overlap | dvc | ok | pipeline_split_overlap:data/train_features.csv | - | 1/1 | 3.7 | 141 |
| disjoint split in dag json | clean | dag_json | ok | - | - | - | 1.2 | 136 |
| dag json dependency has no producer | pipeline_unresolved_lineage | dag_json | ok | pipeline_unresolved_lineage:data/absent.parquet | - | 1/1 | 1.0 | 133 |
| unresolved stage dependency | pipeline_unresolved_lineage | dvc | ok | pipeline_unresolved_lineage:data/missing.csv | - | 1/1 | 2.5 | 136 |
| notebook reads a missing artifact | pipeline_unresolved_lineage | notebook | ok | pipeline_unresolved_lineage:data/absent.csv | - | 1/1 | 1.0 | 131 |
| locked parameter drift | pipeline_param_drift | dvc_lock | ok | pipeline_param_drift:params.yaml:seed | - | 1/1 | 3.2 | 137 |
| locked parameter key removed | pipeline_param_drift | dvc_lock | ok | pipeline_param_drift:params.yaml:max_depth | - | 1/1 | 3.6 | 137 |
| locked parameters still match | clean | dvc_lock | ok | - | - | - | 3.3 | 137 |
| locked output without digest | pipeline_missing_digest | dvc_lock | ok | pipeline_missing_digest:model.pkl | - | 1/1 | 1.8 | 133 |
| unpinned requirements entry | pipeline_unpinned_dependency | requirements | ok | pipeline_unpinned_dependency:requirements.txt, pipeline_unpinned_dependency:requirements/dev.txt | - | 2/2 | 1.3 | 134 |
| requirements without any version | pipeline_unpinned_dependency | requirements | ok | pipeline_unpinned_dependency:requirements.txt | - | 1/1 | 0.7 | 131 |
| pinned requirements only | clean | requirements | ok | - | - | - | 0.8 | 131 |
| training stage without a seed | no_random_state | dvc | ok | no_random_state:dvc.yaml:train | - | 1/1 | 2.7 | 136 |
| seed makes the training stage reproducible | clean | dvc | ok | - | - | - | 2.6 | 136 |
| notebook cells in execution order | clean | notebook | ok | - | - | - | 1.3 | 133 |
| dvc file declares outputs | clean | dvc_file | ok | - | - | - | 1.8 | 134 |
| malformed dvc yaml is a skip | clean | malformed | no_artifacts | - | malformed_yaml=1 | - | 0.9 | 131 |
| malformed dag json is a skip | clean | malformed | no_artifacts | - | malformed_json=1 | - | 0.6 | 131 |
| malformed notebook is a skip | clean | malformed | no_artifacts | - | malformed_notebook=1 | - | 0.6 | 131 |
| empty repository | clean | empty | no_artifacts | - | - | - | 0.3 | 4 |

## Misclassified cases

- none

## Limitations

- The corpus is synthetic and deliberately small; every positive case is a strong signal. Real DVC/DAG exports vary and can produce formats this corpus does not cover.
- `pipeline_split_overlap` only compares stages whose names identify a train and an eval side, and only data-like artifacts. A differently named pipeline is not checked.
- Pipeline findings ship `scored=False` (and are diagnostics), so none of the metrics above affect the trust score or the exit code.
- Thresholds were not tuned against this corpus; the cases define expected output and any change to them is a deliberate detector change.
