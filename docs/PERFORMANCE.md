# Scanner performance

LeakLens optimizes only measured bottlenecks. This page records the scanner
baseline, the shared-parse/cache change, the grouped node index, and why parallel
scanning is not enabled.

## Method

Measurements were taken locally on Windows with CPython 3.14.3 and pandas,
scikit-learn and joblib from the project virtual environment. They are
representative timings for this checkout, not a performance guarantee.

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
# Cold and warm code scans over the package
@'
import time
from leaklens.scanner import scan_path
for i in range(3):
    t0 = time.perf_counter()
    findings = scan_path("leaklens")
    print(i, round(time.perf_counter() - t0, 4), len(findings))
'@ | .venv\Scripts\python.exe -
```

The dataset baseline used a 50,000-row frame with four features and a binary
target, split 40,000/10,000. Profiling used `cProfile` and `time.perf_counter`.

## Baseline

| Workload | Time |
|---|---:|
| `scan_path("leaklens")`, 65 Python files, cold process | 1.70 s |
| 50k-row `scan(df, target)` | 0.20 s |
| `feature_overlap` on 40k/10k rows | 0.019 s |
| `near_duplicate_overlap` on 20k/10k rows | 0.016 s |

The code scanner dominated wall time. Profiling showed every source was parsed
three times (core rules, flow analysis and cross-validation analysis) and the
same structural AST digests were computed more than once. Dataset hashing was
not a meaningful bottleneck: scikit-learn cross-validation inside
`single_feature_leaks` accounted for nearly all dataset time.

## Change

`scan_source` now parses once and shares the AST, its walked node tuple and the
stable structural keys with all three passes. A bounded process-local cache also
reuses the parsed tree, node tuple and digests for repeated scans of identical
text.

The cache key is `SHA-256(source)`. It never trusts mtime or file size, so file
modification and replacement are safe: the caller re-reads the file and changed
content misses. The tree and keys are label-independent; locations, findings and
fingerprints are still computed per call, so changing only the label cannot
change findings. The cache is protected by a lock, capped at 128 entries and
2,000,000 source characters, skips sources over 1,000,000 characters, and does
not store malformed input.

### Grouped node index

With parsing shared, the remaining warm cost was the per-pass loop over every
node checking `isinstance` against the few types that pass cares about. Profiling
of 125,304 nodes across the package showed 2.36 million `isinstance` calls per
scan, and a filtered loop over all nodes cost ~13 ms per pass.

`_node_index(nodes)` groups the already-walked nodes by exact type once per
`scan_source` call. Each pass then iterates only its own node types, in the same
`ast.walk` order, so per-type ordering, "last assignment wins" behaviour and
finding order are unchanged. The index is transient: it is rebuilt from the
cached node tuple for each file and is not stored in the source cache.

## Result

| Workload | Original | Shared parse | + grouped index |
|---|---:|---:|---:|
| Cold `scan_path("leaklens")` | 1.70 s | 0.68 s | 0.66 s |
| Repeated `scan_path("leaklens")`, one process | 1.70 s | 0.20 s | 0.15 s |
| 50k-row `scan(df, target)` | 0.20 s | 0.20 s | 0.20 s |

The shared parse is the main win; the grouped index adds ~22% on warm scans
(from 0.192–0.203 s to 0.150–0.158 s). Cold time is dominated by `ast.parse`, so
the index does not change it measurably. Per-pass timings with instrumented
wrappers:

| Pass | Before | After |
|---|---:|---:|
| `_scan_core` | 110 ms | 57 ms |
| `scan_cv` | 78 ms | 54 ms |
| `scan_flow` | 19 ms | 5 ms |
| index build | — | 10 ms |

Warm peak memory stayed at ~0.42 MiB with `tracemalloc`, because the index is
released after each file. Dataset results are unchanged because the change does
not touch dataset detectors.

### Rejected

- Caching the index in the source cache: it would save ~10 ms per warm scan but
  keep an extra reference per node for up to 128 cached sources.
- A descendant index for `_all_names`/`_strings`: those helpers walk small
  per-expression subtrees, and storing a descendant list per node would cost far
  more memory than the ~45 ms it could save.

## Parallel scanning

Parallelism was measured, not assumed. On the same 65 files:

| Strategy | Time |
|---|---:|
| Sequential | 0.54 s |
| 4-worker thread pool | 1.00 s |
| 4-worker process pool | 2.84 s |

The workload is Python-level AST work, so threads do not help; process startup
and inter-process transfer dominate on Windows. Parallel scanning remains
disabled, which keeps finding order, evidence ids and scores exactly sequential.

## Limits

- The cache is per process and in memory. Separate CLI invocations start cold.
- Cache entry sizes are bounded by source character count, not measured AST
  memory; very deeply nested but small sources can still use more memory than
  their text length.
- Cached objects are never mutated by the scanner. Any future pass that mutates
  an AST must copy it before reuse.
