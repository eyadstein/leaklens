# Vision benchmark (Phase 17 M2)

Generated, deterministic image relationships for the Phase 17 M2 clustering contract. The corpus is produced by `leaklens.bench_vision`; no third-party image ships and no real-world accuracy is claimed. Threshold selection uses the calibration partition only; the holdout partition is scored once at the selected threshold.

Corpus version `1.0.0`, 180 cases (90 calibration, 90 holdout), digest `435b91e21a355723e061b9e3a8828c8dc9d554d939331f4890ab0da57c96d55a`.

## Threshold sweep (calibration only)

| Threshold | Supported recall | Unsupported recall | Negative merges | Known collisions |
|---:|---:|---:|---:|---:|
| 0 | 0.42 | 0.00 | 0 | 0/5 |
| 1 | 0.60 | 0.00 | 0 | 0/5 |
| 2 | 0.75 | 0.00 | 0 | 0/5 |
| 3 | 0.93 | 0.00 | 0 | 0/5 |
| 4 | 0.97 | 0.13 | 0 | 0/5 |
| 5 | 0.97 | 0.20 | 0 | 0/5 |
| 6 | 1.00 | 0.27 | 0 | 5/5 |
| 7 | 1.00 | 0.27 | 0 | 5/5 |
| 8 | 1.00 | 0.27 | 0 | 5/5 |
| 9 | 1.00 | 0.33 | 0 | 5/5 |
| 10 | 1.00 | 0.33 | 0 | 5/5 |
| 11 | 1.00 | 0.33 | 0 | 5/5 |
| 12 | 1.00 | 0.33 | 0 | 5/5 |

Selected threshold: **6 bits** (best supported recall with zero negative merges; smaller distance on a tie).

## Selected-threshold results

| Partition | Supported recall | Unsupported recall | Negative merges | Known collisions | Exact |
|---|---:|---:|---:|---:|---:|
| Calibration | 1.00 | 0.27 | 0 | 5/5 | 5/5 |
| Holdout | 1.00 | 0.27 | 0 | 5/5 | 5/5 |

## Per-family results

| Partition | Family | Label | Cases | Detected | Exact |
|---|---|---|---:|---:|---:|
| calibration | identical | exact | 5 | 5 | 5 |
| calibration | jpeg | duplicate | 5 | 5 | 0 |
| calibration | resize | duplicate | 5 | 5 | 0 |
| calibration | brightness | duplicate | 5 | 5 | 0 |
| calibration | contrast | duplicate | 5 | 5 | 0 |
| calibration | watermark | duplicate | 5 | 5 | 0 |
| calibration | rotate2 | duplicate | 5 | 5 | 0 |
| calibration | crop97 | duplicate | 5 | 5 | 0 |
| calibration | crop95 | duplicate | 5 | 5 | 0 |
| calibration | crop92 | unsupported | 5 | 4 | 0 |
| calibration | crop75 | unsupported | 5 | 0 | 0 |
| calibration | letterbox | unsupported | 5 | 0 | 0 |
| calibration | unrelated | negative | 5 | 0 | 0 |
| calibration | permuted | negative | 5 | 0 | 0 |
| calibration | shared_background | negative | 5 | 0 | 0 |
| calibration | flat_colour | negative | 5 | 0 | 0 |
| calibration | crop50 | negative | 5 | 0 | 0 |
| calibration | gradient_colour | known_collision | 5 | 5 | 0 |
| holdout | identical | exact | 5 | 5 | 5 |
| holdout | jpeg | duplicate | 5 | 5 | 0 |
| holdout | resize | duplicate | 5 | 5 | 0 |
| holdout | brightness | duplicate | 5 | 5 | 0 |
| holdout | contrast | duplicate | 5 | 5 | 0 |
| holdout | watermark | duplicate | 5 | 5 | 0 |
| holdout | rotate2 | duplicate | 5 | 5 | 0 |
| holdout | crop97 | duplicate | 5 | 5 | 0 |
| holdout | crop95 | duplicate | 5 | 5 | 0 |
| holdout | crop92 | unsupported | 5 | 4 | 0 |
| holdout | crop75 | unsupported | 5 | 0 | 0 |
| holdout | letterbox | unsupported | 5 | 0 | 0 |
| holdout | unrelated | negative | 5 | 0 | 0 |
| holdout | permuted | negative | 5 | 0 | 0 |
| holdout | shared_background | negative | 5 | 0 | 0 |
| holdout | flat_colour | negative | 5 | 0 | 0 |
| holdout | crop50 | negative | 5 | 0 | 0 |
| holdout | gradient_colour | known_collision | 5 | 5 | 0 |

## Clustering (calibration collection)

| Metric | Value |
|---|---:|
| Clusters | 5 |
| Expected pairs | 5 |
| Found pairs | 5 |
| False merges | 0 |
| False splits | 0 |

## Scaling and decode (clustering stage)

| Collection | Images | Seconds | Clusters |
|---|---:|---:|---:|
| random | 200 | 0.001 | 0 |
| random | 1000 | 0.015 | 0 |
| random | 4000 | 0.532 | 0 |
| repetitive | 1000 | 0.309 | 1 |

Decode throughput: 100 images in 0.184 s (544.7 images/s). Runtime is runner-dependent and is recorded, not gated.

## Misses

- none

## Method and limits

- Hashes: 8x8 difference hash (64 bits), computed through the M-S5 decode budgets; every case is rendered from a fixed seed.
- Exact: byte-identical files only, by SHA-256 of the file bytes.
- Perceptual: Hamming distance at or below the selected threshold, excluding uniform hashes (all-zero or all-one) from clusters.
- Clusters: connected components over direct comparisons; a cluster member is not necessarily within the threshold of every other member.
- Negatives: unrelated images, 180-degree layout rotation, a shared background with different content, different flat colours and a half-image crop. A smooth gradient with a colour-only difference is counted as a known collision, not as a passing negative: dHash is grayscale and ignores colour.
- Crops beyond 5% and letterboxing are reported as unsupported, not as passing positives: the threshold does not claim them.
- Synthetic structured images are easier than real photographs; these numbers are a regression guard, not a real-world precision/recall claim.
