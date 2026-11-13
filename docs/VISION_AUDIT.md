# Vision audit — exact duplicates and perceptual clusters (Phase 17 M2)

`leaklens images TRAIN TEST` groups images that are byte-identical or visually near-identical
across two declared split folders. Nothing here proves data leakage: a cluster is evidence for
investigation, and every finding stays diagnostic in the sense that it is the same `image_overlap`
finding that shipped before M2. M2 adds the grouping views and their contract; it does not add a
second finding, change `leakage.score()`, change confidence or change the exit code.

```sh
leaklens images data/train data/test            # text: pairs, exact groups, cluster summary
leaklens images data/train data/test --json     # adds the "clustering" payload
leaklens images data/train data/test --max-distance 6
```

The image intake is unchanged: Pillow is required (the `images` extra) and every file passes the
[M-S5 decode budgets](IMAGE_DECODE_BUDGETS.md) before it is hashed.

## Exact duplicate contract

- Exact means **byte equality**: SHA-256 over the file bytes. Two files are in one exact group only
  when their digests match and both are present in the other split.
- A different encoding of the same picture (PNG vs JPEG, different JPEG quality) is **not** exact,
  even when the decoded pixels are identical. LeakLens does not report decoded-pixel equality as a
  separate relationship; perceptual clustering covers that case with a distance of 0 and says so.
- Only images that passed every M-S5 budget are digested. A refused, oversized, animated,
  malformed or unreadable file never appears in an exact group, so a refusal can never be reported
  as a duplicate.
- Groups are keyed by digest and sorted, with sorted member lists, so enumeration order and file
  timestamps cannot change the result.
- The JSON payload reports `exact_duplicates` (number of test images whose bytes also exist in
  train) and `exact_groups` (`digest`, `train` members, `test` members). The text output reports the
  count. Exact groups are cross-split only; duplicates entirely inside one folder are not reported.

## Perceptual contract

- Signature: the existing 8x8 difference hash (64 bits, `images.dhash`). Grayscale gradient
  direction, so colour is ignored except as it changes luminance.
- Distance: Hamming distance between the two 64-bit hashes (`images.hamming`).
- Threshold: **6 bits**, calibrated in [VISION_BENCHMARK.md](VISION_BENCHMARK.md) from generated
  families; the sweep and the trade-off are published there, not asserted here.
- Uniform images (hash popcount 0 or all 64 bits set) are excluded from clusters: every flat-colour
  image hashes the same way, so they would all merge into one meaningless group. They still appear
  in `pairs` and, when byte-identical, in exact groups. The payload counts them as
  `uninformative`.
- The threshold is a distance bound, not a similarity guarantee. A pair at distance 6 is reported
  as near-duplicate evidence; a pair above it is not.

### Measured envelope (calibration partition, max distance observed)

| Transformation | Max distance | Claim |
|---|---:|---|
| Identical bytes | 0 | exact group |
| JPEG quality 80 re-encode | 1 | supported |
| Resize to 75% | 0 | supported |
| Brightness +12% | 2 | supported |
| Contrast +15% | 3 | supported |
| 30 px corner watermark | 3 | supported |
| 2 degree rotation | 5 | supported |
| 3% border crop | 5 | supported |
| 5% border crop | 6 | supported, at the boundary |
| 8% border crop | 11 | not claimed (4 of 5 cases happened to fall inside) |
| 25% border crop | 23 | not claimed |
| Letterbox padding into a canvas | 29 | not claimed |

The last three families are published as `unsupported` in the benchmark: the benchmark does not
count them as passing positives or as negatives, and the tool does not promise them.

### Known blind spots

- **Colour-only differences.** dHash is grayscale. Two images with the same luminance structure but
  different colours collide; on a smooth gradient with a differently coloured disc the measured
  distance is exactly 6, so the selected threshold reports that pair. This is a measured false
  positive, recorded as `known_collision` in the benchmark and never hidden.
- **Crops beyond the supported envelope.** Cropping changes the resampled layout, so 8% crops sit
  around 11 bits and 25% crops around 23 bits. Some 8% crops fall inside the threshold by luck; the
  contract does not claim them.
- **Padding, letterboxing and collage layouts** change the grid the hash sees and typically land
  outside the threshold.
- **Mirrors, rotations beyond a couple of degrees, and heavy augmentation** are not modelled.
- **Very small (e.g. 16x16 icons) and very low-contrast images** have few meaningful gradient bits;
  expect unstable distances.
- **Real photographs are harder than the generated corpus.** Structured random images are a
  regression guard for the envelope, not a real-world accuracy claim.

## Cluster semantics

Clusters are **connected components** of the pairwise relation, not representative-based groups and
not cliques:

- Two images are linked when their Hamming distance is at or below the threshold and both hashes are
  informative.
- Links are transitive by union-find. A cluster of three can contain a pair whose direct distance
  exceeds the threshold. Never read a cluster as "every pair here is within 6 bits".
- Every linked image is a member; there is no representative and no member is discarded.
- Singletons are omitted. Only components with at least two members are returned.
- Members are namespaced `train:<relative path>` and `test:<relative path>`, so a file with the same
  relative name in both folders stays two distinct members.
- IDs are the shared `near_duplicate.cluster_key` over the sorted member list: a 16-hex digest, so the
  same component always gets the same id, independent of input order, discovery order or process.
- Clusters are sorted by id, members are sorted, and both the pair list and the family rows are
  built from sorted inputs. Two runs over the same tree produce identical payloads.
- Each cluster reports `train` and `test` member counts and `exact_pairs`, the number of
  train-times-test digest matches inside it. A cluster can be entirely within one folder; the counts
  make that visible instead of implying a cross-split leak.
- Borderline comparisons are ordinary members: distance 6 is inside, 7 is outside, no buffering and
  no manual review flag. The distance stays in the `pairs` list so a reviewer can judge.

## Cost and resource boundaries

- Exact digests are read only after an image passed the decode budgets, once per image.
- Candidate generation is exhaustive but chunked: `images.pairs_within` computes a block of rows at
  a time with numpy and yields pairs lazily, so a repetitive collection never materialises its
  quadratic edge set as Python tuples. A 64-bit hash has no sketch to band, and byte-band LSH would
  emit roughly a quarter of all pairs as tuples; the measured cost is in the benchmark.
- Union-find is streaming (`near_duplicate.clusters_from_pairs`), so memory grows with members and
  distinct components, not with the number of edges.
- Decoding, pixel budgets, frame limits and refusal reasons are exactly M-S5's; clustering consumes
  only the hashes of accepted images and never re-decodes a file.
- `image_overlap()` keeps its original signature and cost: it skips the digest read and the pairwise
  clustering pass. The CLI calls `analyze_images()`, which adds the clustering views.

## Benchmark

`python -m leaklens.bench_vision` regenerates the corpus from fixed seeds and prints the sweep,
selected threshold, per-family results, cluster check, scaling and decode numbers. It writes
[VISION_BENCHMARK.md](VISION_BENCHMARK.md) only with `--write`; without the flag it prints and
changes nothing. Calibration and holdout seeds are separate, the threshold is selected on
calibration only, and the holdout is scored once at the selected threshold. The corpus digest is
pinned, so an accidental change to the generator or the case table fails the benchmark instead of
silently moving the numbers.
