# Vision audit — exact duplicates, perceptual clusters and declared splits (Phase 17 M2–M3)

`leaklens images TRAIN TEST` groups images that are byte-identical or visually near-identical
across two declared split folders. `leaklens splits DATASET` reads the split declarations a COCO,
YOLO or Pascal VOC dataset already ships and reports image content that appears in more than one
partition. Nothing here proves data leakage: a duplicate, a cluster or a reused path is evidence for
investigation. Every `images` finding is the same `image_overlap` finding that shipped before M2;
M2 adds the grouping views and their contract and does not change `leakage.score()`, confidence or
the exit code. M3 adds one opt-in finding kind, `image_split_overlap` (see below).

```sh
leaklens images data/train data/test            # text: pairs, exact groups, cluster summary
leaklens images data/train data/test --json     # adds the "clustering" payload
leaklens images data/train data/test --max-distance 6

leaklens splits datasets/coco                   # detect the declared format
leaklens splits datasets/voc --format voc --json
leaklens splits datasets/yolo --samples 10
```

The image intake is unchanged: Pillow is required for `leaklens images` (the `images` extra) and
every file passes the [M-S5 decode budgets](IMAGE_DECODE_BUDGETS.md) before it is hashed.
`leaklens splits` only streams file bytes for SHA-256, so it needs neither Pillow nor a decoder.

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

## Declared-split disjointness (M3)

`leaklens splits DATASET` answers one question: does the same image content appear in more than one
declared train/val/test partition? It reads declarations, not pixels.

### Supported declarations

Format auto-detection is deterministic and ordered: **VOC** (`ImageSets/Main/` exists), then
**YOLO** (`data.yaml` or `dataset.yaml` exists), then **COCO** (`annotations/*.json` exists).
`--format auto|coco|yolo|voc` overrides the detection.

| Format | Split declaration | Notes |
|---|---|---|
| COCO | `annotations/*.json` with an `images` list | The split comes from the first `train`/`val`/`validation`/`test` token in the annotation file's name (`instances_train2017.json`, `val.json`, ...). `images[].file_name` is the asset; `images[].id` is **not** used as content identity. No token means the file is not a split declaration. |
| YOLO | `train`, `val`, `test` keys in `data.yaml`/`dataset.yaml` | Each value is a directory, one image path, a list of those, or a `.txt` list with one path per line. Paths resolve against the dataset root (the config's directory); absolute paths are accepted only inside that root. `names`/`nc` are not split declarations. |
| VOC | `ImageSets/Main/{train,val,test}.txt` | Identifiers are extensionless stems resolved through `JPEGImages/`. An identifier that already has an image suffix (or a path separator) is resolved as a path. `trainval.txt` is the documented union of train and val and is **not** audited as a separate partition. |

Canonical partition names are `train`, `val` and `test`; `validation`, `valid` and `testing` normalise
to them. Keys and tokens are matched case-insensitively after lowercasing.

### Path resolution and identity

- Every declared path is resolved with `paths.resolve_within`: a traversal (`..`), a foreign drive,
  a UNC/device anchor or a symlink that points out of the dataset root is refused **before** the file
  is opened.
- COCO `file_name` is tried relative to the dataset root, then relative to `images/`. If both exist,
  the declaration is ambiguous and is not guessed.
- YOLO `.txt` entries are dataset-root-relative; YOLO directory entries are the image files found
  under that directory.
- VOC identifiers with no suffix are looked up by exact stem in `JPEGImages/`. Two files with the
  same stem but different image suffixes are ambiguous, not guessed.
- Identity is the resolved path relative to the dataset root, in POSIX form. A declaration that does
  not resolve to a file with a supported image suffix is not an asset.
- Identity across platforms: the audit is case-sensitive at the declaration level wherever the
  filesystem is; it never invents a name it did not see.

### Disjointness contract

- Content identity is **SHA-256 over the file bytes**, reusing `images.file_digest`. Two files with
  the same digest are byte-identical; a PNG and a JPEG of the same picture are not. M3 does not use
  dHash, does not change M2's 6-bit threshold, and does not claim that different digests are
  different photographs.
- Only files that resolved inside the root and passed the size check are digested. A refused or
  unreadable declaration is counted with a stable reason and can never appear as a duplicate.

The audit distinguishes five outcomes:

| Outcome | Meaning | Where it appears |
|---|---|---|
| Cross-split exact duplicate | One digest appears in two or more canonical partitions | `high` `image_split_overlap` finding, `cross_split` groups |
| Within-split duplicate | One digest appears more than once in a single partition | `within_split` group, no finding |
| Conflicting declaration | One image path is declared in two or more partitions, without a digest to compare | `high` `image_split_overlap` finding, `identity_conflicts` |
| Unresolved membership | A declaration cannot be mapped to a verified image (missing, outside root, ambiguous, control characters, unsupported suffix, oversized) | `unresolved` reason counts |
| Parse limitation | A manifest or list could not be read safely, or a declaration form is unsupported | `limitations` list and exit code 2 when nothing could be verified |

### Finding and exit contract

- The kind is `image_split_overlap`, surface `image`, severity `high`, `diagnostic=False`. The
  roadmap's M3 acceptance explicitly requires a `high` finding; this is the only reason M3 is
  scored. It is emitted only by the opt-in `leaklens splits` command and never alters an existing
  `leaklens`, API, workspace or report result.
- One finding is emitted per cross-split digest group (up to a bound) plus one bounded summary
  finding when groups are omitted. Evidence carries the digest prefix, the partition names, bounded
  paths and the declaration files; it never carries image bytes. Wording says "integrity concern,
  not proof of leakage".
- Trust score is `score(findings)`, unchanged. Exit codes: **0** audit ran and found no cross-split
  overlap, **1** at least one finding, **2** no audit could run (unknown layout, missing optional
  parser dependency, or nothing could be verified). A partial audit still exits 0/1 and prints its
  limitations; it is never reported as a clean result.

### Resource and security bounds

`MAX_MANIFEST_BYTES` (64 MiB per JSON/YAML document), `MAX_LIST_BYTES` (8 MiB per text list),
`MAX_DECLARATIONS` (100,000 records per audit), `MAX_PATH_CHARS` (1,024 per declaration),
`MAX_IMAGE_BYTES` (64 MiB per file, reused from M-S5) and `MAX_DIGEST_BYTES` (1 GiB aggregate) are
enforced before work grows. YAML is read with `safe_load`; a Python/object tag or any other parser
error is a typed limitation and is never executed. Remote (`scheme://`) declarations are reported,
never fetched. The audit opens only declared paths inside the root and writes nothing to the dataset
tree. A digest that cannot be read or a budget that is exhausted leaves the remaining members in the
identity checks and records the limitation instead of crashing.

### Known limitations

- COCO has no standard in-file partition marker, so a file name without a `train`/`val`/`test` token
  is ignored rather than guessed. Point the tool at annotation files that follow the convention, or
  use `--format` with a supported layout.
- VOC resolution assumes the standard `JPEGImages/` directory and `ImageSets/Main/`. XML annotations
  are not parsed (that is M4), and `trainval.txt` alone cannot split train from val.
- YOLO configuration is read only from `data.yaml`/`dataset.yaml`; a `.txt` list's entries are
  dataset-root-relative, and glob patterns (`*.jpg`) are not expanded.
- The audit checks declarations; it does not validate that an image decodes, nor that its annotation
  is valid, nor that a near-duplicate (non-identical) image is shared. Those are M4/M5/M6 concerns.
- A hardlink or two byte-identical files are indistinguishable by content; the finding says the
  content is duplicated, not which file is "the original".

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
