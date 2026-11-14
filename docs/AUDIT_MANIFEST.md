# Audit manifest and offline verification (M-S3)

`leaklens FILE --target COL --manifest audit.json` writes a deterministic JSON
manifest of what the audit saw and concluded. `leaklens verify audit.json`
recomputes it with the standard library, offline, and reports any change.

The manifest is **evidence, not a certificate**. It proves that the document
(except `generated_at`) and the declared input files were not changed after the
manifest was written. It does not prove who wrote it and it does not prove the
conclusion is correct. A deliberate attacker who can edit the manifest can also
recompute its digest; record `manifest_digest` somewhere else (a report, a
ticket, a CI log) when a rewrite must be detected.

## Usage

```sh
leaklens data.csv --target is_fraud --test holdout.csv --manifest audit.json
leaklens verify audit.json
leaklens verify audit.json --root .              # resolve inputs from the project root
leaklens verify audit.json --json                # machine-readable result
```

`--manifest` is opt-in. Without it the payload, trust score, findings and exit
code are identical to today.

## What is recorded

| Key | Content |
|---|---|
| `schema`, `schema_version` | `leaklens-audit-manifest`, `1`. The verifier refuses any other version. |
| `tool` | LeakLens name and version that wrote the manifest. |
| `generated_at` | UTC timestamp. **Excluded from the digest** so two identical runs match. |
| `inputs` | Declared files/directories with `kind` and a `sha256:` digest. |
| `options` | Result-affecting options: target, group, time, threshold, SQL/drift/pipeline/AI flags and the opt-in failure flags. Output paths are not recorded. |
| `detectors` | The shipped detector catalogue: kind, surface, severity, scored flag, introducing release, required extra. |
| `result` | Serialized findings, counts, policy section, resolved baseline entries, trust score and the gate decision. |
| `section_digests` | One digest per section above, so a mismatch names the changed section. |
| `manifest_digest` | Digest of the whole document, excluding `generated_at` and itself. |

Inputs are resolved against the current directory when written. Paths inside
that directory are stored relative (portable); paths outside it are stored
absolute. Verification resolves them inside `--root` (default: the current
directory) and refuses any entry that resolves outside it, including `..`,
another drive, UNC and device paths, and a symlink whose target is outside the
root.

## Digest recipe

One implementation writes and verifies, so the recipe is simple to reproduce:

1. Canonical JSON: sorted keys, `,` and `:` separators, UTF-8, no NaN/Infinity.
2. `sha256(b"leaklens:audit-manifest:v1\0" + canonical_json(value))`, displayed
   as `sha256:<hex>`.
3. A file input digests its bytes. A directory input digests sorted
   `relative path + NUL + SHA-256(file bytes) + LF` records, skipping `.git`,
   `__pycache__`, `.venv`, `node_modules`, `dist`, `build`, `*.egg-info` and
   symlinks (so verification cannot read outside the declared tree).
4. `manifest_digest` covers the document without `generated_at` and itself.

`section_digests` lets the verifier report `findings changed`, `trust_score
changed`, `detectors changed` and so on. The top-level digest catches edits to
anything outside those sections.

## Verification outcomes

| Outcome | Exit | Meaning |
|---|---:|---|
| `manifest OK` | 0 | Digests and every declared input match. |
| `manifest FAILED` | 1 | A section changed, a digest is missing, an input is missing/modified, a path is unsafe, an input is not a regular file or directory, or the document has an unknown key. |
| `leaklens verify: ...` | 2 | The manifest is unreadable, is not valid UTF-8/JSON, repeats a JSON key, is not an object, or uses an unsupported schema/version. |

A repeated JSON key would let two readers see two different documents, so it is
refused instead of silently keeping the last value. Declared inputs must be
regular files or directories: a FIFO, socket or device fails with
`not a regular file`/`not a directory` and is never opened.

At schema version 1 every key is known, so an unknown key is a failure, not a
warning. A newer writer must bump `schema_version`; this verifier then refuses
the file explicitly instead of guessing. `generated_at` is deliberately outside
the digest, so a changed timestamp is not an integrity failure.

Verification reads bytes only. It never imports, unpickles, deserializes or
executes the audited project, and it imports no third-party module.

## Limits

- This is not a signature, not provenance and not a compliance certificate.
  Signature/Sigstore/attestation work is deliberately out of scope.
- Verification detects change relative to the digest stored in the same file.
  Keep the `manifest_digest` value outside the file to detect a full rewrite.
- Directory digests read every audited file, so they add I/O proportional to the
  input tree. That cost exists only when `--manifest` or `verify` is used.
- Input digests are computed when the manifest is written, after the audit read
  the data. An input changed between those two moments is recorded at its later
  state; treat a manifest as evidence of the bytes present at write time.
- Parsing is bounded (16 MiB manifest, 10,000 inputs, 200,000 files per
  directory input); a larger manifest or tree fails explicitly. A declared
  directory that cannot be fully listed fails verification instead of silently
  skipping the unreadable part.

See §T9 in [ARTIFACT_SECURITY_STRATEGY.md](ARTIFACT_SECURITY_STRATEGY.md).
