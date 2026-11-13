# Image decode budgets (M-S5)

`leaklens images TRAIN TEST` hashes images with a 64-bit dHash. Image files are
untrusted input, so `leaklens/images.py` bounds every decode before pixel work
and reports a refusal reason instead of treating a hostile file as a clean image.

## Limits

| Limit | Default | Why |
|---|---|---|
| Encoded bytes per file | 64 MiB | An 8k photographic image is well below this; a larger file is more likely an archive or a padding attack than a dataset image. |
| Width or height | 20,000 px | Beyond consumer cameras and common dataset exports; prevents pathological line buffers before the pixel product is considered. |
| Decoded pixels per file | 64,000,000 (64 MP) | Roughly an 8000x8000 image. Pillow's own limit (89.4 MP) is left unchanged, so this is a stricter preflight, not a replacement. |
| Animation frames | 64 | dHash samples one frame; a long animation adds decode work without adding signal. The frame count is read from the container, not by decoding every frame. |
| Pixels per audit | 2,000,000,000 | One budget is shared by both split folders: ~10,000 images at 200,000 pixels, or ~31 images at the per-file maximum. |

The values are module constants, not CLI flags. A flag that silently raises a
security budget is worse than a documented constant; adjust the constants in a
reviewed change if a real dataset needs it.

## Flow

1. Skip files whose suffix is not in `IMAGE_SUFFIXES` (the existing discovery rule).
2. Check the compressed size with `stat()` before opening the decoder.
3. Open the header with Pillow, then check the decoded format against the
   allowlist (PNG, JPEG, BMP, GIF, WEBP, TIFF), dimensions, pixel count and frame
   count. These checks use metadata only.
4. Check the shared audit budget. If the image cannot fit, it is refused before
   decoding and no pixels are charged.
5. Only then convert to grayscale and resize to the 9x8 hash input. The pixel
   count is charged only after a successful decode, so a corrupt file with a
   readable header cannot exhaust the budget. Images are closed with context
   managers, including on failure.

Pillow's `DecompressionBombWarning` is promoted to an error inside the decode
block and both the warning and `DecompressionBombError` become a
`decompression_bomb` refusal. The process-wide `Image.MAX_IMAGE_PIXELS` is not
modified.

## Refusal reasons

`stats["refused"]` is a reason → count map. `stats["unreadable"]` remains an
integer and still counts only files the decoder could not read at all (truncated
data, malformed headers, I/O errors). A refusal never becomes a clean hash and
never disappears into `unreadable`.

| Reason | Trigger |
|---|---|
| `too_large` | Compressed file exceeds 64 MiB. |
| `too_large_dimensions` | Width or height exceeds 20,000 px. |
| `too_many_pixels` | Width × height exceeds 64 MP. |
| `too_many_frames` | Container declares more than 64 frames. |
| `unsupported_format` | Decoded format is outside the allowlist. |
| `decompression_bomb` | Pillow raised its bomb warning or error. |
| `budget_exhausted` | The audit's shared pixel budget cannot cover the image. |

The CLI prints the refusal counts on the text surface; `--json` includes the same
`refused` map and `decoded_pixels`.

## Limits of this control

- This bounds decoder work; it does not make Pillow a sandbox. A decoder memory
  bug remains a decoder memory bug.
- The budget is per process/audit, not per host. Parallel audits each get their
  own budget.
- A refused file is not hashed and not compared. A split whose images are all
  refused therefore produces no finding, a 100/100 trust score and exit code 0;
  the refusal line and `refused` map are the signal that nothing was compared.
  Treat a non-zero `refused` count as "audit incomplete", not "clean".
- `hash_dir` with no budget creates a throwaway budget: it still enforces the
  per-file limits, but the refusal counts are not returned. Pass a budget when a
  caller needs to see them.
- `decoded_pixels` counts pixels whose decode completed; a corrupt file with a
  readable header does not consume the shared budget.
- Video decoding, EXIF-based transforms and full animation decoding are out of
  scope. dHash uses the first decoded frame.
- Verification of image *content* is not attempted: a small, valid image can
  still contain anything a human or model cares about.
