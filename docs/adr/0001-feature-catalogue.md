# Keep FEATURE_NAMES as the single feature catalogue

The learned detector now treats `FEATURE_NAMES` as the authoritative catalogue for the
emitted feature vector, and `MODEL_FEATURE_NAMES` as the derived model-eligible view.
Explanation metadata in `leaklens.ai.learned` is validated against that catalogue at import,
and `extract_features` fails loudly if it produces an indicator that the catalogue does not
register. We intentionally do **not** refactor the fit-event scanning in
`leaklens.ai.features` into a separate indicator factory: the current one-pass analysis is
small enough to keep local, and splitting it would add a new seam without reducing error risk.

## Consequences

Adding an indicator now requires registering its name in one place. Adding a name only to an
explanation table fails the import guard. Reordering `FEATURE_NAMES` changes the positional
vector, so it must remain a deliberate change with retraining and benchmark checks.
