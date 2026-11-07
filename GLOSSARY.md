# LeakLens

LeakLens vocabulary for leakage auditing: what a finding is, how trust is scored,
and how the optional learned intelligence differs from the deterministic detectors.

## Leakage auditing

**Data leakage**:
Information from evaluation data, future observations, or the label itself becoming available to training, making performance appear better than the model's real-world ability.
_Avoid_: model error, overfitting

**Finding**:
A single deterministic detector result that names a rule, a severity, and the line or column it observed.
_Avoid_: alert, warning, AI verdict

**Trust score**:
A scalar summary of deterministic findings that falls as severity-weighted findings accumulate.
_Avoid_: confidence, accuracy, model probability

**Deterministic finding**:
A scanner result derived from an inspectable rule, with its severity and trust-score contribution fixed by that rule.
_Avoid_: learned score, hypothesis

**Target leakage**:
A feature or column that contains or directly proxies the label being predicted, so the model mostly learns a copy of the answer.
_Avoid_: strong feature, predictive column

**Split overlap**:
Rows from the evaluation partition appearing in, or being nearly identical to rows in, the training partition.
_Avoid_: duplicate data, memorisation failure

**Preprocessing before split**:
A transformation that is fitted on the full data, or on evaluation data, before the split that is supposed to keep evaluation data unseen.
_Avoid_: early transform, fit error

## Learned intelligence

**AI hypothesis**:
An experimental, evidence-bound inference about a possible leakage pattern that never replaces a deterministic finding or changes the trust score.
_Avoid_: finding, verdict, score

**Feature catalogue**:
The authoritative registry of indicator names, their order, and which indicators are eligible for the learned model.
_Avoid_: feature list, name table

**Feature vector**:
The ordered numeric encoding of the indicators for one source unit, positionally aligned with the feature catalogue.
_Avoid_: token vector, embedding

**Model eligibility**:
Whether an emitted indicator is allowed to influence the learned model, separate from whether it is useful diagnostically.
_Avoid_: excluded feature, dead feature

**Semantic risk indicator**:
A context-aware indicator that identifies a concrete leakage path rather than only the presence of a common construct.
_Avoid_: shape signal, raw feature

**Local contribution**:
The direction and size of one feature's influence on one unit's learned probability.
_Avoid_: global importance, causal proof

## Evaluation evidence

**Case**:
One labeled source unit used by a benchmark, with its expected leakage verdict.
_Avoid_: sample, test point, example

**Locked test**:
A benchmark set that is scored once after selection and must not be used for training or tuning.
_Avoid_: validation set, holdout, test data

**Holdout**:
Data preserved from model selection to provide an untouched later check; in LeakLens, the holdout may be authored, real-world, or both.
_Avoid_: test set, validation set

**Prerecorded fixture**:
A static website result generated from the real scanners at build time and replayed in the browser without executing Python.
_Avoid_: live scan, simulation, mock result

**Live scan**:
A real audit performed by a connected LeakLens backend, never inferred from a static fixture.
_Avoid_: recorded scan, demo result
