# Missingness audit

Missing values are not automatically a leak, so the audit only reports patterns where *why* a
value is missing can carry information that is unavailable at prediction time.

| Kind | Severity | Signal |
|---|---|---|
| `missingness_target_signal` | medium | missingness is associated with the target (a point-biserial correlation for numeric targets, or a missing-rate difference between target classes with `z >= 3`) |
| `missingness_split_shift` | medium | missingness differs substantially between the training and test splits, including a feature populated in one split and absent from the other |
| `missingness_group_signal` | low | missingness differs across `--group` entities |

Every check needs a minimum number of rows, missing values and present values, plus a minimum
effect size. Insufficient samples, constant columns, entirely missing columns and missing split
metadata are reported as skips rather than findings. Evidence contains only aggregate rates and
counts, never raw rows or group identifiers.

Statistics are deterministic, use pandas/numpy only (no SciPy), and are linear-time per column.
Findings are a reason to investigate, not proof of leakage: severity never exceeds `medium`, the
data and splits are never modified, and the default exit behavior is unchanged (no missingness
finding can trigger the high-severity gate).

The labeled regression benchmark and its limits are in
[MISSINGNESS_BENCHMARK.md](MISSINGNESS_BENCHMARK.md).
