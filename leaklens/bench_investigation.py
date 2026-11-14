"""Synthetic benchmark for the Phase 18 investigation plan and its narration.

Cases are fixed and deterministic so this guards against regressions instead of
estimating real-world accuracy. It also proves the M4 invariant: a plan with
narration removed is byte-identical to the plan built with narration disabled.

Usage: ``python -m leaklens.bench_investigation``
"""

from __future__ import annotations

import json

from .ai.reasoning import narrate_investigation
from .intelligence import build_investigation
from .leakage import Finding


class _Echo:
    """Deterministic stand-in for a completion provider: rephrases, never adds."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        plan = json.loads(prompt[prompt.index('{"root_causes"') :].strip())
        return json.dumps(
            {
                "summary": "Deterministic findings grouped into investigation steps.",
                "notes": [
                    {
                        "group_id": group["group_id"],
                        "explanation": "Rephrased from the cited evidence only.",
                    }
                    for group in plan["root_causes"]
                ],
            }
        )


def _cases() -> list[tuple[str, list[Finding]]]:
    return [
        (
            "single target leak",
            [Finding("high", "target_leakage", "amount", "copied target")],
        ),
        (
            "identifier cluster",
            [
                Finding("medium", "id_like", "row_id", "sequential identifier"),
                Finding("medium", "id_like", "user_key", "sequential identifier"),
            ],
        ),
        (
            "mixed severities",
            [
                Finding("high", "target_leakage", "amount", "copied target"),
                Finding("low", "categorical_target_leakage", "city", "rare categories"),
                Finding("medium", "group_missing", "group", "groups absent"),
            ],
        ),
        (
            "unknown detector",
            [Finding("high", "deliberately_unknown_kind", "x", "unregistered")],
        ),
        ("clean scan", []),
    ]


def main(argv: list[str] | None = None) -> int:
    rows = []
    for name, findings in _cases():
        plan = build_investigation(findings)
        narrated = narrate_investigation(plan, _Echo())
        removed = narrated.without_narration()
        rows.append(
            {
                "case": name,
                "findings": plan.finding_count,
                "steps": len(plan.steps),
                "tests": sum(
                    1 for step in plan.steps if step.step_type == "suggested_test"
                ),
                "omitted_tests": len(plan.omitted_tests),
                "narration": narrated.narration,
                "identical": json.dumps(removed.to_dict(), sort_keys=True)
                == json.dumps(plan.to_dict(), sort_keys=True),
            }
        )

    width = max(len(row["case"]) for row in rows)
    print(f"{'case':{width}}  findings  steps  tests  omitted  narration  identical")
    for row in rows:
        print(
            f"{row['case']:{width}}  {row['findings']:>8}  {row['steps']:>5}  "
            f"{row['tests']:>5}  {row['omitted_tests']:>7}  {row['narration']:<11}  "
            f"{'yes' if row['identical'] else 'NO'}"
        )
    bad = [row["case"] for row in rows if not row["identical"]]
    print(
        f"\n{len(rows)} cases, {sum(row['steps'] for row in rows)} steps, "
        f"{sum(row['omitted_tests'] for row in rows)} omitted tests, "
        f"{len(bad)} narration mismatches"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
