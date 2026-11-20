from pathlib import Path

from leaklens import llm
from leaklens.leakage import Finding

FIXTURES = Path(__file__).parent / "fixtures" / "prompts"


def test_default_prompt_matches_golden_file():
    finding = Finding(
        "high",
        "preprocess_before_split",
        "customer_id",
        "scaler fitted before split",
    )

    expected = (
        (FIXTURES / "explain_default.txt").read_text(encoding="utf-8").rstrip("\n")
    )

    assert llm._prompt(finding) == expected


def test_custom_prompt_matches_golden_file():
    finding = Finding("medium", "id_like", "customer_id", "98% unique values")
    template = (
        "Review {kind} at {location}.\n"
        'JSON: {"fix": "use a pipeline"}\n'
        "Details: {detail}"
    )

    expected = (
        (FIXTURES / "explain_custom.txt").read_text(encoding="utf-8").rstrip("\n")
    )

    assert llm._prompt(finding, template=template) == expected
