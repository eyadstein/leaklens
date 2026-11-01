from leaklens.report import render


def test_report_renders_intelligence_sections():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "intelligence": {
                "confidence": {
                    "score": 0.85,
                    "level": "high",
                    "reasons": ["Two supporting observations"],
                },
                "root_causes": [
                    {
                        "category": "preprocessing_order",
                        "title": "Preprocessing before split",
                        "explanation": "The transformation may have seen held-out data.",
                    }
                ],
                "recommendations": [
                    {
                        "title": "Move preprocessing into pipeline",
                        "description": "Fit transformations only on training data.",
                        "priority": "high",
                        "actions": ["Split before fitting"],
                        "validation_steps": ["Check held-out data is never fitted"],
                    }
                ],
            },
        }
    )

    assert "Intelligence analysis" in report
    assert "Root-cause analysis" in report
    assert "Remediation recommendations" in report
    assert "Two supporting observations" in report
    assert "Move preprocessing into pipeline" in report


def test_intelligence_report_escapes_untrusted_text():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "intelligence": {
                "confidence": {
                    "score": 0.5,
                    "level": "moderate",
                    "reasons": ["<script>alert(1)</script>"],
                },
                "root_causes": [
                    {
                        "category": "test",
                        "title": "<img src=x onerror=alert(1)>",
                        "explanation": "<script>bad()</script>",
                    }
                ],
                "recommendations": [
                    {
                        "title": "<b>Unsafe</b>",
                        "description": "Unsafe <tag>",
                        "priority": "high",
                        "actions": ["<script>bad()</script>"],
                        "validation_steps": ["<img src=x>"],
                    }
                ],
            },
        }
    )

    assert "<script>" not in report
    assert "<img src=x" not in report
    assert "&lt;script&gt;" in report
    assert "&lt;img" in report


def test_report_renders_dataset_impact_and_escapes_dataset_values():
    report = render({
        "trust_score": 80,
        "findings": [],
        "impact": {
            "train_rows": 100,
            "test_rows": 20,
            "duplicate_test_rows": 2,
            "duplicate_test_pct": 10.0,
            "group_column": "<script>bad()</script>",
            "shared_group_count": 1,
            "shared_group_test_rows": 3,
            "shared_group_test_pct": 15.0,
            "time_column": "event_time",
            "temporal_overlap": True,
            "train_max_time": "2025-01-05T00:00:00",
            "test_min_time": "2025-01-04T00:00:00",
        },
    })

    assert "Dataset impact" in report
    assert "Duplicate test rows" in report
    assert "Shared group values" in report
    assert "Temporal boundary overlap" in report
    assert "&lt;script&gt;bad()&lt;/script&gt;" in report
    assert "<script>bad()</script>" not in report


def test_report_without_intelligence_remains_compatible():
    report = render({"trust_score": 100, "findings": []})

    assert "100/100" in report
    assert "Intelligence analysis" not in report
