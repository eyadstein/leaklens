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


def test_report_without_intelligence_remains_compatible():
    report = render({"trust_score": 100, "findings": []})

    assert "100/100" in report
    assert "Intelligence analysis" not in report
