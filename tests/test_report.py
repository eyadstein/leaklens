from leaklens.cli import main
from leaklens.report import render


def test_render_escapes_and_scores():
    rep = {
        "trust_score": 50,
        "findings": [{"severity": "high", "kind": "k", "column": "<b>", "detail": "d"}],
    }
    h = render(rep)
    assert (
        "50/100" in h
        and "&lt;b&gt;" in h
        and "<b>" not in h.split("<h2>Findings</h2>")[1]
    )


def test_cli_html(tmp_path):
    g = tmp_path / "g.py"
    g.write_text("x = 1\n", encoding="utf-8")
    o = tmp_path / "r.html"
    assert main(["--scan", str(g), "--html", str(o)]) == 0
    assert "100/100" in o.read_text(encoding="utf-8")


def test_html_report_renders_policy_and_status():
    from leaklens.baseline import Baseline
    from leaklens.leakage import Finding
    from leaklens.policy import apply_policy
    from leaklens.suppression import parse_rules

    finding = Finding("high", "target_leakage", "leak", "predicts target")
    outcome = apply_policy(
        [finding],
        baseline=Baseline.from_findings([finding]),
        rules=parse_rules(
            [{"fingerprint": finding.fingerprint, "reason": "<reviewed>"}]
        ),
    )
    rep = {
        "trust_score": 100,
        "findings": [outcome.finding_payload(finding)],
        "policy": outcome.summary(baseline_path="base.json", fail_on_new=True),
    }
    page = render(rep)
    assert "Baseline and suppression policy" in page
    assert "<td>suppressed</td>" in page
    assert "&lt;reviewed&gt;" in page
    assert "<reviewed>" not in page
    assert "fail when new findings appear" in page


def test_html_report_has_severity_legend():
    rep = {
        "trust_score": 75,
        "findings": [
            {
                "severity": "medium",
                "kind": "potential_leak",
                "column": "a",
                "detail": "check",
            }
        ],
    }
    page = render(rep)
    assert 'aria-label="Severity legend"' in page
    assert '<strong class="high">High</strong>: significant leakage risk' in page
    assert '<strong class="medium">Medium</strong>: possible leakage risk' in page
    assert '<strong class="low">Low</strong>: minor or informational finding' in page
    assert page.index("Severity legend:") < page.index("<table>")


def test_html_report_severity_legend_without_findings():
    page = render({"trust_score": 100, "findings": []})
    assert "Severity legend:" in page
    assert "No leakage found." in page
