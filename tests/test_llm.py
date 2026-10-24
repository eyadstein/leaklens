import json

from leaklens import cli, llm
from leaklens.leakage import Finding


class FakeProvider:
    def __init__(self, response="plain-language explanation and remediation"):
        self.response = response
        self.findings = []

    def explain(self, finding):
        self.findings.append(finding)
        return self.response


def test_fake_provider_explains_finding():
    finding = Finding("medium", "id_like", "customer_id", "98% unique values")
    provider = FakeProvider()

    assert llm.explain_findings([finding], provider) == [
        "plain-language explanation and remediation"
    ]
    assert provider.findings == [finding]


def test_cli_uses_provider_only_when_llm_is_enabled(monkeypatch, capsys):
    finding = Finding("medium", "id_like", "customer_id", "98% unique values")
    monkeypatch.setattr(cli, "scan_path", lambda _path: [finding])
    provider = FakeProvider("IDs can let a model memorize rows; remove the identifier.")

    def build_provider(*_args):
        return provider

    monkeypatch.setattr(llm, "create_provider", build_provider)

    assert cli.main(["--scan", "sample.py", "--llm", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["findings"][0]["llm_explanation"] == provider.response
    assert provider.findings == [finding]

    def unexpected_provider(*_args):
        raise AssertionError("provider must not be created without --llm")

    monkeypatch.setattr(llm, "create_provider", unexpected_provider)
    assert cli.main(["--scan", "sample.py", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "llm_explanation" not in report["findings"][0]
