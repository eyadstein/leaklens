import json

import pytest

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


def test_prompt_redacts_absolute_paths_but_keeps_filename_line_and_detail():
    location = r"C:\Users\alice\private-project\notebook.ipynb[cell 2]:14"
    finding = Finding(
        "high",
        "preprocess_before_split",
        location,
        "StandardScaler.fit_transform(X) runs before the split at /Users/alice/private project/notebook.ipynb[cell 2]:8",
    )

    prompt = llm._prompt(finding)

    assert r"C:\Users\alice\private-project" not in prompt
    assert "/Users/alice/private-project" not in prompt
    assert "/Users/alice/private project" not in prompt
    assert "notebook.ipynb[cell 2]:14" in prompt
    assert "notebook.ipynb[cell 2]:8" in prompt
    assert "StandardScaler.fit_transform(X)" in prompt


def test_explanations_are_capped_by_severity_and_sanitized():
    low = Finding("low", "parse_error", "source.py:3", "parse warning")
    high = Finding("high", "preprocess_before_split", "source.py:8", "leak")
    medium = Finding("medium", "fit_on_full_data", "source.py:10", "refit")
    provider = FakeProvider("safe\x1b[31m output\x07")

    explanations = llm.explain_findings([low, high, medium], provider, max_findings=2)

    assert provider.findings == [high, medium]
    assert explanations == [None, "safe output", "safe output"]


def test_cli_reports_truncation_and_keeps_llm_text_out_of_sarif(monkeypatch, capsys, tmp_path):
    high = Finding("high", "preprocess_before_split", "model.py:8", "leak")
    low = Finding("low", "parse_error", "data.py:2", "parse warning")
    monkeypatch.setattr(cli, "scan_path", lambda _path: [low, high])
    monkeypatch.setattr(llm, "create_provider", lambda *_args: FakeProvider("suggestion"))

    sarif_path = tmp_path / "report.sarif"
    assert cli.main(
        [
            "--scan",
            "sample.py",
            "--llm",
            "--llm-max-findings",
            "1",
            "--json",
            "--sarif",
            str(sarif_path),
        ]
    ) == 1

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert "llm_explanation" not in report["findings"][0]
    assert report["findings"][1]["llm_explanation"] == "suggestion"
    assert "skipped 1 finding(s)" in captured.err
    assert "llm_explanation" not in sarif_path.read_text(encoding="utf-8")


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


def test_provider_rejects_non_http_urls():
    with pytest.raises(llm.LLMError, match="http"):
        llm.create_provider("openai-compatible", "ftp://api.example.com/v1", api_key="secret")


def test_openai_compatible_provider_refuses_api_key_over_remote_http():
    with pytest.raises(llm.LLMError, match="plain HTTP"):
        llm.create_provider("openai-compatible", "http://api.example.com/v1", api_key="secret")


def test_openai_compatible_provider_allows_https_and_local_http():
    secure = llm.create_provider("openai-compatible", "https://api.example.com/v1", api_key="secret")
    local = llm.create_provider("openai-compatible", "http://localhost:8000/v1", api_key="secret")

    assert isinstance(secure, llm.OpenAICompatibleProvider)
    assert isinstance(local, llm.OpenAICompatibleProvider)



def test_openai_provider_refuses_redirects_without_following_them(monkeypatch):
    from urllib.error import HTTPError

    finding = Finding("high", "id_like", "data.csv:2", "identifier")
    redirects = []

    class FakeOpener:
        def __init__(self, handlers):
            self.handler = handlers[0]()

        def open(self, request, timeout):
            redirects.append(
                self.handler.redirect_request(
                    request,
                    None,
                    307,
                    "Temporary Redirect",
                    {},
                    "https://attacker.example/collect",
                )
            )
            raise HTTPError(request.full_url, 307, "Temporary Redirect", {}, None)

    monkeypatch.setattr(llm, "build_opener", lambda *handlers: FakeOpener(handlers))
    provider = llm.OpenAICompatibleProvider(
        "https://api.example.com/v1", "test-model", "secret-key"
    )

    with pytest.raises(llm.LLMError, match="request failed"):
        provider.explain(finding)

    assert redirects == [None]
