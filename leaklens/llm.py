from __future__ import annotations

import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Raised when an explanation provider cannot return a response."""


class ExplanationProvider(Protocol):
    def explain(self, finding) -> str: ...


def _prompt(finding) -> str:
    return (
        "Explain this machine-learning data leakage finding in plain language, "
        "then suggest a concise remediation and a minimal illustrative patch or code change. "
        "Be specific about what is known, and do not invent source-code context. "
        "The patch is a suggestion and may need adaptation to the user's code.\n\n"
        f"Severity: {finding.severity}\n"
        f"Finding: {finding.kind}\n"
        f"Column: {finding.column}\n"
        f"Details: {finding.detail}"
    )


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b", timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def explain(self, finding) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _prompt(finding)}],
            "stream": False,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        try:
            return result["message"]["content"].strip()
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMError("Ollama returned an invalid chat response") from exc


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def explain(self, finding) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": _prompt(finding)}],
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise LLMError(f"LLM API request failed: {exc}") from exc
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("LLM API returned an invalid chat response") from exc


def create_provider(
    name: str = "ollama",
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ExplanationProvider:
    """Build a provider only when the caller explicitly enables LLM explanations."""
    if name == "ollama":
        return OllamaProvider(
            base_url=base_url or os.getenv("LEAKLENS_LLM_BASE_URL", "http://localhost:11434"),
            model=model or os.getenv("LEAKLENS_LLM_MODEL", "qwen2.5:7b"),
        )
    if name == "openai-compatible":
        endpoint = base_url or os.getenv("LEAKLENS_LLM_BASE_URL")
        if not endpoint:
            raise LLMError("--llm-base-url or LEAKLENS_LLM_BASE_URL is required for this provider")
        key = api_key or os.getenv("LEAKLENS_LLM_API_KEY")
        if not key:
            raise LLMError("LEAKLENS_LLM_API_KEY is required for the openai-compatible provider")
        selected_model = model or os.getenv("LEAKLENS_LLM_MODEL", "gpt-4o-mini")
        return OpenAICompatibleProvider(endpoint, selected_model, key)
    raise LLMError(f"unknown LLM provider: {name}")


def explain_findings(findings, provider: ExplanationProvider) -> list[str]:
    """Return one provider explanation per finding, using the injectable provider interface."""
    return [provider.explain(finding) for finding in findings]
