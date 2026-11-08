"""Minimal client for Ollama's local REST API. No SDK dependency —
Ollama's HTTP API is small enough that `requests` is simpler than
pulling in another library for this.
"""

import os

import requests


class GenerationTruncatedError(ValueError):
    """The model hit the output-token cap mid-answer (Ollama done_reason
    "length"). A ValueError so callers handle it like any other
    unusable response; the partial text is kept for debugging."""

    def __init__(self, partial_response: str, max_tokens: int):
        super().__init__(
            f"Model output was cut off at the {max_tokens}-token cap before finishing."
        )
        self.partial_response = partial_response


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        # Falls back to .env.example's OLLAMA_HOST/OLLAMA_MODEL so the
        # local (llama3.2:3b) vs. GCP (llama3.1:8b) profiles actually
        # take effect without every call site having to pass them.
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 800, num_ctx: int = 8192) -> str:
        """
        Calls POST {host}/api/generate (non-streaming) and returns the
        response text.

        num_ctx is set explicitly because Ollama's default context window
        (4,096 tokens here) is close to what one gap-analysis prompt
        needs (~3,100 prompt + output), and Ollama drops prompt text that
        doesn't fit WITHOUT raising -- the resume could vanish from the
        prompt silently.

        Raises:
            GenerationTruncatedError: output hit max_tokens (incomplete).
            requests.ConnectionError: if Ollama isn't running -- the UI
                catches this and shows an "is Ollama running?" message.
        """
        resp = requests.post(f"{self.host}/api/generate", json={
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx},
        }, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        if body.get("done_reason") == "length":
            raise GenerationTruncatedError(body["response"], max_tokens)
        return body["response"]
