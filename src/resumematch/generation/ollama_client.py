"""Minimal client for Ollama's local REST API. No SDK dependency:
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


DEFAULT_MODEL = "llama3.2:3b"


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        # OLLAMA_HOST / OLLAMA_MODEL from the environment (.env), else the
        # defaults below. The default model is the one every measurement
        # used and the README tells people to pull; it used to be
        # llama3.1:8b, so a run without OLLAMA_MODEL set asked for a model
        # nobody had pulled (see docs/engineering-log.md, model-default).
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        num_ctx: int = 8192,
        schema: dict | None = None,
    ) -> str:
        """
        Calls POST {host}/api/generate (non-streaming) and returns the
        response text.

        num_ctx is set explicitly because Ollama's default context window
        (4,096 tokens here) is close to what one gap-analysis prompt
        needs (~3,100 prompt + output), and Ollama drops prompt text that
        doesn't fit WITHOUT raising -- the resume could vanish from the
        prompt silently.

        schema (a JSON Schema dict) turns on constrained decoding: Ollama
        compiles it to a grammar and only lets the model emit tokens that
        keep the output valid against it, so the response is always
        schema-shaped JSON (list lengths, integer ranges and required
        fields included) unless it is cut off by max_tokens.

        Raises:
            GenerationTruncatedError: output hit max_tokens (incomplete).
            requests.ConnectionError: if Ollama isn't running -- the UI
                catches this and shows an "is Ollama running?" message.
        """
        payload = {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        if body.get("done_reason") == "length":
            raise GenerationTruncatedError(body["response"], max_tokens)
        return body["response"]
