"""Minimal client for Ollama's local REST API. No SDK dependency —
Ollama's HTTP API is small enough that `requests` is simpler than
pulling in another library for this.
"""

import os

import requests


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        # Falls back to .env.example's OLLAMA_HOST/OLLAMA_MODEL so the
        # local (llama3.2:3b) vs. GCP (llama3.1:8b) profiles actually
        # take effect without every call site having to pass them.
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
        """
        Calls POST {host}/api/generate (non-streaming) and returns the
        response text.

        Raises:
            requests.ConnectionError: if Ollama isn't running locally —
                catch this at the call site and give the user a clear
                "is Ollama running?" message, don't let it bubble up as
                a cryptic traceback.
        """
        resp = requests.post(f"{self.host}/api/generate", json={
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
