"""The client's defaults must match what the README tells people to run."""

from resumematch.generation.ollama_client import DEFAULT_MODEL, OllamaClient


def test_default_model_is_the_documented_one(monkeypatch):
    # Real bug: the default was llama3.1:8b, so running without OLLAMA_MODEL
    # set asked Ollama for a model the setup steps never pulled.
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert OllamaClient().model == DEFAULT_MODEL == "llama3.2:3b"


def test_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
    assert OllamaClient().model == "llama3.1:8b"
